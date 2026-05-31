/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Open-loop NR-V2X experiment driver for MATLAB/DeePC data and baselines.
 *
 * This scratch program is intentionally open-loop only:
 * - PRBS input scheduling can drive Tx power / beacon interval for Hankel data.
 * - Fixed Tx power / beacon interval can be used for open-loop sanity baselines.
 * - No MATLAB DeePC bridge is present in this file.
 * - Use nr_v2x_ngsim_deepc_closed_loop_deepc for closed-loop DeePC runs.
 */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/btp.h"
#include "ns3/caBasicService.h"
#include "ns3/core-module.h"
#include "ns3/geonet.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/nr-sl-ue-rrc.h"
#include "ns3/point-to-point-module.h"
#include "ns3/tag.h"
#include "ns3/vdp.h"

#include "../src/cbr-logger.cc"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NrV2xNgsimDeepc");

struct MobilityRow
{
    double timeS;
    uint32_t vehicleId;
    double xM;
    double yM;
    double vxMps;
    double vyMps;
    double speedMps;
    uint32_t laneId;
};

struct InputCommand
{
    double timeS;
    double txPowerDbm;
    double beaconIntervalS;
};

struct TxEvent
{
    double timeS;
    uint32_t txNodeId;
    uint64_t seq;
    uint32_t eligibleRxCount;
    std::map<uint32_t, double> rxDistanceAtTxM;
};

struct RxEvent
{
    double timeS;
    uint32_t txNodeId;
    uint32_t rxNodeId;
    uint64_t seq;
    double pirS;
};

class KpiPacketTag : public Tag
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::KpiPacketTag")
                                .SetParent<Tag>()
                                .SetGroupName("Applications")
                                .AddConstructor<KpiPacketTag>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return 16;
    }

    void Serialize(TagBuffer i) const override
    {
        i.WriteU64(m_sequence);
        i.WriteU64(static_cast<uint64_t>(m_txTimeNs));
    }

    void Deserialize(TagBuffer i) override
    {
        m_sequence = i.ReadU64();
        m_txTimeNs = static_cast<int64_t>(i.ReadU64());
    }

    void Print(std::ostream& os) const override
    {
        os << "sequence=" << m_sequence << ",txTimeNs=" << m_txTimeNs;
    }

    void SetSequence(uint64_t sequence)
    {
        m_sequence = sequence;
    }

    uint64_t GetSequence() const
    {
        return m_sequence;
    }

    void SetTxTimeNs(int64_t txTimeNs)
    {
        m_txTimeNs = txTimeNs;
    }

    int64_t GetTxTimeNs() const
    {
        return m_txTimeNs;
    }

  private:
    uint64_t m_sequence = 0;
    int64_t m_txTimeNs = 0;
};

static std::map<uint32_t, std::vector<MobilityRow>> g_mobilityByVehicle;
static std::vector<InputCommand> g_inputSchedule;

static std::map<uint32_t, uint32_t> g_vehicleIdToNodeId;
static std::map<uint32_t, uint32_t> g_nodeIdToVehicleId;
static std::map<uint32_t, Ptr<Node>> g_nodeIdToNode;
static std::map<uint32_t, std::pair<double, double>> g_nodeIdToActiveTimeRangeS;

/****************************************************************
 * Helpers
 ****************************************************************/

static double
Distance2d(const Vector& a, const Vector& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kDotOneMicro = 1e7;
static constexpr double kLocalOriginLatitudeDeg = 37.0;
static constexpr double kLocalOriginLongitudeDeg = -122.0;

static long
ClampToLong(double value, long minValue, long maxValue)
{
    const double clamped = std::max(static_cast<double>(minValue),
                                    std::min(static_cast<double>(maxValue), value));
    return static_cast<long>(std::llround(clamped));
}

static long
EncodeLatitudeFromLocalY(double yM)
{
    const double latDeg = kLocalOriginLatitudeDeg + yM / 111320.0;
    return ClampToLong(latDeg * kDotOneMicro, -900000000, 900000000);
}

static long
EncodeLongitudeFromLocalX(double xM)
{
    const double originLatRad = kLocalOriginLatitudeDeg * kPi / 180.0;
    const double metersPerDegLon = 111320.0 * std::cos(originLatRad);
    const double lonDeg = kLocalOriginLongitudeDeg + xM / metersPerDegLon;
    return ClampToLong(lonDeg * kDotOneMicro, -1799999999, 1800000000);
}

static bool
IsInsideCoreRegion(double coordinate, double coreMin, double coreMax)
{
    return coordinate >= coreMin && coordinate <= coreMax;
}

static bool
IsNodeActiveAtTime(uint32_t nodeId, double timeS)
{
    auto it = g_nodeIdToActiveTimeRangeS.find(nodeId);
    if (it == g_nodeIdToActiveTimeRangeS.end())
    {
        return true;
    }
    constexpr double eps = 1e-9;
    return timeS + eps >= it->second.first && timeS <= it->second.second + eps;
}

static std::pair<double, double>
ComputeMobilityTimeRange(const std::vector<MobilityRow>& rows)
{
    double minTimeS = std::numeric_limits<double>::infinity();
    double maxTimeS = -std::numeric_limits<double>::infinity();
    for (const auto& row : rows)
    {
        minTimeS = std::min(minTimeS, row.timeS);
        maxTimeS = std::max(maxTimeS, row.timeS);
    }
    NS_ABORT_MSG_IF(!std::isfinite(minTimeS) || !std::isfinite(maxTimeS),
                    "Cannot compute active time range from empty mobility track");
    return {minTimeS, maxTimeS};
}

static std::pair<double, double>
ComputeMobilityTimeRangeForVehicles(const std::vector<uint32_t>& vehicleIds)
{
    double minTimeS = std::numeric_limits<double>::infinity();
    double maxTimeS = -std::numeric_limits<double>::infinity();
    for (uint32_t vehicleId : vehicleIds)
    {
        const auto range = ComputeMobilityTimeRange(g_mobilityByVehicle[vehicleId]);
        minTimeS = std::min(minTimeS, range.first);
        maxTimeS = std::max(maxTimeS, range.second);
    }
    NS_ABORT_MSG_IF(!std::isfinite(minTimeS) || !std::isfinite(maxTimeS),
                    "Cannot compute selected mobility coverage from empty vehicle set");
    return {minTimeS, maxTimeS};
}

static std::pair<double, double>
ComputeInputScheduleTimeRange()
{
    double minTimeS = std::numeric_limits<double>::infinity();
    double maxTimeS = -std::numeric_limits<double>::infinity();
    for (const auto& input : g_inputSchedule)
    {
        minTimeS = std::min(minTimeS, input.timeS);
        maxTimeS = std::max(maxTimeS, input.timeS);
    }
    NS_ABORT_MSG_IF(!std::isfinite(minTimeS) || !std::isfinite(maxTimeS),
                    "Cannot compute input schedule coverage from empty input schedule");
    return {minTimeS, maxTimeS};
}

static void
ValidateTenMinuteDataCoverage(double simTimeSeconds,
                              double evalStartS,
                              double evalEndS,
                              bool useInputSchedule,
                              bool requireFullDataCoverage,
                              const std::vector<uint32_t>& vehicleIds)
{
    if (!requireFullDataCoverage)
    {
        return;
    }

    constexpr double eps = 1e-6;
    const auto mobilityRange = ComputeMobilityTimeRangeForVehicles(vehicleIds);
    NS_ABORT_MSG_IF(mobilityRange.first > evalStartS + eps,
                    "Mobility CSV starts at " << mobilityRange.first
                                             << " s, but KPI evaluation starts at "
                                             << evalStartS
                                             << " s. Regenerate mobility with an earlier "
                                                "--time-start or increase --warmup.");
    NS_ABORT_MSG_IF(evalEndS >= 0.0 && mobilityRange.second + eps < evalEndS,
                    "Mobility CSV ends at " << mobilityRange.second
                                           << " s, but KPI evaluation ends at " << evalEndS
                                           << " s. Regenerate mobility with a longer "
                                              "--time-end or reduce --simTime.");

    if (useInputSchedule)
    {
        const auto inputRange = ComputeInputScheduleTimeRange();
        NS_ABORT_MSG_IF(inputRange.first > eps,
                        "Input schedule starts at " << inputRange.first
                                                   << " s, but the simulation starts at 0 s.");
        NS_ABORT_MSG_IF(inputRange.second + eps < simTimeSeconds,
                        "Input schedule ends at " << inputRange.second
                                                 << " s, but simTime is " << simTimeSeconds
                                                 << " s. Regenerate PRBS inputs with "
                                                    "--duration at least simTime.");
    }
}

static std::pair<double, double>
ComputeStudyBounds(const std::string& axis)
{
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();

    for (const auto& kv : g_mobilityByVehicle)
    {
        for (const auto& row : kv.second)
        {
            const double value = (axis == "x") ? row.xM : row.yM;
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
    }

    NS_ABORT_MSG_IF(!std::isfinite(minValue) || !std::isfinite(maxValue),
                    "Cannot compute study-area limits from empty mobility data");
    return {minValue, maxValue};
}

static std::string
ResolveCoreAxis(const std::string& requestedAxis)
{
    if (requestedAxis == "x" || requestedAxis == "y")
    {
        return requestedAxis;
    }
    NS_ABORT_MSG_IF(requestedAxis != "auto", "coreAxis must be auto, x, or y");

    const auto xBounds = ComputeStudyBounds("x");
    const auto yBounds = ComputeStudyBounds("y");
    const double xSpan = xBounds.second - xBounds.first;
    const double ySpan = yBounds.second - yBounds.first;
    return (ySpan > xSpan) ? "y" : "x";
}

static double
GetCoreCoordinate(const Vector& position, const std::string& axis)
{
    return (axis == "x") ? position.x : position.y;
}

static void
LoadMobilityCsv(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        NS_FATAL_ERROR("Cannot open mobility CSV: " << path);
    }

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        MobilityRow row;

        std::getline(ss, token, ',');
        row.timeS = std::stod(token);

        std::getline(ss, token, ',');
        row.vehicleId = static_cast<uint32_t>(std::stoul(token));

        std::getline(ss, token, ',');
        row.xM = std::stod(token);

        std::getline(ss, token, ',');
        row.yM = std::stod(token);

        std::getline(ss, token, ',');
        row.vxMps = std::stod(token);

        std::getline(ss, token, ',');
        row.vyMps = std::stod(token);

        std::getline(ss, token, ',');
        row.speedMps = std::stod(token);

        std::getline(ss, token, ',');
        row.laneId = static_cast<uint32_t>(std::stoul(token));

        g_mobilityByVehicle[row.vehicleId].push_back(row);
    }

    std::cout << "Loaded vehicles from mobility CSV: " << g_mobilityByVehicle.size() << std::endl;
}

static void
LoadInputScheduleCsv(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        NS_FATAL_ERROR("Cannot open input schedule CSV: " << path);
    }

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        InputCommand u;

        std::getline(ss, token, ',');
        u.timeS = std::stod(token);

        std::getline(ss, token, ',');
        u.txPowerDbm = std::stod(token);

        std::getline(ss, token, ',');
        u.beaconIntervalS = std::stod(token);

        g_inputSchedule.push_back(u);
    }

    std::cout << "Loaded PRBS input steps: " << g_inputSchedule.size() << std::endl;
}

static void
InstallWaypointMobility(Ptr<Node> node, const std::vector<MobilityRow>& rows)
{
    NS_ABORT_MSG_IF(rows.empty(), "Cannot install waypoint mobility from empty NGSIM track");

    Ptr<WaypointMobilityModel> mob = CreateObject<WaypointMobilityModel>();
    node->AggregateObject(mob);

    const Vector inactiveParkingPosition(
        -10000.0 - static_cast<double>(node->GetId()),
        -10000.0,
        0.0);
    constexpr double waypointEpsS = 1e-6;

    if (rows.front().timeS > 0.0)
    {
        mob->AddWaypoint(Waypoint(Seconds(0.0), inactiveParkingPosition));
        if (rows.front().timeS > waypointEpsS)
        {
            mob->AddWaypoint(Waypoint(Seconds(rows.front().timeS - waypointEpsS),
                                      inactiveParkingPosition));
        }
    }

    for (const auto& r : rows)
    {
        Waypoint wp(Seconds(r.timeS), Vector(r.xM, r.yM, 0.0));
        mob->AddWaypoint(wp);
    }

    mob->AddWaypoint(Waypoint(Seconds(rows.back().timeS + waypointEpsS),
                              inactiveParkingPosition));
}

static void
GetSlBitmapFromString(const std::string& slBitMapString, std::vector<std::bitset<1>>& slBitMapVector)
{
    static std::unordered_map<std::string, uint8_t> lookupTable = {
        {"0", 0},
        {"1", 1},
    };

    std::stringstream ss(slBitMapString);
    std::string token;
    std::vector<std::string> extracted;

    while (std::getline(ss, token, '|'))
    {
        extracted.push_back(token);
    }

    for (const auto& v : extracted)
    {
        if (lookupTable.find(v) == lookupTable.end())
        {
            NS_FATAL_ERROR("Invalid sidelink bitmap token: " << v);
        }
        slBitMapVector.emplace_back(lookupTable[v] & 0x01);
    }
}

/****************************************************************
 * KPI Logger
 *
 * Notes:
 * - PRR within awareness range is implemented.
 * - PIR is implemented.
 * - CBR is supplied by CbrLogger from NR SL sensing, with NrSpectrumPhy::ChannelOccupied fallback.
 ****************************************************************/

class KpiLogger : public Object
{
  public:
    void Configure(const NodeContainer& nodes,
                   CbrLogger* cbrLogger,
                   const std::string& csvPath,
                   double awarenessRangeM,
                   double kpiWindowS,
                   double sampleTimeS,
                   double evalStartS,
                   double evalEndS,
                   bool enableCoreFilter,
                   const std::string& coreAxis,
                   double coreMin,
                   double coreMax)
    {
        m_cbrLogger = cbrLogger;
        m_awarenessRangeM = awarenessRangeM;
        m_kpiWindowS = kpiWindowS;
        m_sampleTimeS = sampleTimeS;
        m_evalStartS = evalStartS;
        m_evalEndS = evalEndS;
        m_enableCoreFilter = enableCoreFilter;
        m_coreAxis = coreAxis;
        m_coreMin = coreMin;
        m_coreMax = coreMax;

        for (uint32_t i = 0; i < nodes.GetN(); ++i)
        {
            Ptr<Node> n = nodes.Get(i);
            m_nodes.push_back(n);
        }
        m_out.open(csvPath.c_str(), std::ios::out | std::ios::trunc);
        if (!m_out.is_open())
        {
            NS_FATAL_ERROR("Cannot open KPI output CSV: " << csvPath);
        }
        m_out << "time_s,tx_power_dbm,beacon_interval_s,"
              << "active_vehicle_count_core,prr_awareness,pir_s,cbr\n";
    }

    void RegisterNodeIpv4(uint32_t nodeId, const Ipv4Address& addr)
    {
        m_ipToNodeId[addr.Get()] = nodeId;
    }

    uint32_t ResolveNodeIdFromIpv4(const Ipv4Address& addr) const
    {
        auto it = m_ipToNodeId.find(addr.Get());
        if (it == m_ipToNodeId.end())
        {
            return std::numeric_limits<uint32_t>::max();
        }
        return it->second;
    }

    void SetCurrentInputs(double pDbm, double tbS)
    {
        m_currentP = pDbm;
        m_currentTb = tbS;
    }

    void LogTx(uint32_t txNodeId, uint64_t seq, Time tTx, const Vector& posTx)
    {
        const double timeS = tTx.GetSeconds();
        if (!IsNodeActiveAtTime(txNodeId, timeS) || !IsInsideEvaluationWindow(timeS) ||
            !IsInsideCore(posTx))
        {
            return;
        }

        std::map<uint32_t, double> rxDistanceAtTxM;

        for (const auto& node : m_nodes)
        {
            if (node->GetId() == txNodeId)
            {
                continue;
            }
            if (!IsNodeActiveAtTime(node->GetId(), timeS))
            {
                continue;
            }

            Vector posRx = node->GetObject<MobilityModel>()->GetPosition();
            if (!IsInsideCore(posRx))
            {
                continue;
            }

            const double distanceM = Distance2d(posTx, posRx);
            if (distanceM <= m_awarenessRangeM)
            {
                rxDistanceAtTxM[node->GetId()] = distanceM;
            }
        }

        TxEvent ev;
        ev.timeS = timeS;
        ev.txNodeId = txNodeId;
        ev.seq = seq;
        ev.eligibleRxCount = static_cast<uint32_t>(rxDistanceAtTxM.size());
        ev.rxDistanceAtTxM = rxDistanceAtTxM;

        m_txEvents.push_back(ev);

        PruneOldEvents();
    }

    void LogRx(uint32_t txNodeId,
           uint32_t rxNodeId,
           uint64_t seq,
           Time tRx,
           const Vector& posRx)
    {
        const double timeS = tRx.GetSeconds();
        if (!IsInsideEvaluationWindow(timeS) || !IsNodeActiveAtTime(txNodeId, timeS) ||
            !IsNodeActiveAtTime(rxNodeId, timeS))
        {
            return;
        }

        const TxEvent* txEvent = FindTxEvent(txNodeId, seq);
        if (!txEvent)
        {
            return;
        }

        if (!IsInsideCore(posRx))
        {
            return;
        }
        auto distIt = txEvent->rxDistanceAtTxM.find(rxNodeId);
        if (distIt == txEvent->rxDistanceAtTxM.end())
        {
            return;
        }

        auto rxTriple = std::make_tuple(txNodeId, rxNodeId, seq);
        if (m_loggedRxTriples.find(rxTriple) != m_loggedRxTriples.end())
        {
            return;
        }
        m_loggedRxTriples.insert(rxTriple);

        double pir = std::numeric_limits<double>::quiet_NaN();
        auto key = std::make_pair(txNodeId, rxNodeId);

        auto it = m_lastRxTimePerPair.find(key);
        if (it != m_lastRxTimePerPair.end())
        {
            pir = tRx.GetSeconds() - it->second;
        }
        m_lastRxTimePerPair[key] = tRx.GetSeconds();

        RxEvent ev;
        ev.timeS = timeS;
        ev.txNodeId = txNodeId;
        ev.rxNodeId = rxNodeId;
        ev.seq = seq;
        ev.pirS = pir;

        m_rxEvents.push_back(ev);

        PruneOldEvents();
    }

    void SampleAndWrite()
    {
        PruneOldEvents();

        const double now = Simulator::Now().GetSeconds();

        uint64_t denom = 0;
        std::set<std::tuple<uint32_t, uint64_t>> activeTxKeys;
        for (const auto& tx : m_txEvents)
        {
            denom += tx.eligibleRxCount;
            activeTxKeys.insert(std::make_tuple(tx.txNodeId, tx.seq));
        }

        std::set<std::tuple<uint32_t, uint32_t, uint64_t>> uniqueRxTriples;
        for (const auto& rx : m_rxEvents)
        {
            if (activeTxKeys.find(std::make_tuple(rx.txNodeId, rx.seq)) != activeTxKeys.end())
            {
                uniqueRxTriples.insert(std::make_tuple(rx.txNodeId, rx.rxNodeId, rx.seq));
            }
        }

        const uint64_t numer = uniqueRxTriples.size();
        const double prr = (denom > 0) ? static_cast<double>(numer) / static_cast<double>(denom) : 0.0;

        double pirMean = 0.0;
        uint64_t pirCount = 0;
        for (const auto& rx : m_rxEvents)
        {
            if (!std::isnan(rx.pirS))
            {
                pirMean += rx.pirS;
                pirCount++;
            }
        }
        pirMean = (pirCount > 0) ? pirMean / static_cast<double>(pirCount) : 0.0;

        const double cbr = m_cbrLogger ? m_cbrLogger->ComputeCbr(now) : 0.0;
        const uint32_t activeVehicleCount = CountActiveCoreVehicles(now);
        MaybePrintProgress(now, activeVehicleCount);

        m_out << std::fixed << std::setprecision(6) << now << ","
              << m_currentP << ","
              << m_currentTb << ","
              << activeVehicleCount << ","
              << prr << ","
              << pirMean << ","
              << cbr << "\n";

        Simulator::Schedule(Seconds(m_sampleTimeS), &KpiLogger::SampleAndWrite, this);
    }

    ~KpiLogger() override
    {
        if (m_out.is_open())
        {
            m_out.close();
        }
    }

  private:
    uint32_t CountActiveCoreVehicles(double timeS) const
    {
        uint32_t count = 0;
        for (const auto& node : m_nodes)
        {
            if (!IsNodeActiveAtTime(node->GetId(), timeS))
            {
                continue;
            }
            const Vector pos = node->GetObject<MobilityModel>()->GetPosition();
            if (IsInsideCore(pos))
            {
                count++;
            }
        }
        return count;
    }

    void MaybePrintProgress(double timeS, uint32_t activeVehicleCount)
    {
        const int64_t second = static_cast<int64_t>(std::floor(timeS + 1e-9));
        if (second >= 1 && second != m_lastProgressSecond)
        {
            std::cout << "t=" << second << ", v=" << activeVehicleCount << std::endl;
            m_lastProgressSecond = second;
        }
    }

    void PruneOldEvents()
    {
        const double now = Simulator::Now().GetSeconds();
        const double cutoff = now - m_kpiWindowS;
        std::set<std::tuple<uint32_t, uint64_t>> activeTxKeys;

        while (!m_txEvents.empty() && m_txEvents.front().timeS < cutoff)
        {
            m_txEvents.pop_front();
        }
        for (const auto& tx : m_txEvents)
        {
            activeTxKeys.insert(std::make_tuple(tx.txNodeId, tx.seq));
        }
        while (!m_rxEvents.empty() && m_rxEvents.front().timeS < cutoff)
        {
            m_loggedRxTriples.erase(std::make_tuple(m_rxEvents.front().txNodeId,
                                                    m_rxEvents.front().rxNodeId,
                                                    m_rxEvents.front().seq));
            m_rxEvents.pop_front();
        }
        m_rxEvents.erase(std::remove_if(m_rxEvents.begin(),
                                        m_rxEvents.end(),
                                        [this, &activeTxKeys](const RxEvent& rx) {
                                            const auto txKey =
                                                std::make_tuple(rx.txNodeId, rx.seq);
                                            if (activeTxKeys.find(txKey) != activeTxKeys.end())
                                            {
                                                return false;
                                            }

                                            m_loggedRxTriples.erase(std::make_tuple(rx.txNodeId,
                                                                                    rx.rxNodeId,
                                                                                    rx.seq));
                                            return true;
                                        }),
                         m_rxEvents.end());
    }

    bool IsInsideEvaluationWindow(double timeS) const
    {
        return timeS >= m_evalStartS && (m_evalEndS < 0.0 || timeS <= m_evalEndS);
    }

    bool IsInsideCore(const Vector& position) const
    {
        return !m_enableCoreFilter ||
               IsInsideCoreRegion(GetCoreCoordinate(position, m_coreAxis), m_coreMin, m_coreMax);
    }

    const TxEvent* FindTxEvent(uint32_t txNodeId, uint64_t seq) const
    {
        for (auto it = m_txEvents.rbegin(); it != m_txEvents.rend(); ++it)
        {
            if (it->txNodeId == txNodeId && it->seq == seq)
            {
                return &(*it);
            }
        }
        return nullptr;
    }


    double m_awarenessRangeM = 150.0;
    double m_kpiWindowS = 1.0;
    double m_sampleTimeS = 0.5;
    double m_evalStartS = 0.0;
    double m_evalEndS = -1.0;
    bool m_enableCoreFilter = true;
    std::string m_coreAxis = "y";
    double m_coreMin = 0.0;
    double m_coreMax = 0.0;
    CbrLogger* m_cbrLogger = nullptr;

    double m_currentP = 23.0;
    double m_currentTb = 0.1;
    int64_t m_lastProgressSecond = -1;

    std::vector<Ptr<Node>> m_nodes;
    std::unordered_map<uint32_t, uint32_t> m_ipToNodeId;

    std::deque<TxEvent> m_txEvents;
    std::deque<RxEvent> m_rxEvents;
    std::set<std::tuple<uint32_t, uint32_t, uint64_t>> m_loggedRxTriples;
    std::map<std::pair<uint32_t, uint32_t>, double> m_lastRxTimePerPair;

    std::ofstream m_out;
};

class NgsimVehicleDataProvider : public VDP
{
  public:
    explicit NgsimVehicleDataProvider(Ptr<Node> node)
        : m_node(node)
    {
        const Vector pos = GetPositionVector();
        m_lastDistancePosition = pos;
        m_lastAccelerationSpeedMps = getSpeedValue();
        m_lastAccelerationTime = Simulator::Now();
    }

    CAM_mandatory_data_t getCAMMandatoryData() override
    {
        CAM_mandatory_data_t data;
        const Vector pos = GetPositionVector();

        data.speed = VDPValueConfidence<>(ClampToLong(getSpeedValue() * 100.0,
                                                      SpeedValue_standstill,
                                                      SpeedValue_outOfRange - 1),
                                          SpeedConfidence_unavailable);
        data.longitude = EncodeLongitudeFromLocalX(pos.x);
        data.latitude = EncodeLatitudeFromLocalY(pos.y);
        data.lane = 0;
        data.altitude = VDPValueConfidence<>(AltitudeValue_unavailable,
                                             AltitudeConfidence_unavailable);
        data.posConfidenceEllipse = {SemiAxisLength_unavailable,
                                     SemiAxisLength_unavailable,
                                     Wgs84AngleValue_unavailable};
        data.longAcceleration = VDPValueConfidence<>(GetLongitudinalAccelerationValue(),
                                                     AccelerationConfidence_unavailable);
        data.heading = VDPValueConfidence<>(ClampToLong(getHeadingValue() * 10.0,
                                                        0,
                                                        HeadingValue_doNotUse - 1),
                                            HeadingConfidence_unavailable);
        data.driveDirection = DriveDirection_forward;
        data.curvature = VDPValueConfidence<>(CurvatureValue_unavailable,
                                              CurvatureConfidence_unavailable);
        data.curvature_calculation_mode = CurvatureCalculationMode_unavailable;
        data.VehicleLength = VDPValueConfidence<long, long>(
            45,
            VehicleLengthConfidenceIndication_noTrailerPresent);
        data.VehicleWidth = 18;
        data.yawRate = VDPValueConfidence<>(YawRateValue_unavailable,
                                            YawRateConfidence_unavailable);
        return data;
    }

    CPM_mandatory_data_t getCPMMandatoryData() override
    {
        const CAM_mandatory_data_t cam = getCAMMandatoryData();
        CPM_mandatory_data_t cpm;
        cpm.speed = cam.speed;
        cpm.longitude = cam.longitude;
        cpm.latitude = cam.latitude;
        cpm.altitude = cam.altitude;
        cpm.posConfidenceEllipse = cam.posConfidenceEllipse;
        cpm.longAcceleration = cam.longAcceleration;
        cpm.heading = cam.heading;
        cpm.driveDirection = cam.driveDirection;
        cpm.curvature = cam.curvature;
        cpm.curvature_calculation_mode = cam.curvature_calculation_mode;
        cpm.VehicleLength = cam.VehicleLength;
        cpm.VehicleWidth = cam.VehicleWidth;
        cpm.yawRate = cam.yawRate;
        return cpm;
    }

    MCM_mandatory_data_t getMCMMandatoryData() override
    {
        const CAM_mandatory_data_t cam = getCAMMandatoryData();
        MCM_mandatory_data_t mcm;
        mcm.speed = cam.speed;
        mcm.longitude = cam.longitude;
        mcm.latitude = cam.latitude;
        mcm.lane = cam.lane;
        mcm.altitude = cam.altitude;
        mcm.posConfidenceEllipse = cam.posConfidenceEllipse;
        mcm.longAcceleration = cam.longAcceleration;
        mcm.heading = cam.heading;
        mcm.driveDirection = cam.driveDirection;
        mcm.curvature = cam.curvature;
        mcm.curvature_calculation_mode = cam.curvature_calculation_mode;
        mcm.VehicleLength = cam.VehicleLength;
        mcm.VehicleWidth = cam.VehicleWidth;
        mcm.yawRate = cam.yawRate;
        return mcm;
    }

    double getSpeedValue() override
    {
        const Vector velocity = GetVelocityVector();
        return std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    }

    double getTravelledDistance() override
    {
        const Vector pos = GetPositionVector();
        m_travelledDistanceM += Distance2d(pos, m_lastDistancePosition);
        m_lastDistancePosition = pos;
        return m_travelledDistanceM;
    }

    double getHeadingValue() override
    {
        const Vector velocity = GetVelocityVector();
        if (std::abs(velocity.x) < 1e-9 && std::abs(velocity.y) < 1e-9)
        {
            return m_lastHeadingDeg;
        }

        double headingDeg = 90.0 - std::atan2(velocity.y, velocity.x) * 180.0 / kPi;
        while (headingDeg < 0.0)
        {
            headingDeg += 360.0;
        }
        while (headingDeg >= 360.0)
        {
            headingDeg -= 360.0;
        }
        m_lastHeadingDeg = headingDeg;
        return headingDeg;
    }

    VDP_position_latlon_t getPosition() override
    {
        const Vector pos = GetPositionVector();
        return {static_cast<double>(EncodeLatitudeFromLocalY(pos.y)) / kDotOneMicro,
                static_cast<double>(EncodeLongitudeFromLocalX(pos.x)) / kDotOneMicro,
                DBL_MAX};
    }

    VDP_position_cartesian_t getPositionXY() override
    {
        const Vector pos = GetPositionVector();
        return {pos.x, pos.y, pos.z};
    }

    VDP_position_cartesian_t getXY(double lon, double lat) override
    {
        const double originLatRad = kLocalOriginLatitudeDeg * kPi / 180.0;
        const double metersPerDegLon = 111320.0 * std::cos(originLatRad);
        return {(lon - kLocalOriginLongitudeDeg) * metersPerDegLon,
                (lat - kLocalOriginLatitudeDeg) * 111320.0,
                DBL_MAX};
    }

    double getCartesianDist(double lon1, double lat1, double lon2, double lat2) override
    {
        const VDP_position_cartesian_t p1 = getXY(lon1, lat1);
        const VDP_position_cartesian_t p2 = getXY(lon2, lat2);
        const double dx = p1.x - p2.x;
        const double dy = p1.y - p2.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    VDPDataItem<int> getLanePosition() override
    {
        return VDPDataItem<int>(false);
    }

    VDPDataItem<uint8_t> getExteriorLights() override
    {
        return VDPDataItem<uint8_t>(false);
    }

  private:
    Vector GetPositionVector() const
    {
        Ptr<MobilityModel> mobility = m_node->GetObject<MobilityModel>();
        return mobility ? mobility->GetPosition() : Vector();
    }

    Vector GetVelocityVector() const
    {
        Ptr<MobilityModel> mobility = m_node->GetObject<MobilityModel>();
        return mobility ? mobility->GetVelocity() : Vector();
    }

    long GetLongitudinalAccelerationValue()
    {
        const Time now = Simulator::Now();
        const double speedMps = getSpeedValue();
        const double dt = (now - m_lastAccelerationTime).GetSeconds();
        long accelerationValue = AccelerationValue_unavailable;
        if (dt > 1e-9)
        {
            accelerationValue = ClampToLong((speedMps - m_lastAccelerationSpeedMps) * 10.0,
                                            -160,
                                            161);
        }
        m_lastAccelerationSpeedMps = speedMps;
        m_lastAccelerationTime = now;
        return accelerationValue;
    }

    Ptr<Node> m_node;
    double m_travelledDistanceM = 0.0;
    double m_lastHeadingDeg = 0.0;
    double m_lastAccelerationSpeedMps = 0.0;
    Time m_lastAccelerationTime;
    Vector m_lastDistancePosition;
};

class CamApplication : public Application
{
  public:
    void Setup(Ptr<Node> node,
               Ptr<KpiLogger> logger,
               Ipv4Address groupIpv4,
               uint16_t port)
    {
        m_node = node;
        m_logger = logger;
        m_groupIpv4 = groupIpv4;
        m_port = port;
    }

    void SetBeaconInterval(double tb)
    {
        m_tGenCamMaxS = std::max(0.1, tb);
        m_caService.T_GenCamMax_ms = static_cast<long>(std::llround(m_tGenCamMaxS * 1000.0));
        if (!m_useEtsiCamGeneration)
        {
            m_caService.T_GenCamMin_ms = m_caService.T_GenCamMax_ms;
        }
    }

    void SetEtsiCamGeneration(bool enable)
    {
        m_useEtsiCamGeneration = enable;
    }

  protected:
    void StartApplication() override
    {
        m_running = true;

        m_socket = Socket::CreateSocket(m_node, UdpSocketFactory::GetTypeId());
        m_socket->SetAllowBroadcast(true);

        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        if (m_socket->Bind(local) < 0)
        {
            NS_FATAL_ERROR("Bind failed on node " << m_node->GetId());
        }
        InetSocketAddress remote = InetSocketAddress(m_groupIpv4, m_port);
        if (m_socket->Connect(remote) < 0)
        {
            NS_FATAL_ERROR("Connect failed on node " << m_node->GetId());
        }

        m_vdp = std::make_unique<NgsimVehicleDataProvider>(m_node);
        m_geoNet = CreateObject<GeoNet>();
        m_btp = CreateObject<btp>();
        m_btp->setGeoNet(m_geoNet);

        m_caService.setBTP(m_btp);
        m_caService.setVDP(m_vdp.get());
        m_btp->setVDP(m_vdp.get());
        m_caService.setStationProperties(m_node->GetId(), StationType_passengerCar);
        m_caService.setRealTime(false);
        m_caService.setSocketTx(m_socket);
        m_caService.setSocketRx(m_socket);
        m_caService.addCATxPacketCallback(
            std::bind(&CamApplication::TagAndLogTx, this, std::placeholders::_1));
        m_caService.addCARxPacketCallback(
            std::bind(&CamApplication::ReceiveCam,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3));

        const double desyncS = GetDeterministicDesync();
        m_caService.startCamDissemination(desyncS);
    }

    void StopApplication() override
    {
        m_running = false;
        m_caService.terminateDissemination();

        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

  private:
    void TagAndLogTx(Ptr<Packet> packet)
    {
        if (!m_running || !packet)
        {
            return;
        }

        const uint64_t kpiSeq = m_nextKpiSequence++;
        KpiPacketTag kpiTag;
        kpiTag.SetSequence(kpiSeq);
        kpiTag.SetTxTimeNs(Simulator::Now().GetNanoSeconds());
        packet->AddPacketTag(kpiTag);

        Vector pos = m_node->GetObject<MobilityModel>()->GetPosition();
        m_logger->LogTx(m_node->GetId(), kpiSeq, Simulator::Now(), pos);
    }

    double GetDeterministicDesync() const
    {
        const double fractional = std::fmod((m_node->GetId() + 1) * 0.6180339887498948, 1.0);
        return fractional * std::max(0.1, m_tGenCamMaxS);
    }

    void ReceiveCam(asn1cpp::Seq<CAM> cam, Address from, Ptr<Packet> packet)
    {
        if (!packet || packet->GetSize() == 0)
        {
            return;
        }

        KpiPacketTag kpiTag;
        if (!packet->PeekPacketTag(kpiTag))
        {
            return;
        }

        InetSocketAddress src = InetSocketAddress::ConvertFrom(from);
        Ipv4Address srcIp = src.GetIpv4();
        uint32_t txNodeId = m_logger->ResolveNodeIdFromIpv4(srcIp);
        if (txNodeId == std::numeric_limits<uint32_t>::max())
        {
            return;
        }
        const uint64_t kpiSeq = kpiTag.GetSequence();
        const uint32_t stationId = asn1cpp::getField(cam->header.stationId, uint32_t);

        if (stationId != txNodeId)
        {
            return;
        }

        Vector posRx = m_node->GetObject<MobilityModel>()->GetPosition();

        m_logger->LogRx(txNodeId,
                        m_node->GetId(),
                        kpiSeq,
                        Simulator::Now(),
                        posRx);
    }

    Ptr<Node> m_node;
    Ptr<KpiLogger> m_logger;
    Ptr<Socket> m_socket;
    Ptr<btp> m_btp;
    Ptr<GeoNet> m_geoNet;
    CABasicService m_caService;
    std::unique_ptr<NgsimVehicleDataProvider> m_vdp;

    bool m_running = false;
    bool m_useEtsiCamGeneration = true;

    double m_tGenCamMaxS = 1.0;

    uint64_t m_nextKpiSequence = 0;

    Ipv4Address m_groupIpv4;
    uint16_t m_port = 8000;
};

static std::map<uint32_t, Ptr<CamApplication>> g_apps;

static void
ApplyTxPowerToAllUes(const NetDeviceContainer& ueDevs, double pDbm)
{
    for (auto it = ueDevs.Begin(); it != ueDevs.End(); ++it)
    {
        Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(*it);
        NS_ABORT_MSG_IF(ueDev == nullptr, "Device is not NrUeNetDevice");

        for (uint32_t bwp = 0; bwp < ueDev->GetCcMapSize(); ++bwp)
        {
            Ptr<NrUePhy> phy = ueDev->GetPhy(static_cast<uint8_t>(bwp));
            NS_ABORT_MSG_IF(phy == nullptr, "Missing NrUePhy for BWP " << bwp);
            phy->SetTxPower(pDbm);
        }
    }
}

static void
ApplyInputsAtTime(Ptr<KpiLogger> logger, NetDeviceContainer ueDevs, double pDbm, double tbS)
{
    logger->SetCurrentInputs(pDbm, tbS);

    for (auto& kv : g_apps)
    {
        kv.second->SetBeaconInterval(tbS);
    }

    ApplyTxPowerToAllUes(ueDevs, pDbm);
}

static void
AssignDeterministicSourceL2Ids(const NetDeviceContainer& ueDevs)
{
    uint32_t ueIndex = 0;
    for (auto it = ueDevs.Begin(); it != ueDevs.End(); ++it, ++ueIndex)
    {
        // SCI format 2 carries an 8-bit source id in this ns-3 NR model.
        // Keep it nonzero on the wire; 255 is left for the group destination.
        const uint32_t sourceL2Id = 1 + (ueIndex % 254);
        Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(*it);
        NS_ABORT_MSG_IF(ueDev == nullptr, "Device is not NrUeNetDevice");
        Ptr<LteUeRrc> lteRrc = ueDev->GetRrc();
        Ptr<NrSlUeRrc> nrSlRrc = lteRrc->GetObject<NrSlUeRrc>();
        NS_ABORT_MSG_IF(nrSlRrc == nullptr, "Missing NrSlUeRrc while assigning source L2 IDs");
        nrSlRrc->SetSourceL2Id(sourceL2Id);
        NS_ABORT_MSG_IF(lteRrc->GetSourceL2Id() == 0,
                        "Failed to assign nonzero source L2 ID for UE device");

        for (uint32_t bwp = 0; bwp < ueDev->GetCcMapSize(); ++bwp)
        {
            Ptr<NrSlUeMac> slMac =
                ueDev->GetMac(static_cast<uint8_t>(bwp))->GetObject<NrSlUeMac>();
            if (slMac)
            {
                slMac->GetNrSlUeCmacSapProvider()->SetSourceL2Id(sourceL2Id);
            }
        }
    }
}

static void
ScheduleInputUpdates(Ptr<KpiLogger> logger, const NetDeviceContainer& ueDevs)
{
    for (const auto& u : g_inputSchedule)
    {
        Simulator::Schedule(Seconds(u.timeS),
                            &ApplyInputsAtTime,
                            logger,
                            ueDevs,
                            u.txPowerDbm,
                            u.beaconIntervalS);
    }
}

static void
ConnectCbrTraces(CbrLogger* cbrLogger)
{
    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy/NrSpectrumPhy/"
                    "ChannelOccupied",
                    MakeCallback(&CbrLogger::RecordChannelOccupied, cbrLogger));
}

static std::string
DeriveSidecarPath(const std::string& csvPath, const std::string& suffix)
{
    std::string path = csvPath;
    std::size_t pos = path.rfind(".csv");
    if (pos != std::string::npos)
    {
        path.replace(pos, 4, suffix);
    }
    else
    {
        path += suffix;
    }
    return path;
}

static std::string
JsonEscape(const std::string& value)
{
    std::ostringstream os;
    for (char c : value)
    {
        switch (c)
        {
        case '\\':
            os << "\\\\";
            break;
        case '"':
            os << "\\\"";
            break;
        case '\n':
            os << "\\n";
            break;
        case '\r':
            os << "\\r";
            break;
        case '\t':
            os << "\\t";
            break;
        default:
            os << c;
            break;
        }
    }
    return os.str();
}

static void
WriteMetadataJson(const std::string& kpiCsv,
                  const std::string& mobilityCsv,
                  const std::string& inputCsv,
                  double simTimeSeconds,
                  double sampleTimeS,
                  double kpiWindowS,
                  double warmupS,
                  double cooldownS,
                  double evalStartS,
                  double evalEndS,
                  bool enableCoreFilter,
                  const std::string& coreAxis,
                  double studyMin,
                  double studyMax,
                  double coreGuardBandM,
                  double coreMin,
                  double coreMax,
                  double awarenessRangeM,
                  uint32_t maxCamSizeBytes,
                  bool useInputSchedule,
                  double fixedBeaconIntervalS,
                  bool etsiCamGeneration,
                  double centralFrequencyHz,
                  uint16_t bandwidth100KhzUnits,
                  uint16_t mcs,
                  bool enableSensing,
                  bool harqEnabled,
                  bool enableChannelRandomness,
                  uint16_t slSensingWindow,
                  uint16_t slSelectionWindow,
                  uint16_t reservationPeriod,
                  uint16_t t1,
                  uint16_t t2,
                  int slThresPsschRsrp,
                  uint32_t vehicleCount,
                  uint32_t seed,
                  uint32_t run)
{
    const std::string metadataPath = DeriveSidecarPath(kpiCsv, "_metadata.json");
    std::ofstream out(metadataPath.c_str(), std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        NS_FATAL_ERROR("Cannot open metadata JSON: " << metadataPath);
    }

    out << std::boolalpha
        << "{\n"
        << "  \"dataset\": \"NGSIM US-101\",\n"
        << "  \"mobility_csv\": \"" << JsonEscape(mobilityCsv) << "\",\n"
        << "  \"input_csv\": \"" << JsonEscape(inputCsv) << "\",\n"
        << "  \"kpi_csv\": \"" << JsonEscape(kpiCsv) << "\",\n"
        << "  \"sim_time_s\": " << simTimeSeconds << ",\n"
        << "  \"sample_time_s\": " << sampleTimeS << ",\n"
        << "  \"kpi_window_s\": " << kpiWindowS << ",\n"
        << "  \"warmup_s\": " << warmupS << ",\n"
        << "  \"cooldown_s\": " << cooldownS << ",\n"
        << "  \"evaluation_start_s\": " << evalStartS << ",\n"
        << "  \"evaluation_end_s\": " << evalEndS << ",\n"
        << "  \"core_filter_enabled\": " << enableCoreFilter << ",\n"
        << "  \"core_axis\": \"" << JsonEscape(coreAxis) << "\",\n"
        << "  \"study_core_axis_min_m\": " << studyMin << ",\n"
        << "  \"study_core_axis_max_m\": " << studyMax << ",\n"
        << "  \"core_guard_band_m\": " << coreGuardBandM << ",\n"
        << "  \"core_min_m\": " << coreMin << ",\n"
        << "  \"core_max_m\": " << coreMax << ",\n"
        << "  \"awareness_range_m\": " << awarenessRangeM << ",\n"
        << "  \"max_cam_encoded_size_bytes\": " << maxCamSizeBytes << ",\n"
        << "  \"cam_payload_format\": \"ETSI CAM ASN.1 UPER encoded payload with "
           "ItsPduHeader, generationDeltaTime, BasicContainer, and "
           "BasicVehicleContainerHighFrequency vehicle fields\",\n"
        << "  \"use_input_schedule\": " << useInputSchedule << ",\n"
        << "  \"fixed_beacon_interval_s\": " << fixedBeaconIntervalS << ",\n"
        << "  \"etsi_cam_generation\": " << etsiCamGeneration << ",\n"
        << "  \"cam_generation_rule\": \""
        << (etsiCamGeneration
                ? "ETSI-style decentralized triggering: deterministic per-node desynchronization, "
                  "100 ms minimum check/generation interval, dynamic triggers for heading >4 deg, "
                  "position >4 m, speed >0.5 m/s, and maximum elapsed interval from input/fixed Tb"
                : "fixed periodic generation at fixed/input beacon interval")
        << "\",\n"
        << "  \"carrier_frequency_hz\": " << centralFrequencyHz << ",\n"
        << "  \"bandwidth_100khz_units\": " << bandwidth100KhzUnits << ",\n"
        << "  \"mcs\": " << mcs << ",\n"
        << "  \"sensing_enabled\": " << enableSensing << ",\n"
        << "  \"harq_enabled\": " << harqEnabled << ",\n"
        << "  \"channel_randomness_enabled\": " << enableChannelRandomness << ",\n"
        << "  \"sl_sensing_window_ms\": " << slSensingWindow << ",\n"
        << "  \"sl_selection_window_slots\": " << slSelectionWindow << ",\n"
        << "  \"reservation_period_ms\": " << reservationPeriod << ",\n"
        << "  \"t1_slots\": " << t1 << ",\n"
        << "  \"t2_slots\": " << t2 << ",\n"
        << "  \"sl_thres_pssch_rsrp_dbm\": " << slThresPsschRsrp << ",\n"
        << "  \"vehicle_count\": " << vehicleCount << ",\n"
        << "  \"seed\": " << seed << ",\n"
        << "  \"run\": " << run << ",\n"
        << "  \"prr_definition\": \"unique successful Tx-Rx-seq receptions with TX-time distance <= "
           "awareness_range_m divided by TX-time eligible receiver opportunities; TX and RX "
           "events are filtered to the evaluation time window and core corridor region when enabled\",\n"
        << "  \"traffic_context_definition\": \"time-varying measured context sampled at the KPI "
           "timestamp: active_vehicle_count_core is the number of evaluated vehicles active and "
           "inside the core region\",\n"
        << "  \"cbr_definition\": \"PHY busy-time fraction over the 1.0 s window from "
           "NrSpectrumPhy::ChannelOccupied, averaged over evaluated UE nodes; evaluated nodes "
           "with no busy interval in the window contribute zero busy time\",\n"
        << "  \"cbr_primary_source\": \"NrSpectrumPhy::ChannelOccupied busy-time trace\",\n"
        << "  \"cbr_alignment_method\": \"CBR and PRR/PIR are sampled by the simulator at the same "
        << sampleTimeS << " s timestamps; no interpolation is applied\",\n"
        << "  \"deepc_dataset_columns\": "
           "\"time_s,tx_power_dbm,beacon_interval_s,active_vehicle_count_core,"
           "prr_awareness,pir_s,cbr\"\n"
        << "}\n";
}

/****************************************************************
 * main
 ****************************************************************/

int
main(int argc, char* argv[])
{
    const auto __ns3_total_compute_start = std::chrono::steady_clock::now();
    // File paths
    std::string mobilityCsv = "data/processed/ngsim_us101_mainline_active20_250_densest_600s.csv";
    std::string inputCsv = "data/processed/prbs_schedule.csv";
    std::string kpiCsv = "data/output/deepc_open_loop_250veh/kpi_timeseries.csv";

    // Simulation time
    double simTimeSeconds = 600.0;
    double warmupS = 10.0;
    double cooldownS = 10.0;
    uint32_t seed = 12345; // 12345, 12346, 12347 for runs 1, 2, 3
    uint32_t run = 1;

    // Traffic / app
    uint32_t maxCamSizeBytes = 300;
    uint16_t port = 8000;
    bool useInputSchedule = true;
    bool etsiCamGeneration = true;
    double fixedBeaconIntervalS = 0.1;

    // SL bearer activation
    Time slBearersActivationTime = Seconds(2.0);
    Time camApplicationStartDelay = MilliSeconds(200);
    bool harqEnabled = true;
    Time delayBudget = Seconds(0);

    // NR-V2X radio baseline
    double centralFrequencyBandSl = 5.9e9;
    uint16_t bandwidthBandSl = 100; // 10 MHz in units of 100 kHz
    double txPower = 20.0;
    std::string tddPattern = "DL|DL|DL|F|UL|UL|UL|UL|UL|UL|";
    std::string slBitMap = "1|1|1|1|1|1|0|0|0|1|1|1";
    uint16_t numerologyBwpSl = 0;
    uint16_t slSensingWindow = 100;
    uint16_t slSelectionWindow = 5;
    uint16_t slSubchannelSize = 50;
    uint16_t slMaxNumPerReserve = 3;
    double slProbResourceKeep = 0.0;
    uint16_t slMaxTxTransNumPssch = 5;
    uint16_t reservationPeriod = 100;
    bool enableSensing = false;
    uint16_t t1 = 2;
    uint16_t t2 = 33;
    int slThresPsschRsrp = -128;
    bool enableChannelRandomness = true;
    uint16_t channelUpdatePeriod = 500;
    uint16_t mcs = 6;
    double awarenessRangeM = 300.0;
    double kpiWindowS = 1.0;
    double sampleTimeS = 0.5;
    uint32_t maxVehicles = 250; // 0 means all vehicles
    bool enableCoreFilter = true;
    std::string coreAxis = "auto";
    double coreGuardBandM = 120.0;
    double coreMin = std::numeric_limits<double>::quiet_NaN();
    double coreMax = std::numeric_limits<double>::quiet_NaN();
    bool requireFullDataCoverage = true;
    bool validateOnly = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("mobilityCsv", "Processed NGSIM mobility CSV", mobilityCsv);
    cmd.AddValue("inputCsv", "PRBS schedule CSV", inputCsv);
    cmd.AddValue("kpiCsv", "Output KPI CSV", kpiCsv);
    cmd.AddValue("simTime", "Simulation time [s]", simTimeSeconds);
    cmd.AddValue("warmup", "Warm-up duration excluded from KPI evaluation [s]", warmupS);
    cmd.AddValue("cooldown", "Final cool-down duration excluded from KPI evaluation [s]", cooldownS);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run number", run);
    cmd.AddValue("packetSize", "Maximum encoded CAM packet size [bytes]", maxCamSizeBytes);
    cmd.AddValue("useInputSchedule", "Use PRBS inputCsv updates; false keeps fixed txPower/Tb", useInputSchedule);
    cmd.AddValue("etsiCamGeneration",
                 "Use ETSI-style decentralized CAM triggering instead of fixed periodic beacons",
                 etsiCamGeneration);
    cmd.AddValue("fixedBeaconInterval", "Fixed CAM beacon interval when useInputSchedule=false [s]", fixedBeaconIntervalS);
    cmd.AddValue("mcs", "Fixed sidelink MCS", mcs);
    cmd.AddValue("txPower", "Baseline Tx power [dBm]", txPower);
    cmd.AddValue("enableSensing", "Enable NR sidelink sensing-based resource selection", enableSensing);
    cmd.AddValue("enableChannelRandomness", "Enable channel update randomness", enableChannelRandomness);
    cmd.AddValue("awarenessRange", "PRR awareness range [m]", awarenessRangeM);
    cmd.AddValue("kpiWindow", "KPI sliding window [s]", kpiWindowS);
    cmd.AddValue("sampleTime", "KPI output sample time [s]", sampleTimeS);
    cmd.AddValue("maxVehicles", "Maximum number of vehicles to simulate; 0 means all vehicles", maxVehicles);
    cmd.AddValue("enableCoreFilter", "Filter KPI evaluation to the central corridor core", enableCoreFilter);
    cmd.AddValue("coreAxis", "Longitudinal coordinate for core filter: auto, x, or y", coreAxis);
    cmd.AddValue("coreGuardBand", "Guard band removed from each corridor edge [m]", coreGuardBandM);
    cmd.AddValue("coreMin", "Manual core-region minimum coordinate [m]; NaN derives from mobility CSV", coreMin);
    cmd.AddValue("coreMax", "Manual core-region maximum coordinate [m]; NaN derives from mobility CSV", coreMax);
    cmd.AddValue("requireFullDataCoverage",
                 "Abort if mobility/input data do not cover [0, simTime]",
                 requireFullDataCoverage);
    cmd.AddValue("validateOnly",
                 "Validate inputs and selected mobility coverage, then exit before building NR devices",
                 validateOnly);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(std::abs(kpiWindowS - 1.0) > 1e-9,
                    "CBR window is fixed by the experiment definition: kpiWindow must be 1.0 s");
    NS_ABORT_MSG_IF(warmupS < 0.0 || cooldownS < 0.0,
                    "warmup and cooldown must be non-negative");
    NS_ABORT_MSG_IF(warmupS + cooldownS >= simTimeSeconds,
                    "warmup + cooldown must be smaller than simTime");
    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    // Load data
    LoadMobilityCsv(mobilityCsv);
    coreAxis = ResolveCoreAxis(coreAxis);
    const auto studyBounds = ComputeStudyBounds(coreAxis);
    const double studyMin = studyBounds.first;
    const double studyMax = studyBounds.second;
    if (std::isnan(coreMin))
    {
        coreMin = studyMin + coreGuardBandM;
    }
    if (std::isnan(coreMax))
    {
        coreMax = studyMax - coreGuardBandM;
    }
    if (coreMin > coreMax)
    {
        std::swap(coreMin, coreMax);
    }
    NS_ABORT_MSG_IF(enableCoreFilter && coreMin >= coreMax,
                    "Invalid core region. Reduce coreGuardBand or set coreMin/coreMax.");

    const double evalStartS = warmupS;
    const double evalEndS = simTimeSeconds - cooldownS;
    std::cout << "Core axis = " << coreAxis << std::endl;
    std::cout << "Study " << coreAxis << "-range = [" << studyMin << ", " << studyMax << "] m"
              << std::endl;
    std::cout << "Core " << coreAxis << "-range = [" << coreMin << ", " << coreMax << "] m"
              << (enableCoreFilter ? " (enabled)" : " (disabled)") << std::endl;
    std::cout << "Evaluation time window = [" << evalStartS << ", " << evalEndS << "] s"
              << std::endl;

    if (useInputSchedule)
    {
        LoadInputScheduleCsv(inputCsv);
    }
    else
    {
        std::cout << "Input schedule disabled; fixed Tx power = " << txPower
                  << " dBm, fixed beacon interval = " << fixedBeaconIntervalS << " s"
                  << ", ETSI CAM generation = " << etsiCamGeneration
                  << std::endl;
    }

    std::vector<uint32_t> vehicleIds;
    vehicleIds.reserve(g_mobilityByVehicle.size());
    for (const auto& kv : g_mobilityByVehicle)
    {
        vehicleIds.push_back(kv.first);
    }
    std::sort(vehicleIds.begin(), vehicleIds.end());

    if (maxVehicles > 0 && maxVehicles < vehicleIds.size())
    {
        vehicleIds.resize(maxVehicles);
    }
    ValidateTenMinuteDataCoverage(simTimeSeconds,
                                  evalStartS,
                                  evalEndS,
                                  useInputSchedule,
                                  requireFullDataCoverage,
                                  vehicleIds);
    if (validateOnly)
    {
        const auto mobilityRange = ComputeMobilityTimeRangeForVehicles(vehicleIds);
        std::cout << "Validation-only mode passed." << std::endl;
        std::cout << "Selected mobility coverage = [" << mobilityRange.first << ", "
                  << mobilityRange.second << "] s" << std::endl;
        if (useInputSchedule)
        {
            const auto inputRange = ComputeInputScheduleTimeRange();
            std::cout << "Input schedule coverage = [" << inputRange.first << ", "
                      << inputRange.second << "] s" << std::endl;
        }
        return 0;
    }

    // Create nodes from vehicleIds and install mobility
    NodeContainer allSlUesContainer;
    allSlUesContainer.Create(vehicleIds.size());

    for (uint32_t i = 0; i < vehicleIds.size(); ++i)
    {
        uint32_t vehicleId = vehicleIds[i];
        Ptr<Node> node = allSlUesContainer.Get(i);

        g_vehicleIdToNodeId[vehicleId] = node->GetId();
        g_nodeIdToVehicleId[node->GetId()] = vehicleId;
        g_nodeIdToNode[node->GetId()] = node;
        g_nodeIdToActiveTimeRangeS[node->GetId()] =
            ComputeMobilityTimeRange(g_mobilityByVehicle[vehicleId]);

        InstallWaypointMobility(node, g_mobilityByVehicle[vehicleId]);
    }

    std::cout << "Total UEs = " << allSlUesContainer.GetN() << std::endl;

    /**************** NR core + helper ****************/
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetEpcHelper(epcHelper);

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConfSl(centralFrequencyBandSl,
                                                     bandwidthBandSl,
                                                     numCcPerBand,
                                                     BandwidthPartInfo::V2V_Highway);

    OperationBandInfo bandSl = ccBwpCreator.CreateOperationBandContiguousCc(bandConfSl);

    if (enableChannelRandomness)
    {
        Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod",
                           TimeValue(MilliSeconds(channelUpdatePeriod)));
        nrHelper->SetChannelConditionModelAttribute("UpdatePeriod",
                                                    TimeValue(MilliSeconds(channelUpdatePeriod)));
        nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    }
    else
    {
        Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));
        nrHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
        nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    }

    nrHelper->InitializeOperationBand(&bandSl);
    allBwps = CcBwpCreator::GetAllBwps({bandSl});

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(txPower));

    nrHelper->SetUeMacTypeId(NrSlUeMac::GetTypeId());
    nrHelper->SetUeMacAttribute("EnableSensing", BooleanValue(enableSensing));
    nrHelper->SetUeMacAttribute("T1", UintegerValue(static_cast<uint8_t>(t1)));
    nrHelper->SetUeMacAttribute("T2", UintegerValue(t2));
    nrHelper->SetUeMacAttribute("ActivePoolId", UintegerValue(0));
    nrHelper->SetUeMacAttribute("SlThresPsschRsrp", IntegerValue(slThresPsschRsrp));

    uint8_t bwpIdForGbrMcptt = 0;
    nrHelper->SetBwpManagerTypeId(TypeId::LookupByName("ns3::NrSlBwpManagerUe"));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_MC_PUSH_TO_TALK",
                                                UintegerValue(bwpIdForGbrMcptt));

    std::set<uint8_t> bwpIdContainer;
    bwpIdContainer.insert(bwpIdForGbrMcptt);

    NetDeviceContainer allSlUesNetDeviceContainer =
        nrHelper->InstallUeDevice(allSlUesContainer, allBwps);

    for (auto it = allSlUesNetDeviceContainer.Begin(); it != allSlUesNetDeviceContainer.End(); ++it)
    {
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();
    }

    /**************** NR sidelink helper ****************/
    Ptr<NrSlHelper> nrSlHelper = CreateObject<NrSlHelper>();
    nrSlHelper->SetEpcHelper(epcHelper);

    std::string errorModel = "ns3::NrEesmIrT1";
    nrSlHelper->SetSlErrorModel(errorModel);
    nrSlHelper->SetUeSlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nrSlHelper->SetNrSlSchedulerTypeId(NrSlUeMacSchedulerFixedMcs::GetTypeId());
    nrSlHelper->SetUeSlSchedulerAttribute("Mcs", UintegerValue(mcs));
    nrSlHelper->PrepareUeForSidelink(allSlUesNetDeviceContainer, bwpIdContainer);
    AssignDeterministicSourceL2Ids(allSlUesNetDeviceContainer);

    /**************** SL preconfiguration ****************/
    LteRrcSap::SlResourcePoolNr slResourcePoolNr;
    Ptr<NrSlCommResourcePoolFactory> ptrFactory = Create<NrSlCommResourcePoolFactory>();

    std::vector<std::bitset<1>> slBitMapVector;
    GetSlBitmapFromString(slBitMap, slBitMapVector);
    NS_ABORT_MSG_IF(slBitMapVector.empty(), "Failed to generate sidelink bitmap");

    ptrFactory->SetSlTimeResources(slBitMapVector);
    ptrFactory->SetSlSensingWindow(slSensingWindow);
    ptrFactory->SetSlSelectionWindow(slSelectionWindow);
    ptrFactory->SetSlFreqResourcePscch(10);
    ptrFactory->SetSlSubchannelSize(slSubchannelSize);
    ptrFactory->SetSlMaxNumPerReserve(slMaxNumPerReserve);
    std::list<uint16_t> resourceReservePeriodList = {0, reservationPeriod};
    ptrFactory->SetSlResourceReservePeriodList(resourceReservePeriodList);

    LteRrcSap::SlResourcePoolNr pool = ptrFactory->CreatePool();
    slResourcePoolNr = pool;

    LteRrcSap::SlResourcePoolConfigNr slresoPoolConfigNr;
    slresoPoolConfigNr.haveSlResourcePoolConfigNr = true;

    uint16_t poolId = 0;
    LteRrcSap::SlResourcePoolIdNr slResourcePoolIdNr;
    slResourcePoolIdNr.id = poolId;
    slresoPoolConfigNr.slResourcePoolId = slResourcePoolIdNr;
    slresoPoolConfigNr.slResourcePool = slResourcePoolNr;

    LteRrcSap::SlBwpPoolConfigCommonNr slBwpPoolConfigCommonNr;
    slBwpPoolConfigCommonNr.slTxPoolSelectedNormal[slResourcePoolIdNr.id] = slresoPoolConfigNr;

    LteRrcSap::Bwp bwp;
    bwp.numerology = numerologyBwpSl;
    bwp.symbolsPerSlots = 14;
    bwp.rbPerRbg = 1;
    bwp.bandwidth = bandwidthBandSl;

    LteRrcSap::SlBwpGeneric slBwpGeneric;
    slBwpGeneric.bwp = bwp;
    slBwpGeneric.slLengthSymbols = LteRrcSap::GetSlLengthSymbolsEnum(14);
    slBwpGeneric.slStartSymbol = LteRrcSap::GetSlStartSymbolEnum(0);

    LteRrcSap::SlBwpConfigCommonNr slBwpConfigCommonNr;
    slBwpConfigCommonNr.haveSlBwpGeneric = true;
    slBwpConfigCommonNr.slBwpGeneric = slBwpGeneric;
    slBwpConfigCommonNr.haveSlBwpPoolConfigCommonNr = true;
    slBwpConfigCommonNr.slBwpPoolConfigCommonNr = slBwpPoolConfigCommonNr;

    LteRrcSap::SlFreqConfigCommonNr slFreConfigCommonNr;
    for (const auto& it : bwpIdContainer)
    {
        slFreConfigCommonNr.slBwpList[it] = slBwpConfigCommonNr;
    }

    LteRrcSap::TddUlDlConfigCommon tddUlDlConfigCommon;
    tddUlDlConfigCommon.tddPattern = tddPattern;

    LteRrcSap::SlPreconfigGeneralNr slPreconfigGeneralNr;
    slPreconfigGeneralNr.slTddConfig = tddUlDlConfigCommon;

    LteRrcSap::SlUeSelectedConfig slUeSelectedPreConfig;
    NS_ABORT_MSG_UNLESS(slProbResourceKeep <= 1.0,
                        "slProbResourceKeep must be in [0,1]");
    slUeSelectedPreConfig.slProbResourceKeep = slProbResourceKeep;

    LteRrcSap::SlPsschTxParameters psschParams;
    psschParams.slMaxTxTransNumPssch = static_cast<uint8_t>(slMaxTxTransNumPssch);

    LteRrcSap::SlPsschTxConfigList pscchTxConfigList;
    pscchTxConfigList.slPsschTxParameters[0] = psschParams;
    slUeSelectedPreConfig.slPsschTxConfigList = pscchTxConfigList;

    LteRrcSap::SidelinkPreconfigNr slPreConfigNr;
    slPreConfigNr.slPreconfigGeneral = slPreconfigGeneralNr;
    slPreConfigNr.slUeSelectedPreConfig = slUeSelectedPreConfig;
    slPreConfigNr.slPreconfigFreqInfoList[0] = slFreConfigCommonNr;

    nrSlHelper->InstallNrSlPreConfiguration(allSlUesNetDeviceContainer, slPreConfigNr);
    AssignDeterministicSourceL2Ids(allSlUesNetDeviceContainer);

    /**************** IP stack + SL bearer ****************/
    InternetStackHelper internet;
    internet.Install(allSlUesContainer);

    uint32_t dstL2Id = 255;
    Ipv4Address groupAddress4("225.0.0.0");

    Ptr<LteSlTft> tft;
    SidelinkInfo slInfo;
    slInfo.m_castType = SidelinkInfo::CastType::Groupcast;
    slInfo.m_dstL2Id = dstL2Id;
    slInfo.m_rri = MilliSeconds(reservationPeriod);
    slInfo.m_dynamic = false;
    slInfo.m_pdb = delayBudget;
    slInfo.m_harqEnabled = harqEnabled;

    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(allSlUesNetDeviceContainer);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < allSlUesContainer.GetN(); ++u)
    {
        Ptr<Node> ueNode = allSlUesContainer.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    tft = Create<LteSlTft>(LteSlTft::Direction::TRANSMIT, groupAddress4, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime, allSlUesNetDeviceContainer, tft);

    tft = Create<LteSlTft>(LteSlTft::Direction::RECEIVE, groupAddress4, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime, allSlUesNetDeviceContainer, tft);

    /**************** KPI logger ****************/
    CbrLogger cbrLogger;
    cbrLogger.Setup(1.0);
    cbrLogger.SetEvaluationWindow(evalStartS, evalEndS);
    std::vector<uint32_t> evaluatedNodeIds;
    evaluatedNodeIds.reserve(allSlUesContainer.GetN());
    for (uint32_t i = 0; i < allSlUesContainer.GetN(); ++i)
    {
        evaluatedNodeIds.push_back(allSlUesContainer.Get(i)->GetId());
    }
    cbrLogger.SetEvaluatedNodeIds(evaluatedNodeIds);
    cbrLogger.SetNodeEvaluationFilter(
        [enableCoreFilter, coreAxis, coreMin, coreMax](uint32_t nodeId, double timeS) {
            if (!IsNodeActiveAtTime(nodeId, timeS))
            {
                return false;
            }
            if (!enableCoreFilter)
            {
                return true;
            }

            auto it = g_nodeIdToNode.find(nodeId);
            if (it == g_nodeIdToNode.end() || !it->second)
            {
                return false;
            }

            const Vector pos = it->second->GetObject<MobilityModel>()->GetPosition();
            return IsInsideCoreRegion(GetCoreCoordinate(pos, coreAxis), coreMin, coreMax);
        });
    ConnectCbrTraces(&cbrLogger);

    Ptr<KpiLogger> logger = CreateObject<KpiLogger>();
    logger->Configure(allSlUesContainer,
                      &cbrLogger,
                      kpiCsv,
                      awarenessRangeM,
                      kpiWindowS,
                      sampleTimeS,
                      evalStartS,
                      evalEndS,
                      enableCoreFilter,
                      coreAxis,
                      coreMin,
                      coreMax);

    for (uint32_t i = 0; i < allSlUesContainer.GetN(); ++i)
    {
        Ipv4Address localAddr =
            allSlUesContainer.Get(i)->GetObject<Ipv4L3Protocol>()->GetAddress(1, 0).GetLocal();
        logger->RegisterNodeIpv4(allSlUesContainer.Get(i)->GetId(), localAddr);
    }
    logger->SetCurrentInputs(txPower, fixedBeaconIntervalS);
    ApplyTxPowerToAllUes(allSlUesNetDeviceContainer, txPower);

    /**************** Automotive CAM apps ****************/
    for (uint32_t i = 0; i < allSlUesContainer.GetN(); ++i)
    {
        Ptr<Node> node = allSlUesContainer.Get(i);

        Ptr<CamApplication> app = CreateObject<CamApplication>();
        app->Setup(node, logger, groupAddress4, port);
        app->SetEtsiCamGeneration(etsiCamGeneration);
        app->SetBeaconInterval(fixedBeaconIntervalS);
        node->AddApplication(app);
        const auto activeRange = g_nodeIdToActiveTimeRangeS[node->GetId()];
        const double appStartS =
            std::max((slBearersActivationTime + camApplicationStartDelay).GetSeconds(),
                     activeRange.first);
        const double appStopS = std::min(simTimeSeconds, activeRange.second);
        if (appStopS > appStartS)
        {
            app->SetStartTime(Seconds(appStartS));
            app->SetStopTime(Seconds(appStopS));
            g_apps[node->GetId()] = app;
        }
    }

    /**************** Time-varying input schedule ****************/
    if (useInputSchedule)
    {
        ScheduleInputUpdates(logger, allSlUesNetDeviceContainer);
    }

    /**************** Periodic KPI sampling ****************/
    Simulator::Schedule(Seconds(sampleTimeS), &KpiLogger::SampleAndWrite, logger);

    WriteMetadataJson(kpiCsv,
                      mobilityCsv,
                      inputCsv,
                      simTimeSeconds,
                      sampleTimeS,
                      kpiWindowS,
                      warmupS,
                      cooldownS,
                      evalStartS,
                      evalEndS,
                      enableCoreFilter,
                      coreAxis,
                      studyMin,
                      studyMax,
                      coreGuardBandM,
                      coreMin,
                      coreMax,
                      awarenessRangeM,
                      maxCamSizeBytes,
                      useInputSchedule,
                      fixedBeaconIntervalS,
                      etsiCamGeneration,
                      centralFrequencyBandSl,
                      bandwidthBandSl,
                      mcs,
                      enableSensing,
                      harqEnabled,
                      enableChannelRandomness,
                      slSensingWindow,
                      slSelectionWindow,
                      reservationPeriod,
                      t1,
                      t2,
                      slThresPsschRsrp,
                      allSlUesContainer.GetN(),
                      seed,
                      run);

    /**************** Simulation ****************/
    Simulator::Stop(Seconds(simTimeSeconds));
    Simulator::Run();
    Simulator::Destroy();

    const auto __ns3_total_compute_end = std::chrono::steady_clock::now();
    const double __ns3_total_compute_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(__ns3_total_compute_end - __ns3_total_compute_start)
            .count();
    std::cout << "Total computation time: " << std::fixed << std::setprecision(3)
              << __ns3_total_compute_seconds << " s" << std::endl;

    return 0;
}
