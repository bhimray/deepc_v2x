/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/CAM.h"
#include "ns3/uper_decoder.h"
#include "ns3/uper_encoder.h"

#include "../src/cbr-logger.cc"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
    uint32_t seq;
    Vector posTx;
    uint32_t eligibleRxCount;
    std::vector<uint32_t> eligibleRxCountsByBin;
    std::map<uint32_t, double> rxDistanceAtTxM;
};

struct RxEvent
{
    double timeS;
    uint32_t txNodeId;
    uint32_t rxNodeId;
    uint32_t seq;
    Vector posTx;
    Vector posRx;
    double distanceAtTxM;
    double distanceAtRxM;
    double pirS;
    double delayS;
};

static std::map<uint32_t, std::vector<MobilityRow>> g_mobilityByVehicle;
static std::vector<InputCommand> g_inputSchedule;

static std::map<uint32_t, uint32_t> g_vehicleIdToNodeId;
static std::map<uint32_t, uint32_t> g_nodeIdToVehicleId;
static std::map<uint32_t, Ptr<Node>> g_nodeIdToNode;

/****************************************************************
 * Helpers
 ****************************************************************/

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kDotOneMicro = 1e7;
static constexpr double kLocalOriginLatitudeDeg = 37.0;
static constexpr double kLocalOriginLongitudeDeg = -122.0;

static double
Distance2d(const Vector& a, const Vector& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

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

static uint32_t
GenerationDeltaTimeMs(Time now)
{
    return static_cast<uint32_t>(now.GetMilliSeconds() % 65536);
}

static double
ComputeCamDelayS(uint32_t generationDeltaTimeMs, Time rxTime)
{
    const int64_t rxMs = rxTime.GetMilliSeconds() % 65536;
    int64_t delayMs = rxMs - static_cast<int64_t>(generationDeltaTimeMs);
    if (delayMs < 0)
    {
        delayMs += 65536;
    }
    return static_cast<double>(delayMs) / 1000.0;
}

static bool
IsInsideCoreRegion(double coordinate, double coreMin, double coreMax)
{
    return coordinate >= coreMin && coordinate <= coreMax;
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
    Ptr<WaypointMobilityModel> mob = CreateObject<WaypointMobilityModel>();
    node->AggregateObject(mob);

    for (const auto& r : rows)
    {
        Waypoint wp(Seconds(r.timeS), Vector(r.xM, r.yM, 0.0));
        mob->AddWaypoint(wp);
    }
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
                   double coreMax,
                   double densityLengthM)
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
        m_densityLengthM = densityLengthM;

        for (uint32_t i = 0; i < nodes.GetN(); ++i)
        {
            Ptr<Node> n = nodes.Get(i);
            m_nodes.push_back(n);
            m_nodeIds.insert(n->GetId());
        }
        m_out.open(csvPath.c_str(), std::ios::out | std::ios::trunc);
        if (!m_out.is_open())
        {
            NS_FATAL_ERROR("Cannot open KPI output CSV: " << csvPath);
        }
        m_out << "time_s,tx_power_dbm,beacon_interval_s,"
              << "active_vehicle_count_core,density_veh_per_km_core,"
              << "mean_neighbors_150m,mean_neighbors_300m,"
              << "prr_150m,pir_s,cbr,sensing_exclusion_ratio\n";

        std::string txLogPath = csvPath;
        std::string rxLogPath = csvPath;

        std::size_t pos = txLogPath.rfind(".csv");
        if (pos != std::string::npos)
        {
            txLogPath.replace(pos, 4, "_tx_packet_log.csv");
            rxLogPath.replace(pos, 4, "_rx_packet_log.csv");
        }
        else
        {
            txLogPath += "_tx_packet_log.csv";
            rxLogPath += "_rx_packet_log.csv";
        }

        m_txOut.open(txLogPath.c_str(), std::ios::out | std::ios::trunc);
        if (!m_txOut.is_open())
        {
            NS_FATAL_ERROR("Cannot open TX packet log CSV: " << txLogPath);
        }
        m_txOut << "time_s,tx_node_id,seq,tx_x_m,tx_y_m,eligible_rx_count_150m";
        for (uint32_t i = 0; i + 1 < m_distanceBinEdgesM.size(); ++i)
        {
            m_txOut << "," << BuildDistanceBinColumnName(i);
        }
        m_txOut << "\n";

        m_rxOut.open(rxLogPath.c_str(), std::ios::out | std::ios::trunc);
        if (!m_rxOut.is_open())
        {
            NS_FATAL_ERROR("Cannot open RX packet log CSV: " << rxLogPath);
        }
        m_rxOut << "time_s,tx_node_id,rx_node_id,seq,tx_x_m,tx_y_m,rx_x_m,rx_y_m,"
                << "distance_m,rx_distance_m,pir_s,delay_s\n";
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

    void LogTx(uint32_t txNodeId, uint32_t seq, Time tTx, const Vector& posTx)
    {
        const double timeS = tTx.GetSeconds();
        if (!IsInsideEvaluationWindow(timeS) || !IsInsideCore(posTx))
        {
            return;
        }

        uint32_t eligible = 0;
        std::vector<uint32_t> eligibleByBin(m_distanceBinEdgesM.size() - 1, 0);
        std::map<uint32_t, double> rxDistanceAtTxM;

        for (const auto& node : m_nodes)
        {
            if (node->GetId() == txNodeId)
            {
                continue;
            }

            Vector posRx = node->GetObject<MobilityModel>()->GetPosition();
            if (!IsInsideCore(posRx))
            {
                continue;
            }

            const double distanceM = Distance2d(posTx, posRx);
            const int bin = GetDistanceBinIndex(distanceM);
            if (bin >= 0)
            {
                eligibleByBin[static_cast<uint32_t>(bin)]++;
                rxDistanceAtTxM[node->GetId()] = distanceM;
            }

            if (distanceM <= m_awarenessRangeM)
            {
                eligible++;
            }
        }

        TxEvent ev;
        ev.timeS = timeS;
        ev.txNodeId = txNodeId;
        ev.seq = seq;
        ev.posTx = posTx;
        ev.eligibleRxCount = eligible;
        ev.eligibleRxCountsByBin = eligibleByBin;
        ev.rxDistanceAtTxM = rxDistanceAtTxM;

        m_txEvents.push_back(ev);

        if (m_txOut.is_open())
        {
            m_txOut << std::fixed << std::setprecision(6)
                    << ev.timeS << ","
                    << ev.txNodeId << ","
                    << ev.seq << ","
                    << ev.posTx.x << ","
                    << ev.posTx.y << ","
                    << ev.eligibleRxCount;
            for (const auto& count : ev.eligibleRxCountsByBin)
            {
                m_txOut << "," << count;
            }
            m_txOut << "\n";
            m_txOut.flush();
        }

        PruneOldEvents();
    }

    void LogRx(uint32_t txNodeId,
           uint32_t rxNodeId,
           uint32_t seq,
           Time tRx,
           const Vector& posTx,
           const Vector& posRx,
           double delayS)
    {
        const double timeS = tRx.GetSeconds();
        if (!IsInsideEvaluationWindow(timeS))
        {
            return;
        }

        const TxEvent* txEvent = FindTxEvent(txNodeId, seq);
        if (!txEvent)
        {
            return;
        }

        if (!IsInsideCore(txEvent->posTx) || !IsInsideCore(posRx))
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

        Vector txPosAtTx = txEvent->posTx;
        double distanceAtTxM = Distance2d(txEvent->posTx, posRx);
        auto distIt = txEvent->rxDistanceAtTxM.find(rxNodeId);
        if (distIt != txEvent->rxDistanceAtTxM.end())
        {
            distanceAtTxM = distIt->second;
        }

        const double distanceAtRxM = Distance2d(posTx, posRx);

        RxEvent ev;
        ev.timeS = timeS;
        ev.txNodeId = txNodeId;
        ev.rxNodeId = rxNodeId;
        ev.seq = seq;
        ev.posTx = txPosAtTx;
        ev.posRx = posRx;
        ev.distanceAtTxM = distanceAtTxM;
        ev.distanceAtRxM = distanceAtRxM;
        ev.pirS = pir;
        ev.delayS = delayS;

        m_rxEvents.push_back(ev);

        if (m_rxOut.is_open())
        {
            m_rxOut << std::fixed << std::setprecision(6)
                    << ev.timeS << ","
                    << ev.txNodeId << ","
                    << ev.rxNodeId << ","
                    << ev.seq << ","
                    << ev.posTx.x << ","
                    << ev.posTx.y << ","
                    << ev.posRx.x << ","
                    << ev.posRx.y << ","
                    << ev.distanceAtTxM << ","
                    << ev.distanceAtRxM << ",";

            if (std::isnan(ev.pirS))
            {
                m_rxOut << "nan,";
            }
            else
            {
                m_rxOut << ev.pirS << ",";
            }
            m_rxOut << ev.delayS << "\n";
            m_rxOut.flush();
        }

        PruneOldEvents();
    }

    void SampleAndWrite()
    {
        PruneOldEvents();

        const double now = Simulator::Now().GetSeconds();

        uint64_t denom = 0;
        std::set<std::tuple<uint32_t, uint32_t>> activeTxKeys;
        for (const auto& tx : m_txEvents)
        {
            denom += tx.eligibleRxCount;
            activeTxKeys.insert(std::make_tuple(tx.txNodeId, tx.seq));
        }

        std::set<std::tuple<uint32_t, uint32_t, uint32_t>> uniqueRxTriples;
        for (const auto& rx : m_rxEvents)
        {
            if (activeTxKeys.find(std::make_tuple(rx.txNodeId, rx.seq)) != activeTxKeys.end() &&
                rx.distanceAtTxM <= m_awarenessRangeM)
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
        const double sensingExclusionRatio =
            m_cbrLogger ? m_cbrLogger->ComputeSensingExclusionRatio(now) : 0.0;
        const TrafficContext traffic = ComputeTrafficContext();

        m_out << std::fixed << std::setprecision(6) << now << ","
              << m_currentP << ","
              << m_currentTb << ","
              << traffic.activeVehicleCount << ","
              << traffic.densityVehPerKm << ","
              << traffic.meanNeighbors150m << ","
              << traffic.meanNeighbors300m << ","
              << prr << ","
              << pirMean << ","
              << cbr << ","
              << sensingExclusionRatio << "\n";

        m_out.flush();

        Simulator::Schedule(Seconds(m_sampleTimeS), &KpiLogger::SampleAndWrite, this);
    }

    ~KpiLogger() override
    {
        if (m_out.is_open())
        {
            m_out.close();
        }
        if (m_txOut.is_open())
        {
            m_txOut.close();
        }
        if (m_rxOut.is_open())
        {
            m_rxOut.close();
        }
    }

  private:
    struct TrafficContext
    {
        uint32_t activeVehicleCount = 0;
        double densityVehPerKm = 0.0;
        double meanNeighbors150m = 0.0;
        double meanNeighbors300m = 0.0;
    };

    TrafficContext ComputeTrafficContext() const
    {
        std::vector<Vector> activePositions;
        activePositions.reserve(m_nodes.size());

        for (const auto& node : m_nodes)
        {
            const Vector pos = node->GetObject<MobilityModel>()->GetPosition();
            if (IsInsideCore(pos))
            {
                activePositions.push_back(pos);
            }
        }

        TrafficContext context;
        context.activeVehicleCount = static_cast<uint32_t>(activePositions.size());
        if (m_densityLengthM > 0.0)
        {
            context.densityVehPerKm =
                static_cast<double>(context.activeVehicleCount) / (m_densityLengthM / 1000.0);
        }

        if (activePositions.empty())
        {
            return context;
        }

        uint64_t neighborCount150m = 0;
        uint64_t neighborCount300m = 0;
        for (uint32_t i = 0; i < activePositions.size(); ++i)
        {
            for (uint32_t j = 0; j < activePositions.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const double distanceM = Distance2d(activePositions[i], activePositions[j]);
                if (distanceM <= 150.0)
                {
                    neighborCount150m++;
                }
                if (distanceM <= 300.0)
                {
                    neighborCount300m++;
                }
            }
        }

        const double activeCount = static_cast<double>(activePositions.size());
        context.meanNeighbors150m = static_cast<double>(neighborCount150m) / activeCount;
        context.meanNeighbors300m = static_cast<double>(neighborCount300m) / activeCount;
        return context;
    }

    void PruneOldEvents()
    {
        const double now = Simulator::Now().GetSeconds();
        const double cutoff = now - m_kpiWindowS;
        std::set<std::tuple<uint32_t, uint32_t>> activeTxKeys;

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

    int GetDistanceBinIndex(double distanceM) const
    {
        for (uint32_t i = 0; i + 1 < m_distanceBinEdgesM.size(); ++i)
        {
            const double low = m_distanceBinEdgesM[i];
            const double high = m_distanceBinEdgesM[i + 1];
            const bool isLastBin = (i + 2 == m_distanceBinEdgesM.size());
            if (distanceM >= low && (distanceM < high || (isLastBin && distanceM <= high)))
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    std::string BuildDistanceBinColumnName(uint32_t index) const
    {
        std::ostringstream os;
        os << "eligible_"
           << static_cast<uint32_t>(m_distanceBinEdgesM[index]) << "_"
           << static_cast<uint32_t>(m_distanceBinEdgesM[index + 1]) << "m";
        return os.str();
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

    const TxEvent* FindTxEvent(uint32_t txNodeId, uint32_t seq) const
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
    double m_sampleTimeS = 0.1;
    double m_evalStartS = 0.0;
    double m_evalEndS = -1.0;
    bool m_enableCoreFilter = true;
    std::string m_coreAxis = "y";
    double m_coreMin = 0.0;
    double m_coreMax = 0.0;
    double m_densityLengthM = 1.0;
    CbrLogger* m_cbrLogger = nullptr;

    double m_currentP = 23.0;
    double m_currentTb = 0.1;
    std::vector<double> m_distanceBinEdgesM = {0.0, 50.0, 100.0, 150.0, 200.0, 300.0};

    std::vector<Ptr<Node>> m_nodes;
    std::set<uint32_t> m_nodeIds;
    std::unordered_map<uint32_t, uint32_t> m_ipToNodeId;

    std::deque<TxEvent> m_txEvents;
    std::deque<RxEvent> m_rxEvents;
    std::set<std::tuple<uint32_t, uint32_t, uint32_t>> m_loggedRxTriples;
    std::map<std::pair<uint32_t, uint32_t>, double> m_lastRxTimePerPair;

    std::ofstream m_out;
    std::ofstream m_txOut;
    std::ofstream m_rxOut;
};

/****************************************************************
 * Custom CAM App
 ****************************************************************/

class CamApp : public Application
{
  public:
    void Setup(Ptr<Node> node,
               Ptr<KpiLogger> logger,
               Ipv4Address localIpv4,
               Ipv4Address groupIpv4,
               uint16_t port,
               uint32_t maxCamSizeBytes)
    {
        m_node = node;
        m_logger = logger;
        m_localIpv4 = localIpv4;
        m_groupIpv4 = groupIpv4;
        m_port = port;
        m_maxCamSizeBytes = maxCamSizeBytes;
    }

    void SetBeaconInterval(double tb)
    {
        m_tGenCamMaxS = std::max(m_tGenCamMinS, tb);
        if (!m_useEtsiCamGeneration)
        {
            m_beaconIntervalS = tb;
            if (m_running && m_txEvent.IsPending())
            {
                Simulator::Cancel(m_txEvent);
                m_txEvent = Simulator::Schedule(Seconds(m_beaconIntervalS), &CamApp::SendPacket, this);
            }
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

        m_socket->SetRecvCallback(MakeCallback(&CamApp::ReceivePacket, this));

        const double desyncS = GetDeterministicDesync();
        if (m_useEtsiCamGeneration)
        {
            m_checkEvent = Simulator::Schedule(Seconds(desyncS), &CamApp::CheckCamConditions, this);
        }
        else
        {
            m_txEvent = Simulator::Schedule(Seconds(desyncS), &CamApp::SendPacket, this);
        }
    }

    void StopApplication() override
    {
        m_running = false;

        if (m_txEvent.IsPending())
        {
            Simulator::Cancel(m_txEvent);
        }

        if (m_checkEvent.IsPending())
        {
            Simulator::Cancel(m_checkEvent);
        }

        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

  private:
    void SendPacket()
    {
        if (!m_running || !m_socket)
        {
            return;
        }

        Vector pos = m_node->GetObject<MobilityModel>()->GetPosition();
        const uint32_t camId = GenerationDeltaTimeMs(Simulator::Now());
        const std::string encodedCam = BuildEncodedCam(pos, camId);
        if (encodedCam.empty())
        {
            NS_FATAL_ERROR("Failed to encode CAM on node " << m_node->GetId());
        }

        Ptr<Packet> packet = Create<Packet>(
            reinterpret_cast<const uint8_t*>(encodedCam.data()),
            encodedCam.size());

        m_logger->LogTx(m_node->GetId(), camId, Simulator::Now(), pos);

        InetSocketAddress remote = InetSocketAddress(m_groupIpv4, m_port);
        m_socket->SendTo(packet, 0, remote);

        StoreLastCamState(pos);

        if (!m_useEtsiCamGeneration)
        {
            m_txEvent = Simulator::Schedule(Seconds(m_beaconIntervalS), &CamApp::SendPacket, this);
        }
    }

    std::string BuildEncodedCam(const Vector& pos, uint32_t generationDeltaTimeMs) const
    {
        CAM_t cam{};

        cam.header.protocolVersion = 2;
        cam.header.messageId = MessageId_cam;
        cam.header.stationId = m_node->GetId();

        cam.cam.generationDeltaTime = generationDeltaTimeMs;
        cam.cam.camParameters.basicContainer.stationType =
            TrafficParticipantType_passengerCar;
        cam.cam.camParameters.basicContainer.referencePosition.latitude =
            EncodeLatitudeFromLocalY(pos.y);
        cam.cam.camParameters.basicContainer.referencePosition.longitude =
            EncodeLongitudeFromLocalX(pos.x);
        cam.cam.camParameters.basicContainer.referencePosition.positionConfidenceEllipse
            .semiMajorAxisLength = SemiAxisLength_unavailable;
        cam.cam.camParameters.basicContainer.referencePosition.positionConfidenceEllipse
            .semiMinorAxisLength = SemiAxisLength_unavailable;
        cam.cam.camParameters.basicContainer.referencePosition.positionConfidenceEllipse
            .semiMajorAxisOrientation = Wgs84AngleValue_unavailable;
        cam.cam.camParameters.basicContainer.referencePosition.altitude.altitudeValue =
            AltitudeValue_unavailable;
        cam.cam.camParameters.basicContainer.referencePosition.altitude.altitudeConfidence =
            AltitudeConfidence_unavailable;

        BasicVehicleContainerHighFrequency_t& vehicle =
            cam.cam.camParameters.highFrequencyContainer.choice.basicVehicleContainerHighFrequency;
        cam.cam.camParameters.highFrequencyContainer.present =
            HighFrequencyContainer_PR_basicVehicleContainerHighFrequency;
        vehicle.heading.headingValue =
            ClampToLong(GetEtsiHeadingDeg() * 10.0, 0, HeadingValue_doNotUse - 1);
        vehicle.heading.headingConfidence = HeadingConfidence_unavailable;
        vehicle.speed.speedValue =
            ClampToLong(GetSpeedMps() * 100.0, SpeedValue_standstill, SpeedValue_outOfRange - 1);
        vehicle.speed.speedConfidence = SpeedConfidence_unavailable;
        vehicle.driveDirection = DriveDirection_forward;
        vehicle.vehicleLength.vehicleLengthValue = 45; // 4.5 m in 0.1 m units.
        vehicle.vehicleLength.vehicleLengthConfidenceIndication =
            VehicleLengthConfidenceIndication_noTrailerPresent;
        vehicle.vehicleWidth = 18; // 1.8 m in 0.1 m units.
        vehicle.longitudinalAcceleration.value = AccelerationValue_unavailable;
        vehicle.longitudinalAcceleration.confidence = AccelerationConfidence_unavailable;
        vehicle.curvature.curvatureValue = CurvatureValue_unavailable;
        vehicle.curvature.curvatureConfidence = CurvatureConfidence_unavailable;
        vehicle.curvatureCalculationMode = CurvatureCalculationMode_unavailable;
        vehicle.yawRate.yawRateValue = YawRateValue_unavailable;
        vehicle.yawRate.yawRateConfidence = YawRateConfidence_unavailable;

        std::vector<uint8_t> buffer(m_maxCamSizeBytes);
        asn_enc_rval_t result = uper_encode_to_buffer(&asn_DEF_CAM,
                                                       nullptr,
                                                       &cam,
                                                       buffer.data(),
                                                       buffer.size());
        if (result.encoded < 0)
        {
            return {};
        }

        const std::size_t encodedBytes = static_cast<std::size_t>((result.encoded + 7) / 8);
        if (encodedBytes > m_maxCamSizeBytes)
        {
            NS_FATAL_ERROR("Encoded CAM size " << encodedBytes
                                               << " exceeds configured packetSize/max buffer "
                                               << m_maxCamSizeBytes);
        }
        return std::string(reinterpret_cast<const char*>(buffer.data()), encodedBytes);
    }

    void CheckCamConditions()
    {
        if (!m_running || !m_socket)
        {
            return;
        }

        const Vector pos = m_node->GetObject<MobilityModel>()->GetPosition();
        const double nowS = Simulator::Now().GetSeconds();

        if (!m_hasLastCamState)
        {
            SendPacket();
            ScheduleNextCamConditionCheck();
            return;
        }

        const double elapsedS = nowS - m_lastCamTimeS;
        if (elapsedS + 1e-9 < m_tGenCamMinS)
        {
            ScheduleNextCamConditionCheck();
            return;
        }

        const double headingDiffDeg = SmallestHeadingDiffDeg(GetHeadingDeg(), m_lastHeadingDeg);
        const double distanceDiffM = Distance2d(pos, m_lastCamPosition);
        const double speedDiffMps = std::abs(GetSpeedMps() - m_lastSpeedMps);

        const bool dynamicTrigger = std::abs(headingDiffDeg) > m_headingDeltaThresholdDeg ||
                                    distanceDiffM > m_positionDeltaThresholdM ||
                                    speedDiffMps > m_speedDeltaThresholdMps;
        const bool maxIntervalTrigger = elapsedS + 1e-9 >= m_tGenCamMaxS;

        if (dynamicTrigger || maxIntervalTrigger)
        {
            SendPacket();
        }

        ScheduleNextCamConditionCheck();
    }

    void ScheduleNextCamConditionCheck()
    {
        if (m_running)
        {
            m_checkEvent =
                Simulator::Schedule(Seconds(m_tCheckCamGenS), &CamApp::CheckCamConditions, this);
        }
    }

    void StoreLastCamState(const Vector& pos)
    {
        m_lastCamPosition = pos;
        m_lastHeadingDeg = GetHeadingDeg();
        m_lastSpeedMps = GetSpeedMps();
        m_lastCamTimeS = Simulator::Now().GetSeconds();
        m_hasLastCamState = true;
    }

    double GetSpeedMps() const
    {
        const Vector velocity = m_node->GetObject<MobilityModel>()->GetVelocity();
        return std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    }

    double GetHeadingDeg() const
    {
        const Vector velocity = m_node->GetObject<MobilityModel>()->GetVelocity();
        if (std::abs(velocity.x) < 1e-9 && std::abs(velocity.y) < 1e-9)
        {
            return m_hasLastCamState ? m_lastHeadingDeg : 0.0;
        }

        constexpr double pi = 3.14159265358979323846;
        double headingDeg = std::atan2(velocity.y, velocity.x) * 180.0 / pi;
        if (headingDeg < 0.0)
        {
            headingDeg += 360.0;
        }
        return headingDeg;
    }

    double GetEtsiHeadingDeg() const
    {
        double headingDeg = 90.0 - GetHeadingDeg();
        while (headingDeg < 0.0)
        {
            headingDeg += 360.0;
        }
        while (headingDeg >= 360.0)
        {
            headingDeg -= 360.0;
        }
        return headingDeg;
    }

    static double SmallestHeadingDiffDeg(double currentDeg, double previousDeg)
    {
        double diff = currentDeg - previousDeg;
        while (diff > 180.0)
        {
            diff -= 360.0;
        }
        while (diff < -180.0)
        {
            diff += 360.0;
        }
        return diff;
    }

    double GetDeterministicDesync() const
    {
        const double fractional = std::fmod((m_node->GetId() + 1) * 0.6180339887498948, 1.0);
        const double windowS = m_useEtsiCamGeneration ? m_tGenCamMaxS : m_beaconIntervalS;
        return fractional * std::max(m_tCheckCamGenS, windowS);
    }

    void ReceivePacket(Ptr<Socket> socket)
    {
        Address from;
        Ptr<Packet> packet = socket->RecvFrom(from);

        if (!packet || packet->GetSize() == 0)
        {
            return;
        }

        std::vector<uint8_t> buffer(packet->GetSize());
        packet->CopyData(buffer.data(), buffer.size());

        CAM_t* cam = nullptr;
        asn_dec_rval_t decodeResult = uper_decode_complete(nullptr,
                                                           &asn_DEF_CAM,
                                                           reinterpret_cast<void**>(&cam),
                                                           buffer.data(),
                                                           buffer.size());
        if (decodeResult.code != RC_OK || !cam)
        {
            if (cam)
            {
                ASN_STRUCT_FREE(asn_DEF_CAM, cam);
            }
            return;
        }

        InetSocketAddress src = InetSocketAddress::ConvertFrom(from);
        Ipv4Address srcIp = src.GetIpv4();
        uint32_t txNodeId = m_logger->ResolveNodeIdFromIpv4(srcIp);
        if (txNodeId == std::numeric_limits<uint32_t>::max())
        {
            return;
        }
        const uint32_t camId = static_cast<uint32_t>(cam->cam.generationDeltaTime);
        const double delayS = ComputeCamDelayS(camId, Simulator::Now());
        const uint32_t stationId = static_cast<uint32_t>(cam->header.stationId);
        ASN_STRUCT_FREE(asn_DEF_CAM, cam);

        if (stationId != txNodeId)
        {
            return;
        }

        Ptr<Node> txNode = g_nodeIdToNode[txNodeId];
        if (!txNode)
        {
            return;
        }

        Vector posTx = txNode->GetObject<MobilityModel>()->GetPosition();
        Vector posRx = m_node->GetObject<MobilityModel>()->GetPosition();

        m_logger->LogRx(txNodeId,
                        m_node->GetId(),
                        camId,
                        Simulator::Now(),
                        posTx,
                        posRx,
                        delayS);
    }

    Ptr<Node> m_node;
    Ptr<KpiLogger> m_logger;
    Ptr<Socket> m_socket;

    bool m_running = false;
    bool m_useEtsiCamGeneration = true;
    EventId m_txEvent;
    EventId m_checkEvent;

    double m_beaconIntervalS = 0.1;
    double m_tCheckCamGenS = 0.1;
    double m_tGenCamMinS = 0.1;
    double m_tGenCamMaxS = 1.0;
    double m_headingDeltaThresholdDeg = 4.0;
    double m_positionDeltaThresholdM = 4.0;
    double m_speedDeltaThresholdMps = 0.5;
    uint32_t m_maxCamSizeBytes = 300;

    bool m_hasLastCamState = false;
    double m_lastCamTimeS = 0.0;
    double m_lastHeadingDeg = 0.0;
    double m_lastSpeedMps = 0.0;
    Vector m_lastCamPosition;

    Ipv4Address m_localIpv4;
    Ipv4Address m_groupIpv4;
    uint16_t m_port = 8000;
};

static std::map<uint32_t, Ptr<CamApp>> g_apps;

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
    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUeMac/SensingAlgorithm",
                    MakeCallback(&CbrLogger::RecordSensingAlgorithm, cbrLogger));

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
                  const std::string& cbrCsv,
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
                  double densityLengthM,
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
        << "  \"cbr_csv\": \"" << JsonEscape(cbrCsv) << "\",\n"
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
        << "  \"density_length_m\": " << densityLengthM << ",\n"
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
           "timestamp: active_vehicle_count_core is the number of evaluated vehicles inside the "
           "core region, density_veh_per_km_core divides that count by density_length_m, and "
           "mean_neighbors_150m/300m are per-vehicle instantaneous neighbor counts among active "
           "core vehicles using 2D Euclidean distance\",\n"
        << "  \"cbr_definition\": \"PHY busy-time fraction over the 1.0 s window from "
           "NrSpectrumPhy::ChannelOccupied, averaged over evaluated UE nodes; evaluated nodes "
           "with no busy interval in the window contribute zero busy time\",\n"
        << "  \"cbr_primary_source\": \"NrSpectrumPhy::ChannelOccupied busy-time trace\",\n"
        << "  \"sensing_exclusion_ratio_definition\": \"N_excluded_candidate_resources / "
           "N_initial_candidate_resources over the 1.0 s window from "
           "NrSlUeMac::SensingAlgorithm; this is a resource-selection diagnostic and is not "
           "used as CBR\",\n"
        << "  \"cbr_alignment_method\": \"CBR and PRR/PIR are sampled by the simulator at the same "
           "0.1 s timestamps; no interpolation is applied\",\n"
        << "  \"deepc_dataset_columns\": "
           "\"time_s,tx_power_dbm,beacon_interval_s,active_vehicle_count_core,"
           "density_veh_per_km_core,mean_neighbors_150m,mean_neighbors_300m,"
           "prr_150m,pir_s,cbr,sensing_exclusion_ratio\",\n"
        << "  \"distance_bins_m\": [0, 50, 100, 150, 200, 300]\n"
        << "}\n";
}

/****************************************************************
 * main
 ****************************************************************/

int
main(int argc, char* argv[])
{
    // File paths
    std::string mobilityCsv = "data/processed/ngsim_us101_mainline_0p1s.csv";
    std::string inputCsv = "data/processed/prbs_schedule.csv";
    std::string kpiCsv = "data/output/kpi_timeseries_run01.csv";
    std::string cbrCsv = "data/output/cbr_timeseries_run01.csv";

    // Simulation time
    double simTimeSeconds = 10.0;
    double warmupS = 10.0;
    double cooldownS = 10.0;
    uint32_t seed = 12345;
    uint32_t run = 1;

    // Traffic / app
    bool useIPv6 = false;
    uint32_t maxCamSizeBytes = 300;
    uint16_t port = 8000;
    bool useInputSchedule = false;
    bool etsiCamGeneration = true;
    double fixedBeaconIntervalS = 0.1;

    // SL bearer activation
    Time slBearersActivationTime = Seconds(2.0);
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
    bool enableSensing = true;
    uint16_t t1 = 2;
    uint16_t t2 = 33;
    int slThresPsschRsrp = -128;
    bool enableChannelRandomness = true;
    uint16_t channelUpdatePeriod = 500;
    uint16_t mcs = 6;
    double awarenessRangeM = 150.0;
    double kpiWindowS = 1.0;
    double sampleTimeS = 0.1;
    uint32_t maxVehicles = 0; // 0 means all vehicles
    bool enableCoreFilter = true;
    std::string coreAxis = "auto";
    double coreGuardBandM = 120.0;
    double coreMin = std::numeric_limits<double>::quiet_NaN();
    double coreMax = std::numeric_limits<double>::quiet_NaN();

    CommandLine cmd(__FILE__);
    cmd.AddValue("mobilityCsv", "Processed NGSIM mobility CSV", mobilityCsv);
    cmd.AddValue("inputCsv", "PRBS schedule CSV", inputCsv);
    cmd.AddValue("kpiCsv", "Output KPI CSV", kpiCsv);
    cmd.AddValue("cbrCsv", "Output standalone CBR CSV", cbrCsv);
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
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(std::abs(kpiWindowS - 1.0) > 1e-9,
                    "CBR sensing window is fixed by the experiment definition: kpiWindow must be 1.0 s");
    NS_ABORT_MSG_IF(std::abs(sampleTimeS - 0.1) > 1e-9,
                    "CBR output sampling is fixed by the experiment definition: sampleTime must be 0.1 s");
    NS_ABORT_MSG_IF(!enableSensing,
                    "CBR requires NR sidelink sensing. Run with --enableSensing=true.");
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

    const double densityLengthM = enableCoreFilter ? (coreMax - coreMin) : (studyMax - studyMin);
    NS_ABORT_MSG_IF(densityLengthM <= 0.0, "Density evaluation length must be positive");

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

    // Create nodes
    NodeContainer allSlUesContainer;
    allSlUesContainer.Create(vehicleIds.size());

    for (uint32_t i = 0; i < vehicleIds.size(); ++i)
    {
        uint32_t vehicleId = vehicleIds[i];
        Ptr<Node> node = allSlUesContainer.Get(i);

        g_vehicleIdToNodeId[vehicleId] = node->GetId();
        g_nodeIdToVehicleId[node->GetId()] = vehicleId;
        g_nodeIdToNode[node->GetId()] = node;

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
    nrHelper->SetUeMacAttribute("EnableSensing", BooleanValue(true));
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

    /**************** IP stack + SL bearer ****************/
    InternetStackHelper internet;
    internet.Install(allSlUesContainer);

    uint32_t dstL2Id = 255;
    Ipv4Address groupAddress4("225.0.0.0");
    Address remoteAddress;
    Address localAddress;

    Ptr<LteSlTft> tft;
    SidelinkInfo slInfo;
    slInfo.m_castType = SidelinkInfo::CastType::Groupcast;
    slInfo.m_dstL2Id = dstL2Id;
    slInfo.m_rri = MilliSeconds(reservationPeriod);
    slInfo.m_dynamic = false;
    slInfo.m_pdb = delayBudget;
    slInfo.m_harqEnabled = harqEnabled;

    NS_ABORT_MSG_IF(useIPv6, "This merged file currently supports IPv4 only.");

    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(allSlUesNetDeviceContainer);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < allSlUesContainer.GetN(); ++u)
    {
        Ptr<Node> ueNode = allSlUesContainer.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    remoteAddress = InetSocketAddress(groupAddress4, port);
    localAddress = InetSocketAddress(Ipv4Address::GetAny(), port);

    tft = Create<LteSlTft>(LteSlTft::Direction::TRANSMIT, groupAddress4, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime, allSlUesNetDeviceContainer, tft);

    tft = Create<LteSlTft>(LteSlTft::Direction::RECEIVE, groupAddress4, slInfo);
    nrSlHelper->ActivateNrSlBearer(slBearersActivationTime, allSlUesNetDeviceContainer, tft);

    /**************** KPI logger ****************/
    CbrLogger cbrLogger;
    cbrLogger.Setup(1.0, 0.1, cbrCsv);
    cbrLogger.SetEvaluationWindow(evalStartS, evalEndS);
    std::vector<uint32_t> evaluatedNodeIds;
    evaluatedNodeIds.reserve(allSlUesContainer.GetN());
    for (uint32_t i = 0; i < allSlUesContainer.GetN(); ++i)
    {
        evaluatedNodeIds.push_back(allSlUesContainer.Get(i)->GetId());
    }
    cbrLogger.SetEvaluatedNodeIds(evaluatedNodeIds);
    cbrLogger.SetNodeEvaluationFilter(
        [enableCoreFilter, coreAxis, coreMin, coreMax](uint32_t nodeId, double) {
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
    cbrLogger.Start();

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
                      coreMax,
                      densityLengthM);

    for (uint32_t i = 0; i < allSlUesContainer.GetN(); ++i)
    {
        Ipv4Address localAddr =
            allSlUesContainer.Get(i)->GetObject<Ipv4L3Protocol>()->GetAddress(1, 0).GetLocal();
        logger->RegisterNodeIpv4(allSlUesContainer.Get(i)->GetId(), localAddr);
    }
    logger->SetCurrentInputs(txPower, fixedBeaconIntervalS);
    ApplyTxPowerToAllUes(allSlUesNetDeviceContainer, txPower);

    /**************** Custom CAM apps ****************/
    for (uint32_t i = 0; i < allSlUesContainer.GetN(); ++i)
    {
        Ptr<Node> node = allSlUesContainer.Get(i);
        Ipv4Address localAddr =
            node->GetObject<Ipv4L3Protocol>()->GetAddress(1, 0).GetLocal();

        Ptr<CamApp> app = CreateObject<CamApp>();
        app->Setup(node, logger, localAddr, groupAddress4, port, maxCamSizeBytes);
        app->SetEtsiCamGeneration(etsiCamGeneration);
        app->SetBeaconInterval(fixedBeaconIntervalS);
        node->AddApplication(app);
        app->SetStartTime(slBearersActivationTime);
        app->SetStopTime(Seconds(simTimeSeconds));
        g_apps[node->GetId()] = app;
    }

    /**************** Time-varying input schedule ****************/
    if (useInputSchedule)
    {
        ScheduleInputUpdates(logger, allSlUesNetDeviceContainer);
    }

    /**************** Periodic KPI sampling ****************/
    Simulator::Schedule(Seconds(sampleTimeS), &KpiLogger::SampleAndWrite, logger);

    WriteMetadataJson(kpiCsv,
                      cbrCsv,
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
                      densityLengthM,
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

    return 0;
}
