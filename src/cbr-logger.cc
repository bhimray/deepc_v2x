#include "cbr-logger.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cstdlib>
#include <map>

using namespace ns3;

void
CbrLogger::Setup(double windowS, double stepS, const std::string& filename)
{
    m_window = windowS;
    m_step = stepS;

    m_out.open(filename.c_str(), std::ios::out | std::ios::trunc);
    if (!m_out.is_open())
    {
        NS_FATAL_ERROR("Cannot open CBR output CSV: " << filename);
    }
    m_out << "time_s,cbr,sensing_exclusion_ratio\n";
}

void
CbrLogger::SetEvaluationWindow(double startS, double endS)
{
    m_evalStartS = startS;
    m_evalEndS = endS;
}

void
CbrLogger::SetNodeEvaluationFilter(std::function<bool(uint32_t, double)> filter)
{
    m_nodeEvaluationFilter = filter;
}

void
CbrLogger::SetEvaluatedNodeIds(const std::vector<uint32_t>& nodeIds)
{
    m_evaluatedNodeIds.clear();
    m_evaluatedNodeIds.insert(nodeIds.begin(), nodeIds.end());
}

void
CbrLogger::RecordObservation(double timeS, bool busy)
{
    if (!IsInsideEvaluationWindow(timeS))
    {
        return;
    }

    Obs o;
    o.time = timeS;
    o.busy = busy;

    m_obs.push_back(o);
    Prune(timeS);
}

void
CbrLogger::RecordResourceObservation(double timeS, uint32_t busyResources, uint32_t totalResources)
{
    if (!IsInsideEvaluationWindow(timeS))
    {
        return;
    }

    if (totalResources == 0)
    {
        return;
    }

    ResourceObs o;
    o.time = timeS;
    o.busy = std::min(busyResources, totalResources);
    o.total = totalResources;

    m_resourceObs.push_back(o);
    Prune(timeS);
}

void
CbrLogger::RecordSensingAlgorithm(std::string context,
                                  const NrSlUeMac::SensingTraceReport& report,
                                  const std::list<SlResourceInfo>& candidateResources,
                                  const std::list<SensingData>& sensingData,
                                  const std::list<SfnSf>& transmitHistory)
{
    (void)context;
    (void)sensingData;
    (void)transmitHistory;

    const double now = Simulator::Now().GetSeconds();
    if (!ShouldRecordContext(context, now))
    {
        return;
    }

    const uint32_t totalResources = report.m_initialCandidateResourcesSize;
    if (totalResources == 0)
    {
        return;
    }

    const uint32_t availableResources =
        std::min<uint32_t>(candidateResources.size(), totalResources);
    const uint32_t busyResources = totalResources - availableResources;
    RecordResourceObservation(now, busyResources, totalResources);
}

void
CbrLogger::RecordChannelOccupied(std::string context, Time duration)
{
    const double now = Simulator::Now().GetSeconds();
    if (!ShouldRecordContext(context, now))
    {
        return;
    }

    if (duration.IsZero() || duration.IsNegative())
    {
        return;
    }

    BusyInterval interval;
    interval.startS = now;
    interval.endS = now + duration.GetSeconds();
    m_busyIntervalsByContext[context].push_back(interval);
    Prune(now);
}

double
CbrLogger::ComputeSensingCbr(double now) const
{
    const double cutoff = now - m_window;
    uint64_t busy = 0;
    uint64_t total = 0;

    for (const auto& o : m_resourceObs)
    {
        if (o.time >= cutoff && o.time <= now)
        {
            busy += o.busy;
            total += o.total;
        }
    }

    if (total > 0)
    {
        return static_cast<double>(busy) / static_cast<double>(total);
    }

    uint64_t scalarBusy = 0;
    uint64_t scalarTotal = 0;
    for (const auto& o : m_obs)
    {
        if (o.time >= cutoff && o.time <= now)
        {
            scalarTotal++;
            if (o.busy)
            {
                scalarBusy++;
            }
        }
    }

    return (scalarTotal > 0) ? static_cast<double>(scalarBusy) / static_cast<double>(scalarTotal)
                             : -1.0;
}

double
CbrLogger::ComputeBusyFractionForContext(const std::deque<BusyInterval>& intervals,
                                         double now) const
{
    const double cutoff = now - m_window;
    std::vector<BusyInterval> clipped;
    for (const auto& interval : intervals)
    {
        const double start = std::max(interval.startS, cutoff);
        const double end = std::min(interval.endS, now);
        if (end > start)
        {
            clipped.push_back({start, end});
        }
    }

    if (clipped.empty())
    {
        return 0.0;
    }

    std::sort(clipped.begin(),
              clipped.end(),
              [](const BusyInterval& a, const BusyInterval& b) {
                  return a.startS < b.startS;
              });

    double busyS = 0.0;
    double mergedStart = clipped.front().startS;
    double mergedEnd = clipped.front().endS;
    for (uint32_t i = 1; i < clipped.size(); ++i)
    {
        if (clipped[i].startS <= mergedEnd)
        {
            mergedEnd = std::max(mergedEnd, clipped[i].endS);
        }
        else
        {
            busyS += mergedEnd - mergedStart;
            mergedStart = clipped[i].startS;
            mergedEnd = clipped[i].endS;
        }
    }
    busyS += mergedEnd - mergedStart;

    return std::min(1.0, busyS / m_window);
}

double
CbrLogger::ComputeChannelOccupiedCbr(double now) const
{
    std::map<uint32_t, std::pair<double, uint32_t>> busyFractionByNode;
    for (const auto& kv : m_busyIntervalsByContext)
    {
        uint32_t nodeId = 0;
        if (!TryParseNodeId(kv.first, nodeId))
        {
            continue;
        }

        if (m_nodeEvaluationFilter && !m_nodeEvaluationFilter(nodeId, now))
        {
            continue;
        }

        auto& aggregate = busyFractionByNode[nodeId];
        aggregate.first += ComputeBusyFractionForContext(kv.second, now);
        aggregate.second++;
    }

    double cbrSum = 0.0;
    for (const auto& kv : busyFractionByNode)
    {
        if (kv.second.second > 0)
        {
            cbrSum += kv.second.first / static_cast<double>(kv.second.second);
        }
    }

    if (!m_evaluatedNodeIds.empty())
    {
        uint32_t denominator = 0;
        for (const auto nodeId : m_evaluatedNodeIds)
        {
            if (!m_nodeEvaluationFilter || m_nodeEvaluationFilter(nodeId, now))
            {
                denominator++;
            }
        }
        return (denominator > 0) ? cbrSum / static_cast<double>(denominator) : 0.0;
    }

    if (busyFractionByNode.empty())
    {
        return 0.0;
    }

    return cbrSum / static_cast<double>(busyFractionByNode.size());
}

double
CbrLogger::ComputeCbr(double now)
{
    Prune(now);
    return std::min(1.0, std::max(0.0, ComputeChannelOccupiedCbr(now)));
}

double
CbrLogger::ComputeSensingExclusionRatio(double now)
{
    Prune(now);

    const double sensingRatio = ComputeSensingCbr(now);
    return (sensingRatio >= 0.0) ? std::min(1.0, std::max(0.0, sensingRatio)) : 0.0;
}

bool
CbrLogger::IsInsideEvaluationWindow(double timeS) const
{
    return timeS >= m_evalStartS && (m_evalEndS < 0.0 || timeS <= m_evalEndS);
}

bool
CbrLogger::TryParseNodeId(const std::string& context, uint32_t& nodeId) const
{
    const std::string marker = "/NodeList/";
    const std::size_t start = context.find(marker);
    if (start == std::string::npos)
    {
        return false;
    }

    const std::size_t idStart = start + marker.size();
    const std::size_t idEnd = context.find('/', idStart);
    const std::string idText = context.substr(idStart, idEnd - idStart);
    if (idText.empty())
    {
        return false;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(idText.c_str(), &end, 10);
    if (end == idText.c_str() || *end != '\0')
    {
        return false;
    }

    nodeId = static_cast<uint32_t>(parsed);
    return true;
}

bool
CbrLogger::ShouldRecordContext(const std::string& context, double timeS) const
{
    if (!IsInsideEvaluationWindow(timeS))
    {
        return false;
    }

    if (!m_nodeEvaluationFilter)
    {
        return true;
    }

    uint32_t nodeId = 0;
    if (!TryParseNodeId(context, nodeId))
    {
        return false;
    }

    return m_nodeEvaluationFilter(nodeId, timeS);
}

double
CbrLogger::GetCurrentCbr()
{
    return ComputeCbr(Simulator::Now().GetSeconds());
}

void
CbrLogger::Prune(double now)
{
    const double cutoff = now - m_window;

    while (!m_obs.empty() && m_obs.front().time < cutoff)
    {
        m_obs.pop_front();
    }
    while (!m_resourceObs.empty() && m_resourceObs.front().time < cutoff)
    {
        m_resourceObs.pop_front();
    }

    for (auto& kv : m_busyIntervalsByContext)
    {
        auto& intervals = kv.second;
        while (!intervals.empty() && intervals.front().endS < cutoff)
        {
            intervals.pop_front();
        }
    }
}

void
CbrLogger::Sample()
{
    const double now = Simulator::Now().GetSeconds();
    const double cbr = ComputeCbr(now);
    const double sensingExclusionRatio = ComputeSensingExclusionRatio(now);

    if (IsInsideEvaluationWindow(now))
    {
        m_out << now << "," << cbr << "," << sensingExclusionRatio << "\n";
        m_out.flush();
    }

    Simulator::Schedule(Seconds(m_step), &CbrLogger::Sample, this);
}

void
CbrLogger::Start()
{
    Simulator::Schedule(Seconds(m_step), &CbrLogger::Sample, this);
}
