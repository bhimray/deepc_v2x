#include "cbr-logger.h"

#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>

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
    m_out << "time_s,cbr\n";
}

void
CbrLogger::RecordObservation(double timeS, bool busy)
{
    Obs o;
    o.time = timeS;
    o.busy = busy;

    m_obs.push_back(o);
    Prune(timeS);
}

void
CbrLogger::RecordResourceObservation(double timeS, uint32_t busyResources, uint32_t totalResources)
{
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
    if (duration.IsZero() || duration.IsNegative())
    {
        return;
    }

    const double now = Simulator::Now().GetSeconds();
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
    if (m_busyIntervalsByContext.empty())
    {
        return 0.0;
    }

    double cbrSum = 0.0;
    for (const auto& kv : m_busyIntervalsByContext)
    {
        cbrSum += ComputeBusyFractionForContext(kv.second, now);
    }
    return cbrSum / static_cast<double>(m_busyIntervalsByContext.size());
}

double
CbrLogger::ComputeCbr(double now)
{
    Prune(now);

    const double sensingCbr = ComputeSensingCbr(now);
    if (sensingCbr >= 0.0)
    {
        return std::min(1.0, std::max(0.0, sensingCbr));
    }

    return std::min(1.0, std::max(0.0, ComputeChannelOccupiedCbr(now)));
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

    m_out << now << "," << cbr << "\n";
    m_out.flush();

    Simulator::Schedule(Seconds(m_step), &CbrLogger::Sample, this);
}

void
CbrLogger::Start()
{
    Simulator::Schedule(Seconds(m_step), &CbrLogger::Sample, this);
}
