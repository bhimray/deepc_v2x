#pragma once

#include "ns3/nr-sl-phy-mac-common.h"
#include "ns3/nr-sl-ue-mac.h"
#include "ns3/nstime.h"

#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

class CbrLogger
{
  public:
    void Setup(double windowS);
    void Setup(double windowS, double stepS, const std::string& filename);
    void SetEvaluationWindow(double startS, double endS);
    void SetNodeEvaluationFilter(std::function<bool(uint32_t, double)> filter);
    void SetEvaluatedNodeIds(const std::vector<uint32_t>& nodeIds);

    void RecordObservation(double timeS, bool busy);
    void RecordResourceObservation(double timeS, uint32_t busyResources, uint32_t totalResources);
    void RecordSensingAlgorithm(
        std::string context,
        const ns3::NrSlUeMac::SensingTraceReport& report,
        const std::list<ns3::SlResourceInfo>& candidateResources,
        const std::list<ns3::SensingData>& sensingData,
        const std::list<ns3::SfnSf>& transmitHistory);
    void RecordChannelOccupied(std::string context, ns3::Time duration);

    double GetCurrentCbr();
    double ComputeCbr(double now);
    double ComputeSensingExclusionRatio(double now);

    void Start();

  private:
    struct Obs
    {
        double time;
        bool busy;
    };

    struct ResourceObs
    {
        double time;
        uint32_t busy;
        uint32_t total;
    };

    struct BusyInterval
    {
        double startS;
        double endS;
    };

    double ComputeSensingCbr(double now) const;
    double ComputeChannelOccupiedCbr(double now) const;
    double ComputeBusyFractionForContext(const std::deque<BusyInterval>& intervals,
                                         double now) const;
    bool IsInsideEvaluationWindow(double timeS) const;
    bool ShouldRecordContext(const std::string& context, double timeS) const;
    bool TryParseNodeId(const std::string& context, uint32_t& nodeId) const;
    void Prune(double now);
    void Sample();

    std::deque<Obs> m_obs;
    std::deque<ResourceObs> m_resourceObs;
    std::map<std::string, std::deque<BusyInterval>> m_busyIntervalsByContext;
    std::function<bool(uint32_t, double)> m_nodeEvaluationFilter;
    std::set<uint32_t> m_evaluatedNodeIds;
    double m_window{1.0};
    double m_step{0.1};
    double m_evalStartS{0.0};
    double m_evalEndS{-1.0};
    bool m_writeCsv{false};
    std::ofstream m_out;
};
