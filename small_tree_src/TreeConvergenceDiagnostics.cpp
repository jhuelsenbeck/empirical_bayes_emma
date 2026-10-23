#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "Msg.hpp"
#include "TreeConvergenceDiagnostics.hpp"



TreeConvergenceDiagnostics::TreeConvergenceDiagnostics(TreeCache* tc) :
    treeCache(tc), credible95Mass(0.0), numCredibleSet95Trees(0) {

    if (treeCache == nullptr)
        Msg::error("TreeConvergenceDiagnostics given null TreeCache");

    TreeCacheMap& cache = treeCache->getCache();

    posteriorRank.reserve(cache.size());

    for (auto& [hash, info] : cache)
        {
        if (info == nullptr)
            Msg::error("Null TreeInfo in TreeCache");

        posteriorRank.push_back(info);
        }

    std::sort(
        posteriorRank.begin(),
        posteriorRank.end(),
        [](const TreeInfo* a, const TreeInfo* b)
            {
            if (a->posteriorProbability != b->posteriorProbability)
                return a->posteriorProbability > b->posteriorProbability;

            return a->hash < b->hash;
            }
        );

    double cumulative = 0.0;

    for (size_t i=0; i<posteriorRank.size(); i++)
        {
        TreeInfo* info = posteriorRank[i];

        if (cumulative < 0.95)
            {
            inCredibleSet95[info->hash] = true;
            cumulative += info->posteriorProbability;
            credible95Mass += info->posteriorProbability;
            numCredibleSet95Trees++;
            }
        }
}

void TreeConvergenceDiagnostics::writeStatsHeader(std::ostream& os, size_t numReplicates) {

    os << "numReps";
    os << "\tmeanDiscoveredMass\tseDiscoveredMass";
    os << "\tmeanUndiscoveredMass\tseUndiscoveredMass";
    os << "\tmeanCredible95MassDiscovered\tseCredible95MassDiscovered";
    os << "\ttau95Fraction\tmeanTau95\tseTau95";
    os << "\tmeanMeanTrueP\tseMeanTrueP";

    for (size_t i=0; i<numReplicates; i++)
        os << "\tmapFirstHitCycle_rep" << i+1;
}

// Advance one chain's running totals over the samples it has drawn since the last call. In sample
// order, as computeChainStats() scans: a tree's posterior joins the discovered mass when it is first
// seen, tau95 is the sample at which that mass first reaches 0.95, and every sample adds its tree's
// posterior to sumP. The discovered masses are therefore summed in discovery order, where
// computeChainStats() summed them in hash-map order; the totals agree to rounding.
void TreeConvergenceDiagnostics::advance(ChainCursor& cur, const TreeSamples* samples) {

    const std::vector<uint64_t>& trace = samples->getTrees();
    uint64_t mapHash = posteriorRank.empty() ? 0 : posteriorRank[0]->hash;
    for (size_t i = cur.processed; i < trace.size(); i++)
        {
        uint64_t h = trace[i];
        if (cur.lastInfo == nullptr || h != cur.lastHash)
            {
            cur.lastInfo = treeCache->getTreeInfo(h);
            cur.lastHash = h;
            if (cur.lastInfo == nullptr)
                Msg::error("Sampled tree not found in TreeCache");
            }
        double p = cur.lastInfo->posteriorProbability;

        if (cur.visited.empty() == true)
            cur.visited.assign(treeCache->size(), 0);
        unsigned char& seen = cur.visited[cur.lastInfo->index];
        if (seen == 0)
            {
            seen = 1;
            cur.discoveredMass += p;
            if (inCredibleSet95.find(h) != inCredibleSet95.end())
                cur.credible95MassDiscovered += p;
            }
        if (cur.tau95WasReached == false && cur.discoveredMass >= 0.95)
            {
            cur.tau95WasReached = true;
            cur.tau95 = (double)(i + 1);
            }
        if (cur.mapFirstHitCycle < 0 && h == mapHash && posteriorRank.empty() == false)
            cur.mapFirstHitCycle = (int)(i + 1);
        cur.sumP += p;
        }
    cur.processed = trace.size();
}

void TreeConvergenceDiagnostics::writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec) {

    if (cursors.size() != sampleVec.size())
        cursors.assign(sampleVec.size(), ChainCursor());

    std::vector<ChainStats> stats;
    stats.reserve(sampleVec.size());

    for (size_t c = 0; c < sampleVec.size(); c++)
        {
        const TreeSamples* s = sampleVec[c];
        ChainStats st;
        if (s != nullptr && s->getNumSamples() > 0)
            {
            ChainCursor& cur = cursors[c];
            advance(cur, s);
            st.numSamples = s->getNumSamples();
            st.discoveredMass = cur.discoveredMass;
            st.undiscoveredMass = 1.0 - cur.discoveredMass;
            if (st.undiscoveredMass < 0.0 && st.undiscoveredMass > -1.0e-12)
                st.undiscoveredMass = 0.0;
            st.credible95MassDiscovered = cur.credible95MassDiscovered;
            st.tau95WasReached = cur.tau95WasReached;
            st.tau95 = cur.tau95;
            st.mapFirstHitCycle = cur.mapFirstHitCycle;
            st.meanTruePosteriorProbability = cur.sumP / (double)st.numSamples;
            }
        stats.push_back(st);
        }

    std::vector<double> discoveredMass;
    std::vector<double> undiscoveredMass;
    std::vector<double> credible95MassDiscovered;
    std::vector<double> tau95Hit;
    std::vector<double> tau95;
    std::vector<bool>   tau95WasReached;
    std::vector<double> meanTrueP;

    discoveredMass.reserve(stats.size());
    undiscoveredMass.reserve(stats.size());
    credible95MassDiscovered.reserve(stats.size());
    tau95Hit.reserve(stats.size());
    tau95.reserve(stats.size());
    tau95WasReached.reserve(stats.size());
    meanTrueP.reserve(stats.size());

    for (const ChainStats& s : stats)
        {
        discoveredMass.push_back(s.discoveredMass);
        undiscoveredMass.push_back(s.undiscoveredMass);
        credible95MassDiscovered.push_back(s.credible95MassDiscovered);

        tau95Hit.push_back(s.tau95WasReached ? 1.0 : 0.0);
        tau95.push_back(s.tau95);
        tau95WasReached.push_back(s.tau95WasReached);

        meanTrueP.push_back(s.meanTruePosteriorProbability);
        }

    os << stats.size();

    writeMeanSe(os, summarize(discoveredMass));
    writeMeanSe(os, summarize(undiscoveredMass));
    writeMeanSe(os, summarize(credible95MassDiscovered));

    SummaryStats tau95HitStats = summarize(tau95Hit);
    os << '\t' << tau95HitStats.mean;
    writeMeanSe(os, summarizeConditional(tau95, tau95WasReached));

    writeMeanSe(os, summarize(meanTrueP));

    for (const ChainStats& s : stats)
        os << '\t' << s.mapFirstHitCycle;
}

TreeConvergenceDiagnostics::ChainStats
TreeConvergenceDiagnostics::computeChainStats(const TreeSamples* samples) const {

    ChainStats stats;

    if (samples == nullptr || samples->getNumSamples() == 0)
        return stats;

    const TreeCountMap& counts = samples->getTreeCounts();
    const std::vector<uint64_t>& trace = samples->getTrees();

    stats.numSamples = samples->getNumSamples();

    for (auto& [hash, cnt] : counts)
        {
        (void)cnt;

        TreeInfo* info = treeCache->getTreeInfo(hash);

        if (info == nullptr)
            Msg::error("Sampled tree not found in TreeCache");

        stats.discoveredMass += info->posteriorProbability;

        if (inCredibleSet95.find(hash) != inCredibleSet95.end())
            stats.credible95MassDiscovered += info->posteriorProbability;
        }

    stats.undiscoveredMass = 1.0 - stats.discoveredMass;

    if (stats.undiscoveredMass < 0.0 && stats.undiscoveredMass > -1.0e-12)
        stats.undiscoveredMass = 0.0;

    if (posteriorRank.empty() == false)
        {
        uint64_t mapHash = posteriorRank[0]->hash;
        stats.mapFirstHitCycle = firstHitCycle(trace, mapHash);
        }

    std::unordered_set<uint64_t> visited;
    visited.reserve(trace.size());

    double discoveredMassThroughTime = 0.0;

    for (size_t i=0; i<trace.size(); i++)
        {
        uint64_t h = trace[i];

        if (visited.find(h) == visited.end())
            {
            visited.insert(h);

            TreeInfo* info = treeCache->getTreeInfo(h);

            if (info == nullptr)
                Msg::error("Sampled tree not found in TreeCache");

            discoveredMassThroughTime += info->posteriorProbability;
            }

        if (stats.tau95WasReached == false && discoveredMassThroughTime >= 0.95)
            {
            stats.tau95WasReached = true;
            stats.tau95 = (double)(i + 1);
            }
        }

    double sumP = 0.0;

    for (uint64_t h : trace)
        {
        TreeInfo* info = treeCache->getTreeInfo(h);

        if (info == nullptr)
            Msg::error("Sampled tree not found in TreeCache");

        sumP += info->posteriorProbability;
        }

    stats.meanTruePosteriorProbability = sumP / (double)stats.numSamples;

    return stats;
}

int TreeConvergenceDiagnostics::firstHitCycle(const std::vector<uint64_t>& trace,
                                              uint64_t hash) const {

    for (size_t i=0; i<trace.size(); i++)
        {
        if (trace[i] == hash)
            return (int)(i + 1);
        }

    return -1;
}

TreeConvergenceDiagnostics::SummaryStats
TreeConvergenceDiagnostics::summarize(const std::vector<double>& values) const {

    SummaryStats stats;

    if (values.empty() == true)
        return stats;

    stats.n = (double)values.size();

    double sum = 0.0;

    for (double x : values)
        sum += x;

    stats.mean = sum / stats.n;

    if (values.size() > 1)
        {
        double ss = 0.0;

        for (double x : values)
            {
            double d = x - stats.mean;
            ss += d * d;
            }

        double var = ss / (stats.n - 1.0);
        stats.se = std::sqrt(var / stats.n);
        }

    return stats;
}

TreeConvergenceDiagnostics::SummaryStats
TreeConvergenceDiagnostics::summarizeConditional(const std::vector<double>& values,
                                                 const std::vector<bool>& include) const {

    std::vector<double> kept;

    for (size_t i=0; i<values.size() && i<include.size(); i++)
        {
        if (include[i] == true)
            kept.push_back(values[i]);
        }

    return summarize(kept);
}

void TreeConvergenceDiagnostics::writeMeanSe(std::ostream& os,
                                             const SummaryStats& s) const {

    os << '\t' << s.mean << '\t' << s.se;
}
