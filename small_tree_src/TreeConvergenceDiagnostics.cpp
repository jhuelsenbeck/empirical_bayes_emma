#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
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

    std::sort(posteriorRank.begin(), posteriorRank.end(),
              [](const TreeInfo* a, const TreeInfo* b)
                  {
                  if (a->posteriorProbability != b->posteriorProbability)
                      return a->posteriorProbability > b->posteriorProbability;
                  return a->hash < b->hash;
                  });

    double cumulative = 0.0;
    for (size_t i=0; i<posteriorRank.size(); i++)
        {
        TreeInfo* info = posteriorRank[i];
        rankByHash[info->hash] = (int)i + 1;

        if (cumulative < 0.95)
            {
            inCredibleSet95[info->hash] = true;
            cumulative += info->posteriorProbability;
            credible95Mass += info->posteriorProbability;
            numCredibleSet95Trees++;
            }
        }
}

void TreeConvergenceDiagnostics::writeStatsHeader(std::ostream& os) {

    os << "numReps";
    os << "\tmeanUniqueTrees\tseUniqueTrees";
    os << "\tmeanDiscoveredMass\tseDiscoveredMass";
    os << "\tmeanCredible95MassDiscovered\tseCredible95MassDiscovered";
    os << "\tmeanCredible95TreeCoverage\tseCredible95TreeCoverage";
    os << "\tmeanTop5Coverage\tseTop5Coverage";
    os << "\tmeanTop10Coverage\tseTop10Coverage";
    os << "\tmeanTop50Coverage\tseTop50Coverage";
    os << "\tmapFoundFraction\tmeanMapFirstHit\tseMapFirstHit";
    os << "\ttau50Fraction\tmeanTau50\tseTau50";
    os << "\ttau90Fraction\tmeanTau90\tseTau90";
    os << "\ttau95Fraction\tmeanTau95\tseTau95";
    os << "\tmeanCurrentTrueP\tseCurrentTrueP";
    os << "\tmeanCurrentRank\tseCurrentRank";
    os << "\tmeanMeanTrueP\tseMeanTrueP";
    os << "\tmeanTVExact\tseTVExact";
    os << "\tmeanJSExact\tseJSExact";
}

void TreeConvergenceDiagnostics::writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec) const {

    std::vector<ChainStats> stats;
    stats.reserve(sampleVec.size());
    for (TreeSamples* s : sampleVec)
        stats.push_back(computeChainStats(s));

    std::vector<double> uniqueTrees, discoveredMass, credibleMass, credibleCoverage;
    std::vector<double> top5, top10, top50;
    std::vector<double> mapHit, mapFirstHit;
    std::vector<double> tau50Hit, tau90Hit, tau95Hit, tau50, tau90, tau95;
    std::vector<double> currentTrueP, currentRank, meanTrueP, tvExact, jsExact;
    std::vector<bool> mapHitTf, tau50Tf, tau90Tf, tau95Tf;

    for (const ChainStats& s : stats)
        {
        uniqueTrees.push_back((double)s.numUniqueTrees);
        discoveredMass.push_back(s.discoveredMass);
        credibleMass.push_back(s.credible95MassDiscovered);
        credibleCoverage.push_back(s.credible95TreeCoverage);
        top5.push_back(s.top5Coverage);
        top10.push_back(s.top10Coverage);
        top50.push_back(s.top50Coverage);
        mapHit.push_back(s.mapWasFound ? 1.0 : 0.0);
        mapFirstHit.push_back(s.mapFirstHit);
        mapHitTf.push_back(s.mapWasFound);
        tau50Hit.push_back(s.tau50WasReached ? 1.0 : 0.0);
        tau90Hit.push_back(s.tau90WasReached ? 1.0 : 0.0);
        tau95Hit.push_back(s.tau95WasReached ? 1.0 : 0.0);
        tau50.push_back(s.tau50);
        tau90.push_back(s.tau90);
        tau95.push_back(s.tau95);
        tau50Tf.push_back(s.tau50WasReached);
        tau90Tf.push_back(s.tau90WasReached);
        tau95Tf.push_back(s.tau95WasReached);
        currentTrueP.push_back(s.currentTreePosteriorProbability);
        currentRank.push_back(s.currentTreeRank);
        meanTrueP.push_back(s.meanTruePosteriorProbability);
        tvExact.push_back(s.tvExact);
        jsExact.push_back(s.jsExact);
        }

    os << stats.size();
    writeMeanSe(os, summarize(uniqueTrees));
    writeMeanSe(os, summarize(discoveredMass));
    writeMeanSe(os, summarize(credibleMass));
    writeMeanSe(os, summarize(credibleCoverage));
    writeMeanSe(os, summarize(top5));
    writeMeanSe(os, summarize(top10));
    writeMeanSe(os, summarize(top50));

    SummaryStats mapHitStats = summarize(mapHit);
    os << '\t' << mapHitStats.mean;
    writeMeanSe(os, summarizeConditional(mapFirstHit, mapHitTf));

    SummaryStats tau50HitStats = summarize(tau50Hit);
    os << '\t' << tau50HitStats.mean;
    writeMeanSe(os, summarizeConditional(tau50, tau50Tf));

    SummaryStats tau90HitStats = summarize(tau90Hit);
    os << '\t' << tau90HitStats.mean;
    writeMeanSe(os, summarizeConditional(tau90, tau90Tf));

    SummaryStats tau95HitStats = summarize(tau95Hit);
    os << '\t' << tau95HitStats.mean;
    writeMeanSe(os, summarizeConditional(tau95, tau95Tf));

    writeMeanSe(os, summarize(currentTrueP));
    writeMeanSe(os, summarize(currentRank));
    writeMeanSe(os, summarize(meanTrueP));
    writeMeanSe(os, summarize(tvExact));
    writeMeanSe(os, summarize(jsExact));
}

TreeConvergenceDiagnostics::ChainStats TreeConvergenceDiagnostics::computeChainStats(const TreeSamples* samples) const {

    ChainStats stats;
    if (samples == nullptr || samples->getNumSamples() == 0)
        return stats;

    const TreeCountMap& counts = samples->getTreeCounts();
    const std::vector<uint64_t>& trace = samples->getTrees();

    stats.numSamples = samples->getNumSamples();
    stats.numUniqueTrees = counts.size();

    for (auto& [hash, cnt] : counts)
        {
        TreeInfo* info = treeCache->getTreeInfo(hash);
        if (info == nullptr)
            Msg::error("Sampled tree not found in TreeCache");

        stats.discoveredMass += info->posteriorProbability;
        if (inCredibleSet95.find(hash) != inCredibleSet95.end())
            stats.credible95MassDiscovered += info->posteriorProbability;
        }

    if (credible95Mass > 0.0)
        stats.credible95TreeCoverage = fractionFound(counts, 1, numCredibleSet95Trees);

    stats.top5Coverage  = fractionFound(counts, 1, 5);
    stats.top10Coverage = fractionFound(counts, 1, 10);
    stats.top50Coverage = fractionFound(counts, 1, 50);

    if (posteriorRank.empty() == false)
        {
        uint64_t mapHash = posteriorRank[0]->hash;
        int hit = firstHitCycle(trace, mapHash);
        if (hit > 0)
            {
            stats.mapWasFound = true;
            stats.mapFirstHit = (double)hit;
            }
        }

    std::unordered_set<uint64_t> visited;
    visited.reserve(trace.size());
    double mass = 0.0;
    for (size_t i=0; i<trace.size(); i++)
        {
        uint64_t h = trace[i];
        if (visited.find(h) == visited.end())
            {
            visited.insert(h);
            TreeInfo* info = treeCache->getTreeInfo(h);
            if (info == nullptr)
                Msg::error("Sampled tree not found in TreeCache");
            mass += info->posteriorProbability;
            }

        if (stats.tau50WasReached == false && mass >= 0.50)
            {
            stats.tau50WasReached = true;
            stats.tau50 = (double)(i + 1);
            }
        if (stats.tau90WasReached == false && mass >= 0.90)
            {
            stats.tau90WasReached = true;
            stats.tau90 = (double)(i + 1);
            }
        if (stats.tau95WasReached == false && mass >= 0.95)
            {
            stats.tau95WasReached = true;
            stats.tau95 = (double)(i + 1);
            }
        }

    double n = (double)stats.numSamples;
    for (TreeInfo* info : posteriorRank)
        {
        double p = info->posteriorProbability;
        double q = empiricalProbability(samples, info->hash);

        stats.tvExact += std::fabs(q - p);

        double m = 0.5 * (p + q);
        if (p > 0.0 && m > 0.0)
            stats.jsExact += 0.5 * p * (std::log(p) - std::log(m));
        if (q > 0.0 && m > 0.0)
            stats.jsExact += 0.5 * q * (std::log(q) - std::log(m));
        }
    stats.tvExact *= 0.5;

    double sumP = 0.0;
    for (uint64_t h : trace)
        {
        TreeInfo* info = treeCache->getTreeInfo(h);
        if (info == nullptr)
            Msg::error("Sampled tree not found in TreeCache");
        sumP += info->posteriorProbability;
        }
    stats.meanTruePosteriorProbability = sumP / n;

    uint64_t currentHash = trace.back();
    TreeInfo* currentInfo = treeCache->getTreeInfo(currentHash);
    if (currentInfo == nullptr)
        Msg::error("Current tree not found in TreeCache");
    stats.currentTreePosteriorProbability = currentInfo->posteriorProbability;

    auto rankIt = rankByHash.find(currentHash);
    if (rankIt != rankByHash.end())
        stats.currentTreeRank = (double)rankIt->second;

    return stats;
}

double TreeConvergenceDiagnostics::empiricalProbability(const TreeSamples* samples, uint64_t hash) const {

    if (samples == nullptr || samples->getNumSamples() == 0)
        return 0.0;

    const TreeCountMap& counts = samples->getTreeCounts();
    TreeCountMap::const_iterator it = counts.find(hash);
    if (it == counts.end())
        return 0.0;

    return (double)it->second / (double)samples->getNumSamples();
}

int TreeConvergenceDiagnostics::firstHitCycle(const std::vector<uint64_t>& trace, uint64_t hash) const {

    for (size_t i=0; i<trace.size(); i++)
        {
        if (trace[i] == hash)
            return (int)(i + 1);
        }
    return 0;
}

double TreeConvergenceDiagnostics::fractionFound(const TreeCountMap& counts, size_t firstRank, size_t lastRank) const {

    if (posteriorRank.empty() == true || firstRank > lastRank)
        return 0.0;

    if (lastRank > posteriorRank.size())
        lastRank = posteriorRank.size();
    if (firstRank < 1)
        firstRank = 1;
    if (firstRank > lastRank)
        return 0.0;

    size_t found = 0;
    for (size_t i=firstRank-1; i<lastRank; i++)
        {
        uint64_t h = posteriorRank[i]->hash;
        if (counts.find(h) != counts.end())
            found++;
        }

    return (double)found / (double)(lastRank - firstRank + 1);
}

TreeConvergenceDiagnostics::SummaryStats TreeConvergenceDiagnostics::summarize(const std::vector<double>& values) const {

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

TreeConvergenceDiagnostics::SummaryStats TreeConvergenceDiagnostics::summarizeConditional(const std::vector<double>& values, const std::vector<bool>& include) const {

    std::vector<double> kept;
    for (size_t i=0; i<values.size() && i<include.size(); i++)
        {
        if (include[i] == true)
            kept.push_back(values[i]);
        }
    return summarize(kept);
}

void TreeConvergenceDiagnostics::writeMeanSe(std::ostream& os, const SummaryStats& s) const {

    os << '\t' << s.mean << '\t' << s.se;
}
