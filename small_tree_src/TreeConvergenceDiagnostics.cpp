#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include "Msg.hpp"
#include "TreeConvergenceDiagnostics.hpp"

TreeConvergenceDiagnostics::TreeConvergenceDiagnostics(TreeCache* tc) :
    treeCache(tc), numCredibleSet95Trees(0) {

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
        rankByHash[info->hash] = (int)i + 1; // 1 = true MAP tree
        if (cumulative < 0.95)
            {
            inCredibleSet95[info->hash] = true;
            cumulative += info->posteriorProbability;
            numCredibleSet95Trees++;
            }
        }
}

void TreeConvergenceDiagnostics::writeStatsHeader(std::ostream& os, size_t numChains) {

    os << "meanPairwiseTreeTV\tmaxPairwiseTreeTV\tmeanPairwiseTreeJS\tmaxPairwiseTreeJS";
    for (size_t c=0; c<numChains; c++) os << "\tvisitedMass_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tcredible95Coverage_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\ttvExact_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tklEmpToExact_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tklExactToEmp_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tjsExact_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tmeanTrueP_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tmeanLnTrueP_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tcurrentTreeHash_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tcurrentTreeP_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tcurrentTreeRank_chain" << (c+1);
}

void TreeConvergenceDiagnostics::writeStatsLine(std::ostream& os, const std::vector<TreeSamples*>& sampleVec) const {

    PairStats pairStats = computePairStats(sampleVec);
    os << pairStats.meanPairwiseTv << '\t'
       << pairStats.maxPairwiseTv  << '\t'
       << pairStats.meanPairwiseJs << '\t'
       << pairStats.maxPairwiseJs;

    std::vector<ChainStats> stats;
    stats.reserve(sampleVec.size());
    for (TreeSamples* s : sampleVec)
        stats.push_back(computeChainStats(s));

    for (const ChainStats& s : stats) os << '\t' << s.visitedPosteriorMass;
    for (const ChainStats& s : stats) os << '\t' << s.credibleSet95Coverage;
    for (const ChainStats& s : stats) os << '\t' << s.totalVariation;
    for (const ChainStats& s : stats) os << '\t' << s.klEmpiricalToTrue;
    for (const ChainStats& s : stats) os << '\t' << s.klTrueToEmpirical;
    for (const ChainStats& s : stats) os << '\t' << s.jsDivergence;
    for (const ChainStats& s : stats) os << '\t' << s.meanTruePosteriorProbability;
    for (const ChainStats& s : stats) os << '\t' << s.meanLnTruePosteriorProbability;
    for (const ChainStats& s : stats) os << '\t' << s.currentTreeHash;
    for (const ChainStats& s : stats) os << '\t' << s.currentTreePosteriorProbability;
    for (const ChainStats& s : stats) os << '\t' << s.currentTreeRank;
}

TreeConvergenceDiagnostics::ChainStats TreeConvergenceDiagnostics::computeChainStats(const TreeSamples* samples) const {

    ChainStats stats;
    if (samples == nullptr || samples->getNumSamples() == 0)
        return stats;

    const TreeCountMap& counts = samples->getTreeCounts();
    const std::vector<uint64_t>& trace = samples->getTrees();
    stats.numSamples = samples->getNumSamples();
    stats.numUniqueTrees = counts.size();

    double n = (double)stats.numSamples;
    double pseudo = 0.5 / (n + 0.5 * (double)posteriorRank.size());

    size_t credibleHits = 0;
    for (auto& [hash, cnt] : counts)
        {
        TreeInfo* info = treeCache->getTreeInfo(hash);
        if (info == nullptr)
            Msg::error("Sampled tree not found in TreeCache");
        stats.visitedPosteriorMass += info->posteriorProbability;
        if (inCredibleSet95.find(hash) != inCredibleSet95.end())
            credibleHits++;
        }

    if (numCredibleSet95Trees > 0)
        stats.credibleSet95Coverage = (double)credibleHits / (double)numCredibleSet95Trees;

    for (TreeInfo* info : posteriorRank)
        {
        double p = info->posteriorProbability;
        double q = empiricalProbability(samples, info->hash);
        double qSmooth = q;
        if (qSmooth <= 0.0)
            qSmooth = pseudo;

        stats.totalVariation += std::fabs(q - p);

        if (q > 0.0 && p > 0.0)
            stats.klEmpiricalToTrue += q * (std::log(q) - std::log(p));
        if (p > 0.0)
            stats.klTrueToEmpirical += p * (std::log(p) - std::log(qSmooth));

        double m = 0.5 * (p + q);
        if (p > 0.0 && m > 0.0)
            stats.jsDivergence += 0.5 * p * (std::log(p) - std::log(m));
        if (q > 0.0 && m > 0.0)
            stats.jsDivergence += 0.5 * q * (std::log(q) - std::log(m));
        }
    stats.totalVariation *= 0.5;

    double sumP = 0.0;
    double sumLnP = 0.0;
    for (uint64_t h : trace)
        {
        TreeInfo* info = treeCache->getTreeInfo(h);
        if (info == nullptr)
            Msg::error("Sampled tree not found in TreeCache");
        double p = info->posteriorProbability;
        sumP += p;
        sumLnP += safeLog(p);
        }
    stats.meanTruePosteriorProbability = sumP / n;
    stats.meanLnTruePosteriorProbability = sumLnP / n;

    stats.currentTreeHash = trace.back();
    TreeInfo* currentInfo = treeCache->getTreeInfo(stats.currentTreeHash);
    if (currentInfo == nullptr)
        Msg::error("Current sampled tree not found in TreeCache");
    stats.currentTreePosteriorProbability = currentInfo->posteriorProbability;
    auto rankIt = rankByHash.find(stats.currentTreeHash);
    if (rankIt != rankByHash.end())
        stats.currentTreeRank = rankIt->second;

    return stats;
}

TreeConvergenceDiagnostics::PairStats TreeConvergenceDiagnostics::computePairStats(const std::vector<TreeSamples*>& sampleVec) const {

    PairStats stats;
    if (sampleVec.size() < 2)
        return stats;

    int numPairs = 0;
    for (size_t i=0; i<sampleVec.size(); i++)
        {
        for (size_t j=i+1; j<sampleVec.size(); j++)
            {
            double tv = 0.0;
            double js = 0.0;
            for (TreeInfo* info : posteriorRank)
                {
                double p = empiricalProbability(sampleVec[i], info->hash);
                double q = empiricalProbability(sampleVec[j], info->hash);
                tv += std::fabs(p - q);
                double m = 0.5 * (p + q);
                if (p > 0.0 && m > 0.0)
                    js += 0.5 * p * (std::log(p) - std::log(m));
                if (q > 0.0 && m > 0.0)
                    js += 0.5 * q * (std::log(q) - std::log(m));
                }
            tv *= 0.5;
            stats.meanPairwiseTv += tv;
            stats.meanPairwiseJs += js;
            if (tv > stats.maxPairwiseTv)
                stats.maxPairwiseTv = tv;
            if (js > stats.maxPairwiseJs)
                stats.maxPairwiseJs = js;
            numPairs++;
            }
        }

    if (numPairs > 0)
        {
        stats.meanPairwiseTv /= (double)numPairs;
        stats.meanPairwiseJs /= (double)numPairs;
        }
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

double TreeConvergenceDiagnostics::safeLog(double x) const {

    if (x <= 0.0)
        return -std::numeric_limits<double>::infinity();
    return std::log(x);
}
