#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "Msg.hpp"
#include "TreeSamples.hpp"



TreeSamples::TreeSamples(TreeCache* tc) : treeCache(tc), numSamples(0) {

}

void TreeSamples::clear(void) {

    treeCounts.clear();
    trees.clear();
    numSamples = 0;
}

void TreeSamples::compareSamples(std::vector<TreeSamples*>& sampleVec) {

    size_t numChains = sampleVec.size();
    if (numChains < 2)
        {
        std::cout << "Need at least two chains to compare tree samples" << std::endl;
        return;
        }

    int numCycles = sampleVec[0]->numSamples;
    for (size_t i=1; i<numChains; i++)
        {
        if (sampleVec[i]->numSamples != numCycles)
            {
            std::cout << "Cannot compare: chains have different sample counts" << std::endl;
            return;
            }
        }
    if (numCycles == 0)
        return;

    TreeCountMap combinedCounts;
    for (size_t i=0; i<numChains; i++)
        for (auto& [hash, cnt] : sampleVec[i]->treeCounts)
            combinedCounts[hash] += cnt;

    std::vector<std::pair<uint64_t,int>> ranked;
    ranked.reserve(combinedCounts.size());
    for (auto& [hash, cnt] : combinedCounts)
        ranked.emplace_back(hash, cnt);
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
        });

    size_t topK = std::min<size_t>(ranked.size(), 10);
    double totalSamples = (double)numCycles * numChains;

    std::vector<std::vector<double>> chainProb(topK, std::vector<double>(numChains, 0.0));
    std::vector<std::vector<double>> chainESS (topK, std::vector<double>(numChains, 0.0));
    std::vector<std::vector<int>>    chainHit (topK, std::vector<int>   (numChains, 0));

    for (size_t r=0; r<topK; r++)
        {
        uint64_t h = ranked[r].first;
        for (size_t c=0; c<numChains; c++)
            {
            TreeCountMap::iterator it = sampleVec[c]->treeCounts.find(h);
            if (it != sampleVec[c]->treeCounts.end())
                chainProb[r][c] = (double)it->second / sampleVec[c]->numSamples;

            chainESS[r][c] = computeIndicatorESS(sampleVec[c]->trees, h);

            const std::vector<uint64_t>& trace = sampleVec[c]->trees;
            for (size_t t=0; t<trace.size(); t++)
                {
                if (trace[t] == h)
                    {
                    chainHit[r][c] = (int)(t + 1);
                    break;
                    }
                }
            }
        }

    std::cout << "   Cross-chain tree sample diagnostics (cycle " << numCycles << ")\n";
    std::cout << "   * Number of chains   = " << numChains << "\n";
    std::cout << "   * Total unique trees = " << combinedCounts.size() << "\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n   P(tree) per chain (combP = pooled probability):\n";
    std::cout << "   " << std::setw(4) << "#" << " " << std::setw(20) << "hash"
              << "  " << std::setw(8) << "combP";
//    for (size_t c=0; c<numChains; c++)
//        std::cout << "  " << std::setw(8) << ("chain" + std::to_string(c+1));
    std::cout << "\n";
    for (size_t r=0; r<topK; r++)
        {
        double pCombined = (double)ranked[r].second / totalSamples;
        std::cout << "   " << std::setw(4) << (r+1)
                  << " " << std::setw(20) << ranked[r].first
                  << " " << std::setw(6) << pCombined;
        for (size_t c=0; c<numChains; c++)
            {
            std::cout << " " << std::setw(6) << chainProb[r][c];
            if ((c+1) % 10 == 0 && c+1 != numChains)
                {
                std::cout << std::endl;
                std::cout << "                            ";
                }
            }
        std::cout << "\n";
        }

    std::cout << std::setprecision(1);
    std::cout << "\n   ESS for the indicator trace 1[T_t == tree]:\n";
//    std::cout << "   " << std::setw(4) << "#";
//    for (size_t c=0; c<numChains; c++)
//        std::cout << "  " << std::setw(10) << ("chain" + std::to_string(c+1));
//    std::cout << "\n";
    for (size_t r=0; r<topK; r++)
        {
        std::cout << "   " << std::setw(4) << (r+1);
        for (size_t c=0; c<numChains; c++)
            {
            std::cout << " " << std::setw(10) << chainESS[r][c];
            if ((c+1) % 10 == 0 && c+1 != numChains)
                {
                std::cout << std::endl;
                std::cout << "       ";
                }
            }
        std::cout << "\n";
        }

    std::cout << "\n   First-hit cycle (--- = never sampled):\n";
//    std::cout << "   " << std::setw(4) << "#";
//    for (size_t c=0; c<numChains; c++)
//        std::cout << "  " << std::setw(10) << ("chain" + std::to_string(c+1));
//    std::cout << "\n";
    for (size_t r=0; r<topK; r++)
        {
        std::cout << "   * " << std::setw(4) << (r+1);
        for (size_t c=0; c<numChains; c++)
            {
            if (chainHit[r][c] == 0)
                std::cout << " " << std::setw(6) << "---";
            else
                std::cout << " " << std::setw(6) << chainHit[r][c];
            if ((c+1) % 10 == 0 && c+1 != numChains)
                {
                std::cout << std::endl;
                std::cout << "         ";
                }
            }
        std::cout << "\n";
        }

    std::cout << "\n   Per-chain unique trees:" << std::endl;
    std::cout << "     ";
    for (size_t c=0; c<numChains; c++)
        {
        std::cout << " " << std::setw(8) << sampleVec[c]->treeCounts.size();
        if ((c+1) % 10 == 0 && c+1 != numChains)
            {
            std::cout << std::endl;
            std::cout << "     ";
            }
        }
    std::cout << "\n";
}

double TreeSamples::computeIndicatorESS(const std::vector<uint64_t>& trace, uint64_t target) {

    size_t N = trace.size();
    if (N < 2)
        return (double)N;

    size_t hits = 0;
    for (uint64_t h : trace)
        if (h == target)
            hits++;
    if (hits == 0 || hits == N)
        return (double)N;

    double mean = (double)hits / N;
    double var  = mean * (1.0 - mean);

    size_t maxLag = N / 4;
    if (maxLag > 2048)
        maxLag = 2048;

    double tauSum = 1.0;
    for (size_t k=1; k<=maxLag; k++)
        {
        double gamma = 0.0;
        for (size_t i=0; i+k<N; i++)
            {
            double xi  = (trace[i]   == target) ? 1.0 : 0.0;
            double xik = (trace[i+k] == target) ? 1.0 : 0.0;
            gamma += (xi - mean) * (xik - mean);
            }
        gamma /= (double)N;
        double rho = gamma / var;

        if (rho <= 0.0)
            break;
        tauSum += 2.0 * rho;
        }

    if (tauSum < 1.0)
        tauSum = 1.0;
    return (double)N / tauSum;
}

double TreeSamples::getTreeProbability(uint64_t treeHash) const {

    TreeCountMap::const_iterator it = treeCounts.find(treeHash);
    if (it != treeCounts.end() && numSamples > 0)
        return (double)it->second / numSamples;
    return 0.0;
}

void TreeSamples::print(void) const {

    std::vector<std::pair<uint64_t, std::size_t>> entries;
    entries.reserve(treeCounts.size());
    for (const auto &kv : treeCounts)
        entries.emplace_back(kv.first, kv.second);

    std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    double sumExact = 0.0, sumMcmc = 0.0;
    int i = 0;
    for (const auto &e : entries)
        {
        TreeInfo* tInfo = treeCache->getTreeInfo(e.first);
        if (tInfo == nullptr)
            Msg::error("Could not find tree in cache");
        double lnL = tInfo->lnLikelihood;
        double prob = (numSamples > 0 ? (double)e.second / numSamples : 0.0);
        sumMcmc += prob;
        sumExact += tInfo->posteriorProbability;
        std::cout << std::setw(5) << ++i << " -- ";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(21) << e.first << " " << lnL << " " << " " << tInfo->posteriorProbability << " ";
        std::cout << prob << " " << sumMcmc <<  " -- ";
        std::cout << "count=" << e.second;
        std::cout << std::endl;
        if (sumExact > 0.99 && sumMcmc > 0.99)
            break;
        }
}

void TreeSamples::combinedPrint(std::vector<TreeSamples*>& sampleVec) {

    if (sampleVec.size() == 0)
        {
        std::cout << "No tree samples to combine" << std::endl;
        return;
        }

    TreeCountMap combinedCounts;
    combinedCounts.insert(sampleVec[0]->treeCounts.begin(), sampleVec[0]->treeCounts.end());
    for (size_t i=1; i<sampleVec.size(); i++)
        {
        TreeCountMap& cnts = sampleVec[i]->treeCounts;
        for (auto& [key,val] : cnts)
            {
            TreeCountMap::iterator it = combinedCounts.find(key);
            if (it == combinedCounts.end())
                combinedCounts.insert(std::make_pair(key,val));
            else
                it->second += val;
            }
        }

    std::vector<std::pair<uint64_t, double>> entries;
    for (const auto &kv : combinedCounts)
        entries.emplace_back(kv.first, kv.second);

    std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    std::vector<double> cumulativeProb(sampleVec.size(), 0.0);
    std::cout << "   Tree posterior probabilities:" << std::endl;
    int i = 0;
    for (const auto &e : entries)
        {
        std::cout << "   * ";
        std::cout << std::left << std::setw(4) << ++i << " " << std::setw(20) << e.first << " -- ";
        int cnt = 0;
        for (TreeSamples* s : sampleVec)
            {
            cnt++;
            TreeCountMap::iterator it = s->treeCounts.find(e.first);
            double x = 0.0;
            if (it != s->treeCounts.end() && s->numSamples > 0)
                x = (double)it->second / s->numSamples;
            cumulativeProb[cnt-1] += x;
            std::cout << std::setprecision(3) << x << " ";
            if (cnt % 10 == 0 && cnt != sampleVec.size())
                {
                std::cout << std::endl;
                std::cout << "                                  ";
                }
            }
        std::cout << std::endl;
        
        bool allGreater = true;
        for (size_t k=0; k<cumulativeProb.size(); k++)
            {
            if (cumulativeProb[k] <=0.999)
                allGreater = false;
            }
        if (allGreater == true)
            break;
        }
    std::cout << std::endl;
}

void TreeSamples::sampleTree(uint64_t treeHash) {

    numSamples++;
    TreeCountMap::iterator it = treeCounts.find(treeHash);
    if (it == treeCounts.end())
        treeCounts.insert(std::make_pair(treeHash,1));
    else
        it->second++;

    trees.push_back(treeHash);
}

void TreeSamples::writeStatsHeader(std::ostream& os, size_t numChains) {

    os << "numUniqueTrees\tmapHash\tcombinedPMap";
    for (size_t c=0; c<numChains; c++) os << "\tPMap_chain"        << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tESSMap_chain"      << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tFirstHitMap_chain" << (c+1);
    for (size_t c=0; c<numChains; c++) os << "\tUniqueTrees_chain" << (c+1);
}

void TreeSamples::writeStatsLine(std::ostream& os, std::vector<TreeSamples*>& sampleVec) {

    size_t numChains = sampleVec.size();
    if (numChains < 2)
        return;

    int numCycles = sampleVec[0]->numSamples;
    if (numCycles == 0)
        {
        os << 0 << '\t' << 0 << '\t' << 0;
        for (int pass=0; pass<4; pass++)
            for (size_t c=0; c<numChains; c++)
                os << '\t' << 0;
        return;
        }

    TreeCountMap combinedCounts;
    for (size_t i=0; i<numChains; i++)
        for (auto& [hash, cnt] : sampleVec[i]->treeCounts)
            combinedCounts[hash] += cnt;

    uint64_t mapHash = 0;
    int      mapCnt  = -1;
    for (auto& [hash, cnt] : combinedCounts)
        {
        if (cnt > mapCnt || (cnt == mapCnt && hash < mapHash))
            {
            mapCnt  = cnt;
            mapHash = hash;
            }
        }
    double combinedPMap = (double)mapCnt / ((double)numCycles * numChains);

    os << combinedCounts.size() << '\t' << mapHash << '\t' << combinedPMap;

    for (size_t c=0; c<numChains; c++)
        {
        TreeCountMap::iterator it = sampleVec[c]->treeCounts.find(mapHash);
        double p = (it != sampleVec[c]->treeCounts.end() && sampleVec[c]->numSamples > 0) ? (double)it->second / sampleVec[c]->numSamples : 0.0;
        os << '\t' << p;
        }

    for (size_t c=0; c<numChains; c++)
        os << '\t' << computeIndicatorESS(sampleVec[c]->trees, mapHash);

    for (size_t c=0; c<numChains; c++)
        {
        const std::vector<uint64_t>& trace = sampleVec[c]->trees;
        int hit = 0;
        for (size_t t=0; t<trace.size(); t++)
            {
            if (trace[t] == mapHash)
                {
                hit = (int)(t + 1);
                break;
                }
            }
        os << '\t' << hit;
        }

    for (size_t c=0; c<numChains; c++)
        os << '\t' << sampleVec[c]->treeCounts.size();
}
