#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <string>
#include <map>
#include <queue>
#include <unordered_set>
#include <utility>
#include "Msg.hpp"
#include "Peak.hpp"
#include "TreeKey.hpp"
#include "TreeSamples.hpp"
#include "TreeSpace.hpp"

typedef struct {

    unsigned    n;
    double      mean;
    double      M2;
} RunningStats;

void initRunningStats(RunningStats *s) {

    s->n = 0;
    s->mean = 0.0;
    s->M2 = 0.0;
}

void addObservation(RunningStats *s, double x) {

    s->n++;

    double delta = x - s->mean;
    s->mean += delta / s->n;
    double delta2 = x - s->mean;
    s->M2 += delta * delta2;
}

double mean(const RunningStats *s) {

    return s->mean;
}

double populationVariance(const RunningStats *s) {

    return (s->n > 0) ? s->M2 / s->n : 0.0;
}

double sampleVariance(const RunningStats *s) {

    return (s->n > 1) ? s->M2 / (s->n - 1) : 0.0;
}

double populationStdDev(const RunningStats *s) {

    return sqrt(populationVariance(s));
}

double sampleStdDev(const RunningStats *s) {

    return sqrt(sampleVariance(s));
}




TreeSpace::TreeSpace(TreeCache* tc, std::string st) : treeCache(tc), swapType(st) {
        
    // construct graph
    TreeCacheMap& tCache = treeCache->getCache();

    int barWidth = 60, numAsterices = 0;
    size_t numTrees = tCache.size();
    size_t treeCnt = 0;

    std::cout << "   Constructing tree space graph:" << std::endl;
    std::cout << "   * [";
    for (int i=0; i<barWidth; i++)
        {
        if ((i+1) % (int)(barWidth*0.1) == 0 && i+1 != barWidth)
            std::cout << "|";
        else
            std::cout << "-";
        }
    std::cout << "]" << std::endl;
    std::cout << "   * [";

    RunningStats degreeStats;
    initRunningStats(&degreeStats);
    for (auto& [key,val] : tCache)
        {
        TreeSpaceNode* thisTree = getTree(key);
        std::vector<TreeInfo*>& ndeNeighbors = val->neighbors;
        for (size_t i=0; i<ndeNeighbors.size(); i++)
            {
            TreeSpaceNode* neighboringTree = getTree(ndeNeighbors[i]->hash);
            neighboringTree->neighbors.insert(thisTree);
            thisTree->neighbors.insert(neighboringTree);
            }
            
        averageDegree += thisTree->neighbors.size();
        addObservation(&degreeStats, thisTree->neighbors.size());

        treeCnt++;

        double progress = static_cast<double>(treeCnt) / numTrees;
        int filledWidth = static_cast<int>(progress * barWidth);

        for (int i=0; i<filledWidth-numAsterices; i++)
            std::cout << "*" << std::flush;
        numAsterices = filledWidth;
        }
    averageDegree = mean(&degreeStats);
    varianceDegree = sampleVariance(&degreeStats);

    std::cout << "]" << std::endl << std::endl;        
       
    // calculate the exact posterior probability
    double bestLnL = std::numeric_limits<double>::lowest();
    for (auto& [key,val] : tCache)
        {
        if (val->lnLikelihood > bestLnL)
            bestLnL = val->lnLikelihood;
        treeProbabilities.insert( std::make_pair(key,val->lnLikelihood) );
        }
    double sum = 0.0;
    for (auto& [key,val] : treeProbabilities)
        {
        double x = std::exp(val - bestLnL);
        val = x;
        sum += x;
        }
    double factor = 1.0 / sum;
    for (auto& [key,val] : treeProbabilities)
        {
        val *= factor;    
        
        TreeInfo* tInfo = treeCache->getTreeInfo(key);
        if (tInfo == nullptr)
            Msg::error("Could not find tree in tree cache");
        tInfo->posteriorProbability = val;
        }
}

TreeSpace::~TreeSpace(void) {

    clearPeaks();
    for (auto& [key,val] : treeNodes)
        delete val;
}

void TreeSpace::adjacentTreeDistances(uint64_t treeHash) {

    TreeSpaceNode* p = getTree(treeHash);
    std::set<TreeSpaceNode*>& pNeighbors = p->neighbors;
    
    if (p->distance == -1)
        {
        int minD = 1e9;
        for (TreeSpaceNode* n : pNeighbors)
            {
            if (n->distance < minD && n->distance != -1)
                minD = n->distance;
            }
        p->distance = minD + 1;
        }
        
    for (TreeSpaceNode* n : pNeighbors)
        {
        if (n->distance == -1)
            adjacentTreeDistances(n->treeHash);
        }
}

void TreeSpace::calculateDistances(uint64_t treeHash) {

    for (auto& [key,val] : treeNodes)
        val->distance = -1;
        
    getTree(treeHash)->distance = 0;
    adjacentTreeDistances(treeHash);
}

void TreeSpace::characterize(void) {

    clearPeaks();

    // find the peaks
    for (auto& [key,val] : treeNodes)
        {
        uint64_t peakHash = steepestAscent(key); 
        double treeProb = getTreeProbabiity(key);
        Peak* peak = findPeak(peakHash);
        if (peak == nullptr)
            {
            Peak* newPeak = new Peak;
            newPeak->setPeakTreeHash(peakHash);
            newPeak->setPeakTreeProbability(getTreeProbabiity(peakHash));
            newPeak->addTreeToPeak(key, treeProb);
            peaks.insert( std::make_pair(peakHash, newPeak) );
            }
        else 
            {
            peak->addTreeToPeak(key, treeProb);
            }
        }
        
    // check that the peaks contain all of the trees in the graph and that the basin probabilities sum to one
    int n = 0;
    double sum = 0.0;
    for (auto& [key,val] : peaks)
        {
        TreeSet& treesInSet = val->getTrees();
        n += treesInSet.size();
        sum += val->getPeakProbability();
        }
    if (n != treeNodes.size())
        {
        std::cout << "treeNodes.size() = " << treeNodes.size() << std::endl;
        std::cout << "Sum member trees = " << n << std::endl;
        Msg::error("Mismatch between peak member tree size and number of tree space vertices");
        }
    if (fabs(1.0 - sum) > 0.01)
        {
        std::cout << "Sum of basin probabilities = " << sum << std::endl;
        Msg::error("Problem with basin probabilities");
        }

    // label the peaks in order from highest to lowest
    std::vector<std::pair<uint64_t, Peak*>> vec(peaks.begin(), peaks.end());
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.second->getPeakTreeProbability() > b.second->getPeakTreeProbability(); // or: a.second->getProbability() > b.second->getProbability();
        });
    int peakId = 0;
    double bMax = 0.0, peakMassEntropy = 0.0, firstProb = 0.0, secondProb = 0.0;
    for (const auto& [key, peak] : vec) 
        {
        peak->setPeakId(peakId++);
        double peakMass = peak->getPeakProbability();
        if (peakMass > bMax)
            bMax = peakMass;
        peakMassEntropy += -peakMass * std::log(peakMass);
        if (peakMass > firstProb)
            {
            secondProb = firstProb;
            firstProb = peakMass;
            }
        else if (peakMass > secondProb)
            {
            secondProb = peakMass;
            }
        }
    double dominanceRatio = 0.0;
    if (peaks.size() > 1)
        dominanceRatio = firstProb / secondProb;
        
    // number of graph edges connecting each pair of basins (basin adjacency)
    int nPeaks = (int)peaks.size();
    std::vector<int> peakPaths(nPeaks * nPeaks, 0);

    std::unordered_map<uint64_t,uint64_t> assignment = computePeakAssignments();
    std::unordered_map<uint64_t,int> peakHashToId;
    for (const auto& [key, peak] : peaks)
        peakHashToId[key] = peak->getPeakId();

    for (const auto& [hash, node] : treeNodes)
        {
        auto ia = assignment.find(hash);
        if (ia == assignment.end())
            continue;
        int idA = peakHashToId[ia->second];
        for (TreeSpaceNode* nb : node->neighbors)
            {
            if (hash > nb->treeHash)
                continue;                       // count each undirected edge once
            auto ib = assignment.find(nb->treeHash);
            if (ib == assignment.end())
                continue;
            int idB = peakHashToId[ib->second];
            if (idA == idB)
                continue;                       // same basin, not a crossing edge
            peakPaths[idA * nPeaks + idB]++;
            peakPaths[idB * nPeaks + idA]++;
            }
        }
        
    // print peak information
    TreeKey& tKey = TreeKey::treeKey();
    std::cout << "   Peaks (" << swapType << "):" << std::endl;
    sum = 0.0;
    for (const auto& [key, peak] : vec) 
        {
        sum += peak->getPeakProbability();
        std::cout << "   * " << peak->getPeakId() << ": " << std::left << std::setw(8) << tKey.numberForTreeHash(key) << " " 
                  << " " << std::setw(8) << peak->getNumTrees() << " " << " " << peak->getPeakProbability() << " " << sum << "\n";
        }
    std::cout << std::endl;

//    std::cout << "   Number of paths between peaks" << std::endl;
//    for (int i=0; i<nPeaks; i++)
//        {
//        for (int j=i+1; j<nPeaks; j++)
//            std::cout << "   * " << i << "-" << j << ": " << peakPaths[i * nPeaks + j] << std::endl;
//        }
//    std::cout << std::endl;
    
    std::cout << "   Tree space characteristics (" << swapType << "):" << std::endl;
    std::cout << "   * Number of vertices       = " << treeNodes.size() << std::endl;
    std::cout << "   * Average degree           = " << averageDegree << std::endl;
    std::cout << "   * Variance degree          = " << varianceDegree << std::endl;
    std::cout << "   * Largest basin fraction   = " << bMax << std::endl;
    std::cout << "   * Peak mass entropy        = " << peakMassEntropy << std::endl;
    std::cout << "   * Expected number of peaks = " << std::exp(peakMassEntropy) << std::endl;
    if (peaks.size() > 1)
        std::cout << "   * Dominance ratio          = " << dominanceRatio << std::endl;
    else 
        std::cout << "   * Dominance ratio          = Undefined" << std::endl;
    std::cout << std::endl;
    
#   if 0
    // print tree space information for python graph
    std::cout << "vertex_data = [";
    bool isFirst = true;
    for (const auto& [key, peak] : vec) 
        {
        int basinSize = peak->getNumTrees();
        if (isFirst == false)
            std::cout << ",";
        isFirst = false;
        std::cout << "(" << basinSize << "," << peak->getPeakProbability() << ")";
        }
    std::cout << "]" << std::endl;
    
    std::cout << "edges = [";
    isFirst = true;
    for (int i=0; i<nPeaks; i++)
        {
        for (int j=i+1; j<nPeaks; j++)
            {            
            if (peakPaths[i * nPeaks + j] > 0)
                {
                if (isFirst == false)
                    std::cout << ",";
                isFirst = false;
                std::cout << "(" << i << "," << j << ")";
                }
            }
        }
    std::cout << "]" << std::endl;
    
    std::cout << "edge_widths = [";
    isFirst = true;
    for (int i=0; i<nPeaks; i++)
        {
        for (int j=i+1; j<nPeaks; j++)
            {            
            if (peakPaths[i * nPeaks + j] > 0)
                {
                if (isFirst == false)
                    std::cout << ",";
                isFirst = false;
                //std::cout << "peakPaths[i * nPeaks + j]";
                std::cout << "1";
                }
            }
        }
    std::cout << "]" << std::endl;
#   endif
}

void TreeSpace::characterizeBasins(std::unordered_map<uint64_t,int>& basins) {

    for (auto& [key,val] : treeNodes)
        {
        uint64_t endingTree = steepestAscent(key);
        std::unordered_map<uint64_t,int>::iterator it = basins.find(endingTree);
        if (it == basins.end())
            basins.insert( std::make_pair(endingTree,1) );
        else 
            it->second++;
        }
}

void TreeSpace::clearPeaks(void) {

    for (auto& [key,val] : peaks)
        delete val;
    peaks.clear();
}

std::unordered_map<uint64_t,uint64_t> TreeSpace::computePeakAssignments(void) {

    if (peaks.empty() == true)
        characterize();

    std::unordered_map<uint64_t,uint64_t> assignments;
    assignments.reserve(treeNodes.size());
    for (auto& [peakHash, peak] : peaks)
        {
        TreeSet& memberTrees = peak->getTrees();
        for (uint64_t h : memberTrees)
            assignments[h] = peakHash;
        }
    return assignments;
}

LocalSummary TreeSpace::computeLocalSummary(void) {

    LocalSummary stats;
    stats.numTrees = treeNodes.size();

    if (stats.numTrees == 0)
        return stats;

    double sumUphillFraction = 0.0;
    double sumWeightedUphillFraction = 0.0;
    double sumBestDiff = 0.0;
    double sumWeightedBestDiff = 0.0;
    double sumMeanAbsDiff = 0.0;
    double sumWeightedMeanAbsDiff = 0.0;
    size_t degreeTotal = 0;
    RunningStats degreeStats;
    initRunningStats(&degreeStats);

    for (auto& [hash, node] : treeNodes)
        {
        size_t degree = node->neighbors.size();
        addObservation(&degreeStats, degree);

        double p = getTreeProbabiity(hash);
        size_t uphill = 0;
        double bestDiff = -std::numeric_limits<double>::infinity();
        double sumAbsDiff = 0.0;

        for (TreeSpaceNode* neighbor : node->neighbors)
            {
            double diff = neighbor->lnL - node->lnL;
            if (diff > 0.0)
                uphill++;
            if (diff > bestDiff)
                bestDiff = diff;
            sumAbsDiff += std::fabs(diff);
            }

        double uphillFraction = 0.0;
        double meanAbsDiff = 0.0;
        if (degree > 0)
            {
            uphillFraction = (double)uphill / (double)degree;
            meanAbsDiff = sumAbsDiff / (double)degree;
            }
        if (bestDiff == -std::numeric_limits<double>::infinity())
            bestDiff = 0.0;

        if (uphill == 0)
            {
            stats.numLocalPeaks++;
            stats.posteriorMassOfLocalPeaks += p;
            }

        sumUphillFraction += uphillFraction;
        sumWeightedUphillFraction += p * uphillFraction;
        sumBestDiff += bestDiff;
        sumWeightedBestDiff += p * bestDiff;
        sumMeanAbsDiff += meanAbsDiff;
        sumWeightedMeanAbsDiff += p * meanAbsDiff;
        }

    stats.numEdges = degreeTotal / 2;
    stats.meanDegree = mean(&degreeStats);
    stats.varDegree = sampleVariance(&degreeStats);
    stats.meanUphillFraction = sumUphillFraction / (double)stats.numTrees;
    stats.posteriorWeightedMeanUphillFraction = sumWeightedUphillFraction;
    stats.meanBestNeighborLnLDifference = sumBestDiff / (double)stats.numTrees;
    stats.posteriorWeightedMeanBestNeighborLnLDifference = sumWeightedBestDiff;
    stats.meanNeighborAbsLnLDifference = sumMeanAbsDiff / (double)stats.numTrees;
    stats.posteriorWeightedMeanNeighborAbsLnLDifference = sumWeightedMeanAbsDiff;

    return stats;
}

BasinSummary TreeSpace::computeBasinSummary(void) {

    if (peaks.empty() == true)
        characterize();

    BasinSummary stats;
    stats.numPeaks = peaks.size();

    if (stats.numPeaks == 0)
        return stats;

    double bestTreeProb = -1.0;
    for (auto& [hash, prob] : treeProbabilities)
        {
        if (prob > bestTreeProb)
            {
            bestTreeProb = prob;
            stats.mapTreeHash = hash;
            }
        }

    Peak* mapPeak = findPeakForTreeWithHash(stats.mapTreeHash);
    if (mapPeak != nullptr)
        {
        stats.mapPeakHash = mapPeak->getPeakTreeHash();
        stats.mapPeakId = mapPeak->getPeakId();
        stats.mapBasinMass = mapPeak->getPeakProbability();
        }
    stats.posteriorMassOutsideMapBasin = 1.0 - stats.mapBasinMass;

    for (auto& [peakHash, peak] : peaks)
        {
        double mass = peak->getPeakProbability();
        int size = peak->getNumTrees();
        if (mass > 0.01)
            stats.numPeaksMassGreater01++;
        if (mass > 0.05)
            stats.numPeaksMassGreater05++;
        if (size > stats.largestBasinSize)
            stats.largestBasinSize = size;
        if (mass > stats.largestBasinMass)
            stats.largestBasinMass = mass;
        if (mass > 0.0)
            stats.basinMassEntropy += -mass * std::log(mass);
        }
    stats.effectiveNumBasins = std::exp(stats.basinMassEntropy);

    return stats;
}

std::vector<uint64_t> TreeSpace::credibleSet(double probability) {

    std::vector<std::pair<uint64_t,double>> sorted(treeProbabilities.begin(), treeProbabilities.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b)
      {
      if (a.second != b.second)
          return a.second > b.second;
      return a.first < b.first;
      });

    std::vector<uint64_t> result;
    double cumulative = 0.0;
    for (auto& [hash, prob] : sorted)
        {
        if (cumulative >= probability)
            break;
        result.push_back(hash);
        cumulative += prob;
        }
    return result;
}

AscentSummary TreeSpace::computeAscentSummary(void) {

    AscentSummary stats;
    if (treeNodes.empty() == true)
        return stats;

    std::unordered_set<uint64_t> credible95;
    for (uint64_t h : credibleSet(0.95))
        credible95.insert(h);

    double sumLength = 0.0;
    double sumWeightedLength = 0.0;
    double sumLength95 = 0.0;
    double sumWeightedLength95 = 0.0;
    double mass95 = 0.0;
    size_t n95 = 0;

    for (auto& [hash, node] : treeNodes)
        {
        int len = steepestAscentLength(hash);
        double p = getTreeProbabiity(hash);
        sumLength += (double)len;
        sumWeightedLength += p * (double)len;
        if (len > stats.maxAscentLength)
            stats.maxAscentLength = len;

        if (credible95.find(hash) != credible95.end())
            {
            n95++;
            mass95 += p;
            sumLength95 += (double)len;
            sumWeightedLength95 += p * (double)len;
            }
        }

    stats.meanAscentLength = sumLength / (double)treeNodes.size();
    stats.posteriorWeightedMeanAscentLength = sumWeightedLength;
    if (n95 > 0)
        stats.meanAscentLengthCredible95 = sumLength95 / (double)n95;
    if (mass95 > 0.0)
        stats.posteriorWeightedMeanAscentLengthCredible95 = sumWeightedLength95 / mass95;

    return stats;
}

CredibleComponentSummary TreeSpace::computeCredible95ComponentSummary(void) {

    CredibleComponentSummary stats;

    std::vector<uint64_t> credibleVec = credibleSet(0.95);
    std::unordered_set<uint64_t> credible(credibleVec.begin(), credibleVec.end());
    stats.credible95NumTrees = credible.size();
    for (uint64_t h : credible)
        stats.credible95Mass += getTreeProbabiity(h);

    if (credible.empty() == true)
        return stats;

    BasinSummary basinStats = computeBasinSummary();
    uint64_t mapHash = basinStats.mapTreeHash;

    std::unordered_set<uint64_t> visited;
    visited.reserve(credible.size());
    for (uint64_t startHash : credible)
        {
        if (visited.find(startHash) != visited.end())
            continue;

        stats.credible95NumComponents++;
        size_t componentSize = 0;
        double componentMass = 0.0;
        bool containsMap = false;

        std::queue<uint64_t> q;
        q.push(startHash);
        visited.insert(startHash);
        while (q.empty() == false)
            {
            uint64_t h = q.front();
            q.pop();
            componentSize++;
            componentMass += getTreeProbabiity(h);
            if (h == mapHash)
                containsMap = true;

            TreeSpaceNode* node = getTree(h);
            for (TreeSpaceNode* nb : node->neighbors)
                {
                uint64_t nh = nb->treeHash;
                if (credible.find(nh) == credible.end())
                    continue;
                if (visited.find(nh) != visited.end())
                    continue;
                visited.insert(nh);
                q.push(nh);
                }
            }

        if (componentSize > stats.credible95LargestComponentSize)
            stats.credible95LargestComponentSize = componentSize;
        if (componentMass > stats.credible95LargestComponentMass)
            stats.credible95LargestComponentMass = componentMass;
        if (containsMap == true)
            {
            stats.credible95MapComponentSize = componentSize;
            stats.credible95MapComponentMass = componentMass;
            }
        }

    return stats;
}

BarrierSummary TreeSpace::computeBarrierSummary(std::vector<SaddleInfo>& saddles) {

    if (peaks.empty() == true)
        characterize();

    BarrierSummary stats;
    BasinSummary basinStats = computeBasinSummary();
    std::unordered_map<uint64_t,uint64_t> assignments = computePeakAssignments();

    std::map<std::pair<uint64_t,uint64_t>,SaddleInfo> bestSaddle;

    for (auto& [hash, node] : treeNodes)
        {
        uint64_t peak1 = assignments[hash];
        for (TreeSpaceNode* nb : node->neighbors)
            {
            uint64_t peak2 = assignments[nb->treeHash];
            if (peak1 == peak2)
                continue;
            if (hash > nb->treeHash)
                continue; // each undirected edge once

            uint64_t a = std::min(peak1, peak2);
            uint64_t b = std::max(peak1, peak2);
            std::pair<uint64_t,uint64_t> key = std::make_pair(a,b);
            double saddleLnL = std::min((double)node->lnL, (double)nb->lnL);

            auto it = bestSaddle.find(key);
            if (it == bestSaddle.end() || saddleLnL > it->second.saddleLnL)
                {
                SaddleInfo s;
                s.peakHash1 = a;
                s.peakHash2 = b;
                Peak* p1 = findPeak(a);
                Peak* p2 = findPeak(b);
                if (p1 == nullptr || p2 == nullptr)
                    Msg::error("Could not find peak while computing saddle");
                s.peakId1 = p1->getPeakId();
                s.peakId2 = p2->getPeakId();
                s.treeHash1 = hash;
                s.treeHash2 = nb->treeHash;
                s.saddleLnL = saddleLnL;
                s.saddleProbability = std::min(getTreeProbabiity(hash), getTreeProbabiity(nb->treeHash));
                s.barrierFromPeak1 = treeCache->getTreeInfo(a)->lnLikelihood - saddleLnL;
                s.barrierFromPeak2 = treeCache->getTreeInfo(b)->lnLikelihood - saddleLnL;
                bestSaddle[key] = s;
                }
            }
        }

    for (auto& [key, val] : bestSaddle)
        saddles.push_back(val);

    std::sort(saddles.begin(), saddles.end(),
              [](const SaddleInfo& a, const SaddleInfo& b)
                  {
                  if (a.saddleLnL != b.saddleLnL)
                      return a.saddleLnL > b.saddleLnL;
                  if (a.peakId1 != b.peakId1)
                      return a.peakId1 < b.peakId1;
                  return a.peakId2 < b.peakId2;
                  });

    stats.basinGraphEdges = saddles.size();
    if (peaks.empty() == true)
        return stats;

    std::unordered_map<uint64_t,std::set<uint64_t>> basinGraph;
    for (auto& [peakHash, peak] : peaks)
        basinGraph[peakHash];
    for (const SaddleInfo& s : saddles)
        {
        basinGraph[s.peakHash1].insert(s.peakHash2);
        basinGraph[s.peakHash2].insert(s.peakHash1);
        }

    size_t degreeSum = 0;
    for (auto& [peakHash, neighbors] : basinGraph)
        degreeSum += neighbors.size();
    stats.meanBasinGraphDegree = (double)degreeSum / (double)peaks.size();
    stats.mapBasinGraphDegree = (int)basinGraph[basinStats.mapPeakHash].size();

    if (peaks.size() == 1)
        stats.basinGraphConnected = true;
    else
        {
        std::unordered_set<uint64_t> visited;
        std::queue<uint64_t> q;
        q.push(basinGraph.begin()->first);
        visited.insert(basinGraph.begin()->first);
        while (q.empty() == false)
            {
            uint64_t h = q.front();
            q.pop();
            for (uint64_t nb : basinGraph[h])
                {
                if (visited.find(nb) == visited.end())
                    {
                    visited.insert(nb);
                    q.push(nb);
                    }
                }
            }
        stats.basinGraphConnected = (visited.size() == peaks.size());
        }

    if (saddles.empty() == true)
        return stats;

    double sumBarrier = 0.0;
    double sumWeightedBarrier = 0.0;
    double weightSum = 0.0;
    stats.maxBarrier = 0.0;
    stats.minBarrierFromMap = std::numeric_limits<double>::infinity();

    for (const SaddleInfo& s : saddles)
        {
        Peak* p1 = findPeak(s.peakHash1);
        Peak* p2 = findPeak(s.peakHash2);
        double m1 = p1->getPeakProbability();
        double m2 = p2->getPeakProbability();
        double meanBarrierForEdge = 0.5 * (s.barrierFromPeak1 + s.barrierFromPeak2);
        double weight = m1 * m2;

        sumBarrier += meanBarrierForEdge;
        sumWeightedBarrier += weight * meanBarrierForEdge;
        weightSum += weight;
        if (s.barrierFromPeak1 > stats.maxBarrier)
            stats.maxBarrier = s.barrierFromPeak1;
        if (s.barrierFromPeak2 > stats.maxBarrier)
            stats.maxBarrier = s.barrierFromPeak2;

        if (s.peakHash1 == basinStats.mapPeakHash && s.barrierFromPeak1 < stats.minBarrierFromMap)
            stats.minBarrierFromMap = s.barrierFromPeak1;
        if (s.peakHash2 == basinStats.mapPeakHash && s.barrierFromPeak2 < stats.minBarrierFromMap)
            stats.minBarrierFromMap = s.barrierFromPeak2;
        }

    stats.meanBarrier = sumBarrier / (double)saddles.size();
    if (weightSum > 0.0)
        stats.posteriorWeightedMeanBarrier = sumWeightedBarrier / weightSum;
    if (stats.minBarrierFromMap == std::numeric_limits<double>::infinity())
        stats.minBarrierFromMap = 0.0;

    return stats;
}


TreeSpaceNode* TreeSpace::findBestNeighbor(TreeSpaceNode* nde) {

    double bestLnL = (*nde->neighbors.begin())->lnL;
    TreeSpaceNode* bestNeighbor = (*nde->neighbors.begin());
    for (TreeSpaceNode* n : nde->neighbors)
        {
        if (n->lnL > bestLnL)
            {
            bestLnL = n->lnL;
            bestNeighbor = n;
            }
        }
    return bestNeighbor;
}

Peak* TreeSpace::findPeak(uint64_t treeHash) {

    PeakMap::iterator it = peaks.find(treeHash);
    if (it != peaks.end())
        return it->second;
    return nullptr;
}

Peak* TreeSpace::findPeakForTreeWithHash(uint64_t treeHash) {

    for (auto& [key,val] : peaks)
        {
        if (val->isTreeInPeak(treeHash) == true)
            return val;
        }
    return nullptr;
}

Peak* TreeSpace::findPeakWithId(int id) {

    for (auto& [key,val] : peaks)
        {
        if (val->getPeakId() == id)
            return val;
        }
    return nullptr;
}

TreeSpaceNode* TreeSpace::getTree(uint64_t treeHash) {

    TreeNodesMap::iterator it = treeNodes.find(treeHash);
    if (it == treeNodes.end())
        {
        TreeSpaceNode* newNode = new TreeSpaceNode;
        TreeInfo* info = treeCache->getTreeInfo(treeHash);
        if (info == nullptr)
            Msg::error("Could not find tree when constructing tree space");
        newNode->treeHash = treeHash;
        newNode->lnL = info->lnLikelihood;
        treeNodes.insert( std::make_pair(treeHash,newNode) );
        return newNode;
        }
    return it->second;
}

double TreeSpace::getTreeProbabiity(uint64_t treeHash) {

    TreeProbMap::iterator it = treeProbabilities.find(treeHash);
    if (it == treeProbabilities.end())
        Msg::error("Could not find tree probability");
    return it->second;
}

int TreeSpace::graphDistance(const TreeSpaceNode* a, const TreeSpaceNode* b) {

    if (a == nullptr || b == nullptr)
        return -1;

    if (a == b)
        return 0;

    std::queue<std::pair<const TreeSpaceNode*, int>> q;
    std::unordered_set<const TreeSpaceNode*> visited;
    visited.reserve(1024); // optional

    visited.insert(a);
    q.push({a, 0});

    while (!q.empty())
        {
        const auto [cur, dist] = q.front();
        q.pop();

        // Explore neighbors (each step is +1 edge)
        for (const TreeSpaceNode* nb : cur->neighbors)
            {
            if (nb == nullptr)
                continue;

            if (visited.find(nb) != visited.end())
                continue;

            if (nb == b)
                return dist + 1;

            visited.insert(nb);
            q.push({nb, dist + 1});
            }
        }

    return -1; // unreachable
}

int TreeSpace::peakIdForTreeWithHash(uint64_t treeHash) {

    TreeNodesMap::iterator it = treeNodes.find(treeHash);
    if (it == treeNodes.end())
        Msg::error("Could not find tree node");
    return it->second->peakId;
}

void TreeSpace::printPosterior(void) {

    // copy into a vector of pairs
    std::vector<std::pair<uint64_t,double>> vec(treeProbabilities.begin(), treeProbabilities.end());

    // sort by value (descending)
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });

    // print keys and values 
    std::cout << "   Exact posterior probability distribution: " << std::endl;
    TreeKey& tKey = TreeKey::treeKey();
    double sum = 0.0;
    for (const auto& [key, value] : vec) 
        {
        Peak* pk = findPeakForTreeWithHash(key);
        int peakId = pk->getPeakId();
        sum += value;
        unsigned tNum = tKey.numberForTreeHash(key);
        std::cout << "      " << std::left << std::setw(10) << tNum << " " << std::setw(4) << peakId << " -- " << value << " " << sum << "\n";
        if (sum > 0.999)
            break;
        }
    std::cout << std::endl;
}

void TreeSpace::printPosterior(std::string fileName) {

    std::ofstream strm(fileName);
    
    // copy into a vector of pairs
    std::vector<std::pair<uint64_t,double>> vec(treeProbabilities.begin(), treeProbabilities.end());

    // sort by value (descending)
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });

    // print keys and values
    double sum = 0.0;
    TreeKey& tKey = TreeKey::treeKey();
    for (const auto& [key, value] : vec) 
        {
        sum += value;
        unsigned tNum = tKey.numberForTreeHash(key);
        strm << "      " << std::setw(10) << tNum << " -- " << value << " " << sum << "\n";
        if (sum > 0.999)
            break;
        }
    
    strm.close();
}

void TreeSpace::printPosterior(TreeSamples* samples) {

    // copy into a vector of pairs
    std::vector<std::pair<uint64_t,double>> vec(treeProbabilities.begin(), treeProbabilities.end());

    // sort by value (descending)
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;  // highest to lowest
        });

    // print keys (and values if you want)
    std::cout << "   True posterior probability distribution: " << std::endl;
    TreeKey& tKey = TreeKey::treeKey();
    double sumTrue = 0.0, sumMcmc = 0.0;
    for (const auto& [key, value] : vec) 
        {
        Peak* peak = findPeakForTreeWithHash(key);
        int peakId = peak->getPeakId();
        double mcmcApprox = samples->getTreeProbability(key);
        sumTrue += value;
        sumMcmc += mcmcApprox;
        unsigned tNum = tKey.numberForTreeHash(key);
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "      " << std::setw(10) << tNum << std::setw(3) << " " << peakId << " -- " << value << " <-> " << mcmcApprox << " " << sumTrue << "\n";
        if (sumTrue > 0.999 && sumMcmc > 0.999)
            break;
        }
}

uint64_t TreeSpace::steepestAscent(uint64_t startTree) {

    TreeSpaceNode* currentTree = getTree(startTree);
    bool stopWalk = false;
    do {
        TreeSpaceNode* bestNeighbor = findBestNeighbor(currentTree);
        if (bestNeighbor == nullptr)
            Msg::error("bestNeighbor is NULL");
        if (bestNeighbor->lnL > currentTree->lnL)
            currentTree = bestNeighbor;
        else 
            stopWalk = true;

        } while(stopWalk == false);

    return currentTree->treeHash;
}




int TreeSpace::steepestAscentLength(uint64_t startTree) {

    TreeSpaceNode* currentTree = getTree(startTree);
    int length = 0;
    bool stopWalk = false;
    do {
        TreeSpaceNode* bestNeighbor = findBestNeighbor(currentTree);
        if (bestNeighbor == nullptr)
            Msg::error("bestNeighbor is NULL");
        if (bestNeighbor->lnL > currentTree->lnL)
            {
            currentTree = bestNeighbor;
            length++;
            }
        else
            stopWalk = true;

        } while(stopWalk == false);

    return length;
}

void TreeSpace::writeRuggednessStatistics(std::string fileName) {

    if (peaks.empty() == true)
        characterize();

    LocalSummary local = computeLocalSummary();
    BasinSummary basin = computeBasinSummary();
    AscentSummary ascent = computeAscentSummary();
    CredibleComponentSummary credible = computeCredible95ComponentSummary();
    std::vector<SaddleInfo> saddles;
    BarrierSummary barrier = computeBarrierSummary(saddles);

    std::ofstream strm(fileName);
    if (strm.is_open() == false)
        Msg::error("Could not open ruggedness statistics file");

    strm << std::fixed << std::setprecision(10);

    strm << "section\tstatistic\tvalue\n";
    strm << "summary\tnumTrees\t" << local.numTrees << "\n";
    strm << "summary\tnumEdges\t" << local.numEdges << "\n";
    strm << "summary\tmeanDegree\t" << local.meanDegree << "\n";
    strm << "local\tnumLocalPeaks\t" << local.numLocalPeaks << "\n";
    strm << "local\tposteriorMassOfLocalPeaks\t" << local.posteriorMassOfLocalPeaks << "\n";
    strm << "local\tmeanUphillFraction\t" << local.meanUphillFraction << "\n";
    strm << "local\tposteriorWeightedMeanUphillFraction\t" << local.posteriorWeightedMeanUphillFraction << "\n";
    strm << "local\tmeanBestNeighborLnLDifference\t" << local.meanBestNeighborLnLDifference << "\n";
    strm << "local\tposteriorWeightedMeanBestNeighborLnLDifference\t" << local.posteriorWeightedMeanBestNeighborLnLDifference << "\n";
    strm << "local\tmeanNeighborAbsLnLDifference\t" << local.meanNeighborAbsLnLDifference << "\n";
    strm << "local\tposteriorWeightedMeanNeighborAbsLnLDifference\t" << local.posteriorWeightedMeanNeighborAbsLnLDifference << "\n";
    strm << "basin\tnumPeaks\t" << basin.numPeaks << "\n";
    strm << "basin\tnumPeaksMassGreater01\t" << basin.numPeaksMassGreater01 << "\n";
    strm << "basin\tnumPeaksMassGreater05\t" << basin.numPeaksMassGreater05 << "\n";
    strm << "basin\tlargestBasinSize\t" << basin.largestBasinSize << "\n";
    strm << "basin\tlargestBasinMass\t" << basin.largestBasinMass << "\n";
    strm << "basin\tbasinMassEntropy\t" << basin.basinMassEntropy << "\n";
    strm << "basin\teffectiveNumBasins\t" << basin.effectiveNumBasins << "\n";
    strm << "basin\tmapTreeHash\t" << basin.mapTreeHash << "\n";
    strm << "basin\tmapPeakHash\t" << basin.mapPeakHash << "\n";
    strm << "basin\tmapPeakId\t" << basin.mapPeakId << "\n";
    strm << "basin\tmapBasinMass\t" << basin.mapBasinMass << "\n";
    strm << "basin\tposteriorMassOutsideMapBasin\t" << basin.posteriorMassOutsideMapBasin << "\n";
    strm << "ascent\tmeanAscentLength\t" << ascent.meanAscentLength << "\n";
    strm << "ascent\tposteriorWeightedMeanAscentLength\t" << ascent.posteriorWeightedMeanAscentLength << "\n";
    strm << "ascent\tmeanAscentLengthCredible95\t" << ascent.meanAscentLengthCredible95 << "\n";
    strm << "ascent\tposteriorWeightedMeanAscentLengthCredible95\t" << ascent.posteriorWeightedMeanAscentLengthCredible95 << "\n";
    strm << "ascent\tmaxAscentLength\t" << ascent.maxAscentLength << "\n";
    strm << "credible95\tcredible95NumTrees\t" << credible.credible95NumTrees << "\n";
    strm << "credible95\tcredible95Mass\t" << credible.credible95Mass << "\n";
    strm << "credible95\tcredible95NumComponents\t" << credible.credible95NumComponents << "\n";
    strm << "credible95\tcredible95LargestComponentSize\t" << credible.credible95LargestComponentSize << "\n";
    strm << "credible95\tcredible95LargestComponentMass\t" << credible.credible95LargestComponentMass << "\n";
    strm << "credible95\tcredible95MapComponentSize\t" << credible.credible95MapComponentSize << "\n";
    strm << "credible95\tcredible95MapComponentMass\t" << credible.credible95MapComponentMass << "\n";
    strm << "barrier\tbasinGraphEdges\t" << barrier.basinGraphEdges << "\n";
    strm << "barrier\tmeanBasinGraphDegree\t" << barrier.meanBasinGraphDegree << "\n";
    strm << "barrier\tmapBasinGraphDegree\t" << barrier.mapBasinGraphDegree << "\n";
    strm << "barrier\tbasinGraphConnected\t" << (barrier.basinGraphConnected ? 1 : 0) << "\n";
    strm << "barrier\tminBarrierFromMap\t" << barrier.minBarrierFromMap << "\n";
    strm << "barrier\tmeanBarrier\t" << barrier.meanBarrier << "\n";
    strm << "barrier\tposteriorWeightedMeanBarrier\t" << barrier.posteriorWeightedMeanBarrier << "\n";
    strm << "barrier\tmaxBarrier\t" << barrier.maxBarrier << "\n";

    strm << "\n# peaks\n";
    strm << "peakId\tpeakHash\tpeakTreeProbability\tbasinNumTrees\tbasinPosteriorMass\n";
    std::vector<Peak*> sortedPeaks;
    sortedPeaks.reserve(peaks.size());
    for (auto& [hash, peak] : peaks)
        sortedPeaks.push_back(peak);
    std::sort(sortedPeaks.begin(), sortedPeaks.end(), [](Peak* a, Peak* b)
      {
      return a->getPeakId() < b->getPeakId();
      });
    for (Peak* peak : sortedPeaks)
        {
        strm << peak->getPeakId() << '\t'
             << peak->getPeakTreeHash() << '\t'
             << peak->getPeakTreeProbability() << '\t'
             << peak->getNumTrees() << '\t'
             << peak->getPeakProbability() << "\n";
        }

    strm << "\n# saddles\n";
    strm << "peakId1\tpeakId2\tpeakHash1\tpeakHash2\tsaddleTree1\tsaddleTree2\tsaddleLnL\tsaddleProbability\tbarrierFromPeak1\tbarrierFromPeak2\n";
    for (const SaddleInfo& s : saddles)
        {
        strm << s.peakId1 << '\t'
             << s.peakId2 << '\t'
             << s.peakHash1 << '\t'
             << s.peakHash2 << '\t'
             << s.treeHash1 << '\t'
             << s.treeHash2 << '\t'
             << s.saddleLnL << '\t'
             << s.saddleProbability << '\t'
             << s.barrierFromPeak1 << '\t'
             << s.barrierFromPeak2 << "\n";
        }

    strm.close();
}
