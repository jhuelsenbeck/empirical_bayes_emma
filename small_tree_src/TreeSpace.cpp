#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <unordered_set>
#include <utility>
#include "Msg.hpp"
#include "Peak.hpp"
#include "TreeLikelihoods.hpp"
#include "TreeNeighbors.hpp"
#include "TreeSamples.hpp"
#include "TreeSpace.hpp"



TreeSpace::TreeSpace(TreeCache* tc, TreeLikelihoods* tl, TreeNeighbors* tn) : 
    treeCache(tc), treeLikelihoods(tl), treeNeighbors(tn) {
        
    // construct graph
    for (auto& [key,val] : *treeCache)
        {
        TreeSpaceNode* thisTree = getTree(key);
        std::vector<TreeInfo*>& ndeNeighbors = val->neighbors;
        for (size_t i=0; i<ndeNeighbors.size(); i++)
            {
            TreeSpaceNode* neighboringTree = getTree(ndeNeighbors[i]->hash);
            neighboringTree->neighbors.insert(thisTree);
            thisTree->neighbors.insert(neighboringTree);
            }
        }
       
    // calculate the exact posterior probability
    double bestLnL = std::numeric_limits<double>::lowest();
    for (auto& [key,val] : *treeCache)
        {
        if (val->lnLikelihood > bestLnL)
            bestLnL = val->lnLikelihood;
        treeProbabilities.insert( std::make_pair(key,val->lnLikelihood) );
        }
    double sum = 0.0;
    for (auto& [key,val] : treeProbabilities)
        {
        double x = std::exp(val-bestLnL);
        val = x;
        sum += x;
        }
    double factor = 1.0 / sum;
    for (auto& [key,val] : treeProbabilities)
        val *= factor;    
}

TreeSpace::~TreeSpace(void) {

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

    // label the peaks in order from highest to lowest
    std::vector<std::pair<uint64_t, Peak*>> vec(peaks.begin(), peaks.end());
    std::sort(vec.begin(), vec.end(),
              [](const auto& a, const auto& b) {
                  return a.second->getPeakTreeProbability() > b.second->getPeakTreeProbability();
                  // or: a.second->getProbability() > b.second->getProbability();
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

    // print peak information
    for (const auto& [key, peak] : vec) 
        {
        std::cout << peak->getPeakId() << ": " << std::setw(20) << key << " " 
                  << peak->getPeakTreeLnProbability() << " " << peak->getNumTrees() << " " << " " << peak->getPeakProbability() << "\n";
        }
    std::cout << "Largest basin fraction   = " << bMax << std::endl;
    std::cout << "Peak mass entropy        = " << peakMassEntropy << std::endl;
    std::cout << "Expected number of peaks = " << std::exp(peakMassEntropy) << std::endl;
    if (peaks.size() > 1)
        std::cout << "Dominance ratio          = " << dominanceRatio << std::endl;
    else 
        std::cout << "Dominance ratio          = Undefined" << std::endl;
        

//
//    // find the peaks and the size of the basin of attraction for each
//    std::unordered_map<uint64_t,PeakInfo> peaks;
//    int numPeaks = 0;
//    for (auto& [key,val] : treeNodes)
//        {
//        uint64_t peak = steepestAscent(key);        
//        val->peakPtr = getTree(peak);
//        std::unordered_map<uint64_t,PeakInfo>::iterator it = peaks.find(peak);
//        if (it == peaks.end())
//            {
//            TreeInfo& tinfo = treeList->getTreeInfo(peak);
//            PeakInfo pinfo;
//            pinfo.basinSize = 1;
//            pinfo.lnL = tinfo.lnL;
//            pinfo.minLnL = val->lnL;
//            pinfo.peakId = numPeaks;
//            peaks.insert( std::make_pair(peak,pinfo) );
//            val->peakId = numPeaks;
//            numPeaks++;
//            }
//        else 
//            {
//            it->second.basinSize++;
//            if (val->lnL < it->second.minLnL)
//                it->second.minLnL = val->lnL;
//            val->peakId = it->second.peakId;
//            }
//        }
//        
//    // find the saddle points
//    std::map<std::pair<int,int>,std::tuple<float,int,int>> saddlePoints;
//    for (auto& [key,val] : treeNodes)
//        {
//        int thisPeakId = val->peakId;
//        for (TreeSpaceNode* n : val->neighbors)
//            {
//            if (n->peakId != thisPeakId)
//                {
//                std::pair<int,int> peakPair;
//                if (thisPeakId < n->peakId)
//                    peakPair = std::make_pair(thisPeakId,n->peakId);
//                else 
//                    peakPair = std::make_pair(n->peakId,thisPeakId);
//                    
//                std::map<std::pair<int,int>,std::tuple<float,int,int>>::iterator it = saddlePoints.find(peakPair);
//                if (it == saddlePoints.end())
//                    {
//                    int d1 = 0, d2 = 0;
//                    if (thisPeakId < n->peakId)
//                        {
//                        d1 = graphDistance(val, val->peakPtr);
//                        d2 = graphDistance(val, n->peakPtr);
//                        }
//                    else 
//                        {
//                        d2 = graphDistance(val, val->peakPtr);
//                        d1 = graphDistance(val, n->peakPtr);
//                        }
//                    std::tuple<float,int,int> v(val->lnL,d1,d2);
//                    saddlePoints.insert( std::make_pair(peakPair,v) );
//                    }
//                else 
//                    {
//                    if (val->lnL > std::get<0>(it->second))
//                        {
//                        saddlePoints.erase(it);
//                        int d1 = 0, d2 = 0;
//                        if (thisPeakId < n->peakId)
//                            {
//                            d1 = graphDistance(val, val->peakPtr);
//                            d2 = graphDistance(val, n->peakPtr);
//                            }
//                        else 
//                            {
//                            d2 = graphDistance(val, val->peakPtr);
//                            d1 = graphDistance(val, n->peakPtr);
//                            }
//                        std::tuple<float,int,int> v(val->lnL,d1,d2);
//                        saddlePoints.insert( std::make_pair(peakPair,v) );
//                        }
//                    }
//                    
//                }
//            }
//        
//        }
//        
//    // calculate distance from saddle points to peaks
//    std::map<std::pair<int,int>,std::pair<int,int>> saddlePointDistances;
//        
//    // sort peaks by lnL in descending order
//    std::vector<std::pair<uint64_t, PeakInfo>> sortedPeaks(peaks.begin(), peaks.end());
//    std::sort(sortedPeaks.begin(), sortedPeaks.end(), [](const auto& a, const auto& b) {
//            return a.second.lnL > b.second.lnL;  // descending order
//        });
//        
//    std::cout << "   Peak information:" << std::endl;
//    for (const auto& [key, val] : sortedPeaks)
//        std::cout << "   * " << std::setw(3) << val.peakId << " " << std::setw(20) << key << " -- " << std::setw(8) << val.basinSize << " " << val.lnL << " " << val.minLnL << std::endl;
//    
//    std::cout << "   Saddle point inforamtion:" << std::endl;
//    for (const auto& [key, val] : saddlePoints)
//        std::cout << "   * " << std::setw(2) << key.first 
//                  << " - " << std::setw(2) << key.second 
//                  << " -- " << std::get<0>(val) << " ("
//                  << std::get<1>(val) << "," << std::get<2>(val) << ")"
//                  << std::endl;

#   if 0
    // find peaks
    std::vector<uint64_t> peaks;
    for (auto& [key,val] : treeNodes)
        {
        double lnL0 = val->lnL;
        val->isPeak = false;
        val->peakId = 0;
        bool isPeak = true;
        for (TreeSpaceNode* n : val->neighbors)
            {
            double lnL = n->lnL;
            if (lnL > lnL0)
                {
                isPeak = false;
                break;
                }
            }
        if (isPeak == true)
            {
            peaks.push_back(key);
            val->isPeak = true;
            val->peakId = (int)(peaks.size());
            }
        }
        
    int numPeaks = (int)peaks.size();
    int** d = new int*[numPeaks];
    d[0] = new int[numPeaks * numPeaks];
    for (size_t i=1; i<numPeaks; i++)
        d[i] = d[i-1] + numPeaks;
    for (size_t i=0; i<numPeaks; i++)
        for (size_t j=0; j<numPeaks; j++)
            d[i][j] = 0;
            
    for (size_t i=0; i<numPeaks; i++)
        {
        calculateDistances(peaks[i]);
        for (size_t j=i+1; j<numPeaks; j++)
            {
            TreeSpaceNode* p = getTree(peaks[j]);
            d[i][j] = p->distance;
            d[j][i] = p->distance;
            }
        }
            
    std::unordered_map<uint64_t,int> basins;
    characterizeBasins(basins);
    std::vector<PeakInfo> sortedPeakInfo;
    for (auto& [key,val] : basins)
        {
        int closestPeakDistance = 10e6;
        double averagePeakDistance = 0.0;
        uint64_t closestPeak = 0;
        int rowIdx = -1;
        for (int i=0; i<numPeaks; i++)
            {
            if (peaks[i] == key)
                {
                rowIdx = i;
                break;
                }
            }
        if (rowIdx == -1)
            Msg::error("Could not find peak");
        for (int i=0; i<numPeaks; i++)
            {
            if (i != rowIdx)
                {
                averagePeakDistance += (double)d[rowIdx][i];
                if (d[rowIdx][i] < closestPeakDistance)
                    {
                    closestPeakDistance = d[rowIdx][i];
                    closestPeak = peaks[i];
                    }
                }
            }
        averagePeakDistance /= (numPeaks-1);

        PeakInfo info;
        info.treeHash = key;
        info.lnL = getTree(key)->lnL;
        info.basinSize = val;
        info.closestPeak = closestPeak;
        info.closestPeakDistance = closestPeakDistance;
        info.averageDistanceToPeaks = averagePeakDistance;
        
        sortedPeakInfo.push_back(info);
        }
    std::sort(sortedPeakInfo.begin(), sortedPeakInfo.end(),
        [](const PeakInfo& a, const PeakInfo& b) {
            return a.lnL > b.lnL;   // descending
        });
    
    std::cout << "   Tree space characteristics:" << std::endl;
    for (size_t i=0; i<sortedPeakInfo.size(); i++)
        {
        PeakInfo& info = sortedPeakInfo[i];
        std::cout << "   * " << i+1 << " -- ";
        std::cout << std::setw(20) << info.treeHash << " ";
        std::cout << info.lnL << " ";
        std::cout << std::setw(5) << info.basinSize << " ";
        std::cout << std::setw(20) << info.closestPeak << " ";
        std::cout << std::setw(8) << info.closestPeakDistance << " ";
        std::cout << std::setw(8) << info.averageDistanceToPeaks << " ";
        std::cout << std::endl;
        }
    std::cout << std::endl;
        
//    for (size_t i=0; i<numPeaks; i++)
//        {
//        for (size_t j=0; j<numPeaks; j++)
//            std::cout << std::setw(2) << d[i][j] << " ";
//        std::cout << std::endl;
//        }
        
    delete [] d[0];
    delete [] d;
    
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

int TreeSpace::findPeakIdForTreeWithHash(uint64_t treeHash) {

    for (auto& [key,val] : peaks)
        {
        if (val->isTreeInPeak(treeHash) == true)
            return val->getPeakId();
        }
    return -1;
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
        TreeCacheMap::iterator it = treeCache->find(treeHash);
        if (it == treeCache->end())
            Msg::error("Could not find tree when constructing tree space");
        TreeInfo* info = it->second;
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

void TreeSpace::printPosterior(void) {

    // copy into a vector of pairs
    std::vector<std::pair<uint64_t,double>> vec(treeProbabilities.begin(), treeProbabilities.end());

    // sort by value (descending)
    std::sort(vec.begin(), vec.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;  // highest to lowest
              });

    // print keys (and values if you want)
    std::cout << "   True posterior probability distribution: " << std::endl;
    double sum = 0.0;
    for (const auto& [key, value] : vec) 
        {
        sum += value;
        std::cout << "      " << std::setw(20) << key << " -- " << value << " " << sum << "\n";
        if (sum > 0.999)
            break;
        }
}

void TreeSpace::printPosterior(TreeSamples* samples) {

    // copy into a vector of pairs
    std::vector<std::pair<uint64_t,double>> vec(treeProbabilities.begin(), treeProbabilities.end());

    // sort by value (descending)
    std::sort(vec.begin(), vec.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;  // highest to lowest
              });

    // print keys (and values if you want)
    std::cout << "   True posterior probability distribution: " << std::endl;
    double sumTrue = 0.0, sumMcmc = 0.0;
    for (const auto& [key, value] : vec) 
        {
        int peakId = findPeakIdForTreeWithHash(key);
        double mcmcApprox = samples->getTreeProbability(key);
        sumTrue += value;
        sumMcmc += mcmcApprox;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "      " << std::setw(20) << key << std::setw(3) << " " << peakId << " -- " << value << " <-> " << mcmcApprox << " " << sumTrue << "\n";
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

