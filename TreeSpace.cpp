#include <algorithm>
#include <iomanip>
#include <iostream>
#include "Msg.hpp"
#include "TreeList.hpp"
#include "TreeNeighborhood.hpp"
#include "TreeSpace.hpp"



TreeSpace::TreeSpace(TreeList* tl) : treeList(tl) {

    TreeNeighborhoodNni neighbors(treeList);
    TreeMap& trees = treeList->getTreeList();
    for (auto& [key,val] : trees)
        neighbors.getNeighbors(key);
    if (neighbors.size() != treeList->size())
        Msg::error("Problem enumerating neighbors of all trees");
        
    for (auto& [key,val] : trees)
        {
        TreeSpaceNode* thisTree = getTree(key);
        TreeHashVec& ndeNeighbors = neighbors.getNeighbors(key);
        for (size_t i=0; i<ndeNeighbors.size(); i++)
            {
            TreeSpaceNode* neighboringTree = getTree(ndeNeighbors[i]);
            neighboringTree->neighbors.insert(thisTree);
            thisTree->neighbors.insert(neighboringTree);
            }
        }
}

TreeSpace::~TreeSpace(void) {

    for (auto& [key,val] : treeNodes)
        delete val;
}

TreeSpaceNode* TreeSpace::getTree(uint64_t treeHash) {

    TreeNodesMap::iterator it = treeNodes.find(treeHash);
    if (it == treeNodes.end())
        {
        TreeSpaceNode* newNode = new TreeSpaceNode;
        TreeInfo& info = treeList->getTreeInfo(treeHash);
        newNode->treeHash = treeHash;
        newNode->lnL = info.lnL;
        treeNodes.insert( std::make_pair(treeHash,newNode) );
        return newNode;
        }
    return it->second;
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

uint64_t TreeSpace::steepestAscent(uint64_t startTree) {

    TreeSpaceNode* currentTree = getTree(startTree);
    bool stopWalk = false;
    do {
        TreeSpaceNode* bestNeighbor = findBestNeighbor(currentTree);
        if (bestNeighbor->lnL > currentTree->lnL)
            currentTree = bestNeighbor;
        else 
            stopWalk = true;

        } while(stopWalk == false);

    return currentTree->treeHash;
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
