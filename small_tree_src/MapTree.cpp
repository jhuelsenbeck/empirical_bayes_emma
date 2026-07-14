#include <iostream>
#include "MapTree.hpp"
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"
#include "TreeKey.hpp"



MapTree::MapTree(TreeCache* c) : treeCache(c) {

    findMapTree();
    numTaxa = mapTree->getNumTips();
    findPartitions();
    print();
}

void MapTree::findMapTree(void) {

    mapHash = 0;
    mapTree = nullptr;
    double bestLnL = 0.0;
    TreeCacheMap& tempMap = treeCache->getCache();
    for (auto& [key,val] : tempMap)
        {
        if (val->hasLnLikelihood == false)
            Msg::error("Expecting tree cache to contain valid ln likelihoods");
            
        if (mapTree == nullptr)
            {
            mapTree = val->tree;
            mapHash = key;
            bestLnL = val->lnLikelihood;
            }
        else    
            {
            if (val->lnLikelihood > bestLnL)
                {
                mapTree = val->tree;
                mapHash = key;
                bestLnL = val->lnLikelihood;
                }
            }
            
        }
}

void MapTree::findPartitions(void) {

    for (uint16_t part : mapTree->getPartitions())
        mapPartitions[part];

    TreeCacheMap& treeMap = treeCache->getCache();

    for (auto& [hash, info] : treeMap)
        {
        const auto treePartitions = info->tree->getPartitions();

        for (auto& [part, trees] : mapPartitions)
            {
            if (treePartitions.contains(part))
                trees.insert(info);
            }
        }    
}

double MapTree::partitionProbability(uint16_t part) {

    PartitionMap::iterator it = mapPartitions.find(part);
    if (it == mapPartitions.end())
        Msg::error("Could not find partition in set of MAP partitions");
    
    std::set<TreeInfo*>& parts = it->second;
    double prob = 0.0;
    for (TreeInfo* info : parts)
        prob += info->posteriorProbability;
    return prob;
}

std::string MapTree::partitionString(uint16_t part) {

    std::string str = "";
    uint16_t mask = 1;
    for (int i=0; i<numTaxa; i++)
        {
        if ((mask & part) == 0)
            str += "0";
        else 
            str += "1";
        mask = mask << 1;
        }
    return str;
}

void MapTree::printPartition(uint16_t part) {

    uint16_t mask = 1;
    for (int i=0; i<numTaxa; i++)
        {
        if ((mask & part) == 0)
            std::cout << "0";
        else 
            std::cout << "1";
        mask = mask << 1;
        }
    std::cout << std::endl;
}

void MapTree::print(void) {

    TreeKey& tKey = TreeKey::treeKey();
    
    std::cout << "   MAP Tree" << std::endl;
    std::cout << "   * ID  = " << tKey.numberForTreeHash(mapHash) << std::endl;
    std::cout << "   * Pr(T=" << tKey.numberForTreeHash(mapHash) << " | X) = " << treeCache->getTreeInfo(mapHash)->posteriorProbability << std::endl;
    for (auto& [part, trees] : mapPartitions)
        {
        std::cout << "   * " << partitionString(part) << " = " << partitionProbability(part) << " (" << trees.size() << ")" << std::endl;;
        }
        
#   if 0
    for (auto& [part, trees] : mapPartitions)
        {
        std::cout << partitionString(part) << ": ";
        for (TreeInfo* info : trees)
            {
            std::cout << tKey.numberForTreeHash(info->hash) << " ";
            }
        std::cout << std::endl;
        }
#   endif

    std::cout << std::endl;
}
