#include <iomanip>
#include <iostream>
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeNeighborGenerator.hpp"
#include "TreeNeighbors.hpp"



TreeNeighbors::TreeNeighbors(TreeCache* tc, TreeNeighborGenerator* ng, int nt) : 
    treeCache(tc), neighborGenerator(ng), numTaxa(nt) {

}

std::vector<TreeInfo*>& TreeNeighbors::neighbors(uint64_t treeHash) {

    TreeCacheMap& tCache = treeCache->getCache();
    TreeCacheMap::iterator it = tCache.find(treeHash);
    if (it == tCache.end())
        Msg::error("Could not find tree hash in cache");

    Tree* t = new Tree(it->second->compactTree, numTaxa);
    std::vector<TreeInfo*>& nbrs = neighbors(t);
    delete t;

    return nbrs;
}

std::vector<TreeInfo*>& TreeNeighbors::neighbors(Tree* t) {

    TreeInfo* tInfo = treeCache->getOrCreateTreeInfo(t);
    if (tInfo->neighbors.size() == 0)
        neighborGenerator->generateNeighbors(t, tInfo->neighbors);
    return tInfo->neighbors;
}

void TreeNeighbors::print(void) {

    int i = 0;
    TreeCacheMap& tCache = treeCache->getCache();
    for (auto& [key,val] : tCache)
        {
        std::cout << std::setw(6) << ++i << " " << std::setw(20) << key << " -- ";
        for (size_t i=0; i<val->neighbors.size(); i++)
            std::cout << val->neighbors[i] << " ";
        std::cout << std::endl;
        }
}
