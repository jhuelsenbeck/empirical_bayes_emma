#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeNeighborGenerator.hpp"
#include "TreeNeighbors.hpp"



TreeNeighbors::TreeNeighbors(TreeCache* tc, TreeNeighborGenerator* ng, int nt) : 
    treeCache(tc), neighborGenerator(ng), numTaxa(nt) {

}

std::vector<TreeInfo*>& TreeNeighbors::neighbors(uint64_t treeHash) {

    TreeCacheMap::iterator it = treeCache->find(treeHash);
    if (it == treeCache->end())
        Msg::error("Could not find tree hash in cache");
    Tree* t = new Tree(it->second->compactTree, numTaxa);
    return neighbors(t);
}

std::vector<TreeInfo*>& TreeNeighbors::neighbors(Tree* t) {

    TreeInfo* tInfo = getOrCreateTreeInfo(treeCache, t);
    if (tInfo->neighbors.size() == 0)
        neighborGenerator->generateNeighbors(t, tInfo->neighbors);
    return tInfo->neighbors;
}

