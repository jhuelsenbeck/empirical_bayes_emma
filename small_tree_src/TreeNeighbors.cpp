#include <iomanip>
#include <iostream>
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeNeighborGenerator.hpp"
#include "TreeNeighbors.hpp"



TreeNeighbors::TreeNeighbors(TreeCache* tc, TreeNeighborGenerator* ng, int nt) :
    treeCache(tc), neighborGenerator(ng), numTaxa(nt) {

}

NeighborList TreeNeighbors::neighbors(uint64_t treeHash) {

    TreeInfo* info = treeCache->getTreeInfo(treeHash);
    if (info == nullptr)
        Msg::error("Could not find tree hash in cache");
    return neighbors(info);
}

NeighborList TreeNeighbors::neighbors(TreeInfo* info) {

    if (treeCache->hasNeighbors(info) == true)
        return treeCache->neighbors(info);

    // The Tree objects are released once the likelihoods and the MAP-tree partitions are done, so a
    // neighborhood generated later (TBR is generated only when it is analyzed) rebuilds the tree from
    // its compact encoding for the duration of the call.
    if (info->tree != nullptr)
        return neighbors(info->tree);
    Tree* t = new Tree(info->compactTree, numTaxa);
    NeighborList nbrs = neighbors(t);
    delete t;
    return nbrs;
}

NeighborList TreeNeighbors::neighbors(Tree* t) {

    TreeInfo* tInfo = treeCache->getOrCreateTreeInfo(t);
    if (treeCache->hasNeighbors(tInfo) == false)
        {
        scratch.clear();
        neighborGenerator->generateNeighbors(t, scratch);
        treeCache->setNeighbors(tInfo, scratch);
        }
    return treeCache->neighbors(tInfo);
}

void TreeNeighbors::print(void) {

    int i = 0;
    TreeCacheMap& tCache = treeCache->getCache();
    for (auto& [key,val] : tCache)
        {
        std::cout << std::setw(6) << ++i << " " << std::setw(20) << key << " -- ";
        for (TreeInfo* nb : treeCache->neighbors(val))
            std::cout << nb << " ";
        std::cout << std::endl;
        }
}
