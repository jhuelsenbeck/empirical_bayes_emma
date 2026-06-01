#include "CompactTree.hpp"
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"


void TreeCache::cleanCacheStatistics(void) {

    for (TreeCacheMap::const_iterator it = treeCache.begin(); it != treeCache.end(); ++it)
        {
        TreeInfo* info = it->second;
        if (info == nullptr)
            Msg::error("Tree information not present");
        info->meanFirstHit = 0.0;
        info->m2FirstHit = 0.0;
        info->meanNumRevisits = 0.0;
        info->m2NumRevisits = 0.0;
        info->meanResidenceCount = 0.0;
        info->m2ResidenceCount = 0.0;
        info->firstHit = std::numeric_limits<unsigned>::max();
        info->residenceCount = 0;
        info->numRevisits = 0;
        info->hasBeenVisited = false;
        }
}

TreeInfo* TreeCache::getTreeInfo(uint64_t treeHash) {

    TreeCacheMap::iterator it = treeCache.find(treeHash);
    if (it != treeCache.end())
        return it->second;
    return nullptr;
}

TreeInfo* TreeCache::getOrCreateTreeInfo(Tree* t) {

    uint64_t h = t->getHash();

    TreeCacheMap::iterator it = treeCache.find(h);
    if (it != treeCache.end())
        return it->second;

    TreeInfo* info = new TreeInfo;
    info->compactTree = t->getCompactRepresentation();
    info->hash = t->getHash();

    treeCache[h] = info;

    return info;
}

size_t TreeCache::cacheSize(void) {

    size_t total = sizeof(TreeCache);

    // hash table bucket array
    total += treeCache.bucket_count() * sizeof(void*);

    // per-node overhead: key + value pointer + next pointer
    const size_t nodeOverhead = sizeof(uint64_t) + sizeof(TreeInfo*) + sizeof(void*);
    total += treeCache.size() * nodeOverhead;

    // each TreeInfo and its owned heap data
    for (TreeCacheMap::const_iterator it = treeCache.begin(); it != treeCache.end(); ++it)
        {
        TreeInfo* info = it->second;
        if (info == nullptr)
            continue;
        total += sizeof(TreeInfo);
        if (info->compactTree != nullptr)
            total += info->compactTree->sizeInBytes();
        total += info->neighbors.capacity() * sizeof(TreeInfo*);
        // info->tree intentionally not counted (size unknown, may be shared/transient)
        }

    return total;
}

void TreeCache::freeTreeCache(void) {

    for (auto& pair : treeCache) 
        {
        delete pair.second->compactTree;
        if (pair.second->tree != nullptr)
            delete pair.second->tree;
        delete pair.second;
        }
    treeCache.clear();
}

void TreeCache::injectTreesAndLikelihoods(TreeCache* tc) {

    TreeCacheMap& otherMap = tc->getCache();
    for (auto& [key,val] : otherMap)
        {
        TreeCacheMap::iterator it = treeCache.find(key);
        if (it == treeCache.end())
            {
            TreeInfo* info = new TreeInfo;
            
            info->hash = val->hash;
            info->lnLikelihood = val->lnLikelihood;
            info->hasLnLikelihood = val->hasLnLikelihood;
            info->posteriorProbability = val->posteriorProbability;
            CompactTree* newCompactTree = new CompactTree(*val->compactTree);
            info->compactTree = newCompactTree;
            if (val->tree != nullptr)
                info->tree = new Tree(newCompactTree, val->tree->getNumTips());
            treeCache.insert( std::make_pair(key,info) );
            }
        else 
            {
            it->second->lnLikelihood = val->lnLikelihood;
            it->second->hasLnLikelihood = val->hasLnLikelihood;
            it->second->posteriorProbability = val->posteriorProbability;
            }
        }
}
