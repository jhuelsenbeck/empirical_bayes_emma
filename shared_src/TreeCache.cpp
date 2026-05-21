#include "CompactTree.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"


TreeInfo* getOrCreateTreeInfo(TreeCache* treeCache, Tree* t) {

    uint64_t h = t->getHash();

    TreeCacheMap::iterator it = treeCache->find(h);
    if (it != treeCache->end())
        return it->second;

    TreeInfo* info = new TreeInfo;
    info->compactTree = t->getCompactRepresentation();
    info->hash = t->getHash();

    (*treeCache)[h] = info;

    return info;
}

size_t cacheSize(TreeCache* treeCache) {

    size_t total = sizeof(TreeCache);

    // hash table bucket array
    total += treeCache->bucket_count() * sizeof(void*);

    // per-node overhead: key + value pointer + next pointer
    const size_t nodeOverhead = sizeof(uint64_t) + sizeof(TreeInfo*) + sizeof(void*);
    total += treeCache->size() * nodeOverhead;

    // each TreeInfo and its owned heap data
    for (TreeCacheMap::const_iterator it = treeCache->begin(); it != treeCache->end(); ++it)
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

void freeTreeCache(TreeCache* cache) {

    for (auto& pair : *cache) 
        {
        delete pair.second->compactTree;
        delete pair.second;
        }
    cache->clear();
}
