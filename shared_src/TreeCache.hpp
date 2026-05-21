#ifndef TreeCache_hpp
#define TreeCache_hpp

#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>
class CompactTree;
class Tree;



struct TreeInfo {

    uint64_t                hash = 0;
    Tree*                   tree = nullptr;
    CompactTree*            compactTree = nullptr;
    std::vector<TreeInfo*>  neighbors;
    double                  lnLikelihood = std::numeric_limits<double>::quiet_NaN();
    bool                    hasNeighbors = false;
    bool                    hasLnLikelihood = false;
};

typedef std::unordered_map<uint64_t, TreeInfo*> TreeCacheMap;
using TreeCache = TreeCacheMap;

size_t      cacheSize(TreeCache* treeCache);
void        freeTreeCache(TreeCache* treeCache);
TreeInfo*   getOrCreateTreeInfo(TreeCache* treeCache, Tree* t);

#endif
