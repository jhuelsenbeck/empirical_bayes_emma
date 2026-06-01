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
    double                  posteriorProbability = 0.0;

    double                  meanFirstHit = 0.0;
    double                  m2FirstHit = 0.0;
    double                  meanNumRevisits = 0.0;
    double                  m2NumRevisits = 0.0;
    double                  meanResidenceCount = 0.0;
    double                  m2ResidenceCount = 0.0;

    unsigned                firstHit = std::numeric_limits<unsigned>::max();
    unsigned                residenceCount = 0;
    unsigned                numRevisits = 0;
    
    bool                    hasBeenVisited = false;
    bool                    hasNeighbors = false;
    bool                    hasLnLikelihood = false;
};


typedef std::unordered_map<uint64_t, TreeInfo*> TreeCacheMap;

class TreeCache {

    public:
        size_t          cacheSize(void);
        void            cleanCacheStatistics(void);
        void            freeTreeCache(void);
        TreeCacheMap&   getCache(void) { return treeCache; }
        TreeInfo*       getTreeInfo(uint64_t treeHash);
        TreeInfo*       getOrCreateTreeInfo(Tree* t);
        void            injectTreesAndLikelihoods(TreeCache* tc);
        size_t          size(void) { return treeCache.size(); }
    
    private:
        TreeCacheMap    treeCache;
};

//using TreeCache = TreeCacheMap;

//size_t      cacheSize(TreeCache* treeCache);
//void        cleanCacheStatistics(TreeCache* treeCache);
//void        freeTreeCache(TreeCache* treeCache);
//TreeInfo*   getTreeInfo(TreeCache* treeCache, uint64_t treeHash);
//TreeInfo*   getOrCreateTreeInfo(TreeCache* treeCache, Tree* t);

#endif
