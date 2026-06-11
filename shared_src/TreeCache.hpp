#ifndef TreeCache_hpp
#define TreeCache_hpp

#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
class CompactTree;
class Tree;



struct TreeInfo {

    uint64_t                hash = 0;
    Tree*                   tree = nullptr;
    CompactTree*            compactTree = nullptr;
    std::vector<TreeInfo*>  neighbors;
    std::vector<double>     neighborProposalProbabilities;
    
    double                  lnLikelihood = std::numeric_limits<double>::quiet_NaN();
    double                  posteriorProbability = 0.0;

    bool                    hasNeighbors = false;
    bool                    hasLnLikelihood = false;
    bool                    hasNeighborProposalProbabilities = false;
    double                  neighborProposalPower = std::numeric_limits<double>::quiet_NaN();
};


typedef std::unordered_map<uint64_t, TreeInfo*> TreeCacheMap;

class TreeCache {

    public:
                        TreeCache(void) = delete;
                        TreeCache(std::string nm);
        size_t          cacheSize(void);
        void            freeTreeCache(void);
        TreeCacheMap&   getCache(void) { return treeCache; }
        TreeInfo*       getTreeInfo(uint64_t treeHash);
        TreeInfo*       getOrCreateTreeInfo(Tree* t);
        std::string     getName(void) { return cacheName; }
        void            injectTreesAndLikelihoods(TreeCache* tc);
        const std::vector<double>& neighborProposalProbabilities(TreeInfo* info, double power);
        void            cacheNeighborProposalProbabilities(double power);
        size_t          size(void) { return treeCache.size(); }
        void            sortNeighborsByLikelihood(void);
    
    private:
        TreeCacheMap    treeCache;
        std::string     cacheName;
};

#endif
