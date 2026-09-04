#ifndef TreeCache_hpp
#define TreeCache_hpp

#include <cstdint>
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
    double                  lnMarginalLikelihood = std::numeric_limits<double>::quiet_NaN();
    double                  posteriorProbability = 0.0;

    bool                    hasNeighbors = false;
    bool                    hasLnLikelihood = false;
    bool                    hasLnMarginalLikelihood = false;
    bool                    hasNeighborProposalProbabilities = false;
    double                  neighborProposalPower = std::numeric_limits<double>::quiet_NaN();

                            // Scalars cached for the benefit of MarkovChainAnalyzer. The index of the
                            // state that holds this tree in the transition kernel is stored here so
                            // that an edge of the kernel can be placed without a lookup: the kernel of
                            // a TBR analysis for ten taxa has some 590 million edges, and a hash lookup
                            // for each one is a hash lookup too many.
    int64_t                 stateIndex = -1;

                            // The largest log likelihood among this tree's neighbors, and the
                            // reciprocal of the normalizing constant of its proposal distribution,
                            // both for the currently cached power. Together they give the probability
                            // that this tree proposes any one of its neighbors, in constant time and
                            // without a second pass over the neighborhood.
    double                  neighborMaxLnL = std::numeric_limits<double>::quiet_NaN();
    double                  neighborProposalNormInv = std::numeric_limits<double>::quiet_NaN();
};


typedef std::unordered_map<uint64_t, TreeInfo*> TreeCacheMap;

class TreeCache {

    public:
                        TreeCache(void) = delete;
                        TreeCache(std::string nm);
        void            cacheNeighborProposalProbabilities(double power);
        size_t          cacheSize(void);
        void            calculatePosteriorProbabilities(void);
        void            freeTreeCache(void);
        TreeCacheMap&   getCache(void) { return treeCache; }
        TreeInfo*       getTreeInfo(uint64_t treeHash);
        TreeInfo*       getOrCreateTreeInfo(Tree* t);
        std::string     getName(void) { return cacheName; }
        void            injectTreesAndLikelihoods(TreeCache* tc);
        const std::vector<double>& neighborProposalProbabilities(TreeInfo* info, double power);
        size_t          size(void) { return treeCache.size(); }
        void            sortNeighborsByLikelihood(void);
    
    private:
        TreeCacheMap    treeCache;
        std::string     cacheName;
};

#endif
