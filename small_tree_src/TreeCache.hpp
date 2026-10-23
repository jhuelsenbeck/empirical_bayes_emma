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
class TreeCache;



// Everything known about one topology that does not depend on the move set. The records are shared:
// the NNI, NNI2, and TBR caches all hold the same TreeInfo objects, and each keeps only its own
// neighborhoods (see TreeCache). At ten taxa there are two million of these, and keeping three copies
// of each -- with a Tree object apiece -- was several gigabytes for no gain.
struct TreeInfo {

    uint64_t                hash = 0;
    Tree*                   tree = nullptr;
    CompactTree*            compactTree = nullptr;
    uint32_t                index = 0;                // position in the shared table; keys the per-move arrays

    double                  lnLikelihood = std::numeric_limits<double>::quiet_NaN();
    double                  lnMarginalLikelihood = std::numeric_limits<double>::quiet_NaN();
    double                  posteriorProbability = 0.0;

    bool                    hasLnLikelihood = false;
    bool                    hasLnMarginalLikelihood = false;

                            // The index of the state that holds this tree in the transition kernel, so
                            // that an edge of the kernel can be placed without a lookup: the kernel of a
                            // TBR analysis for ten taxa has some 590 million edges, and a hash lookup for
                            // each one is a hash lookup too many. Only one kernel exists at a time.
    int64_t                 stateIndex = -1;
};


typedef std::unordered_map<uint64_t, TreeInfo*> TreeCacheMap;


// The neighborhood of one tree under one move set: a view of that tree's row in the move's neighbor
// array. Entries are kept exactly as the generator produced them, duplicates included -- a TBR
// neighbor reachable by two different bisection-reconnection pairs appears twice and is proposed
// twice as often -- in the order fixed by sortNeighborsByLikelihood.
class NeighborList {

    public:
        class iterator {

            public:
                                    iterator(const NeighborList* l, uint32_t k) : list(l), pos(k) { }
                TreeInfo*           operator*(void) const { return (*list)[pos]; }
                iterator&           operator++(void) { ++pos; return *this; }
                bool                operator!=(const iterator& o) const { return pos != o.pos; }

            private:
                const NeighborList* list;
                uint32_t            pos;
        };

                                NeighborList(void) = default;
                                NeighborList(TreeInfo* const* tbl, const uint32_t* e, uint32_t n) : table(tbl), entries(e), count(n) { }
        uint32_t                size(void) const { return count; }
        bool                    empty(void) const { return count == 0; }
        TreeInfo*               operator[](uint32_t k) const { return table[entries[k]]; }
        uint32_t                indexAt(uint32_t k) const { return entries[k]; }
        TreeInfo*               back(void) const { return table[entries[count - 1]]; }
        iterator                begin(void) const { return iterator(this, 0); }
        iterator                end(void) const { return iterator(this, count); }

    private:
        TreeInfo* const*        table = nullptr;
        const uint32_t*         entries = nullptr;
        uint32_t                count = 0;
};


// A move set's view of the tree space. The first cache constructed owns the table of TreeInfo
// records (and the hash map that finds them); a cache constructed with sharing(owner) uses that same
// table. Each cache keeps its own neighborhoods in one flat array of 32-bit tree indices, addressed
// per tree by an offset and a count, and the two scalars that define each tree's proposal
// distribution at the current power. The proposal probabilities themselves are not stored -- at ten
// taxa that was 2.9 kilobytes per tree for TBR -- but recomputed on demand from those scalars, by the
// same expression that used to fill the stored vector, so they are bit-for-bit the values it held.
class TreeCache {

    public:
                            TreeCache(void) = delete;
                            TreeCache(std::string nm);
                            TreeCache(std::string nm, TreeCache* sharing);
                           ~TreeCache(void);
                            TreeCache(const TreeCache&) = delete;
        TreeCache&          operator=(const TreeCache&) = delete;

                            // the shared table
        TreeCacheMap&       getCache(void) { return table->treeMap; }
        TreeInfo*           getTreeInfo(uint64_t treeHash);
        TreeInfo*           getOrCreateTreeInfo(Tree* t);
        TreeInfo*           treeAt(uint32_t idx) const { return table->byIndex[idx]; }
        size_t              size(void) const { return table->byIndex.size(); }
        size_t              cacheSize(void);
        double              calculatePosteriorProbabilities(void);
        void                freeTreeObjects(void);
        std::string         getName(void) { return cacheName; }

                            // this move set's neighborhoods
        NeighborList        neighbors(const TreeInfo* info) const;
        bool                hasNeighbors(const TreeInfo* info) const;
        void                setNeighbors(const TreeInfo* info, const std::vector<TreeInfo*>& nbrs);
        void                setNeighbors(const TreeInfo* info, const uint32_t* nbrIndices, uint32_t count);
        void                reserveNeighborEntries(size_t n) { entries.reserve(n); }
        void                finishNeighbors(void);
        void                sortNeighborsByLikelihood(void);
        void                freeNeighbors(void);
        size_t              numNeighborEntries(void) const { return entries.size(); }

                            // this move set's proposal distribution, q(j|i) proportional to L_j^power
        void                cacheNeighborProposalProbabilities(double power);
        bool                hasNeighborProposalProbabilities(void) const { return proposalReady; }
        double              neighborProposalPower(void) const { return proposalPower; }
        double              neighborMaxLnL(const TreeInfo* info) const { return maxLnL[info->index]; }
        double              neighborProposalNormInv(const TreeInfo* info) const { return normInv[info->index]; }
        double              neighborProposalProbability(const TreeInfo* from, const TreeInfo* to) const;

                            // releases this move's neighborhoods; the owner also releases the shared
                            // table, which must outlive every cache sharing it
        void                freeTreeCache(void);

    private:
        void                ensureRowStorage(void);
        TreeCache*          table;                     // the owner of the shared records (this, for the owner)
        TreeCacheMap        treeMap;                   // owner only
        std::vector<TreeInfo*> byIndex;                // owner only

        std::vector<uint64_t> rowOffset;               // per tree: start of its row in entries
        std::vector<uint32_t> rowCount;                // per tree: length of its row (0 = not generated)
        std::vector<uint32_t> entries;                 // all rows, concatenated, as tree indices
        bool                neighborsSorted = false;

        std::vector<double> maxLnL;                    // per tree, for proposalPower
        std::vector<double> normInv;
        double              proposalPower = std::numeric_limits<double>::quiet_NaN();
        bool                proposalReady = false;

        std::string         cacheName;
};

#endif
