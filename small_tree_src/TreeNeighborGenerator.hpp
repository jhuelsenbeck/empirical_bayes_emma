#ifndef TreeNeighborGenerator_hpp
#define TreeNeighborGenerator_hpp

#include <cstdint>
#include <vector>
#include "TreeCache.hpp"
class ThreadPool;
class Tree;
struct TreeInfo;



class TreeNeighborGenerator {

    public:
                        TreeNeighborGenerator(void) = delete;
                        TreeNeighborGenerator(TreeCache* tc);
        bool            checkTree(Tree* t);
        virtual void    generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors) = 0;
    
    protected:
        TreeCache*      treeCache;
};

class TreeNeighborGeneratorNNI : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorNNI(void) = delete;
                        TreeNeighborGeneratorNNI(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
};

class TreeNeighborGeneratorNNI2 : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorNNI2(void) = delete;
                        TreeNeighborGeneratorNNI2(TreeCache* tc, TreeCache* nniTc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
        void            generateNeighbors(uint64_t treeHash, std::vector<TreeInfo*>& neighbors);
        
    private:
        TreeCache*      nniTreeCache;
};

class TreeNeighborGeneratorTBR : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorTBR(void) = delete;
                        TreeNeighborGeneratorTBR(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
};

// TBR neighborhoods computed from split masks, without building trees.
//
// A tree is identified by its hash, which Tree::setDescriptor forms from the tree's clusters (rooted on
// tip 0, the set of taxa below each interior node other than the root's child), sorted and packed as
// bitsets. Everything a TBR move does can be said in those terms. Cutting the edge whose cluster is Y
// leaves two subtrees, on Y and on X = (all taxa) \ Y; reconnecting edge a of the X side to edge b of
// the Y side gives a tree whose splits are X | Y itself; for every other edge of each side, the part of
// that side lying away from the attachment point; and the two halves of each subdivided edge a and b.
// Each of these is a mask operation on the original clusters, so a neighbor's hash costs a few dozen
// integer operations, a sort of n - 3 small integers and one short hash -- against building, rerooting,
// checking and hashing a whole Tree object as TreeNeighborGeneratorTBR does.
//
// The (cut, a, b) triples enumerated are exactly those of TreeNeighborGeneratorTBR, so the neighbor
// lists are the same multisets, duplicates included: a tree reachable by two triples is listed twice.
// The identity reconnection, which recreates the starting tree, is skipped, as there. Each tree's
// neighborhood depends only on that tree, and the shared table is only read, so the trees are spread
// over the thread pool.
class TbrSplitNeighbors {

    public:
                        TbrSplitNeighbors(void) = delete;
                        TbrSplitNeighbors(TreeCache* tc, int nt);
        void            generate(const TreeInfo* info, std::vector<uint32_t>& out) const;
        void            generateAll(ThreadPool* pool);
        static uint64_t hashOfClusters(uint32_t* clusters, int numClusters, int numTaxa);

    private:
        TreeCache*      treeCache;
        int             numTaxa;
};

class TreeNeighborGeneratorRandomTBR : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorRandomTBR(void) = delete;
                        TreeNeighborGeneratorRandomTBR(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
};

#endif
