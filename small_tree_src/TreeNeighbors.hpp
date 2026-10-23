#ifndef TreeNeighbors_hpp
#define TreeNeighbors_hpp

#include <vector>
#include "TreeCache.hpp"
class Tree;
class TreeNeighborGenerator;


class TreeNeighbors {

    public:
                                TreeNeighbors(void) = delete;
                                TreeNeighbors(TreeCache* tc, TreeNeighborGenerator* ng, int nt);
        NeighborList            neighbors(Tree* t);
        NeighborList            neighbors(uint64_t treeHash);
        NeighborList            neighbors(TreeInfo* info);
        void                    print(void);

    private:
        TreeCache*              treeCache;
        TreeNeighborGenerator*  neighborGenerator;
        int                     numTaxa;
        std::vector<TreeInfo*>  scratch;               // one neighborhood, reused across trees
};

#endif
