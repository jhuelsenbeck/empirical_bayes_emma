#ifndef TreeNeighborGenerator_hpp
#define TreeNeighborGenerator_hpp

#include <vector>
#include "TreeCache.hpp"
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

class TreeNeighborGeneratorRandomTBR : public TreeNeighborGenerator {

    public:
                        TreeNeighborGeneratorRandomTBR(void) = delete;
                        TreeNeighborGeneratorRandomTBR(TreeCache* tc);
        void            generateNeighbors(Tree* tree, std::vector<TreeInfo*>& neighbors);
};

#endif
