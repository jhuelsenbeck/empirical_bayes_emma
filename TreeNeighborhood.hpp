#ifndef TreeNeighborhood_hpp
#define TreeNeighborhood_hpp

#include <cstdint>
#include <unordered_map>
#include <vector>
class Tree;
class TreeList;

typedef std::vector<uint64_t> TreeHashVec;
typedef std::unordered_map<uint64_t,TreeHashVec> TreeNeighbors;


class TreeNeighborhood {

    public:
                                TreeNeighborhood(void) = delete;
                                TreeNeighborhood(TreeList* tl);
        virtual                ~TreeNeighborhood(void);
        virtual TreeHashVec&    getNeighbors(uint64_t treeHash) = 0;
        virtual TreeHashVec&    getNeighbors(Tree* tree) = 0;
        virtual size_t          size(void) = 0;

    protected:
        TreeList*               treeList;
};

class TreeNeighborhoodNni : public TreeNeighborhood {

    public:
                                TreeNeighborhoodNni(void) = delete;
                                TreeNeighborhoodNni(TreeList* tl);
        TreeHashVec&            getNeighbors(uint64_t treeHash);
        TreeHashVec&            getNeighbors(Tree* tree);
        size_t                  size(void) { return nniNeighborhood.size(); }
    
    protected:
        void                    generateNniNeighbors(uint64_t treeHash, std::vector<uint64_t>& neighbors);
        TreeNeighbors           nniNeighborhood;
};

class TreeNeighborhoodNni2 : public TreeNeighborhoodNni {

    public:
                                TreeNeighborhoodNni2(void) = delete;
                                TreeNeighborhoodNni2(TreeList* tl);
        TreeHashVec&            getNeighbors(uint64_t treeHash);
        TreeHashVec&            getNeighbors(Tree* tree);
        size_t                  size(void) { return nniNniNeighborhood.size(); }
    
    private:
        void                    generateNniNniNeighbors(uint64_t treeHash, std::vector<uint64_t>& neighbors);
        TreeNeighbors           nniNniNeighborhood;
};

#endif
