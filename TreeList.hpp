#ifndef TreeList_hpp
#define TreeList_hpp

#include "BitSet.hpp"
#include <unordered_map>
class Tree;

struct TreeInfo {

    TreeInfo(Tree* t, double x, bool tf) : tree(t), lnL(x), likelihoodCalculated(tf) {}
    Tree*   tree;
    double  lnL;
    bool    likelihoodCalculated;
};

typedef std::unordered_map<uint64_t,TreeInfo> TreeMap;
typedef TreeMap::iterator TreeMapIter;



class TreeList {

    public:
                   ~TreeList(void);
        double      lnLikelihood(Tree* t);
        void        addTree(Tree* t);
        void        addTree(Tree* t, double x);
        Tree*       getTree(uint64_t treeHash);
        TreeInfo&   getTreeInfo(uint64_t treeHash);
        void        print(void);
    
    private:
        TreeMap     map;
};

#endif 
