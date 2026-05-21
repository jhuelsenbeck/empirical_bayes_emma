#ifndef ExhaustiveSearch_hpp
#define ExhaustiveSearch_hpp

#include <functional>
#include <map>
#include <string>
#include <vector>
class Alignment;
class Node;
class ThreadPool;
class Tree;
class TreeList;
class TreeSpace;



class ExhaustiveSearch {

    public:
        using TreeCallback = std::function<void(Node* root, int treeNum)>;
    
                                    ExhaustiveSearch(void) = delete;
                                    ExhaustiveSearch(Alignment* aln, TreeList* tl, ThreadPool* tp);
                                   ~ExhaustiveSearch(void);
        void                        search(TreeCallback callback);
        std::vector<Tree*>          searchAndCollect(void);
        int                         getNumTrees(void) const { return numTrees; }
        static int                  numUnrootedTrees(int n);
    
    private:
        void                        buildStartingTree(void);
        void                        collectBranches(Node* p, std::vector<Node*>& branches);
        void                        enumerate(int nextTaxonIdx, TreeCallback& callback);
        void                        enumerateAllTrees(void);
        void                        returnScratchNodes(void);
        Alignment*                  alignment;
        TreeList*                   treeList;
        ThreadPool*                 threadPool;
        std::vector<std::string>    taxonNames;
        double                      bestLnL;
        int                         numTaxa;
        int                         numTrees;
        Node*                       scratchRoot;
        std::vector<Node*>          scratchNodes;
        bool                        treesEnumerated;
};

#endif
