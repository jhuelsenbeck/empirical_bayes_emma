#ifndef TreeLikelihoods_hpp
#define TreeLikelihoods_hpp

#include <cstdlib>
#include "TreeCache.hpp"
class Tree;



class TreeLikelihoods {

    public:
                        TreeLikelihoods(void) = delete;
                        TreeLikelihoods(TreeCache* tc);
        void            addLnLikelihood(Tree* t, double lnL);
        double          lnLikelihood(Tree* t);
        void            print(void);
    
    private:
        TreeCache*      treeCache;
};

#endif
