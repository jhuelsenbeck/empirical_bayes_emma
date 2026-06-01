#include <iomanip>
#include <iostream>
#include "Tree.hpp"
#include "TreeLikelihoods.hpp"



TreeLikelihoods::TreeLikelihoods(TreeCache* tc) : treeCache(tc) {

}

void TreeLikelihoods::addLnLikelihood(Tree* t, double lnL) {

    TreeInfo* info = treeCache->getOrCreateTreeInfo(t);
    info->hasLnLikelihood = true;
    info->lnLikelihood = lnL;
}

double TreeLikelihoods::lnLikelihood(Tree* t) {

    TreeInfo* info = treeCache->getOrCreateTreeInfo(t);
    return info->lnLikelihood;
}

void TreeLikelihoods::print(void) {

    int i = 0;
    TreeCacheMap& tCache = treeCache->getCache();
    for (auto& [key,val] : tCache)
        {
        std::cout << std::setw(6) << ++i << " " << std::setw(20) << key << " -- " << std::fixed << std::setprecision(3) << val->lnLikelihood << std::endl;
        }
}
