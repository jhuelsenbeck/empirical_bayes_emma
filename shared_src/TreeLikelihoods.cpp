#include <iomanip>
#include <iostream>
#include "Tree.hpp"
#include "TreeLikelihoods.hpp"



TreeLikelihoods::TreeLikelihoods(TreeCache* tc) : treeCache(tc) {

}

void TreeLikelihoods::addLnLikelihood(Tree* t, double lnL) {

    TreeInfo* info = getOrCreateTreeInfo(treeCache, t);
    info->hasLnLikelihood = true;
    info->lnLikelihood = lnL;
}

double TreeLikelihoods::lnLikelihood(Tree* t) {

    TreeInfo* info = getOrCreateTreeInfo(treeCache, t);
    return info->lnLikelihood;
}

void TreeLikelihoods::print(void) {

    int i = 0;
    for (auto& [key,val] : *treeCache)
        {
        std::cout << std::setw(6) << ++i << " " << std::setw(20) << key << " -- " << std::fixed << std::setprecision(3) << val->lnLikelihood << std::endl;
        }
}
