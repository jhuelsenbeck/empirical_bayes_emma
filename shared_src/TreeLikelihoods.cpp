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

