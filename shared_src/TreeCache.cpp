#include <algorithm>
#include <cmath>
#include "CompactTree.hpp"
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"


TreeInfo* TreeCache::getTreeInfo(uint64_t treeHash) {

    TreeCacheMap::iterator it = treeCache.find(treeHash);
    if (it != treeCache.end())
        return it->second;
    return nullptr;
}

TreeInfo* TreeCache::getOrCreateTreeInfo(Tree* t) {

    uint64_t h = t->getHash();

    TreeCacheMap::iterator it = treeCache.find(h);
    if (it != treeCache.end())
        return it->second;

    TreeInfo* info = new TreeInfo;
    info->compactTree = t->getCompactRepresentation();
    info->hash = t->getHash();

    treeCache[h] = info;

    return info;
}

size_t TreeCache::cacheSize(void) {

    size_t total = sizeof(TreeCache);

    // hash table bucket array
    total += treeCache.bucket_count() * sizeof(void*);

    // per-node overhead: key + value pointer + next pointer
    const size_t nodeOverhead = sizeof(uint64_t) + sizeof(TreeInfo*) + sizeof(void*);
    total += treeCache.size() * nodeOverhead;

    // each TreeInfo and its owned heap data
    for (TreeCacheMap::const_iterator it = treeCache.begin(); it != treeCache.end(); ++it)
        {
        TreeInfo* info = it->second;
        if (info == nullptr)
            continue;
        total += sizeof(TreeInfo);
        if (info->compactTree != nullptr)
            total += info->compactTree->sizeInBytes();
        total += info->neighbors.capacity() * sizeof(TreeInfo*);
        total += info->neighborProposalProbabilities.capacity() * sizeof(double);
        // info->tree intentionally not counted (size unknown, may be shared/transient)
        }

    return total;
}

void TreeCache::freeTreeCache(void) {

    for (auto& pair : treeCache) 
        {
        delete pair.second->compactTree;
        if (pair.second->tree != nullptr)
            delete pair.second->tree;
        delete pair.second;
        }
    treeCache.clear();
}

void TreeCache::injectTreesAndLikelihoods(TreeCache* tc) {

    TreeCacheMap& otherMap = tc->getCache();
    for (auto& [key,val] : otherMap)
        {
        TreeCacheMap::iterator it = treeCache.find(key);
        if (it == treeCache.end())
            {
            TreeInfo* info = new TreeInfo;
            
            info->hash = val->hash;
            info->lnLikelihood = val->lnLikelihood;
            info->hasLnLikelihood = val->hasLnLikelihood;
            info->posteriorProbability = val->posteriorProbability;
            CompactTree* newCompactTree = new CompactTree(*val->compactTree);
            info->compactTree = newCompactTree;
            if (val->tree != nullptr)
                info->tree = new Tree(newCompactTree, val->tree->getNumTips());
            treeCache.insert( std::make_pair(key,info) );
            }
        else 
            {
            it->second->lnLikelihood = val->lnLikelihood;
            it->second->hasLnLikelihood = val->hasLnLikelihood;
            it->second->posteriorProbability = val->posteriorProbability;
            }
        }
}

const std::vector<double>& TreeCache::neighborProposalProbabilities(TreeInfo* info, double power) {

    if (info == nullptr)
        Msg::error("Cannot cache proposal probabilities for a null TreeInfo");

    if (info->neighbors.size() == 0)
        Msg::error("Cannot cache proposal probabilities before neighbors are generated");

    if (info->hasNeighborProposalProbabilities == true &&
        info->neighborProposalPower == power &&
        info->neighborProposalProbabilities.size() == info->neighbors.size())
        return info->neighborProposalProbabilities;

    double maxLnL = info->neighbors[0]->lnLikelihood;
    for (size_t i=1; i<info->neighbors.size(); i++)
        {
        TreeInfo* neighbor = info->neighbors[i];
        if (neighbor == nullptr)
            Msg::error("Null neighbor in TreeInfo");
        if (neighbor->hasLnLikelihood == false)
            Msg::error("Cannot cache proposal probabilities before likelihoods are calculated");
        if (neighbor->lnLikelihood > maxLnL)
            maxLnL = neighbor->lnLikelihood;
        }

    std::vector<double>& probs = info->neighborProposalProbabilities;
    probs.resize(info->neighbors.size());

    double sum = 0.0;
    for (size_t i=0; i<info->neighbors.size(); i++)
        {
        double x = std::exp((info->neighbors[i]->lnLikelihood - maxLnL) * power);
        probs[i] = x;
        sum += x;
        }

    if (sum <= 0.0 || std::isfinite(sum) == false)
        Msg::error("Bad normalization constant for proposal probabilities");

    double factor = 1.0 / sum;
    for (size_t i=0; i<probs.size(); i++)
        probs[i] *= factor;

    info->neighborProposalPower = power;
    info->hasNeighborProposalProbabilities = true;

    return probs;
}

void TreeCache::cacheNeighborProposalProbabilities(double power) {

    for (auto& [hash, info] : treeCache)
        {
        if (info == nullptr)
            continue;
        if (info->neighbors.size() == 0)
            continue;
        neighborProposalProbabilities(info, power);
        }
}

void TreeCache::sortNeighborsByLikelihood(void) {

    for (auto& [hash, info] : treeCache)
        {
        if (info == nullptr)
            continue;

        std::vector<TreeInfo*>& neighbors = info->neighbors;

        std::sort(neighbors.begin(),
                  neighbors.end(),
                  [](TreeInfo* a, TreeInfo* b)
                  {
                  if (a == nullptr && b == nullptr)
                      return false;
                  if (a == nullptr)
                      return false;
                  if (b == nullptr)
                      return true;

                  // Trees with known likelihoods first
                  if (a->hasLnLikelihood == true &&
                      b->hasLnLikelihood == false)
                      return true;

                  if (a->hasLnLikelihood == false &&
                      b->hasLnLikelihood == true)
                      return false;

                  // Neither has a likelihood
                  if (a->hasLnLikelihood == false &&
                      b->hasLnLikelihood == false)
                      return a->hash < b->hash;

                  // Highest likelihood first
                  if (a->lnLikelihood != b->lnLikelihood)
                      return a->lnLikelihood > b->lnLikelihood;

                  // Deterministic tie breaker
                  return a->hash < b->hash;
                  });

        info->hasNeighborProposalProbabilities = false;
        info->neighborProposalPower = std::numeric_limits<double>::quiet_NaN();
        }
}
