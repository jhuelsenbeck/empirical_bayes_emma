#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include "CompactTree.hpp"
#include "Msg.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"
#include "UserSettings.hpp"



TreeCache::TreeCache(std::string nm) : table(this), cacheName(nm) {

}

TreeCache::TreeCache(std::string nm, TreeCache* sharing) : table(sharing->table), cacheName(nm) {

}

TreeCache::~TreeCache(void) {

    freeTreeCache();
}

TreeInfo* TreeCache::getTreeInfo(uint64_t treeHash) {

    TreeCacheMap& m = table->treeMap;
    TreeCacheMap::iterator it = m.find(treeHash);
    if (it != m.end())
        return it->second;
    return nullptr;
}

TreeInfo* TreeCache::getOrCreateTreeInfo(Tree* t) {

    uint64_t h = t->getHash();

    TreeCacheMap& m = table->treeMap;
    TreeCacheMap::iterator it = m.find(h);
    if (it != m.end())
        return it->second;

    TreeInfo* info = new TreeInfo;
    info->compactTree = t->getCompactRepresentation();
    info->hash = h;
    info->index = static_cast<uint32_t>(table->byIndex.size());

    m[h] = info;
    table->byIndex.push_back(info);

    return info;
}

size_t TreeCache::cacheSize(void) {

    size_t total = sizeof(TreeCache);

    // this move's neighborhoods and proposal scalars
    total += rowOffset.capacity() * sizeof(uint64_t);
    total += rowCount.capacity() * sizeof(uint32_t);
    total += entries.capacity() * sizeof(uint32_t);
    total += (maxLnL.capacity() + normInv.capacity()) * sizeof(double);

    // the shared table (counted by every cache that uses it)
    const TreeCacheMap& m = table->treeMap;
    total += m.bucket_count() * sizeof(void*);
    const size_t nodeOverhead = sizeof(uint64_t) + sizeof(TreeInfo*) + sizeof(void*);
    total += m.size() * nodeOverhead;
    total += table->byIndex.capacity() * sizeof(TreeInfo*);
    for (const TreeInfo* info : table->byIndex)
        {
        if (info == nullptr)
            continue;
        total += sizeof(TreeInfo);
        if (info->compactTree != nullptr)
            total += info->compactTree->sizeInBytes();
        // info->tree intentionally not counted (size unknown, may be shared/transient)
        }

    return total;
}

double TreeCache::calculatePosteriorProbabilities(void) {

    TreeCacheMap& treeCache = table->treeMap;

    // calculate the exact posterior probability
    double bestLnL = std::numeric_limits<double>::lowest();
    double worstLnL = 0.0;
    for (auto& [key,val] : treeCache)
        {
        if (val->lnLikelihood > bestLnL)
            bestLnL = val->lnLikelihood;
        if (val->lnLikelihood < worstLnL)
            worstLnL = val->lnLikelihood;
        }

    std::cout << "   Calculating posterior probabilities" << std::endl;
    std::cout << "   * Maximum Likelihood = " << bestLnL << std::endl;
    std::cout << "   * Minimum Likelihood = " << worstLnL << std::endl;
    std::cout << "   * Difference         = " << bestLnL - worstLnL << std::endl << std::endl;

    double sum = 0.0;
    for (auto& [key,val] : treeCache)
        {
        double x = std::exp(val->lnLikelihood - bestLnL);
        val->posteriorProbability = x;
        sum += x;
        }
    double kl = 0.0;
    double factor = 1.0 / sum;
    double priorProb = 1.0 / treeCache.size();
    for (auto& [key,val] : treeCache)
        {
        val->posteriorProbability *= factor;
        kl += val->posteriorProbability * (log(val->posteriorProbability) - log(priorProb));
        }

    if (treeCache.begin()->second->hasLnMarginalLikelihood == true)
        {
        size_t nBad = 0, nTot = 0;
        double lo =  std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (auto& [key,val] : treeCache)
            {
            ++nTot;
            double m = val->lnMarginalLikelihood;
            if (!std::isfinite(m)) { ++nBad; continue; }
            lo = std::min(lo, m);
            hi = std::max(hi, m);
            }
        if (nBad > 0)
            std::cout << nBad << " / " << nTot << " non-finite; finite range [" << lo << ", " << hi << "]\n";

        double bestLnL = std::numeric_limits<double>::lowest();
        for (auto& [key,val] : treeCache)
            {
            if (val->lnMarginalLikelihood > bestLnL)
                bestLnL = val->lnMarginalLikelihood;
            }
        double sum = 0.0;
        for (auto& [key,val] : treeCache)
            {
            double x = std::exp(val->lnMarginalLikelihood - bestLnL);
            val->lnMarginalLikelihood = x;
            sum += x;
            }
        double factor = 1.0 / sum;
        for (auto& [key,val] : treeCache)
            val->lnMarginalLikelihood *= factor;

        // print file comparing probabilities
        std::string ppCompFileName = UserSettings::userSettings().getOutputFileName() + ".pp_comp.tsv";
        std::ofstream compOut(ppCompFileName);
        if (!compOut)
            throw std::runtime_error("Could not open basin table file: " + ppCompFileName);
        compOut << "hash,x,y" << '\n';
        for (auto& [key,val] : treeCache)
            {
            compOut << key << ",";
            compOut << val->posteriorProbability << ",";
            compOut << val->lnMarginalLikelihood << '\n';
            }
        compOut.close();
        }

    return kl;
}

void TreeCache::freeTreeObjects(void) {

    // Release the heavyweight Tree objects; keep compact encodings, likelihoods and posteriors. Safe
    // once the MapTree partitions (Tree::getPartitions) have been read: neighbor generation rebuilds a
    // tree from its compact encoding when the Tree object is gone. The records are shared, so this
    // releases them for every move set.
    for (TreeInfo* info : table->byIndex)
        {
        if (info == nullptr)
            continue;
        if (info->tree != nullptr)
            {
            delete info->tree;
            info->tree = nullptr;
            }
        }
}

void TreeCache::ensureRowStorage(void) {

    size_t n = table->byIndex.size();
    if (rowOffset.size() < n)
        {
        rowOffset.resize(n, 0);
        rowCount.resize(n, 0);
        }
}

NeighborList TreeCache::neighbors(const TreeInfo* info) const {

    uint32_t idx = info->index;
    if (idx >= rowCount.size() || rowCount[idx] == 0)
        return NeighborList();
    return NeighborList(table->byIndex.data(), entries.data() + rowOffset[idx], rowCount[idx]);
}

bool TreeCache::hasNeighbors(const TreeInfo* info) const {

    uint32_t idx = info->index;
    return idx < rowCount.size() && rowCount[idx] > 0;
}

void TreeCache::setNeighbors(const TreeInfo* info, const std::vector<TreeInfo*>& nbrs) {

    ensureRowStorage();
    uint32_t idx = info->index;
    if (rowCount[idx] != 0)
        Msg::error("Neighbors of a tree were generated twice");
    if (nbrs.size() > std::numeric_limits<uint32_t>::max())
        Msg::error("Neighborhood too large for its row");

    rowOffset[idx] = entries.size();
    rowCount[idx]  = static_cast<uint32_t>(nbrs.size());
    for (TreeInfo* nb : nbrs)
        entries.push_back(nb->index);
    neighborsSorted = false;
    proposalReady = false;
}

void TreeCache::setNeighbors(const TreeInfo* info, const uint32_t* nbrIndices, uint32_t count) {

    ensureRowStorage();
    uint32_t idx = info->index;
    if (rowCount[idx] != 0)
        Msg::error("Neighbors of a tree were generated twice");

    rowOffset[idx] = entries.size();
    rowCount[idx]  = count;
    entries.insert(entries.end(), nbrIndices, nbrIndices + count);
    neighborsSorted = false;
    proposalReady = false;
}

void TreeCache::finishNeighbors(void) {

    // The rows are appended as they are generated, so the array grows by doubling; release the slack
    // once every row is in. At ten taxa that slack is up to a gigabyte for TBR.
    entries.shrink_to_fit();
}

void TreeCache::freeNeighbors(void) {

    std::vector<uint64_t>().swap(rowOffset);              // release capacity, not just size
    std::vector<uint32_t>().swap(rowCount);
    std::vector<uint32_t>().swap(entries);
    std::vector<double>().swap(maxLnL);
    std::vector<double>().swap(normInv);
    neighborsSorted = false;
    proposalReady = false;
    proposalPower = std::numeric_limits<double>::quiet_NaN();
}

void TreeCache::freeTreeCache(void) {

    freeNeighbors();
    if (table != this)
        return;                                            // the shared records belong to the owner

    for (TreeInfo* info : byIndex)
        {
        if (info == nullptr)
            continue;
        delete info->compactTree;
        if (info->tree != nullptr)
            delete info->tree;
        delete info;
        }
    std::vector<TreeInfo*>().swap(byIndex);
    treeMap.clear();
}

double TreeCache::neighborProposalProbability(const TreeInfo* from, const TreeInfo* to) const {

    // exactly the expression that filled the old stored vector: exp((lnL_j - max) * power) / Z
    uint32_t i = from->index;
    return std::exp((to->lnLikelihood - maxLnL[i]) * proposalPower) * normInv[i];
}

void TreeCache::cacheNeighborProposalProbabilities(double power) {

    if (proposalReady == true && proposalPower == power)
        return;

    size_t n = table->byIndex.size();
    maxLnL.assign(n, std::numeric_limits<double>::quiet_NaN());
    normInv.assign(n, std::numeric_limits<double>::quiet_NaN());

    for (TreeInfo* info : table->byIndex)
        {
        if (info == nullptr)
            continue;
        NeighborList nbrs = neighbors(info);
        if (nbrs.empty() == true)
            continue;

        double mx = nbrs[0]->lnLikelihood;
        for (uint32_t k=1; k<nbrs.size(); k++)
            {
            TreeInfo* neighbor = nbrs[k];
            if (neighbor->hasLnLikelihood == false)
                Msg::error("Cannot cache proposal probabilities before likelihoods are calculated");
            if (neighbor->lnLikelihood > mx)
                mx = neighbor->lnLikelihood;
            }

        double sum = 0.0;
        for (uint32_t k=0; k<nbrs.size(); k++)
            sum += std::exp((nbrs[k]->lnLikelihood - mx) * power);
        if (sum <= 0.0 || std::isfinite(sum) == false)
            Msg::error("Bad normalization constant for proposal probabilities");

        // The two scalars that define this tree's proposal distribution. The probability of any one
        // neighbor follows from them in constant time, which the transition kernel needs for the
        // REVERSE proposal q(i|j) -- a property of j's neighborhood rather than of i's.
        maxLnL[info->index]  = mx;
        normInv[info->index] = 1.0 / sum;
        }

    proposalPower = power;
    proposalReady = true;
}

void TreeCache::sortNeighborsByLikelihood(void) {

    // The order depends only on the likelihoods, which do not change between powers, so the rows are
    // sorted once. Earlier versions re-sorted every row at every power.
    if (neighborsSorted == true)
        return;

    TreeInfo* const* tbl = table->byIndex.data();
    for (size_t idx=0; idx<rowCount.size(); idx++)
        {
        uint32_t* row = entries.data() + rowOffset[idx];
        std::sort(row, row + rowCount[idx],
                  [tbl](uint32_t ia, uint32_t ib)
                  {
                  const TreeInfo* a = tbl[ia];
                  const TreeInfo* b = tbl[ib];

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
        }

    neighborsSorted = true;
    proposalReady = false;
}
