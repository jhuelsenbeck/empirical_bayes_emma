#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include "Alignment.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
#include "Msg.hpp"
#include "RandomVariable.hpp"
#include "TreeCache.hpp"
#include "TreeConvergenceDiagnostics.hpp"
#include "TreeSamples.hpp"
#include "UserSettings.hpp"



// Sample k distinct indices uniformly, without replacement, from [0, degree), optionally
// excluding one index, using Floyd's algorithm. Chosen indices are appended to `out` (the
// caller clears it, or pre-seeds it with a forced index as in the reverse reference set).
// k must not exceed the effective pool size; the caller guarantees this by capping the number
// of tries at the degree.
static void sampleDistinctIndices(RandomVariable* rng, int k, int degree, int exclude, std::vector<int>& out) {

    int pool = degree;
    if (exclude >= 0 && exclude < degree)
        pool = degree - 1;                        // `exclude` is not eligible to be drawn
    if (k > pool)
        k = pool;

    for (int j = pool - k; j < pool; j++)
        {
        int t = (int)(rng->uniformRv() * (double)(j + 1));   // uniform in [0, j]
        if (t > j)                                            // guard against uniformRv() == 1
            t = j;
        // map pool position -> real neighbor index, skipping `exclude`
        int cand = (exclude >= 0 && t >= exclude) ? t + 1 : t;
        bool present = false;
        for (int q : out)
            if (q == cand)
                {
                present = true;
                break;
                }
        if (present)
            out.push_back((exclude >= 0 && j >= exclude) ? j + 1 : j);
        else
            out.push_back(cand);
        }
}



Mcmc::Mcmc(RandomVariable* r, TreeCache* tc, Alignment* a, bool tf, std::string cfn, ThreadPool* tp) :
    rng(r), threadPool(tp), alignment(a), treeCache(tc), expandedOutput(tf), convergenceLogFileName(cfn) {

    UserSettings& settings = UserSettings::userSettings();
    temperature     = settings.getTemperature();
    numCycles       = settings.getChainLength();
    printFrequency  = settings.getPrintFrequency();
    sampleFrequency = settings.getSampleFrequency();
}

Mcmc::~Mcmc(void) {

    deleteSamplesAndPartitions();
    for (size_t i=0; i<allocatedCalculators.size(); i++)
        delete allocatedCalculators[i];
}

std::pair<int,int> Mcmc::chooseChains(RandomVariable* r, int numChains) {

    int idx1 = (int)(r->uniformRv()*numChains);
    int idx2 = idx1;
    while (idx1 == idx2)
        idx2 = (int)(r->uniformRv()*numChains);
    return std::make_pair(idx1,idx2);
}

TreeInfo* Mcmc::chooseInitialTreeInfo(void) {

    TreeCacheMap& cache = treeCache->getCache();
    if (cache.empty() == true)
        Msg::error("Cannot initialize MCMC from an empty TreeCache");

    // A draw from the exact posterior starts the chain at stationarity, so that its replicates measure
    // the chain's efficiency alone, without the transient from a start far from the posterior's mass.
    if (startFromPosterior == true)
        {
        if (startTrees.empty() == true)
            {
            double cumulative = 0.0;
            for (auto& [hash, info] : cache)
                {
                cumulative += info->posteriorProbability;
                startTrees.push_back(info);
                startCdf.push_back(cumulative);
                }
            }
        double u = rng->uniformRv() * startCdf.back();
        size_t k = (size_t)(std::upper_bound(startCdf.begin(), startCdf.end(), u) - startCdf.begin());
        if (k >= startTrees.size())
            k = startTrees.size() - 1;
        return startTrees[k];
        }

    size_t which = (size_t)(rng->uniformRv() * cache.size());
    if (which >= cache.size())
        which = cache.size() - 1;

    TreeCacheMap::iterator it = cache.begin();
    std::advance(it, which);
    if (it->second == nullptr)
        Msg::error("Null TreeInfo in TreeCache");
    return it->second;
}

// Draw a tree from the exact posterior. cdf holds the running sums of probs, formed in the same order
// and by the same additions as the linear scan this replaces, so the first entry whose cumulative
// probability exceeds u -- found by binary search rather than by walking the list -- is the tree that
// scan returned.
uint64_t Mcmc::chooseGibbs(RandomVariable* r, std::vector<std::pair<uint64_t,double>>& probs, const std::vector<double>& cdf) {

    double u = r->uniformRv();
    std::vector<double>::const_iterator it = std::upper_bound(cdf.begin(), cdf.end(), u);
    if (it == cdf.end())
        Msg::error("Failed to choose a Gibbs tree");
    return probs[static_cast<size_t>(it - cdf.begin())].first;
}

TreeInfo* Mcmc::chooseTreeInfo(RandomVariable* r, TreeInfo* currentInfo, double& proposalProbability) {

    if (currentInfo == nullptr)
        Msg::error("Current tree is null in Mcmc::chooseTreeInfo");

    NeighborList neighbors = treeCache->neighbors(currentInfo);

    if (neighbors.empty() == true)
        Msg::error("Current tree has no neighbors");
    if (treeCache->hasNeighborProposalProbabilities() == false)
        Msg::error("Proposal probabilities have not been cached");

    // The proposal probabilities are formed as the scan reaches them. The neighbors are sorted by
    // likelihood, so for any positive power the scan usually stops within the first few.
    double u = r->uniformRv();
    double sum = 0.0;
    for (uint32_t i=0; i<neighbors.size(); i++)
        {
        double q = treeCache->neighborProposalProbability(currentInfo, neighbors[i]);
        sum += q;
        if (u < sum)
            {
            proposalProbability = q;
            return neighbors[i];
            }
        }

    // Numerical roundoff can leave sum infinitesimally below one. Fall back to
    // the last neighbor instead of returning null.
    proposalProbability = treeCache->neighborProposalProbability(currentInfo, neighbors.back());
    return neighbors.back();
}

TreeInfo* Mcmc::chooseTreeInfo(RandomVariable* r, std::vector<int>& subset, TreeInfo* currentInfo, double& proposalProbability, int n, double power) {

    if (currentInfo == nullptr)
        Msg::error("Current tree is null in Mcmc::chooseTreeInfo");
        
    if (n == 0)
        return chooseTreeInfo(r, currentInfo, proposalProbability);

    NeighborList neighbors = treeCache->neighbors(currentInfo);

    if (neighbors.empty() == true)
        Msg::error("Current tree has no neighbors");
    // Cap the number of distinct tries at the degree. Note: without-replacement MTM is exactly
    // reversible only when this cap is uniform across states (n <= min degree, or regular degree);
    // NNI is degree-regular and TBR neighborhoods here dwarf n, so the cap does not bite.
    if (n > (int)neighbors.size())
        n = (int)neighbors.size();

    // Multiple-try weights use the reverse single-step kernel T(y,x) = 1/deg(y),
    // so each trial y carries weight L_y^power / deg(y). The current tree's log
    // likelihood is a common offset that cancels against the identical offset
    // applied in findTreeProbability; it only keeps the exponentials near unity.
    double refLnL = currentInfo->lnLikelihood;

    // draw n distinct trials uniformly, without replacement, from the neighborhood
    subset.clear();
    sampleDistinctIndices(r, n, (int)neighbors.size(), -1, subset);

    std::vector<double> weights(subset.size());
    double totalWeight = 0.0;
    for (size_t i=0; i<subset.size(); i++)
        {
        TreeInfo* nbr = neighbors[subset[i]];
        uint32_t nbrDegree = treeCache->neighbors(nbr).size();
        if (nbrDegree == 0)
            Msg::error("Neighbor has no neighbors; cannot form multiple-try weight");
        double w = std::exp((nbr->lnLikelihood - refLnL) * power) / (double)nbrDegree;
        weights[i] = w;
        totalWeight += w;
        }

    double u = r->uniformRv() * totalWeight;
    double sum = 0.0;
    size_t pick = subset.size() - 1;
    for (size_t i=0; i<subset.size(); i++)
        {
        sum += weights[i];
        if (u < sum)
            {
            pick = i;
            break;
            }
        }

    // return the forward multiple-try weight sum (not a single selection prob)
    proposalProbability = totalWeight;
    return neighbors[subset[pick]];
}

double Mcmc::findTreeProbability(TreeInfo* fromInfo, const TreeInfo* toInfo) {

    if (fromInfo == nullptr)
        Msg::error("Null source tree in Mcmc::findTreeProbability");

    // The destination is found by its index in the shared table, read straight from the row, rather
    // than by comparing the hash of every neighbor: for TBR that was some 360 scattered reads of
    // TreeInfo records per step. The first match is the same either way.
    NeighborList neighbors = treeCache->neighbors(fromInfo);
    const uint32_t toIndex = toInfo->index;

    for (uint32_t i=0; i<neighbors.size(); i++)
        {
        if (neighbors.indexAt(i) == toIndex)
            return treeCache->neighborProposalProbability(fromInfo, neighbors[i]);
        }

    std::cout << "neighbors.size()=" << neighbors.size() << std::endl;
    Msg::error("Could not find reverse tree in neighbor list");
    return 0.0;
}

double Mcmc::findTreeProbability(RandomVariable* r, std::vector<int>& subset, TreeInfo* fromInfo, const TreeInfo* toInfo, int n, double power) {

    if (fromInfo == nullptr)
        Msg::error("Null source tree in Mcmc::findTreeProbability");
        
    if (n == 0)
        return findTreeProbability(fromInfo, toInfo);

    NeighborList neighbors = treeCache->neighbors(fromInfo);

    if (n > (int)neighbors.size())
        n = (int)neighbors.size();

    // locate the destination x among Y's neighbors; it is forced in as x_k* = x
    int toIdx = -1;
    const uint32_t toIndex = toInfo->index;
    for (uint32_t i=0; i<neighbors.size(); i++)
        {
        if (neighbors.indexAt(i) == toIndex)
            {
            toIdx = (int)i;
            break;
            }
        }
    if (toIdx == -1)
        Msg::error("Could not find reverse tree in neighbor list");

    // the same offset used in chooseTreeInfo: neighbors[toIdx] is x, so this is curLnL
    double refLnL = neighbors[toIdx]->lnLikelihood;

    // reference set: the forced destination x, plus n-1 distinct draws from N(Y)\{x},
    // sampled uniformly without replacement so the m reference trees are all distinct
    subset.clear();
    subset.push_back(toIdx);
    sampleDistinctIndices(r, n - 1, (int)neighbors.size(), toIdx, subset);

    double totalWeight = 0.0;
    for (const int& idx : subset)
        {
        TreeInfo* nbr = neighbors[idx];
        uint32_t nbrDegree = treeCache->neighbors(nbr).size();
        if (nbrDegree == 0)
            Msg::error("Neighbor has no neighbors; cannot form multiple-try weight");
        totalWeight += std::exp((nbr->lnLikelihood - refLnL) * power) / (double)nbrDegree;
        }

    // return the reverse multiple-try weight sum (not a single selection prob)
    return totalWeight;
}

int Mcmc::coldChainIndex(std::vector<int>& chainIndices) {

    int coldChainIdx = 0;
    for (int i=0; i<chainIndices.size(); i++)
        {
        if (chainIndices[i] == 0)
            {
            coldChainIdx = i;
            break;
            }
        }
    return coldChainIdx;
}

void Mcmc::deleteSamplesAndPartitions(void) {

    for (TreeSamples* s : samples)
        delete s;
    samples.clear();
}

LikelihoodCalculator* Mcmc::getCalculator(void) {

    if (calculatorPool.empty() == true)
        {
        LikelihoodCalculator* newCalculator = new LikelihoodCalculator(alignment);
        allocatedCalculators.push_back(newCalculator);
        calculatorPool.push_back(newCalculator);
        }

    LikelihoodCalculator* c = calculatorPool.back();
    calculatorPool.pop_back();
    return c;
}

double Mcmc::heat(int i, double temperature) {

    return 1.0 / (1.0 + i*temperature);
}

void Mcmc::openConvergenceLog(size_t numReplicates) {

    (void)numReplicates;

    UserSettings& settings = UserSettings::userSettings();
    std::string path = settings.getOutputFileName();
    if (convergenceLogFileName == "")
        path += ".convergence.tsv";
    else
        path += convergenceLogFileName + ".tsv";

    convergenceLog.open(path, std::ios::out | std::ios::trunc);
    if (convergenceLog.is_open() == false)
        {
        std::cout << "Warning: could not open convergence log at " << path << std::endl;
        return;
        }

    convergenceLog << std::fixed << std::setprecision(6);
    diagnostics = std::make_unique<TreeConvergenceDiagnostics>(treeCache);
    convergenceLog << "cycle\t";
    TreeConvergenceDiagnostics::writeStatsHeader(convergenceLog, numReplicates);
    convergenceLog << '\n';
    convergenceLog.flush();
}

void Mcmc::printToScreen(int n, double curLnL, double newLnL) {

    if (n % printFrequency == 0 || n == 1)
        {
        std::cout << "   * " << std::setw(6) << n << " -- ";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << curLnL << " -> " << newLnL << " " << std::setw(8) << newLnL-curLnL << " ";
        std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)treeCache->cacheSize()/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::printToScreen(int n, std::vector<double>& curLnL) {

    if (expandedOutput == false)
        return;

    if (n % printFrequency == 0 || n == 1)
        {
        std::cout << "   * " << std::setw(6) << n << " -- ";
        std::cout << std::fixed << std::setprecision(2);
        for (size_t i=0; i<curLnL.size(); i++)
            {
            std::cout << curLnL[i] << " ";
            if ((i+1) % 10 == 0 && i + 1 != curLnL.size())
                {
                std::cout << std::endl;
                std::cout << "               ";
                }
            }
        //std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)treeCache->cacheSize()/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::printToScreen(int n, std::vector<std::vector<double>>& curLnL, std::vector<std::vector<int>>& indices) {

    if (expandedOutput == false)
        return;

    if (n % printFrequency == 0 || n == 1)
        {
        std::cout << "   * " << std::setw(6) << n << " -- ";
        std::cout << std::fixed << std::setprecision(2);
        for (int i=0; i<curLnL.size(); i++)
            {
            std::cout << curLnL[i][coldChainIndex(indices[i])] << " ";
             if ((i+1) % 10 == 0 && i + 1 != curLnL.size())
                {
                std::cout << std::endl;
                std::cout << "               ";
                }
           }
        //std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)treeCache->cacheSize()/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::returnCalculator(LikelihoodCalculator* calculator) {

    calculatorPool.push_back(calculator);
}

// One body(run) call per replicate, on the thread pool when there is one. The replicates share only
// read-only state -- the tree records, neighbor rows and proposal normalizers -- and each owns its
// generator, scratch and samples, so the results do not depend on how the calls are scheduled.
namespace {

class ReplicateTask : public ThreadTask {

    public:
                                ReplicateTask(const std::function<void(int)>* b, int i) : body(b), index(i) {}
        void                    run(void) override { (*body)(index); }

    private:
        const std::function<void(int)>* body;
        int                     index;
};

}

void Mcmc::forEachRun(int numRuns, const std::function<void(int)>& body) {

    if (threadPool == nullptr || numRuns < 2)
        {
        for (int run=0; run<numRuns; run++)
            body(run);
        return;
        }

    std::vector<ReplicateTask> tasks;
    tasks.reserve(numRuns);
    for (int run=0; run<numRuns; run++)
        tasks.emplace_back(&body, run);
    const size_t batch = 1024;                               // well inside the pool's queue
    for (size_t first=0; first<tasks.size(); first+=batch)
        {
        for (size_t t=first; t<tasks.size() && t<first+batch; t++)
            threadPool->pushTask(&tasks[t]);
        threadPool->wait();
        }
}

// Each replicate draws from a generator of its own, seeded from the master generator in replicate
// order, so a run is reproducible from the program's seed whatever the number of threads.
std::vector<RandomVariable> Mcmc::makeRunGenerators(int numRuns) {

    std::vector<RandomVariable> gens;
    gens.reserve(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        uint32_t seed = 0;
        while (seed == 0)                                    // zero would mean "seed from the device"
            seed = (uint32_t)(rng->uniformRv() * 4294967296.0);
        gens.emplace_back(seed);
        }
    return gens;
}

// The last cycle of the segment that starts at `first`: the replicates advance independently up to
// the next cycle at which the convergence log is written or the screen is updated, where they meet.
uint32_t Mcmc::segmentEnd(uint32_t first) {

    uint32_t n = first;
    while (n < (uint32_t)numCycles)
        {
        if (shouldSample(n) == true || n % printFrequency == 0 || n == 1)
            break;
        n++;
        }
    return n;
}

void Mcmc::run(std::string label, double power, int nNeighbors) {

    std::cout << "   " << label << std::endl;

    // Cache proposal probabilities once for this power. This assumes that all
    // neighborhoods and likelihoods have already been generated.
    treeCache->cacheNeighborProposalProbabilities(power);

    deleteSamplesAndPartitions();
    samples.push_back(new TreeSamples(treeCache));
    samples[0]->reserve(numCycles);

    TreeInfo* currentInfo = chooseInitialTreeInfo();
    double curLnL = currentInfo->lnLikelihood;

    for (uint32_t n=1; n<=numCycles; n++)
        {
        double forwardProbability = 0.0;
        TreeInfo* newInfo = chooseTreeInfo(rng, subsetIndices, currentInfo, forwardProbability, nNeighbors, power);
        if (newInfo == nullptr)
            Msg::error("New tree is null");

        double reverseProbability = findTreeProbability(rng, subsetIndices, newInfo, currentInfo, nNeighbors, power);

        double newLnL = newInfo->lnLikelihood;
        double lnR;
        if (nNeighbors == 0)
            {
            // plain informed Metropolis-Hastings over the full neighborhood
            double lnLikelihoodRatio = newLnL - curLnL;
            double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
            lnR = lnLikelihoodRatio + lnProposalRatio;
            }
        else
            {
            // multiple-try Metropolis: forward/reverse are weight sums, and the
            // target enters through the (power - 1) exponent on the likelihoods
            lnR = (power - 1.0) * (curLnL - newLnL) +
                  std::log(forwardProbability) - std::log(reverseProbability);
            }

        if (std::log(rng->uniformRv()) < lnR)
            {
            currentInfo = newInfo;
            curLnL = newLnL;
            }

        samples[0]->sampleTree(currentInfo->hash);
        printToScreen(n, curLnL, newLnL);
        }

    samples[0]->print();
    std::cout << std::endl;
}

void Mcmc::run(std::string label, int numRuns) {

    if (numRuns < 2)
        Msg::error("Expecting at least two runs for this chain");

    if (expandedOutput == true)
        {
        std::cout << "   " << label << std::endl;
        std::cout << "   * Number of runs = " << numRuns << std::endl;
        }
    else
        std::cout << "   Running " << numRuns << " MCMC chains for " << numCycles << " generations each" << std::endl;
        
    std::vector<std::pair<uint64_t,double>> orderedProbs;
    TreeCacheMap& cache = treeCache->getCache();
    for (auto& [key,val] : cache)
        {
        orderedProbs.emplace_back(key, val->posteriorProbability);
        }
    std::sort(orderedProbs.begin(), orderedProbs.end(),
        [](const auto& a, const auto& b)
        {
            return a.second > b.second;   // descending order
        });
    std::vector<double> gibbsCdf(orderedProbs.size());
    double cumulative = 0.0;
    for (size_t i=0; i<orderedProbs.size(); i++)
        {
        cumulative += orderedProbs[i].second;
        gibbsCdf[i] = cumulative;
        }

    deleteSamplesAndPartitions();
    samples.resize(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        samples[run] = new TreeSamples(treeCache);
        samples[run]->reserve(numCycles);
        }

    openConvergenceLog((size_t)numRuns);

    std::vector<uint64_t> currentTree(numRuns);
    std::vector<double> curLnL(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        currentTree[run] = chooseGibbs(rng, orderedProbs, gibbsCdf);
        TreeInfo* tInfo = treeCache->getTreeInfo(currentTree[run]);
        curLnL[run] = tInfo->lnLikelihood;
        }
    std::vector<RandomVariable> runRng = makeRunGenerators(numRuns);

    for (uint32_t first=1; first<=numCycles; )
        {
        uint32_t last = segmentEnd(first);
        forEachRun(numRuns, [&](int run)
            {
            RandomVariable* r = &runRng[run];
            for (uint32_t n=first; n<=last; n++)
                {
                uint64_t newTree = chooseGibbs(r, orderedProbs, gibbsCdf);
                currentTree[run] = newTree;
                samples[run]->sampleTree(newTree);
                }
            curLnL[run] = treeCache->getTreeInfo(currentTree[run])->lnLikelihood;
            });

        if (shouldSample(last) == true)
            writeConvergenceLine(last);
        printToScreen(last, curLnL);
        first = last + 1;
        }
    std::cout << std::endl;

    TreeSamples::combinedPrint(samples);
    TreeSamples::compareSamples(samples, threadPool);
    writeFunctionalReport();

    std::cout << std::endl;
}

void Mcmc::run(std::string label, double power, int nNeighbors, int numRuns) {

    if (numRuns < 2)
        Msg::error("Expecting at least two runs for this chain");

    if (expandedOutput == true)
        {
        std::cout << "   " << label << std::endl;
        std::cout << "   * Number of runs = " << numRuns << std::endl;
        }
    else
        std::cout << "   Running " << numRuns << " MCMC chains for " << numCycles << " generations each" << std::endl;

    treeCache->cacheNeighborProposalProbabilities(power);

    deleteSamplesAndPartitions();
    samples.resize(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        samples[run] = new TreeSamples(treeCache);
        samples[run]->reserve(numCycles);
        }

    openConvergenceLog((size_t)numRuns);

    std::vector<TreeInfo*> currentInfo(numRuns);
    std::vector<double> curLnL(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        currentInfo[run] = chooseInitialTreeInfo();
        curLnL[run] = currentInfo[run]->lnLikelihood;
        }
    std::vector<RandomVariable> runRng = makeRunGenerators(numRuns);
    std::vector<std::vector<int>> runSubset(numRuns);

    for (uint32_t first=1; first<=numCycles; )
        {
        uint32_t last = segmentEnd(first);
        forEachRun(numRuns, [&](int run)
            {
            // the chain's state lives in locals for the segment, so threads do not share cache lines
            RandomVariable* r = &runRng[run];
            std::vector<int>& subset = runSubset[run];
            TreeInfo* cur = currentInfo[run];
            double lnL = curLnL[run];
            TreeSamples* s = samples[run];
            for (uint32_t n=first; n<=last; n++)
                {
                double forwardProbability = 0.0;
                TreeInfo* newInfo = chooseTreeInfo(r, subset, cur, forwardProbability, nNeighbors, power);
                if (newInfo == nullptr)
                    Msg::error("New tree is null");

                double reverseProbability = findTreeProbability(r, subset, newInfo, cur, nNeighbors, power);

                double newLnL = newInfo->lnLikelihood;
                double lnR;
                if (nNeighbors == 0)
                    {
                    double lnLikelihoodRatio = newLnL - lnL;
                    double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
                    lnR = lnLikelihoodRatio + lnProposalRatio;
                    }
                else
                    {
                    lnR = (power - 1.0) * (lnL - newLnL) + std::log(forwardProbability) - std::log(reverseProbability);
                    }

                if (std::log(r->uniformRv()) < lnR)
                    {
                    cur = newInfo;
                    lnL = newLnL;
                    }

                s->sampleTree(cur->hash);
                }
            currentInfo[run] = cur;
            curLnL[run] = lnL;
            });

        if (shouldSample(last) == true)
            writeConvergenceLine(last);
        printToScreen(last, curLnL);
        first = last + 1;
        }
    std::cout << std::endl;

    TreeSamples::combinedPrint(samples);
    TreeSamples::compareSamples(samples, threadPool);
    writeFunctionalReport();

    std::cout << std::endl;
}

void Mcmc::run(std::string label, double power, int nNeighbors, int numRuns, int numHeatedChains) {

    if (numRuns < 1)
        Msg::error("Expecting at least one run");
    if (numHeatedChains < 2)
        Msg::error("Expecting at least two chains for MCMCMC");

    if (expandedOutput == true)
        {
        std::cout << "   " << label << std::endl;
        std::cout << "   * Number of runs = " << numRuns << std::endl;
        std::cout << "   * Number of chains = " << numHeatedChains << std::endl;
        std::cout << "   * Temperature = " << temperature << std::endl;
        }
    else
        std::cout << "   Running " << numRuns << " MCMCMC analyses (one cold and " << numHeatedChains-1 << " heated chains each) for " << numCycles << " generations" << std::endl;

    treeCache->cacheNeighborProposalProbabilities(power);

    std::vector<std::vector<double>> curLnL(numRuns);
    std::vector<std::vector<TreeInfo*>> currentInfo(numRuns);
    std::vector<std::vector<int>> chainIndex(numRuns);
    for (uint32_t run=0; run<numRuns; run++)
        {
        curLnL[run].resize(numHeatedChains);
        currentInfo[run].resize(numHeatedChains);
        chainIndex[run].resize(numHeatedChains);
        for (int chain=0; chain<numHeatedChains; chain++)
            {
            currentInfo[run][chain] = chooseInitialTreeInfo();
            curLnL[run][chain] = currentInfo[run][chain]->lnLikelihood;
            chainIndex[run][chain] = chain;
            }
        }

    deleteSamplesAndPartitions();
    samples.resize(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        samples[run] = new TreeSamples(treeCache);
        samples[run]->reserve(numCycles);
        }
        
    // swap counts are kept per replicate, so the replicates can run in parallel, and summed at the end
    std::vector<std::vector<int>> numAttemptedSwaps(numHeatedChains);
    std::vector<std::vector<int>> numAcceptedSwaps(numHeatedChains);
    for (size_t i=0; i<numHeatedChains; i++)
        {
        numAttemptedSwaps[i].resize(numHeatedChains,0);
        numAcceptedSwaps[i].resize(numHeatedChains,0);
        }
    std::vector<std::vector<std::vector<int>>> runAttemptedSwaps(numRuns, numAttemptedSwaps);
    std::vector<std::vector<std::vector<int>>> runAcceptedSwaps(numRuns, numAcceptedSwaps);

    openConvergenceLog((size_t)numRuns);

    std::vector<RandomVariable> runRng = makeRunGenerators(numRuns);
    std::vector<std::vector<int>> runSubset(numRuns);

    for (uint32_t first=1; first<=numCycles; )
        {
        uint32_t last = segmentEnd(first);
        forEachRun(numRuns, [&](int run)
            {
            RandomVariable* r = &runRng[run];
            std::vector<int>& subset = runSubset[run];
            std::vector<double>& runLnL = curLnL[run];
            std::vector<TreeInfo*>& runInfo = currentInfo[run];
            std::vector<int>& runIndex = chainIndex[run];
            for (uint32_t n=first; n<=last; n++)
                {
                for (int chain=0; chain<numHeatedChains; chain++)
                    {
                    double forwardProbability = 0.0;
                    TreeInfo* newInfo = chooseTreeInfo(r, subset, runInfo[chain], forwardProbability, nNeighbors, power);
                    if (newInfo == nullptr)
                        Msg::error("New tree is null");

                    double reverseProbability = findTreeProbability(r, subset, newInfo, runInfo[chain], nNeighbors, power);

                    double heatBeta = heat(runIndex[chain], temperature);
                    double newLnL = newInfo->lnLikelihood;
                    double lnR;
                    if (nNeighbors == 0)
                        {
                        double lnLikelihoodRatio = newLnL - runLnL[chain];
                        double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
                        lnR = heatBeta * lnLikelihoodRatio + lnProposalRatio;
                        }
                    else
                        {
                        // chain c targets pi^heatBeta; the proposal still biases by power
                        lnR = (power - heatBeta) * (runLnL[chain] - newLnL) +
                              std::log(forwardProbability) - std::log(reverseProbability);
                        }

                    if (std::log(r->uniformRv()) < lnR)
                        {
                        runInfo[chain] = newInfo;
                        runLnL[chain] = newLnL;
                        }
                    }

                // swap temperatures assigned to two physical chains
                std::pair<int,int> idx = chooseChains(r, numHeatedChains);
                int idx0 = runIndex[idx.first];
                int idx1 = runIndex[idx.second];
                double lnR = (runLnL[idx.first]  * heat(idx1, temperature) +
                              runLnL[idx.second] * heat(idx0, temperature));
                lnR -=      (runLnL[idx.first]  * heat(idx0, temperature) +
                             runLnL[idx.second] * heat(idx1, temperature));
                if (std::log(r->uniformRv()) < lnR)
                    {
                    runIndex[idx.first] = idx1;
                    runIndex[idx.second] = idx0;
                    runAcceptedSwaps[run][idx0][idx1]++;
                    runAcceptedSwaps[run][idx1][idx0]++;
                    }
                runAttemptedSwaps[run][idx0][idx1]++;
                runAttemptedSwaps[run][idx1][idx0]++;

                int coldIdx = coldChainIndex(runIndex);
                samples[run]->sampleTree(runInfo[coldIdx]->hash);
                }
            });

        if (shouldSample(last) == true)
            writeConvergenceLine(last);
        printToScreen(last, curLnL, chainIndex);
        first = last + 1;
        }

    for (int run=0; run<numRuns; run++)
        for (size_t i=0; i<numHeatedChains; i++)
            for (size_t j=0; j<numHeatedChains; j++)
                {
                numAttemptedSwaps[i][j] += runAttemptedSwaps[run][i][j];
                numAcceptedSwaps[i][j] += runAcceptedSwaps[run][i][j];
                }

    TreeSamples::combinedPrint(samples);
    TreeSamples::compareSamples(samples, threadPool);
    writeFunctionalReport();
    
    // show chain swap acceptance information
    std::cout << "        ";
    for (size_t i=0; i<numHeatedChains; i++)
        std::cout << std::setw(4) << i << " ";
    std::cout << std::endl;
    for (size_t i=0; i<numHeatedChains; i++)
        {
        std::cout << "     " << std::setw(2) << i << " ";
        for (size_t j=0; j<numHeatedChains; j++)
            {
            if (i != j)
                std::cout << std::fixed << std::setprecision(2) << (double)numAcceptedSwaps[i][j] / numAttemptedSwaps[i][j] << " ";
            else
                std::cout << std::fixed << std::setprecision(2) << 0.0 << " ";
            }
        std::cout << std::endl;
        }

    std::cout << std::endl;
}

void FunctionalReport::writeHeader(std::ostream& os) {

    os << "sampler\tmoveType\tpower\tfunctional\texactPosterior\tnumReplicates\tsamplesPerReplicate"
          "\tmeanEstimate\tsdAcrossReplicates\trmseFromExact\tpredictedSd\tsdRatio"
          "\tmeanEstimateBurnin\tsdAcrossReplicatesBurnin\trmseFromExactBurnin\tpredictedSdBurnin\tsdRatioBurnin\n";
}

// For each functional, the frequency with which each replicate sampled it, summarized across the
// replicates: their mean, standard deviation and root-mean-square error from the exact posterior
// probability. predictedSd is sqrt(asymptoticVariance / samplesPerReplicate), the standard deviation
// the exact analysis predicts for one replicate, and sdRatio the observed over the predicted. The
// prediction is for a chain at stationarity; each replicate starts from a random tree, and for a
// well-supported functional that start-up transient can dominate the spread. The *Burnin columns repeat
// the summary with the burn-in fraction (-b, default 0.1) of every replicate discarded.
void Mcmc::writeFunctionalReport(void) {

    FunctionalReport* r = functionalReport;
    if (r == nullptr || r->out == nullptr || samples.empty())
        return;
    size_t K = r->labels.size();
    size_t R = samples.size();
    // counts of each functional over the whole replicate, and over its first 10%
    auto addTree = [&](uint64_t hash, double count, std::vector<std::vector<double>>& into, size_t run)
        {
        auto it = r->membership.find(hash);
        if (it == r->membership.end())
            return;
        uint64_t bits = it->second;
        while (bits != 0)
            {
            int k = __builtin_ctzll(bits);
            bits &= bits - 1;
            if ((size_t)k < K)
                into[k][run] += count;
            }
        };
    std::vector<std::vector<double>> total(K, std::vector<double>(R, 0.0)), early(K, std::vector<double>(R, 0.0));
    double N = (double)samples[0]->getNumSamples();
    size_t B = (size_t)(UserSettings::userSettings().getBurnin() * N);
    for (size_t run=0; run<R; run++)
        {
        for (const auto& [hash, count] : samples[run]->getTreeCounts())
            addTree(hash, (double)count, total, run);
        const std::vector<uint64_t>& trace = samples[run]->getTrees();
        for (size_t t=0; t<B && t<trace.size(); t++)
            addTree(trace[t], 1.0, early, run);
        }

    std::ostream& os = *r->out;
    os << std::setprecision(10);
    auto summarize = [&](size_t k, bool burnin)
        {
        double n = burnin ? N - (double)B : N;
        std::vector<double> f(R);
        for (size_t run=0; run<R; run++)
            f[run] = (n > 0.0) ? (total[k][run] - (burnin ? early[k][run] : 0.0)) / n : 0.0;
        double mean = 0.0, sq = 0.0, mse = 0.0;
        for (double x : f)
            mean += x;
        mean /= (double)R;
        for (double x : f)
            {
            sq  += (x - mean) * (x - mean);
            mse += (x - r->exact[k]) * (x - r->exact[k]);
            }
        double sd   = (R > 1) ? std::sqrt(sq / (double)(R - 1)) : std::numeric_limits<double>::quiet_NaN();
        double rmse = std::sqrt(mse / (double)R);
        double av   = (k < r->asymptoticVariance.size()) ? r->asymptoticVariance[k] : std::numeric_limits<double>::quiet_NaN();
        double pred = (av >= 0.0 && n > 0.0) ? std::sqrt(av / n) : std::numeric_limits<double>::quiet_NaN();
        os << '\t' << mean << '\t' << sd << '\t' << rmse << '\t' << pred << '\t'
           << ((pred > 0.0) ? sd / pred : std::numeric_limits<double>::quiet_NaN());
        };
    for (size_t k=0; k<K; k++)
        {
        os << r->sampler << '\t' << r->move << '\t' << r->power << '\t' << r->labels[k] << '\t'
           << r->exact[k] << '\t' << R << '\t' << (long)N;
        summarize(k, false);
        summarize(k, true);
        os << '\n';
        }
    os.flush();
}

bool Mcmc::shouldSample(uint32_t cycle) {

    if (cycle < 10)
        return true;

    uint64_t p10 = 1;
    while (p10 * 10 <= cycle)
        p10 *= 10;

    return (cycle % p10) == 0;
}

void Mcmc::writeConvergenceLine(int cycle) {

    if (convergenceLog.is_open() == false)
        return;

    convergenceLog << cycle << '\t';

    if (diagnostics == nullptr)
        diagnostics = std::make_unique<TreeConvergenceDiagnostics>(treeCache);
    diagnostics->writeStatsLine(convergenceLog, samples);

    convergenceLog << '\n';
    convergenceLog.flush();
}
