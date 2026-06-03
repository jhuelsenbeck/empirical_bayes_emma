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
#include "TreeLikelihoods.hpp"
#include "TreeNeighbors.hpp"
#include "TreePartitions.hpp"
#include "TreeSamples.hpp"
#include "UserSettings.hpp"



Mcmc::Mcmc(RandomVariable* r, ThreadPool* p, TreeCache* tc, TreeLikelihoods* tl, TreeNeighbors* tn, Alignment* a, bool tf, std::string cfn) :
    rng(r), threadPool(p), alignment(a), treeCache(tc), treeLikelihoods(tl), treeNeighbors(tn), expandedOutput(tf), convergenceLogFileName(cfn) {

    UserSettings& settings = UserSettings::userSettings();
    numChains       = settings.getNumChains();
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

std::pair<int,int> Mcmc::chooseChains(int numChains) {

    int idx1 = (int)(rng->uniformRv()*numChains);
    int idx2 = idx1;
    while (idx1 == idx2)
        idx2 = (int)(rng->uniformRv()*numChains);
    return std::make_pair(idx1,idx2);
}

TreeInfo* Mcmc::chooseInitialTreeInfo(void) {

    TreeCacheMap& cache = treeCache->getCache();
    if (cache.empty() == true)
        Msg::error("Cannot initialize MCMC from an empty TreeCache");

    size_t which = (size_t)(rng->uniformRv() * cache.size());
    if (which >= cache.size())
        which = cache.size() - 1;

    TreeCacheMap::iterator it = cache.begin();
    std::advance(it, which);
    if (it->second == nullptr)
        Msg::error("Null TreeInfo in TreeCache");
    return it->second;
}

TreeInfo* Mcmc::chooseTreeInfo(TreeInfo* currentInfo, double& proposalProbability) {

    if (currentInfo == nullptr)
        Msg::error("Current tree is null in Mcmc::chooseTreeInfo");

    std::vector<TreeInfo*>& neighbors = currentInfo->neighbors;
    std::vector<double>& probs = currentInfo->neighborProposalProbabilities;

    if (neighbors.empty() == true)
        Msg::error("Current tree has no neighbors");
    if (neighbors.size() != probs.size())
        Msg::error("Neighbor vector and proposal probability vector differ in size");

    double u = rng->uniformRv();
    double sum = 0.0;
    for (size_t i=0; i<probs.size(); i++)
        {
        sum += probs[i];
        if (u < sum)
            {
            proposalProbability = probs[i];
            return neighbors[i];
            }
        }

    // Numerical roundoff can leave sum infinitesimally below one. Fall back to
    // the last neighbor instead of returning null.
    proposalProbability = probs.back();
    return neighbors.back();
}

double Mcmc::findTreeProbability(TreeInfo* fromInfo, uint64_t toHash) {

    if (fromInfo == nullptr)
        Msg::error("Null source tree in Mcmc::findTreeProbability");

    std::vector<TreeInfo*>& neighbors = fromInfo->neighbors;
    std::vector<double>& probs = fromInfo->neighborProposalProbabilities;

    if (neighbors.size() != probs.size())
        Msg::error("Neighbor vector and proposal probability vector differ in size");

    for (size_t i=0; i<neighbors.size(); i++)
        {
        if (neighbors[i]->hash == toHash)
            return probs[i];
        }

    Msg::error("Could not find reverse tree in neighbor list");
    return 0.0;
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
    for (TreePartitions* p : partitions)
        delete p;
    samples.clear();
    partitions.clear();
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
    convergenceLog << "cycle\t";
    TreeConvergenceDiagnostics::writeStatsHeader(convergenceLog);
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

void Mcmc::run(std::string label, double power) {

    std::cout << "   " << label << std::endl;

    // Cache proposal probabilities once for this power. This assumes that all
    // neighborhoods and likelihoods have already been generated.
    treeCache->cacheNeighborProposalProbabilities(power);

    deleteSamplesAndPartitions();
    samples.push_back(new TreeSamples(treeCache));
    samples[0]->reserve(numCycles);

    TreeInfo* currentInfo = chooseInitialTreeInfo();
    double curLnL = currentInfo->lnLikelihood;

    for (int n=1; n<=numCycles; n++)
        {
        double forwardProbability = 0.0;
        TreeInfo* newInfo = chooseTreeInfo(currentInfo, forwardProbability);
        if (newInfo == nullptr)
            Msg::error("New tree is null");

        double reverseProbability = findTreeProbability(newInfo, currentInfo->hash);

        double newLnL = newInfo->lnLikelihood;
        double lnLikelihoodRatio = newLnL - curLnL;
        double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
        double lnR = lnLikelihoodRatio + lnProposalRatio;

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

void Mcmc::run(std::string label, double power, int numRuns) {

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

    for (int n=1, nextLogPoint=1; n<=numCycles; n++)
        {
        for (int run=0; run<numRuns; run++)
            {
            double forwardProbability = 0.0;
            TreeInfo* newInfo = chooseTreeInfo(currentInfo[run], forwardProbability);
            if (newInfo == nullptr)
                Msg::error("New tree is null");

            double reverseProbability = findTreeProbability(newInfo, currentInfo[run]->hash);

            double newLnL = newInfo->lnLikelihood;
            double lnLikelihoodRatio = newLnL - curLnL[run];
            double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
            double lnR = lnLikelihoodRatio + lnProposalRatio;

            if (std::log(rng->uniformRv()) < lnR)
                {
                currentInfo[run] = newInfo;
                curLnL[run] = newLnL;
                }

            samples[run]->sampleTree(currentInfo[run]->hash);
            }

        if (n == nextLogPoint)
            {
            writeConvergenceLine(n);
            nextLogPoint = (nextLogPoint < 100000) ? nextLogPoint * 10 : nextLogPoint + 100000;
            }

        printToScreen(n, curLnL);
        }
    std::cout << std::endl;

    TreeSamples::combinedPrint(samples);
    TreeSamples::compareSamples(samples);

    std::cout << std::endl;
}

void Mcmc::run(std::string label, double power, int numRuns, int numHeatedChains) {

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
    for (int run=0; run<numRuns; run++)
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

    openConvergenceLog((size_t)numRuns);

    for (int n=1, nextLogPoint=1; n<=numCycles; n++)
        {
        for (int run=0; run<numRuns; run++)
            {
            for (int chain=0; chain<numHeatedChains; chain++)
                {
                double forwardProbability = 0.0;
                TreeInfo* newInfo = chooseTreeInfo(currentInfo[run][chain], forwardProbability);
                if (newInfo == nullptr)
                    Msg::error("New tree is null");

                double reverseProbability = findTreeProbability(newInfo, currentInfo[run][chain]->hash);

                double beta = heat(chainIndex[run][chain], temperature);
                double newLnL = newInfo->lnLikelihood;
                double lnLikelihoodRatio = newLnL - curLnL[run][chain];
                double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
                double lnR = beta * lnLikelihoodRatio + lnProposalRatio;

                if (std::log(rng->uniformRv()) < lnR)
                    {
                    currentInfo[run][chain] = newInfo;
                    curLnL[run][chain] = newLnL;
                    }
                }

            // swap temperatures assigned to two physical chains
            std::pair<int,int> idx = chooseChains(numHeatedChains);
            int idx0 = chainIndex[run][idx.first];
            int idx1 = chainIndex[run][idx.second];
            double lnR = (curLnL[run][idx.first]  * heat(idx1, temperature) +
                          curLnL[run][idx.second] * heat(idx0, temperature));
            lnR -=     (curLnL[run][idx.first]  * heat(idx0, temperature) +
                         curLnL[run][idx.second] * heat(idx1, temperature));
            if (std::log(rng->uniformRv()) < lnR)
                {
                chainIndex[run][idx.first] = idx1;
                chainIndex[run][idx.second] = idx0;
                }

            int coldIdx = coldChainIndex(chainIndex[run]);
            samples[run]->sampleTree(currentInfo[run][coldIdx]->hash);
            }

        if (n == nextLogPoint)
            {
            writeConvergenceLine(n);
            nextLogPoint = (nextLogPoint < 100000) ? nextLogPoint * 10 : nextLogPoint + 100000;
            }

        printToScreen(n, curLnL, chainIndex);
        }

    TreeSamples::combinedPrint(samples);
    TreeSamples::compareSamples(samples);

    std::cout << std::endl;
}

void Mcmc::writeConvergenceLine(int cycle) {

    if (convergenceLog.is_open() == false)
        return;

    convergenceLog << cycle << '\t';

    TreeConvergenceDiagnostics diagnostics(treeCache);
    diagnostics.writeStatsLine(convergenceLog, samples);

    convergenceLog << '\n';
    convergenceLog.flush();
}
