#include <iomanip>
#include <iostream>
#include "Alignment.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
#include "Msg.hpp"
#include "RandomVariable.hpp"
#include "UserSettings.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"
#include "TreeNeighborhood.hpp"
#include "TreeSamples.hpp"
#include "TreeSpace.hpp"



Mcmc::Mcmc(RandomVariable* r, ThreadPool* tp, Alignment* a, TreeList* tl, TreeSpace* ts) : 
    rng(r), threadPool(tp), alignment(a), treeList(tl), treeSpace(ts) {

    UserSettings& settings = UserSettings::userSettings();
    chainLength = settings.getChainLength();
    printFrequency = settings.getPrintFrequency();
    sampleFrequency = settings.getSampleFrequency();
    burn = chainLength * settings.getBurnin();
    
    samples = new TreeSamples(treeList);
}

Mcmc::~Mcmc(void) {

    for (size_t i=0; i<allocatedCalculators.size(); i++)
        delete allocatedCalculators[i];
    delete samples;
}

double Mcmc::calculateMaximumLikelihood(NeighborValues& vals) {

    if (activeCalculators.size() > 0)
        Msg::error("Outstanding calculators");
    
    // TBR neighborhoods can produce O(n^3) reconnection candidates, easily
    // overflowing the thread pool's fixed-size queue. Process in batches
    // bounded by the queue capacity; wait() drains each batch before the
    // next push, so the queue is guaranteed empty at the top of each pass.
    const size_t batchSize = threadPool->getQueueCapacity();
    
    size_t i = 0;
    while (i < vals.size())
        {
        // gather up to batchSize calculators, skipping vals entries whose
        // tree pointer is null (already-known topologies needing no eval)
        while (i < vals.size() && activeCalculators.size() < batchSize)
            {
            Tree* t = std::get<1>(vals[i]);
            if (t != nullptr)
                {
                LikelihoodCalculator* calculator = getCalculator();
                calculator->setTree(t);
                calculator->setOffset(i);
                activeCalculators.push_back(calculator);
                }
            i++;
            }
        
        // dispatch this batch and wait for it to finish
        for (LikelihoodCalculator* calculator : activeCalculators)
            threadPool->pushTask(calculator);
        threadPool->wait();
        
        // collect results, free trees, return calculators to the pool
        for (LikelihoodCalculator* calculator : activeCalculators)
            {
            Tree* t = calculator->getTree();
            double lnL = calculator->getResult();
            size_t idx = calculator->getOffset();
            
            treeList->addTree(t, lnL);
            std::get<1>(vals[idx]) = nullptr;
            std::get<2>(vals[idx]) = lnL;
            delete t;
            
            returnCalculator(calculator);
            }
        activeCalculators.clear();
        }
    
    return 0.0;
}

std::pair<int,int> Mcmc::chooseChains(int numChains) {

    int idx1 = (int)(rng->uniformRv()*numChains);
    int idx2 = idx1;
    while (idx1 == idx2)
        idx2 = (int)(rng->uniformRv()*numChains);
    return std::make_pair(idx1,idx2);
}

double Mcmc::chooseTree(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree) {

    double u = rng->uniformRv();
    double sum = 0.0;
    for (auto x : neighborhoodInfo)
        {
        sum += x.second;
        if (u < sum)
            {
            tree = x.first;
            return x.second;
            }
        }
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

double Mcmc::findTreeProbability(std::vector<std::pair<uint64_t, double>>& neighborhoodInfo, uint64_t& tree) {

    for (auto x : neighborhoodInfo)
        {
        if (x.first == tree)
            return x.second;
        }
    return 0.0;
}

LikelihoodCalculator* Mcmc::getCalculator(void) {
        
    if (calculatorPool.empty() == true)
        {
        // if the pool is empty, we allocate a new block of nodes
        LikelihoodCalculator* newCalculator = new LikelihoodCalculator(alignment);
        allocatedCalculators.push_back(newCalculator);
        calculatorPool.push_back(newCalculator);
        }
    
    // return a calculator from the pool, remembering to remove it from the pool
    LikelihoodCalculator* c = calculatorPool.back();
    calculatorPool.pop_back();
    return c;
}

double Mcmc::heat(int i, double temperature) {

    return 1.0 / (1.0 + i*temperature);
}

void Mcmc::normalize(double power, NeighborValues& neighbors) {

    double maxLnL = std::get<2>(neighbors[0]);
    for (size_t i=1; i<neighbors.size(); i++)
        {
        if (std::get<2>(neighbors[i]) > maxLnL)
            maxLnL = std::get<2>(neighbors[i]);
        }
    double sum = 0.0;
    for (size_t i=0; i<neighbors.size(); i++)
        {
        double x = std::exp((std::get<2>(neighbors[i]) - maxLnL) * power);
        sum += x;
        std::get<2>(neighbors[i]) = x;
        }
    double factor = 1.0 / sum;
    for (size_t i=0; i<neighbors.size(); i++)
        std::get<2>(neighbors[i]) *= factor;
}

void Mcmc::printToScreen(int n, double curLnL, double newLnL, size_t treeListSize) {

    std::cout << "   * " << std::setw(6) << n << " -- ";
    std::cout << std::fixed << std::setprecision(2); 
    std::cout << curLnL << " -> " << newLnL << " " << std::setw(8) << newLnL-curLnL << " ";
    std::cout << "(" << treeListSize << ") ";
    std::cout << std::endl;
}

void Mcmc::returnCalculator(LikelihoodCalculator* calculator) {

    calculatorPool.push_back(calculator);
}

void Mcmc::run(TreeNeighborhood* neighborhood, double power) {

    std::cout << "   Markov chain Monte Carlo:" << std::endl;

    Tree* initialTree = new Tree(rng, alignment->getTaxonNames());
    uint64_t currentTree = initialTree->getHash();

    TreeInfo& treeInfo = treeList->getTreeInfo(currentTree);
    double curLnL;
    if (treeInfo.isLikelihoodCalculated() == true)
        {
        curLnL = treeInfo.lnL;
        }
    else    
        {
        LikelihoodCalculator* calculator = getCalculator();
        calculator->setTree(treeInfo.getTree());
        curLnL = calculator->lnLikelihood();
        treeInfo.setLikelihoodCalculated(true);
        treeInfo.lnL = curLnL;
        returnCalculator(calculator);
        }
    std::cout << "   * Initial likelihood = " << curLnL << std::endl;

    NeighborValues forwardNeighbors;
    NeighborValues reverseNeighborhood;
    
    int numAccepted = 0;
    samples->clear();
    
    for (int n=1; n<=chainLength; n++)
        {
        neighborhood->getNeighbors(currentTree, forwardNeighbors);
        calculateMaximumLikelihood(forwardNeighbors);
        normalize(power, forwardNeighbors);
        uint64_t newTree;
        double forwardProbability = chooseTree(forwardNeighborhood, newTree);
        double newLnL = treeList->getTreeInfo(newTree).lnL;
        
        double reverseProbability = forwardProbability;
        std::vector<uint64_t>& reverseNeighbors = neighborhood->getNeighbors(newTree);
        reverseNeighborhood.clear();
        calculateMaximumLikelihoods(*treeList, newTree, reverseNeighbors, reverseNeighborhood);
        normalize(power, reverseNeighborhood);
        reverseProbability = findTreeProbability(reverseNeighborhood, currentTree);
        
        double lnLikelihoodRatio = newLnL - curLnL;
        double lnPriorRatio = 0.0;
        double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
        double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
        
        bool accept = false;
        if (log(rng->uniformRv()) < lnR)
            accept = true;

        if (n % printFrequency == 0)
            printToScreen(n, curLnL, newLnL, treeList->size());
            
        if (accept == true)
            {
            currentTree = newTree;
            curLnL = newLnL;
            numAccepted++;
            }

        if (n % sampleFrequency == 0 && n >= burn)
            samples->sampleTree(currentTree);
        }
        
    std::cout << "   Acceptance rate: " << ((double)numAccepted / chainLength) * 100.0 << "%" << std::endl << std::endl;
}

void Mcmc::run(TreeNeighborhood* neighborhood, double power, int numChains, double temperature) {

    std::cout << "   Markov chain Monte Carlo:" << std::endl;

    std::vector<int> chainIndex(numChains);
    std::vector<Tree*> initialTree(numChains);
    std::vector<uint64_t> currentTree(numChains);
    for (size_t i=0; i<numChains; i++)
        {
        chainIndex[i] = (int)i;
        initialTree[i] = new Tree(rng, alignment->getTaxonNames());
        currentTree[i] = initialTree[i]->getHash();
        }
        
    LikelihoodCalculator* calculator = getCalculator();
    std::vector<double> curLnL(numChains);
    for (size_t i=0; i<numChains; i++)
        {
        calculator->setTree(initialTree[i]);
        curLnL[i] = calculator->lnLikelihood();
        treeList->addTree(initialTree[i], curLnL[i]);
        }
    returnCalculator(calculator);
    
    std::vector<std::pair<uint64_t, double>> forwardNeighborhood;
    std::vector<std::pair<uint64_t, double>> reverseNeighborhood;
    int numAccepted = 0;
    samples->clear();
    
    for (int n=1; n<=chainLength; n++)
        {
        for (size_t chain=0; chain<numChains; chain++)
            {
            std::vector<uint64_t>& forwardNeighbors = neighborhood->getNeighbors(currentTree[chain]);
            forwardNeighborhood.clear();
            calculateMaximumLikelihoods(*treeList, currentTree[chain], forwardNeighbors, forwardNeighborhood);
            normalize(power, forwardNeighborhood);
            uint64_t newTree;
            double forwardProbability = chooseTree(forwardNeighborhood, newTree);
            double newLnL = treeList->getTreeInfo(newTree).lnL;
            
            double reverseProbability = forwardProbability;
            std::vector<uint64_t>& reverseNeighbors = neighborhood->getNeighbors(newTree);
            reverseNeighborhood.clear();
            calculateMaximumLikelihoods(*treeList, newTree, reverseNeighbors, reverseNeighborhood);
            normalize(power, reverseNeighborhood);
            reverseProbability = findTreeProbability(reverseNeighborhood, currentTree[chain]);
            
            double lnLikelihoodRatio = (newLnL - curLnL[chain]) * heat(chainIndex[chain], temperature);
            double lnPriorRatio = 0.0;
            double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
            double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
            
            bool accept = false;
            if (log(rng->uniformRv()) < lnR)
                accept = true;
            if (chainIndex[chain] == 0 && n % printFrequency == 0)
                printToScreen(n, curLnL[chain], newLnL, treeList->size());
                
            if (accept == true)
                {
                currentTree[chain] = newTree;
                curLnL[chain] = newLnL;
                numAccepted++;
                }
            }
            
        // choose two chains at random
        std::pair<int,int> idx = chooseChains(numChains);
        int idx0 = chainIndex[idx.first];
        int idx1 = chainIndex[idx.second];
        double lnR = (curLnL[idx.first] * heat(idx1, temperature) + curLnL[idx.second] * heat(idx0, temperature));
        lnR -=  (curLnL[idx.first] * heat(idx0, temperature) + curLnL[idx.second] * heat(idx1, temperature));
        if (log(rng->uniformRv()) < lnR)
            {
            chainIndex[idx.first] = idx1;
            chainIndex[idx.second] = idx0;
            }

        if (n % sampleFrequency == 0 && n >= burn)
            {
            int coldChainIdx = 0;
            for (int i=0; i<numChains; i++)
                {
                if (chainIndex[i] == 0)
                    {
                    coldChainIdx = i;
                    break;
                    }
                }
            samples->sampleTree(currentTree[coldChainIdx]);
            }
        }
        
    std::cout << "   Acceptance rate: " << ((double)numAccepted / chainLength) * 100.0 << "%" << std::endl << std::endl;
}


