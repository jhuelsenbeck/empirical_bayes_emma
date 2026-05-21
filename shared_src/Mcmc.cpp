#include "Alignment.hpp"
#include "CompactTree.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
#include "Msg.hpp"
#include "RandomVariable.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"
#include "TreeLikelihoods.hpp"
#include "TreeNeighbors.hpp"
#include "TreePartitions.hpp"
#include "TreeSamples.hpp"
#include "UserSettings.hpp"



Mcmc::Mcmc(RandomVariable* r, ThreadPool* p, TreeCache* tc, TreeLikelihoods* tl, TreeNeighbors* tn, Alignment* a) : 
    rng(r), threadPool(p), treeCache(tc), treeLikelihoods(tl), treeNeighbors(tn), alignment(a) {

    UserSettings& settings = UserSettings::userSettings();
    numChains       = settings.getNumChains();
    temperature     = settings.getTemperature();
    numCycles       = settings.getChainLength();
    printFrequency  = settings.getPrintFrequency();
    sampleFrequency = settings.getSampleFrequency();
}

Mcmc::~Mcmc(void) {

    for (TreeSamples* s : samples)
        delete s;
    for (TreePartitions* p : partitions)
        delete p;
    for (size_t i=0; i<allocatedCalculators.size(); i++)
        delete allocatedCalculators[i];
}

double Mcmc::calculateMaximumLikelihood(Tree* t) {

    if (activeCalculators.size() > 0)
        Msg::error("Outstanding calculators");
    
    double lnL = treeLikelihoods->lnLikelihood(t);
    if (std::isnan(lnL) == true) 
        {
        LikelihoodCalculator* calculator = getCalculator();
        calculator->setTree(t);
        calculator->setOffset(0);

        threadPool->pushTask(calculator);
        threadPool->wait();
        double lnL = calculator->getResult();
        
        returnCalculator(calculator);
        treeLikelihoods->addLnLikelihood(t, lnL);
        
        return lnL;
        }
    return lnL;
}

double Mcmc::calculateMaximumLikelihood(std::vector<TreeInfo*>& vals) {

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
            TreeInfo* tInfo = vals[i];
            if (tInfo->hasLnLikelihood == false)
                {
                Tree* t = tInfo->tree;
                if (t == nullptr)
                    Msg::error("No tree for likelihood calculation");
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
            
            TreeInfo* tInfo = vals[idx];
            tInfo->lnLikelihood = lnL;
            tInfo->hasLnLikelihood = true;
            tInfo->tree = nullptr;
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

double Mcmc::chooseTree(std::vector<TreeInfo*>& neighbors, std::vector<double>& probs, Tree*& tree, double& lnL) {

    double u = rng->uniformRv();
    double sum = 0.0;
    for (size_t i=0; i<probs.size(); i++)
        {
        sum += probs[i];
        if (u < sum)
            {
            tree = new Tree(neighbors[i]->compactTree, alignment->getNumTaxa());
            lnL = neighbors[i]->lnLikelihood;
            return probs[i];
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

double Mcmc::findTreeProbability(std::vector<TreeInfo*>& neighbors, std::vector<double>& probs, uint64_t tree) {

    for (size_t i=0; i<neighbors.size(); i++)
        {
        if (neighbors[i]->hash == tree)
            return probs[i];
        }
    Msg::error("Could not find tree in list");
    return 0.0;
}

LikelihoodCalculator* Mcmc::getCalculator(void) {
        
    if (calculatorPool.empty() == true)
        {
        // if the pool is empty, we allocate a new calculator
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

void Mcmc::normalize(double power, std::vector<TreeInfo*>& neighbors, std::vector<double>& probs) {

    double maxLnL = neighbors[0]->lnLikelihood;
    for (size_t i=1; i<neighbors.size(); i++)
        {
        if (neighbors[i]->lnLikelihood > maxLnL)
            maxLnL = neighbors[i]->lnLikelihood;
        }
        
    probs.resize(neighbors.size());
    double sum = 0.0;
    for (size_t i=0; i<neighbors.size(); i++)
        {
        double x = std::exp((neighbors[i]->lnLikelihood - maxLnL) * power);
        sum += x;
        probs[i] = x;
        }
        
    double factor = 1.0 / sum;
    for (size_t i=0; i<probs.size(); i++)
        probs[i] *= factor;
}

void Mcmc::openConvergenceLog(void) {

    UserSettings& settings = UserSettings::userSettings();
    std::string path = settings.getOutputFileName() + ".convergence.tsv";
    convergenceLog.open(path, std::ios::out | std::ios::trunc);
    if (convergenceLog.is_open() == false)
        {
        std::cout << "Warning: could not open convergence log at " << path << std::endl;
        return;
        }

    // 6 sig figs is enough for split frequencies and ESS;
    // setting it once here means the per-line writers can stay terse
    convergenceLog << std::fixed << std::setprecision(6);

    convergenceLog << "cycle\t";
    TreePartitions::writeStatsHeader(convergenceLog);
    convergenceLog << '\t';
    TreeSamples::writeStatsHeader(convergenceLog, (size_t)numChains);
    convergenceLog << '\n';
    convergenceLog.flush();
}

void Mcmc::printToScreen(int n, double curLnL, double newLnL) {

    if (n % printFrequency == 0 || n == 1)
        {
        std::cout << "   * " << std::setw(6) << n << " -- ";
        std::cout << std::fixed << std::setprecision(2); 
        std::cout << curLnL << " -> " << newLnL << " " << std::setw(8) << newLnL-curLnL << " ";
        std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)cacheSize(treeCache)/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::printToScreen(int n, std::vector<double>& curLnL) {

    if (n % printFrequency == 0 || n == 1)
        {
        std::cout << "   * " << std::setw(6) << n << " -- ";
        std::cout << std::fixed << std::setprecision(2); 
        for (size_t i=0; i<curLnL.size(); i++)
            std::cout << curLnL[i] << " ";
        std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)cacheSize(treeCache)/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::printToScreen(int n, std::vector<std::vector<double>>& curLnL, std::vector<std::vector<int>>& indices) {

    if (n % printFrequency == 0 || n == 1)
        {
        std::cout << "   * " << std::setw(6) << n << " -- ";
        std::cout << std::fixed << std::setprecision(2); 
        for (int i=0; i<curLnL.size(); i++)
            {
            std::cout << curLnL[i][coldChainIndex(indices[i])] << " ";
            }
        std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)cacheSize(treeCache)/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::returnCalculator(LikelihoodCalculator* calculator) {

    calculatorPool.push_back(calculator);
}

void Mcmc::run(double power) {
        
    std::cout << "   MCMC:" << std::endl;

    // initialize chain
    Tree* currentTree = new Tree(rng, alignment->getTaxonNames());
    double curLnL = calculateMaximumLikelihood(currentTree);
    
    // run chain
    std::vector<double> forwardProbabilities, reverseProbabilities;
    samples.push_back(new TreeSamples(treeCache));
    samples[0]->reserve(numCycles);
    partitions.push_back(new TreePartitions(alignment->getNumTaxa()));
    for (int n=1; n<=numCycles; n++)
        {
        std::vector<TreeInfo*>& forwardNeighbors = treeNeighbors->neighbors(currentTree);
        calculateMaximumLikelihood(forwardNeighbors);
        normalize(power, forwardNeighbors, forwardProbabilities);
        Tree* newTree = nullptr;
        double newLnL = 0.0;
        double forwardProbability = chooseTree(forwardNeighbors, forwardProbabilities, newTree, newLnL);
        if (newTree == nullptr)
            Msg::error("newTree is null");

        double reverseProbability = forwardProbability;
        std::vector<TreeInfo*>& reverseNeighbors = treeNeighbors->neighbors(newTree);
        calculateMaximumLikelihood(reverseNeighbors);
        normalize(power, reverseNeighbors, reverseProbabilities);
        reverseProbability = findTreeProbability(reverseNeighbors, reverseProbabilities, currentTree->getHash());
        
        forwardNeighbors.clear();
        reverseNeighbors.clear();
        forwardProbabilities.clear();
        reverseProbabilities.clear();
        
        double lnLikelihoodRatio = newLnL - curLnL;
        double lnPriorRatio = 0.0;
        double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
        double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
        
        bool accept = false;
        if (log(rng->uniformRv()) < lnR)
            accept = true;

        printToScreen(n, curLnL, newLnL);

        if (accept == true)
            {
            curLnL = newLnL;
            Tree* temp = currentTree;
            currentTree = newTree;
            delete temp;
            }
        
        samples[0]->sampleTree(currentTree->getHash());
        partitions[0]->addTree(currentTree);
        }
            
    samples[0]->print();
    partitions[0]->print();
    
    delete currentTree;

    std::cout << std::endl;
}

#if 0
void Mcmc::run(double power, int numRuns) {

    std::cout << "   MCMC:" << std::endl;
    std::cout << "   * Number of runs = " << numRuns << std::endl;

    // initialize chain
    std::vector<double> curLnL(numRuns);
    std::vector<Tree*> currentTree(numRuns);
    samples.resize(numRuns);
    partitions.resize(numRuns);
    for (size_t i=0; i<numRuns; i++)
        {
        currentTree[i] = new Tree(rng, alignment->getTaxonNames());
        curLnL[i] = calculateMaximumLikelihood(currentTree[i]);
        samples[i] = new TreeSamples(treeList);
        samples[i]->reserve(numCycles);
        partitions[i] = new TreePartitions(alignment->getNumTaxa());
        }

    openConvergenceLog();
    
    // run chain
    NeighborValues forwardNeighbors, reverseNeighbors;
    int nextLogPoint = 1;
    for (int n=1; n<=numCycles; n++)
        {
        for (size_t i=0; i<numRuns; i++)
            {
            neighborhood->getNeighbors(currentTree[i], forwardNeighbors);
            calculateMaximumLikelihood(forwardNeighbors);
            normalize(power, forwardNeighbors);
            uint64_t newTree;
            double forwardProbability = chooseTree(forwardNeighbors, newTree);
            double newLnL = treeList->getTreeInfo(newTree)->lnL;

            double reverseProbability = forwardProbability;
            neighborhood->getNeighbors(newTree, reverseNeighbors);
            calculateMaximumLikelihood(reverseNeighbors);
            normalize(power, reverseNeighbors);
            reverseProbability = findTreeProbability(reverseNeighbors, currentTree[i]->getHash());
            
            forwardNeighbors.clear();
            reverseNeighbors.clear();
            
            double lnLikelihoodRatio = newLnL - curLnL[i];
            double lnPriorRatio = 0.0;
            double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
            double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
            
            bool accept = false;
            if (log(rng->uniformRv()) < lnR)
                accept = true;

            if (accept == true)
                {
                curLnL[i] = newLnL;
                Tree* temp = currentTree[i];
                currentTree[i] = treeList->getTree(newTree);
                delete temp;
                }
            }
            
        printToScreen(n, curLnL);
        
        for (size_t i=0; i<numRuns; i++)
            {
            samples[i]->sampleTree(currentTree[i]->getHash());
            partitions[i]->addTree(currentTree[i]);
            }
        if (n == nextLogPoint)
            {
            TreePartitions::comparePartitions(partitions);
            TreeSamples::compareSamples(samples);
            writeConvergenceLine(n);
            nextLogPoint = (nextLogPoint < 100000) ? nextLogPoint * 10 : nextLogPoint + 100000;
            }
        }
            
    TreeSamples::compbinedPrint(samples);
    
    for (size_t i=0; i<numRuns; i++)
        delete currentTree[i];
    
    std::cout << std::endl;
}

void Mcmc::run(double power, int numRuns, int numChains) {

    std::cout << "   MCMC:" << std::endl;
    std::cout << "   * Number of runs = " << numRuns << std::endl;
    std::cout << "   * Number of chains = " << numChains << std::endl;
    std::cout << "   * Temperature = " << temperature << std::endl;
    
    // initialize chain
    std::vector<std::vector<double>> curLnL(numRuns);
    std::vector<std::vector<Tree*>> currentTree(numRuns);
    std::vector<std::vector<int>> chainIndex(numRuns);
    for (size_t i=0; i<numRuns; i++)
        {
        curLnL[i].resize(numChains);
        currentTree[i].resize(numChains);
        chainIndex[i].resize(numChains);
        }
    samples.resize(numRuns);
    partitions.resize(numRuns);
    
    for (size_t i=0; i<numRuns; i++)
        {
        for (size_t j=0; j<numChains; j++)
            {
            currentTree[i][j] = new Tree(rng, alignment->getTaxonNames());
            curLnL[i][j] = calculateMaximumLikelihood(currentTree[i][j]);
            chainIndex[i][j] = (int)j;
            }
        samples[i] = new TreeSamples(treeList);
        samples[i]->reserve(numCycles);
        partitions[i] = new TreePartitions(alignment->getNumTaxa());
        }

    openConvergenceLog();
    
    // run chain
    NeighborValues forwardNeighbors, reverseNeighbors;
    int nextLogPoint = 1;
    for (int n=1; n<=numCycles; n++)
        {
        for (size_t i=0; i<numRuns; i++)
            {
            for (size_t j=0; j<numChains; j++)
                {
                neighborhood->getNeighbors(currentTree[i][j], forwardNeighbors);
                calculateMaximumLikelihood(forwardNeighbors);
                normalize(power, forwardNeighbors);
                uint64_t newTree;
                double forwardProbability = chooseTree(forwardNeighbors, newTree);
                double newLnL = treeList->getTreeInfo(newTree)->lnL;

                double reverseProbability = forwardProbability;
                neighborhood->getNeighbors(newTree, reverseNeighbors);
                calculateMaximumLikelihood(reverseNeighbors);
                normalize(power, reverseNeighbors);
                reverseProbability = findTreeProbability(reverseNeighbors, currentTree[i][j]->getHash());
                
                forwardNeighbors.clear();
                reverseNeighbors.clear();
                
                double lnLikelihoodRatio = (newLnL - curLnL[i][j]) * heat(chainIndex[i][j], temperature);
                double lnPriorRatio = 0.0;
                double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
                double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
                
                bool accept = false;
                if (log(rng->uniformRv()) < lnR)
                    accept = true;

                if (accept == true)
                    {
                    curLnL[i][j] = newLnL;
                    Tree* temp = currentTree[i][j];
                    currentTree[i][j] = treeList->getTree(newTree);
                    delete temp;
                    }
                }

            // swap
            std::pair<int,int> idx = chooseChains(numChains);
            int idx0 = chainIndex[i][idx.first];
            int idx1 = chainIndex[i][idx.second];
            double lnR = (curLnL[i][idx.first] * heat(idx1, temperature) + curLnL[i][idx.second] * heat(idx0, temperature));
            lnR -=  (curLnL[i][idx.first] * heat(idx0, temperature) + curLnL[i][idx.second] * heat(idx1, temperature));
            if (log(rng->uniformRv()) < lnR)
                {
                chainIndex[i][idx.first] = idx1;
                chainIndex[i][idx.second] = idx0;
                }
                
            }
            
        printToScreen(n, curLnL, chainIndex);
        
        for (size_t i=0; i<numRuns; i++)
            {
            samples[i]->sampleTree(currentTree[i][coldChainIndex(chainIndex[i])]->getHash());
            partitions[i]->addTree(currentTree[i][coldChainIndex(chainIndex[i])]);
            }
        if (n == nextLogPoint)
            {
            TreePartitions::comparePartitions(partitions);
            writeConvergenceLine(n);
            nextLogPoint = (nextLogPoint < 100000) ? nextLogPoint * 10 : nextLogPoint + 100000;
            }
        }
            
    TreeSamples::compbinedPrint(samples);
    
    for (size_t i=0; i<numRuns; i++)
        for (size_t j=0; j<numChains; j++)
            delete currentTree[i][j];
            
    std::cout << std::endl;
}
#endif
void Mcmc::writeConvergenceLine(int cycle) {

    if (convergenceLog.is_open() == false)
        return;

    convergenceLog << cycle << '\t';
    TreePartitions::writeStatsLine(convergenceLog, partitions);
    convergenceLog << '\t';
    TreeSamples::writeStatsLine(convergenceLog, samples);
    convergenceLog << '\n';

    // flush every line so the user can `tail -f` during long runs;
    // the cost is negligible at log-spaced trigger frequencies
    convergenceLog.flush();
}
