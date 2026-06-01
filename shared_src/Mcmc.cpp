#include "Alignment.hpp"
#include "CompactTree.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
#include "Msg.hpp"
#include "Peak.hpp"
#include "RandomVariable.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"
#include "TreeConvergenceDiagnostics.hpp"
#include "TreeLikelihoods.hpp"
#include "TreeNeighbors.hpp"
#include "TreePartitions.hpp"
#include "TreeSamples.hpp"
#include "TreeSpace.hpp"
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

void Mcmc::deleteSamplesAndPartitions(void) {

    for (TreeSamples* s : samples)
        delete s;
    for (TreePartitions* p : partitions)
        delete p;
    samples.clear();
    partitions.clear();
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
    std::string path = settings.getOutputFileName() + convergenceLogFileName + ".tsv";

    convergenceLog.open(path, std::ios::out | std::ios::trunc);
    if (convergenceLog.is_open() == false)
        {
        std::cout << "Warning: could not open convergence log at " << path << std::endl;
        return;
        }

    convergenceLog << std::fixed << std::setprecision(6);

    convergenceLog << "cycle\t";
    TreePartitions::writeStatsHeader(convergenceLog);
    convergenceLog << '\t';
    TreeSamples::writeStatsHeader(convergenceLog, (size_t)numChains);
    convergenceLog << '\t';
    TreeConvergenceDiagnostics::writeStatsHeader(convergenceLog, (size_t)numChains);
    convergenceLog << '\n';

    convergenceLog.flush();
}

void Mcmc::openTreeFile(void) {

    UserSettings& settings = UserSettings::userSettings();
    std::string path = settings.getOutputFileName() + ".tre";
    treeStrm.open(path, std::ios::out | std::ios::trunc);
    if (treeStrm.is_open() == false)
        {
        std::cout << "Warning: could not open tree stream at " << path << std::endl;
        return;
        }

    treeStrm << "#NEXUS" << std::endl << std::endl;
    treeStrm << "begin trees;" << std::endl;
    treeStrm << "   translate" << std::endl;
    std::vector<std::string>& tNames = alignment->getTaxonNames();
    for (size_t i=0; i<tNames.size(); i++)
        {
        treeStrm << "      " << i+1 << " " << tNames[i];
        if (i + 1 != tNames.size())
            treeStrm << ",";
        else 
            treeStrm << ";";
        treeStrm << std::endl;
        }
}

void Mcmc::printTreeToFile(int n, Tree* t) {

    treeStrm << "   tree gen." << n << " = " << t->getNewickString() << std::endl;
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
            std::cout << curLnL[i] << " ";
        std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)treeCache->cacheSize()/1000000.0 << "MB) ";
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
            }
        std::cout << "(" << treeCache->size() << " " << std::setprecision(1) << (double)treeCache->cacheSize()/1000000.0 << "MB) ";
        std::cout << std::endl;
        }
}

void Mcmc::recordState(int n, bool accept, uint64_t currentTreeHash, uint64_t newTreeHash) {

    if (accept == true)
        {
        TreeInfo* tInfo = treeCache->getTreeInfo(newTreeHash);
        if (currentTreeHash != newTreeHash)
            {
            if (tInfo->hasBeenVisited == false)
                {
                tInfo->firstHit = n;
                tInfo->hasBeenVisited = true;
                }
            tInfo->numRevisits++;
            }
        tInfo->residenceCount++;
        }
    else
        {
        TreeInfo* tInfo = treeCache->getTreeInfo(currentTreeHash);
        tInfo->residenceCount++;
        }
}

void Mcmc::returnCalculator(LikelihoodCalculator* calculator) {

    calculatorPool.push_back(calculator);
}

void Mcmc::run(double power) {
        
    if (expandedOutput == true)
        std::cout << "   MCMC:" << std::endl;
    else 
        std::cout << "   Running MCMC for " << numCycles << " generations" << std::endl;
    
    openTreeFile();

    // initialize chain
    Tree* currentTree = new Tree(rng, alignment->getTaxonNames());
    double curLnL = calculateMaximumLikelihood(currentTree);
    recordState(1, true, currentTree->getHash()-1, currentTree->getHash());
    
    // run chain
    std::vector<double> forwardProbabilities, reverseProbabilities;
    deleteSamplesAndPartitions();
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
        recordState(n, accept, currentTree->getHash(), newTree->getHash());

        if (accept == true)
            {
            curLnL = newLnL;
            Tree* temp = currentTree;
            currentTree = newTree;
            delete temp;
            }
        
        samples[0]->sampleTree(currentTree->getHash());
        partitions[0]->addTree(currentTree);
        if (n % sampleFrequency == 0 || n == 1)
            printTreeToFile(n, currentTree);
        }
                    
    samples[0]->print();
    partitions[0]->print();
    
    delete currentTree;

    std::cout << std::endl;
}

void Mcmc::run(double power, int numRuns) {

    if (numRuns < 2)
        Msg::error("Expecting at least two runs for this chain");
        
    if (expandedOutput == true)
        {
        std::cout << "   MCMC:" << std::endl;
        std::cout << "   * Number of runs = " << numRuns << std::endl;
        }
    else 
        std::cout << "   Running " << numRuns << " MCMC chains for " << numCycles << " generations each" << std::endl;
    
    openTreeFile();
    openConvergenceLog();

    // initialize chain
    std::vector<Tree*> currentTree(numRuns);
    std::vector<double> curLnL(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        currentTree[run] = new Tree(rng, alignment->getTaxonNames());
        curLnL[run] = calculateMaximumLikelihood(currentTree[run]);
        }
        
    // initialize objects for storing results
    deleteSamplesAndPartitions();
    samples.resize(numRuns);
    partitions.resize(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        samples[run] = new TreeSamples(treeCache);
        samples[run]->reserve(numCycles);
        partitions[run] = new TreePartitions(alignment->getNumTaxa());
        }

    // run chain
    std::vector<double> forwardProbabilities, reverseProbabilities;
    for (int n=1, nextLogPoint=1; n<=numCycles; n++)
        {
        for (int run=0; run<numRuns; run++)
            {
            std::vector<TreeInfo*>& forwardNeighbors = treeNeighbors->neighbors(currentTree[run]);
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
            reverseProbability = findTreeProbability(reverseNeighbors, reverseProbabilities, currentTree[run]->getHash());
            
            forwardNeighbors.clear();
            reverseNeighbors.clear();
            forwardProbabilities.clear();
            reverseProbabilities.clear();
            
            double lnLikelihoodRatio = newLnL - curLnL[run];
            double lnPriorRatio = 0.0;
            double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
            double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
            
            bool accept = false;
            if (log(rng->uniformRv()) < lnR)
                accept = true;

            if (run == 0)
                recordState(n, accept, currentTree[run]->getHash(), newTree->getHash());

            if (accept == true)
                {
                curLnL[run] = newLnL;
                Tree* temp = currentTree[run];
                currentTree[run] = newTree;
                delete temp;
                }
            
            samples[run]->sampleTree(currentTree[run]->getHash());
            partitions[run]->addTree(currentTree[run]);
            if (n % sampleFrequency == 0 || n == 1)
                printTreeToFile(n, currentTree[run]);

            if (n == nextLogPoint)
                {
                TreePartitions::comparePartitions(partitions, false);
                writeConvergenceLine(n);
                nextLogPoint = (nextLogPoint < 100000) ? nextLogPoint * 10 : nextLogPoint + 100000;
                }
            }
        printToScreen(n, curLnL);
        }
            
    TreeSamples::compbinedPrint(samples);
    TreeSamples::compareSamples(samples);
    TreePartitions::comparePartitions(partitions, true);
    
    for (int run=0; run<numRuns; run++)
        delete currentTree[run];
        
    std::cout << std::endl;
}

void Mcmc::run(double power, int numRuns, int numChains) {

    if (expandedOutput == true)
        {
        std::cout << "   MCMC:" << std::endl;
        std::cout << "   * Number of runs = " << numRuns << std::endl;
        std::cout << "   * Number of chains = " << numChains << std::endl;
        std::cout << "   * Temperature = " << temperature << std::endl;
        }
    else 
        std::cout << "   Running " << numRuns << " MCMCMC chains (one cold and " << numChains-1 << "heated) for " << numCycles << " generations each" << std::endl;
    
    openTreeFile();

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
    deleteSamplesAndPartitions();
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
        samples[i] = new TreeSamples(treeCache);
        samples[i]->reserve(numCycles);
        partitions[i] = new TreePartitions(alignment->getNumTaxa());
        }
        
    // initialize objects for storing results
    samples.resize(numRuns);
    partitions.resize(numRuns);
    for (int run=0; run<numRuns; run++)
        {
        samples[run] = new TreeSamples(treeCache);
        samples[run]->reserve(numCycles);
        partitions[run] = new TreePartitions(alignment->getNumTaxa());
        }

    // run chain
    std::vector<double> forwardProbabilities, reverseProbabilities;
    for (int n=1, nextLogPoint=1; n<=numCycles; n++)
        {
        for (int run=0; run<numRuns; run++)
            {
            for (size_t chain=0; chain<numChains; chain++)
                {
                std::vector<TreeInfo*>& forwardNeighbors = treeNeighbors->neighbors(currentTree[run][chain]);
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
                reverseProbability = findTreeProbability(reverseNeighbors, reverseProbabilities, currentTree[run][chain]->getHash());
                
                forwardNeighbors.clear();
                reverseNeighbors.clear();
                forwardProbabilities.clear();
                reverseProbabilities.clear();
                
                double lnLikelihoodRatio = newLnL - curLnL[run][chain];
                double lnPriorRatio = 0.0;
                double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
                double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
                
                bool accept = false;
                if (log(rng->uniformRv()) < lnR)
                    accept = true;

                if (run == coldChainIndex(chainIndex[run]))
                    recordState(n, accept, currentTree[run][chain]->getHash(), newTree->getHash());

                if (accept == true)
                    {
                    curLnL[run][chain] = newLnL;
                    Tree* temp = currentTree[run][chain];
                    currentTree[run][chain] = newTree;
                    delete temp;
                    }
                
                }
                
            // swap
            std::pair<int,int> idx = chooseChains(numChains);
            int idx0 = chainIndex[run][idx.first];
            int idx1 = chainIndex[run][idx.second];
            double lnR = (curLnL[run][idx.first] * heat(idx1, temperature) + curLnL[run][idx.second] * heat(idx0, temperature));
            lnR -=  (curLnL[run][idx.first] * heat(idx0, temperature) + curLnL[run][idx.second] * heat(idx1, temperature));
            if (log(rng->uniformRv()) < lnR)
                {
                chainIndex[run][idx.first] = idx1;
                chainIndex[run][idx.second] = idx0;
                }

            int coldIdx = coldChainIndex(chainIndex[run]);
            samples[run]->sampleTree(currentTree[run][coldIdx]->getHash());
            partitions[run]->addTree(currentTree[run][coldIdx]);
            if (n == nextLogPoint)
                {
                TreePartitions::comparePartitions(partitions, false);
                writeConvergenceLine(n);
                nextLogPoint = (nextLogPoint < 100000) ? nextLogPoint * 10 : nextLogPoint + 100000;
                }
            }
            
        printToScreen(n, curLnL, chainIndex);
        }
            
    TreeSamples::compbinedPrint(samples);
    TreeSamples::compareSamples(samples);
    TreePartitions::comparePartitions(partitions, true);
    
    for (int run=0; run<numRuns; run++)
        for (int chain=0; chain<numChains; chain++)
            delete currentTree[run][chain];
        
    std::cout << std::endl;
}

void Mcmc::welfordSummary(TreeSpace* ts, TreeCache* tc, double n) {

    std::vector<TreeInfo*> trees;
    trees.reserve(tc->size());
    TreeCacheMap& tcCache = tc->getCache();
    for (const auto& kv : tcCache)
        {
        if (kv.second != nullptr)
            trees.push_back(kv.second);
        }

    std::sort(trees.begin(),
              trees.end(),
              [](const TreeInfo* a, const TreeInfo* b)
                  {
                  return a->posteriorProbability > b->posteriorProbability;
                  });

    std::cout << std::fixed << std::setprecision(2);
    int i = 0;
    double sum = 0.0;
    for (const TreeInfo* info : trees)
        {
        Peak* peak = ts->findPeakForTreeWithHash(info->hash);
        if (peak == nullptr)
            Msg::error("Could not find peak");
        int peakId = peak->getPeakId();
        double peakProb = peak->getPeakProbability();
        
        sum += info->posteriorProbability;
        std::cout << ++i << " -- " << std::setw(20) << info->hash << " " << peakId << " " << peakProb << " -- ";
        std::cout << info->lnLikelihood << " " << info->posteriorProbability << " " << sum << " -- ";
        
        double mean = info->meanResidenceCount;
        double var = info->m2ResidenceCount / (n-1);
        double sem = std::sqrt(var/n);
        std::cout << mean << " (" << sem << ") ";
        
        mean = info->meanFirstHit;
        var = info->m2FirstHit / (n-1);
        sem = std::sqrt(var/n);
        std::cout << mean << " (" << sem << ") ";

        mean = info->meanNumRevisits;
        var = info->m2NumRevisits / (n-1);
        sem = std::sqrt(var/n);
        std::cout << mean << " (" << sem << ") ";

        std::cout << std::endl;
                  
        if (sum > 0.99)
            break;
        }
}

void Mcmc::welfordUpdate(double n) {

    TreeCacheMap& tCache = treeCache->getCache();
    for (auto& [key,val] : tCache)
        {
        TreeInfo* info = val;
        
        double x = (double)val->firstHit;
        double delta = x - info->meanFirstHit;
        info->meanFirstHit += delta / n;
        double delta2 = x - info->meanFirstHit;
        info->m2FirstHit += delta * delta2;
        
        x = (double)val->numRevisits;
        delta = x - info->meanNumRevisits;
        info->meanNumRevisits += delta / n;
        delta2 = x - info->meanNumRevisits;
        info->m2NumRevisits += delta * delta2;

        x = (double)val->residenceCount;
        delta = x - info->meanResidenceCount;
        info->meanResidenceCount += delta / n;
        delta2 = x - info->meanResidenceCount;
        info->m2ResidenceCount += delta * delta2;
        
        info->firstHit = std::numeric_limits<unsigned>::max();
        info->numRevisits = 0;
        info->residenceCount = 0;
        info->hasBeenVisited = false;
        }
}

void Mcmc::writeConvergenceLine(int cycle) {

    if (convergenceLog.is_open() == false)
        return;

    convergenceLog << cycle << '\t';

    TreePartitions::writeStatsLine(convergenceLog, partitions);
    convergenceLog << '\t';

    TreeSamples::writeStatsLine(convergenceLog, samples);
    convergenceLog << '\t';

    TreeConvergenceDiagnostics diagnostics(treeCache);
    diagnostics.writeStatsLine(convergenceLog, samples);

    convergenceLog << '\n';
    convergenceLog.flush();
}
