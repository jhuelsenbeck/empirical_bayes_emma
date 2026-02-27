#include <iomanip>
#include <iostream>
#include "Alignment.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
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
}

Mcmc::~Mcmc(void) {

    for (size_t i=0; i<allocatedCalculators.size(); i++)
        delete allocatedCalculators[i];
}

void Mcmc::calculateMaximumLikelihoods(TreeList& treeList, uint64_t currentTree, std::vector<uint64_t>& neighbors, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo) {

    // figure out which likelihoods we have already computed and which need to be computed
    neighborhoodInfo.resize(neighbors.size());
    activeCalculators.clear();
    for (size_t i=0; i<neighbors.size(); i++)
        {
        TreeInfo& treeInfo = treeList.getTreeInfo(neighbors[i]);
        if (treeInfo.isLikelihoodCalculated() == true)
            {
            neighborhoodInfo[i] = std::make_pair(neighbors[i], treeInfo.lnL);
            }
        else    
            {
            LikelihoodCalculator* calculator = getCalculator();
            calculator->setTree(treeInfo.getTree());
            calculator->setOffset(i);
            activeCalculators.push_back(calculator);
            }
        }
        
    // push all of the jobs to the thead pool and wait for completion
    for (LikelihoodCalculator* calculator : activeCalculators)
        threadPool->pushTask(calculator);
    threadPool->wait();
    
    // fill out the information for this neighborhood
    for (LikelihoodCalculator* calculator : activeCalculators)
        neighborhoodInfo[calculator->getOffset()] = std::make_pair(calculator->getTree()->getHash(), calculator->getResult());

    // return all of the calculators to the pool
    for (LikelihoodCalculator* calculator : activeCalculators)
        returnCalculator(calculator);
        
    // add all of the calculated likelihoods to the TreeList object
    for (size_t i=0; i<neighborhoodInfo.size(); i++)
        {
        TreeInfo& treeInfo = treeList.getTreeInfo(neighborhoodInfo[i].first);
        if (treeInfo.isLikelihoodCalculated() == false)
            {
            treeInfo.lnL = neighborhoodInfo[i].second;
            treeInfo.setLikelihoodCalculated(true);
            }
        }
    //treeList.print();
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

void Mcmc::normalize(double power, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo) {

    double maxLnL = neighborhoodInfo[0].second;
    for (size_t i=1; i<neighborhoodInfo.size(); i++)
        {
        if (neighborhoodInfo[i].second > maxLnL)
            maxLnL = neighborhoodInfo[i].second;
        }
    double sum = 0.0;
    for (size_t i=0; i<neighborhoodInfo.size(); i++)
        {
        double x = std::exp((neighborhoodInfo[i].second - maxLnL) * power);
        sum += x;
        neighborhoodInfo[i].second = x;
        }
    double factor = 1.0 / sum;
    for (size_t i=0; i<neighborhoodInfo.size(); i++)
        neighborhoodInfo[i].second *= factor;
}

void Mcmc::print(std::map<uint64_t,std::pair<double,double>>& treeProbabilities) {

    size_t numTrees = treeProbabilities.size();
    std::vector<uint64_t> trees;
    trees.reserve(numTrees);
    for (auto& x : treeProbabilities)
        trees.push_back(x.first);
        
    // tree-to-tree distances
    for (size_t i=0; i<numTrees; i++)
        {
        for (size_t j=i+1; j<numTrees; j++)
            {
            double d = treeList->distance(trees[i], trees[j]);
            std::cout << "      d(" << std::setw(20) << trees[i] << "," << std::setw(20) << trees[j] << ") = " << d << std::endl;
            }
        }
        
    int i = 0;
    for (auto& x : treeProbabilities)
        {
        TreeSpaceNode* nde = treeSpace->getTree(x.first);
        int basinId = nde->peakId;
        std::cout << "   * " << ++i << " -- " << x.first << " " << x.second.first;
        std::cout << " " << x.second.second << " ";
        std::cout << basinId << " ";
        std::cout << std::endl;
        }

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

void Mcmc::run(std::map<uint64_t,std::pair<double,double>>& treeProbabilities, double power) {

    std::cout << "   Markov chain Monte Carlo:" << std::endl;

    Tree* initialTree = new Tree(rng, alignment->getTaxonNames());
    uint64_t currentTree = initialTree->getHash();

    LikelihoodCalculator* calculator = getCalculator();
    calculator->setTree(initialTree);
    double curLnL = calculator->lnLikelihood();
    treeList->addTree(initialTree, curLnL);
    returnCalculator(calculator);
    
    initialTree->print();
    std::cout << initialTree->getNewickString() << std::endl;
    BitSet* bs = initialTree->getCompactRepresentation();
    bs->print();
    Tree tempTree(bs, alignment->getTaxonNames());
    tempTree.print();
    
    TreeNeighborhoodNni treeSpace(treeList);
    TreeSamples samples(treeList);
    std::vector<std::pair<uint64_t, double>> forwardNeighborhood;
    std::vector<std::pair<uint64_t, double>> reverseNeighborhood;
    int numAccepted = 0;
    
    for (int n=1; n<=chainLength; n++)
        {
        std::vector<uint64_t>& forwardNeighbors = treeSpace.getNeighbors(currentTree);
        forwardNeighborhood.clear();
        calculateMaximumLikelihoods(*treeList, currentTree, forwardNeighbors, forwardNeighborhood);
        normalize(power, forwardNeighborhood);
        uint64_t newTree;
        double forwardProbability = chooseTree(forwardNeighborhood, newTree);
        double newLnL = treeList->getTreeInfo(newTree).lnL;
        
        double reverseProbability = forwardProbability;
        std::vector<uint64_t>& reverseNeighbors = treeSpace.getNeighbors(newTree);
        reverseNeighborhood.clear();
        calculateMaximumLikelihoods(*treeList, newTree, reverseNeighbors, reverseNeighborhood);
        normalize(power, reverseNeighborhood);
        reverseProbability = findTreeProbability(reverseNeighborhood, currentTree);
        
        double lnLikelihoodRatio = newLnL - curLnL;
        double lnPriorRatio = 0.0;
        double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
        lnProposalRatio *= power;
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
            samples.sampleTree(currentTree);
        }
        
    samples.print(treeProbabilities);
    std::cout << "   Acceptance rate: " << ((double)numAccepted / chainLength) * 100.0 << "%" << std::endl << std::endl;
    
    print(treeProbabilities);
}

void Mcmc::run(std::map<uint64_t,std::pair<double,double>>& treeProbabilities, double power, int numChains, double temperature) {

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
    
    TreeNeighborhoodNni treeSpace(treeList);
    TreeSamples samples(treeList);
    std::vector<std::pair<uint64_t, double>> forwardNeighborhood;
    std::vector<std::pair<uint64_t, double>> reverseNeighborhood;
    int numAccepted = 0;
    
    for (int n=1; n<=chainLength; n++)
        {
        for (size_t chain=0; chain<numChains; chain++)
            {
            std::vector<uint64_t>& forwardNeighbors = treeSpace.getNeighbors(currentTree[chain]);
            forwardNeighborhood.clear();
            calculateMaximumLikelihoods(*treeList, currentTree[chain], forwardNeighbors, forwardNeighborhood);
            normalize(power, forwardNeighborhood);
            uint64_t newTree;
            double forwardProbability = chooseTree(forwardNeighborhood, newTree);
            double newLnL = treeList->getTreeInfo(newTree).lnL;
            
            double reverseProbability = forwardProbability;
            std::vector<uint64_t>& reverseNeighbors = treeSpace.getNeighbors(newTree);
            reverseNeighborhood.clear();
            calculateMaximumLikelihoods(*treeList, newTree, reverseNeighbors, reverseNeighborhood);
            normalize(power, reverseNeighborhood);
            reverseProbability = findTreeProbability(reverseNeighborhood, currentTree[chain]);
            
            double lnLikelihoodRatio = (newLnL - curLnL[chain]) * heat(chainIndex[chain], temperature);
            double lnPriorRatio = 0.0;
            double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
            lnProposalRatio *= power;
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
            samples.sampleTree(currentTree[coldChainIdx]);
            }
        }
        
    samples.print(treeProbabilities);
    std::cout << "   Acceptance rate: " << ((double)numAccepted / chainLength) * 100.0 << "%" << std::endl << std::endl;
    
    print(treeProbabilities);
}


