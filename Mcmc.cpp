#include <iomanip>
#include <iostream>
#include "Alignment.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
#include "RandomVariable.hpp"
#include "UserSettings.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"
#include "TreeSpace.hpp"



Mcmc::Mcmc(RandomVariable* r, ThreadPool* tp, Alignment* a) : rng(r), threadPool(tp), alignment(a) {

    UserSettings& settings = UserSettings::userSettings();
    chainLength = settings.getChainLength();
    printFrequency = settings.getPrintFrequency();
    sampleFrequency = settings.getSampleFrequency();
}

Mcmc::~Mcmc(void) {

    for (size_t i=0; i<allocatedCalculators.size(); i++)
        delete allocatedCalculators[i];
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

void Mcmc::run(void) {

    TreeList treeList;
    Tree* initialTree = new Tree(rng, alignment->getTaxonNames());
    uint64_t currentTree = initialTree->getHash();

    LikelihoodCalculator* calculator = getCalculator();
    calculator->setTree(initialTree);
    double curLnL = calculator->lnLikelihood();
    treeList.addTree(initialTree, curLnL);
    returnCalculator(calculator);
    
    TreeSpace treeSpace(&treeList);
    std::vector<std::pair<uint64_t, double>> forwardNeighborhood;
    std::vector<std::pair<uint64_t, double>> reverseNeighborhood;
    double power = 0.1;
    for (int n=1; n<=chainLength; n++)
        {
        std::vector<uint64_t>& forwardNeighbors = treeSpace.getNeighbors(currentTree);
        forwardNeighborhood.clear();
        calculateMaximumLikelihoods(treeList, currentTree, forwardNeighbors, forwardNeighborhood);
        normalize(power, forwardNeighborhood);
        uint64_t newTree;
        double forwardProbability = chooseTree(forwardNeighborhood, newTree);
        double newLnL = treeList.getTreeInfo(newTree).lnL;
        
        std::vector<uint64_t>& reverseNeighbors = treeSpace.getNeighbors(newTree);
        reverseNeighborhood.clear();
        calculateMaximumLikelihoods(treeList, newTree, reverseNeighbors, reverseNeighborhood);
        normalize(power, reverseNeighborhood);
        double reverseProbability = findTreeProbability(reverseNeighborhood, currentTree);
        
        double lnLikelihoodRatio = newLnL - curLnL;
        double lnPriorRatio = 0.0;
        double lnProposalRatio = std::log(reverseProbability) - std::log(forwardProbability);
        lnProposalRatio *= power;
        double lnR = lnLikelihoodRatio + lnPriorRatio + lnProposalRatio;
        
        bool accept = false;
        if (log(rng->uniformRv()) < lnR)
            accept = true;
            
        if (accept == true)
            {
            currentTree = newTree;
            curLnL = newLnL;
            }
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(6) << n << " -- " << curLnL << " -> " << newLnL << " " << std::setw(8) << lnLikelihoodRatio << " " << std::setw(8) << lnProposalRatio << std::endl;

            
            
        }
}

void Mcmc::calculateMaximumLikelihoods(TreeList& treeList, uint64_t currentTree, std::vector<uint64_t>& neighbors, std::vector<std::pair<uint64_t, double>>& neighborhoodInfo) {

    // figure out which likelihoods we have already computed and which need to be computed
    neighborhoodInfo.resize(neighbors.size()+1);
    activeCalculators.clear();
    for (size_t i=0; i<neighbors.size(); i++)
        {
        TreeInfo& treeInfo = treeList.getTreeInfo(neighbors[i]);
        if (treeInfo.likelihoodCalculated == true)
            {
            neighborhoodInfo[i] = std::make_pair(neighbors[i], treeInfo.lnL);
            }
        else    
            {
            LikelihoodCalculator* calculator = getCalculator();
            calculator->setTree(treeInfo.tree);
            calculator->setOffset(i);
            activeCalculators.push_back(calculator);
            }
        }
    TreeInfo& curTreeInfo = treeList.getTreeInfo(currentTree);
    if (curTreeInfo.likelihoodCalculated == true)
        neighborhoodInfo[neighbors.size()] = std::make_pair(currentTree, curTreeInfo.lnL);
    else 
        {
        LikelihoodCalculator* calculator = getCalculator();
        calculator->setTree(curTreeInfo.tree);
        calculator->setOffset(neighbors.size());
        activeCalculators.push_back(calculator);
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
        if (treeInfo.likelihoodCalculated == false)
            {
            treeInfo.lnL = neighborhoodInfo[i].second;
            treeInfo.likelihoodCalculated = true;
            }
        }
    //treeList.print();
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

void Mcmc::returnCalculator(LikelihoodCalculator* calculator) {

    calculatorPool.push_back(calculator);
}
