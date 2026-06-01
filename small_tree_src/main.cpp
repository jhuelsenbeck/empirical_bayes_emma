#include <iostream>
#include <cstdlib>
#include "Alignment.hpp"
#include "BitSetFactory.hpp"
#include "ExhaustiveSearch.hpp"
#include "Mcmc.hpp"
#include "RandomVariable.hpp"
#include "Threads.hpp"
#include "TreeCache.hpp"
#include "TreeLikelihoods.hpp"
#include "TreeNeighborGenerator.hpp"
#include "TreeNeighbors.hpp"
#include "TreeSpace.hpp"
#include "UserSettings.hpp"

void printHeader(void);



int main(int argc, char* argv[]) {

    printHeader();
    
    // instantiate random variable and thread objects
    RandomVariable rng;
    ThreadPool threads;

    // read the user settings
    UserSettings::userSettings().readSettings(argc, argv);
    UserSettings::userSettings().print();
    
    // read the alignment file
    int numTaxa = 8;
    Alignment* originalAlignment = new Alignment(UserSettings::userSettings().getInputFileName());
    Alignment* data = originalAlignment;
    if (originalAlignment->getNumTaxa() > numTaxa)
        data = new Alignment(*originalAlignment, numTaxa, &rng);
    data->print(UserSettings::userSettings().getOutputFileName() + ".nex");
    data->summarize();
    data->compress();
    BitSetFactory::getFactory().initialize(data->getNumTaxa());

    // calculate likelihoods of all trees
    TreeCache treeCacheNni;
    TreeLikelihoods treeLikelihoods(&treeCacheNni);
    ExhaustiveSearch exhaustive(data, &treeCacheNni, &threads);

    // generate the NNI neighbors for each tree
    std::cout << "   Generating NNI neighbors for all " << treeCacheNni.size() << " trees" << std::endl;
    TreeNeighborGeneratorNNI treeNeighborGeneratorNni(&treeCacheNni);
    TreeNeighbors treeNeighborsNni(&treeCacheNni, &treeNeighborGeneratorNni, data->getNumTaxa());
    TreeCacheMap& nniCache = treeCacheNni.getCache();
    for (auto& [key,val] : nniCache)
        treeNeighborsNni.neighbors(val->tree);
        
    // generate the TBR neighbors for each tree
    TreeCache treeCacheTbr;
    treeCacheTbr.injectTreesAndLikelihoods(&treeCacheNni);
    std::cout << "   Generating TBR neighbors for all " << treeCacheTbr.size() << " trees" << std::endl;
    TreeNeighborGeneratorTBR treeNeighborGeneratorTbr(&treeCacheTbr);
    TreeNeighbors treeNeighborsTbr(&treeCacheTbr, &treeNeighborGeneratorTbr, data->getNumTaxa());
    TreeCacheMap& tbrCache = treeCacheTbr.getCache();
    for (auto& [key,val] : tbrCache)
        treeNeighborsTbr.neighbors(val->tree);
    
    // determine the tree landscapes for NNI and TBR
    TreeSpace treeSpaceNni(&treeCacheNni);
    treeSpaceNni.characterize();
    treeSpaceNni.printPosterior();
    treeSpaceNni.printPosterior(UserSettings::userSettings().getOutputFileName() + ".true");
    TreeSpace treeSpaceTbr(&treeCacheTbr);
    treeSpaceTbr.characterize();
        
    // Markov chain Monte Carlo exploration of tree space
    std::vector<double> powers = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0 };
    std::vector<TreeCache*> treeCache = { &treeCacheNni, &treeCacheTbr };
    std::vector<TreeSpace*> treeSpace = { &treeSpaceNni, &treeSpaceTbr };
    std::vector<TreeNeighbors*> treeNeighbors = { &treeNeighborsNni, &treeNeighborsTbr };
    std::vector<std::string> swapName = { "nni", "tbr" };
    for (int swapType=0; swapType<2; swapType++)
        {
        int nReps = 5;
        for (double power : powers)
            {
            std::cout << "Analysis: " << swapType << " " << power << std::endl;
            std::string convergenceFileName = "conv_" + swapName[swapType] + "_" + std::to_string(power);
            Mcmc mcmc(&rng, &threads, treeCache[swapType], &treeLikelihoods, treeNeighbors[swapType], data, true, convergenceFileName);
            mcmc.run(power, nReps);
            mcmc.welfordUpdate(nReps);
            Mcmc::welfordSummary(treeSpace[swapType], treeCache[swapType], nReps);
            treeCacheNni.cleanCacheStatistics();
            }
        }
    
    // clean up
    if (originalAlignment->getNumTaxa() > numTaxa)
        delete data;
    delete originalAlignment;
    treeCacheNni.freeTreeCache();
    treeCacheTbr.freeTreeCache();
    
    return EXIT_SUCCESS;
}

void printHeader(void) {

    std::cout << std::endl;
    std::cout << "   Bayesian Inference of Phylogeny using Profile Likelihoods" << std::endl;
    std::cout << "   * Running on " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "   * John P. Huelsenbeck (University of California, Berkeley)" << std::endl;
    std::cout << "   * Emma Gomez (California State University, Fullerton)" << std::endl;
    std::cout << "   * Bruce Rannala (University of California, Davis)" << std::endl;
    std::cout << "   * Levi Yoder Raskin (University of California, Berkeley)" << std::endl;
    std::cout << std::endl;
}
