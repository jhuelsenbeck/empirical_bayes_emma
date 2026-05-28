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
    Alignment* originalAlignment = new Alignment(UserSettings::userSettings().getInputFileName());
    Alignment* data = originalAlignment;
    if (originalAlignment->getNumTaxa() > 10)
        data = new Alignment(*originalAlignment, 10, &rng);
    data->print(UserSettings::userSettings().getOutputFileName() + ".nex");
    data->summarize();
    data->compress();
    BitSetFactory::getFactory().initialize(data->getNumTaxa());

    // calculate likelihoods of all trees
    TreeCache treeCache;
    TreeLikelihoods treeLikelihoods(&treeCache);
    ExhaustiveSearch exhaustive(data, &treeCache, &threads);

    // generate the neighbors for each tree
    TreeNeighborGeneratorNNI treeNeighborGenerator(&treeCache);
    TreeNeighbors treeNeighbors(&treeCache, &treeNeighborGenerator, data->getNumTaxa());
    for (auto& [key,val] : treeCache)
        treeNeighbors.neighbors(val->tree);
    
    // determine the tree landscape
    TreeSpace treeSpace(&treeCache);
    treeSpace.characterize();
    treeSpace.printPosterior();
    treeSpace.printPosterior(UserSettings::userSettings().getOutputFileName() + ".true");
        
    // Markov chain Monte Carlo exploration of tree space
//    Mcmc mcmc(&rng, &threads, &treeCache, &treeLikelihoods, &treeNeighbors, &alignment);
//    mcmc.run(0.1);
//    treeSpace.printPosterior(mcmc.getSamples()[0]);
//    mcmc.run(0.0);
//    treeSpace.printPosterior(mcmc.getSamples()[0]);
    
    // clean up
    if (originalAlignment->getNumTaxa() > 10)
        delete data;
    delete originalAlignment;
    freeTreeCache(&treeCache);
    
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
