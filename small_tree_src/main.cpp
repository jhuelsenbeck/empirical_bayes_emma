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
    Alignment originalAlignment(UserSettings::userSettings().getInputFileName());
    //Alignment& alignment = originalAlignment;
    Alignment alignment(originalAlignment, 9, &rng);
    alignment.print(UserSettings::userSettings().getOutputFileName());
    alignment.summarize();
    alignment.compress();
    
    // simulate an alignment
    std::string newickString = "((((T1:0.1,T2:0.1):0.1,T3:0.1):0.1,(T4:0.1,T5:0.1):0.1):0.1,((T6:0.1,T7:0.1):0.1,(T8:0.1,T9:0.1):0.1):0.1);";
    std::vector<std::string> taxonNames = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"};
    Alignment simulatedAlignment(taxonNames, newickString, 1000, &rng);
    simulatedAlignment.twist(&rng, 20);
    simulatedAlignment.summarize();
    simulatedAlignment.compress();
    
    Alignment& data = alignment;
    BitSetFactory::getFactory().initialize(data.getNumTaxa());

    // calculate likelihoods of all trees
    TreeCache treeCache;
    TreeLikelihoods treeLikelihoods(&treeCache);
    ExhaustiveSearch exhaustive(&data, &treeCache, &treeLikelihoods, &threads);
    treeLikelihoods.print();

    // generate the neighbors for each tree
    TreeNeighborGeneratorNNI treeNeighborGenerator(&treeCache);
    TreeNeighbors treeNeighbors(&treeCache, &treeNeighborGenerator, alignment.getNumTaxa());
    for (auto& [key,val] : treeCache)
        treeNeighbors.neighbors(val->tree);
    treeNeighbors.print();
    
    // determine the tree landscape
    TreeSpace treeSpace(&treeCache, &treeLikelihoods, &treeNeighbors);
    treeSpace.characterize();
    treeSpace.printPosterior();
        
    // Markov chain Monte Carlo exploration of tree space
    Mcmc mcmc(&rng, &threads, &treeCache, &treeLikelihoods, &treeNeighbors, &alignment);
    mcmc.run(0.1);
    treeSpace.printPosterior(mcmc.getSamples()[0]);
    mcmc.run(0.0);
    //treeSpace.printPosterior(mcmc.getSamples());
    
    return EXIT_SUCCESS;
}

void printHeader(void) {

    std::cout << std::endl;
    std::cout << "   EPIC — Empirical Phylogenetic Inference of Clades" << std::endl;
    std::cout << "   * Running on " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "   * John P. Huelsenbeck (University of California, Berkeley)" << std::endl;
    std::cout << "   * Emma Gomez (California State University, Fullerton)" << std::endl;
    std::cout << "   * Bruce Rannala (University of California, Davis)" << std::endl;
    std::cout << "   * Levi Yoder Raskin (University of California, Berkeley)" << std::endl;
    std::cout << std::endl;

}
