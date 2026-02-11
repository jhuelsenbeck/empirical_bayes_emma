#include <iostream>
#include <cstdlib>
#include "Alignment.hpp"
#include "BitSetFactory.hpp"
#include "ExhaustiveSearch.hpp"
#include "Mcmc.hpp"
#include "RandomVariable.hpp"
#include "Threads.hpp"
#include "Tree.hpp"
#include "TreeList.hpp"
#include "UserSettings.hpp"

void printHeader(void);



int main(int argc, char* argv[]) {

    printHeader();
    
    RandomVariable rng; 
    ThreadPool pool;

    // read the user settings
    UserSettings::userSettings().readSettings(argc, argv);
    UserSettings::userSettings().print();
    
    // read the alignment file
    Alignment alignment(UserSettings::userSettings().getInputFileName());
    alignment.twist(&rng, 20);
    alignment.print(UserSettings::userSettings().getOutputFileName());
    alignment.summarize();
    alignment.compress();
    
    BitSetFactory::getFactory().initialize(alignment.getNumTaxa());
    TreeList treeList(alignment.getTaxonNames());
    std::map<uint64_t,std::pair<double,double>> treeProbabilities;
    
    ExhaustiveSearch exhaustive(&alignment, &treeList, &pool);
    exhaustive.enumerateAllTrees(treeProbabilities);
        
    // Markov chain Monte Carlo exploration of tree space
    Mcmc mcmc(&rng, &pool, &alignment, &treeList);
    mcmc.run(treeProbabilities);
    
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
