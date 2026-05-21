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
#include "TreeNeighborhood.hpp"
#include "TreeSpace.hpp"
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
    
    TreeList treeList(data.getTaxonNames());
    TreeNeighborhoodNni nniNeighborhood(&treeList);
    TreeNeighborhoodNni2 nni2Neighborhood(&treeList);
    TreeNeighborhoodTbr tbrNeighborhood(&treeList);
    TreeNeighborhood& neighborhood = tbrNeighborhood;
    
    ExhaustiveSearch exhaustive(&data, &treeList, &pool);
    
    TreeSpace treeSpace(&treeList, &neighborhood);
    treeSpace.characterize();
    treeSpace.printPosterior();
        
    // Markov chain Monte Carlo exploration of tree space
    Mcmc mcmc(&rng, &pool, &data, &treeList, &treeSpace);
    mcmc.run(&neighborhood, 0.1);
    treeSpace.printPosterior(mcmc.getSamples());
    mcmc.run(&neighborhood, 0.0);
    treeSpace.printPosterior(mcmc.getSamples());
    
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
