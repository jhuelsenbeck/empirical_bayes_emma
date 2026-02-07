#include <iostream>
#include <cstdlib>
#include "Alignment.hpp"
#include "BitSetFactory.hpp"
#include "LikelihoodCalculator.hpp"
#include "Mcmc.hpp"
#include "RandomVariable.hpp"
#include "Threads.hpp"
#include "Tree.hpp"
#include "UserSettings.hpp"

void printHeader(void);



int main(int argc, char* argv[]) {

    printHeader();
    
    // read the user settings
    UserSettings::userSettings().readSettings(argc, argv);
    UserSettings::userSettings().print();
    
    // read the alignment file
    Alignment alignment(UserSettings::userSettings().getInputFileName());
    alignment.summarize();
    alignment.compress();
    
    // instantiate and initialize some important objects
    RandomVariable rng; 
    BitSetFactory::getFactory().initialize(alignment.getNumTaxa());
    ThreadPool pool;
    
    // Markov chain Monte Carlo exploration of tree space
    Mcmc mcmc(&rng, &pool, &alignment);
    mcmc.run();
    
    return EXIT_SUCCESS;
}

void printHeader(void) {

    std::cout << std::endl;
    std::cout << "   EPIC — Empirical Phylogenetic Inference of Clades" << std::endl;
    std::cout << "   * Running on " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "   * John P. Huelsenbeck (University of California, Berkeley)" << std::endl;
    std::cout << "   * Emma Gomez (California State University, Fullerton)" << std::endl;
    std::cout << "   * Levi Raskin (University of California, Berkeley)" << std::endl;
    std::cout << std::endl;

}
