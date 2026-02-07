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



int main(int argc, char* argv[]) {
    
    // read the user settings
    UserSettings::userSettings().readSettings(argc, argv);
    UserSettings::userSettings().print();
    
    // read the alignment file
    Alignment alignment(UserSettings::userSettings().getInputFileName());
    alignment.summarize();
    alignment.compress();
    
    // instantiate and initialize some important objects
    RandomVariable rng(1); 
    BitSetFactory::getFactory().initialize(alignment.getNumTaxa());
    ThreadPool pool;
    
    // Markov chain Monte Carlo exploration of tree space
    Mcmc mcmc(&rng, &pool, &alignment);
    mcmc.run();
    
    return EXIT_SUCCESS;
}
