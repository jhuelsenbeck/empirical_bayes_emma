#include <iostream>
#include "Alignment.hpp"
#include "BitSetFactory.hpp"
#include "Mcmc.hpp"
#include "RandomVariable.hpp"
#include "Threads.hpp"
#include "TreeCache.hpp"
#include "TreeLikelihoods.hpp"
#include "TreeNeighborGenerator.hpp"
#include "TreeNeighbors.hpp"
#include "UserSettings.hpp"



int main(int argc, char* argv[]) {

    // instantiate random variable and thread objects
    RandomVariable rng;
    ThreadPool threads;

    // read the user settings
    UserSettings::userSettings().readSettings(argc, argv);
    UserSettings::userSettings().print();
    
    // read the alignment file
    Alignment alignment(UserSettings::userSettings().getInputFileName());
    BitSetFactory::getFactory().initialize(alignment.getNumTaxa());
    alignment.summarize();
    alignment.print(UserSettings::userSettings().getOutputFileName());
    alignment.compress();
    
    // instantiate the TreeList object
    TreeCache treeCache;
    TreeLikelihoods treeLikelihoods(&treeCache);
    TreeNeighborGeneratorNNI treeNeighborGenerator(&treeCache);
    TreeNeighbors treeNeighbors(&treeCache, &treeNeighborGenerator, alignment.getNumTaxa());
        
    // run chain
    Mcmc mcmc1(&rng, &threads, &treeCache, &treeLikelihoods, &treeNeighbors, &alignment);
    mcmc1.run(0.1);

//    Mcmc mcmc2(&rng, &threads, &treeCache, &treeLikelihoods, &treeNeighbors, &alignment);
//    mcmc2.run(0.1, 2);
 
//    Mcmc mcmc3(&rng, &threads, &treeList, neighborhood, &alignment);
//    mcmc3.run(0.1);

    freeTreeCache(&treeCache);

    return EXIT_SUCCESS;
}
