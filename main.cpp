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
    
    // ML = -24390.50
//    std::string mlTree = "(Anolis_ahli,(((((((((Anolis_aliniger,Anolis_coelestinus),(Anolis_equestris,Anolis_luteogularis)),(Anolis_bahorucoensis,Anolis_occultus)),(Anolis_insolitus,Anolis_olssoni)),(Anolis_barahonae,Anolis_cuvieri)),((Anolis_brevirostris,Anolis_distichus),((Anolis_cristatellus,Anolis_stratulus),Anolis_krugi))),((Anolis_alutaceus,Anolis_vanidicus),((Anolis_angusticeps,Anolis_paternus),Anolis_loysiana))),((Anolis_marcanoi,Anolis_strahmi),Diplolaemus_darwinii)),(Anolis_ophiolepis,Anolis_sagrei)),(((Anolis_garmani,Anolis_grahami),Anolis_lineatopus),Anolis_valencienni));";
//    //Tree t(&rng, alignment.getTaxonNames());
//    Tree t(mlTree, alignment.getTaxonNames());
//        
//    LikelihoodCalculator like(&alignment);
//    like.setTree(&t);
//
//    double lnL = like.lnLikelihood();
//    t.print();
//    std::cout << "lnL = " << lnL << std::endl;

    return EXIT_SUCCESS;
}
