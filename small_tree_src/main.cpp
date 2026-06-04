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

void generateNeighbors(TreeCache& treeCache, TreeNeighbors& generator, std::string label);
void printHeader(void);



int main(int argc, char* argv[]) {

    printHeader();
    
    // instantiate random variable and thread objects
    RandomVariable rng;
    ThreadPool threads;

    // read the user settings
    UserSettings& settings = UserSettings::userSettings();
    settings.readSettings(argc, argv);
    settings.print();
    int numTaxa = 10;
    int nReps = 50;
    
    // read the alignment file
    Alignment* originalAlignment = new Alignment(settings.getInputFileName());
    Alignment* data = originalAlignment;
    if (originalAlignment->getNumTaxa() > numTaxa)
        data = new Alignment(*originalAlignment, numTaxa, &rng);
    data->print(settings.getOutputFileName() + ".nex");
    if (settings.getNumTwists() > 0)
        data->twist(&rng, settings.getNumTwists());
    data->summarize();
    data->compress();
    BitSetFactory::getFactory().initialize(data->getNumTaxa());

    // calculate likelihoods of all trees
    TreeCache treeCacheNni;
    TreeLikelihoods treeLikelihoods(&treeCacheNni);
    ExhaustiveSearch exhaustive(data, &treeCacheNni, &threads);

    // generate the NNI neighbors for each tree
    TreeNeighborGeneratorNNI treeNeighborGeneratorNni(&treeCacheNni);
    TreeNeighbors treeNeighborsNni(&treeCacheNni, &treeNeighborGeneratorNni, data->getNumTaxa());
    generateNeighbors(treeCacheNni, treeNeighborsNni, "NNI");
    
    // generate the NNi of the NNI neighbors for each tree
    TreeCache treeCacheNni2;
    treeCacheNni2.injectTreesAndLikelihoods(&treeCacheNni);
    TreeNeighborGeneratorNNI2 treeNeighborGeneratorNni2(&treeCacheNni2, &treeCacheNni);
    TreeNeighbors treeNeighborsNni2(&treeCacheNni2, &treeNeighborGeneratorNni2, data->getNumTaxa());
    generateNeighbors(treeCacheNni2, treeNeighborsNni2, "NNI2");
        
    // generate the TBR neighbors for each tree
    TreeCache treeCacheTbr;
    treeCacheTbr.injectTreesAndLikelihoods(&treeCacheNni);
    TreeNeighborGeneratorTBR treeNeighborGeneratorTbr(&treeCacheTbr);
    TreeNeighbors treeNeighborsTbr(&treeCacheTbr, &treeNeighborGeneratorTbr, data->getNumTaxa());
    generateNeighbors(treeCacheTbr, treeNeighborsTbr, "TBR");
    
    // determine the tree landscapes for NNI, NNI2, and TBR
    TreeSpace treeSpaceNni(&treeCacheNni, "NNI");
    treeSpaceNni.characterize();
    treeSpaceNni.printPosterior();
    treeSpaceNni.printPosterior(settings.getOutputFileName() + ".true");
    TreeSpace treeSpaceNni2(&treeCacheNni2, "NNI2");
    treeSpaceNni2.characterize();
    TreeSpace treeSpaceTbr(&treeCacheTbr, "TBR");
    treeSpaceTbr.characterize();
    treeSpaceNni.writeRuggednessStatistics(settings.getOutputFileName() + ".nni.ruggedness.tsv");
    treeSpaceNni2.writeRuggednessStatistics(settings.getOutputFileName() + ".nni2.ruggedness.tsv");
    treeSpaceTbr.writeRuggednessStatistics(settings.getOutputFileName() + ".tbr.ruggedness.tsv");
        
    // Markov chain Monte Carlo exploration of tree space    
    std::vector<double> powers = { 0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 };
    for (double power : powers)
        {
        std::string convergenceFileName = ".conv_NNI_" + std::to_string(power);
        std::string label = "MCMC (NNI, " + std::to_string(power) + ")";
        Mcmc mcmc1(&rng, &treeCacheNni, data, true, convergenceFileName);
        mcmc1.run(label, power, 0, nReps);

        convergenceFileName = ".conv_NNI2_" + std::to_string(power);
        label = "MCMC (NNI2, " + std::to_string(power) + ")";
        Mcmc mcmc2(&rng, &treeCacheNni2, data, true, convergenceFileName);
        mcmc2.run(label, power, 0, nReps);
        
        convergenceFileName = ".conv_TBR_" + std::to_string(power);
        label = "MCMC (TBR, " + std::to_string(power) + ")";
        Mcmc mcmc3(&rng, &treeCacheTbr, data, true, convergenceFileName);
        mcmc3.run(label, power, 0, nReps);
        
        convergenceFileName = ".conv_rTBR_" + std::to_string(power);
        label = "MCMC (rTBR, " + std::to_string(power) + ")";
        Mcmc mcmc4(&rng, &treeCacheTbr, data, true, convergenceFileName);
        mcmc4.run(label, power, 2*(data->getNumTaxa()-3), nReps);
        
        convergenceFileName = ".conv_mc3_NNI_" + std::to_string(power);
        label = "MCMCMC (NNI, " + std::to_string(power) + ")";
        Mcmc mcmcmc(&rng, &treeCacheNni, data, true, convergenceFileName);
        mcmcmc.run(label, power, 0, nReps, 4);
        }
    
    // clean up
    if (originalAlignment->getNumTaxa() > numTaxa)
        delete data;
    delete originalAlignment;
    treeCacheNni.freeTreeCache();
    treeCacheTbr.freeTreeCache();
    
    return EXIT_SUCCESS;
}

void generateNeighbors(TreeCache& treeCache, TreeNeighbors& treeNeighbors, std::string label) {

    TreeCacheMap& cache = treeCache.getCache();

    int barWidth = 60, numAsterices = 0;
    size_t numTrees = cache.size();
    size_t treeCnt = 0;

    std::cout << "   Generating " << label << " neighbors for all "
              << numTrees << " trees:" << std::endl;

    std::cout << "   * [";
    for (int i=0; i<barWidth; i++)
        {
        if ((i+1) % (int)(barWidth*0.1) == 0 && i+1 != barWidth)
            std::cout << "|";
        else
            std::cout << "-";
        }
    std::cout << "]" << std::endl;

    std::cout << "   * [";

    for (auto& [key,val] : cache)
        {
        treeNeighbors.neighbors(val->tree);

        treeCnt++;

        double progress = static_cast<double>(treeCnt) / numTrees;
        int filledWidth = static_cast<int>(progress * barWidth);

        for (int i=0; i<filledWidth-numAsterices; i++)
            std::cout << "*" << std::flush;

        numAsterices = filledWidth;
        }

    std::cout << "]" << std::endl << std::endl;
    
#   if 0
    int num = 0, sum = 0, min=(int)cache.size() + 1, max=0;
    for (auto& [key,val] : cache)
        {
        num++;
        int x = (int)val->neighbors.size();
        if (x < min) 
            min = x;
        if (x > max)
            max = x;
        sum += x;
        }
    std::cout << "Average number of neighbors = " << (double)sum / num << " (" << min << " " << max << ")" << std::endl;
#   endif
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
