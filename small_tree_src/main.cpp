#include <iostream>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include "Alignment.hpp"
#include "BitSetFactory.hpp"
#include "ExhaustiveSearch.hpp"
#include "LandscapeMixingCollator.hpp"
#include "LikelihoodCalculator.hpp"
#include "MapTree.hpp"
#include "MarkovChainAnalyzer.hpp"
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
std::string powerLabel(double x);


int main(int argc, char* argv[]) {

    printHeader();
    
    // instantiate random variable and thread pool objects
    RandomVariable rng;
    ThreadPool threads;

    // read the user settings
    UserSettings& settings = UserSettings::userSettings();
    settings.readSettings(argc, argv);
    settings.print();
    bool analyticsOnly = false;
    int numTaxa = 10;
    int nReps = 50;
    
    // read the alignment file
    Alignment* originalAlignment = new Alignment(settings.getInputFileName());
    Alignment* data = originalAlignment;
    if (originalAlignment->getNumTaxa() > numTaxa)
        data = new Alignment(*originalAlignment, numTaxa, &rng);
    if (settings.getNumTwists() > 0)
        data->twist(&rng, settings.getNumTwists());
    data->print(settings.getOutputFileName() + ".nex");
    data->summarize();
    data->compress();
    BitSetFactory::getFactory().initialize(data->getNumTaxa());

    // integrate branch lengths out under an IID Exp() prior (Laplace) in addition to the ML fit
    LikelihoodCalculator::setComputeMarginalLikelihood(true);  // set to false to skip the marginal and keep only the profile/ML likelihood
    LikelihoodCalculator::setExponentialPriorRate(10.0);       // change the prior rate here if needed
    LikelihoodCalculator::setMarginalHessianMethod(1);         // Hessian for the Laplace volume term: 0 = diagonal only, 1 = full via finite differences (default)

    // calculate likelihoods of all trees
    TreeCache treeCacheNni("NNI");
    TreeLikelihoods treeLikelihoods(&treeCacheNni);
    ExhaustiveSearch exhaustive(data, &treeCacheNni, &threads);
    treeCacheNni.calculatePosteriorProbabilities();            // normalize likelihoods (profile and hierarchical)
    
    // instantiate tree cache objects for NNI2 and TBR
    TreeCache treeCacheNni2("NNI2");
    treeCacheNni2.injectTreesAndLikelihoods(&treeCacheNni);
    TreeCache treeCacheTbr("TBR");
    treeCacheTbr.injectTreesAndLikelihoods(&treeCacheNni);

    // generate the neighbors for each tree under NNI, NNI2, and TBR
    TreeNeighborGeneratorNNI treeNeighborGeneratorNni(&treeCacheNni);
    TreeNeighbors treeNeighborsNni(&treeCacheNni, &treeNeighborGeneratorNni, data->getNumTaxa());
    generateNeighbors(treeCacheNni, treeNeighborsNni, "NNI");
    TreeNeighborGeneratorNNI2 treeNeighborGeneratorNni2(&treeCacheNni2, &treeCacheNni);
    TreeNeighbors treeNeighborsNni2(&treeCacheNni2, &treeNeighborGeneratorNni2, data->getNumTaxa());
    generateNeighbors(treeCacheNni2, treeNeighborsNni2, "NNI2");
    TreeNeighborGeneratorTBR treeNeighborGeneratorTbr(&treeCacheTbr);
    TreeNeighbors treeNeighborsTbr(&treeCacheTbr, &treeNeighborGeneratorTbr, data->getNumTaxa());
    generateNeighbors(treeCacheTbr, treeNeighborsTbr, "TBR");
    
    // determine the tree landscapes for NNI, NNI2, and TBR
    TreeSpace treeSpaceNni(&treeCacheNni, "NNI");
    treeSpaceNni.characterize();
    treeSpaceNni.printPosterior();
    treeSpaceNni.printPosterior(settings.getOutputFileName() + ".nni.true");
    treeSpaceNni.writeRuggednessStatistics(settings.getOutputFileName() + ".nni.ruggedness.tsv");

    TreeSpace treeSpaceNni2(&treeCacheNni2, "NNI2");
    treeSpaceNni2.characterize();
    treeSpaceNni2.printPosterior();
    treeSpaceNni2.printPosterior(settings.getOutputFileName() + ".nni2.true");
    treeSpaceNni2.writeRuggednessStatistics(settings.getOutputFileName() + ".nni2.ruggedness.tsv");

    TreeSpace treeSpaceTbr(&treeCacheTbr, "TBR");
    treeSpaceTbr.characterize();
    treeSpaceTbr.printPosterior();
    treeSpaceTbr.printPosterior(settings.getOutputFileName() + ".tbr.true");
    treeSpaceTbr.writeRuggednessStatistics(settings.getOutputFileName() + ".tbr.ruggedness.tsv");

    // find the MAP tree
    MapTree mapTree(&treeCacheNni);
            
    // analytics using the kernel of the Markov chain
    std::vector<TreeCache*> caches = { &treeCacheNni, &treeCacheNni2, &treeCacheTbr };
    std::vector<TreeSpace*> spaces = { &treeSpaceNni, &treeSpaceNni2, &treeSpaceTbr };
    std::vector<double> powers = { 0.0, 0.02, 0.05, 0.1, 0.2, 0.3 };
    bool writeSmallStateFiles = (data->getNumTaxa() <= 8);

    // Per-basin barrier table, written once. Barriers are power-independent, so this sits outside the
    // power loop; the per-tree state report is joined to it on (moveType, basinPeakId) during analysis.
    std::string basinTableFileName = settings.getOutputFileName() + ".basins.tsv";
    std::ofstream basinsOut(basinTableFileName);
    if (!basinsOut)
        throw std::runtime_error("Could not open basin table file: " + basinTableFileName);
    TreeSpace::writeBasinTableHeader(basinsOut);
    for (TreeSpace* space : spaces)
        space->writeBasinTable(basinsOut, mapTree.getMapTree());
    basinsOut.close();
    std::cout << "   Basin barrier table written to " << basinTableFileName << "\n";
    
    std::string diagnosticsFileName = settings.getOutputFileName() + ".markov.tsv";
    std::ofstream diagnosticsOut(diagnosticsFileName);
    if (!diagnosticsOut)
        throw std::runtime_error("Could not open Markov-chain diagnostics file: " + diagnosticsFileName);
    std::string efficiencyFileName = settings.getOutputFileName() + ".eff.tsv";
    std::ofstream effOut(efficiencyFileName);
    if (!effOut)
        throw std::runtime_error("Could not open Markov-chain efficiency file: " + efficiencyFileName);
    std::string stateReportFileName = settings.getOutputFileName() + ".state.tsv";
    std::ofstream stateOut(stateReportFileName);
    if (!stateOut)
        throw std::runtime_error("Could not open per-tree state-report file: " + stateReportFileName);
    MarkovChainAnalyzer::writeTsvHeader(diagnosticsOut);
    MarkovChainAnalyzer::writeEfficiencyTsvHeader(effOut);
    LandscapeMixingCollator::writeStateReportHeader(stateOut);

    for (double power : powers)
        {
        for (size_t i=0; i<caches.size(); i++)
            {
            TreeCache* c = caches[i];
            std::cout << "   Analyzing " << c->getName() << " with power " << power << "\n";

            c->sortNeighborsByLikelihood();
            c->cacheNeighborProposalProbabilities(power);

            MarkovChainAnalyzer analyzer(&threads, c, c->getName() + " (" + std::to_string(power) + ")", true); // true -> forces sparse

            // Per-tree join of the kernel's dynamics with the landscape, one row per topology, appended
            // to a single file across every move and power. This replaces the former per-move mfpt file,
            // whose name was not power-stamped and so held only the last power analyzed. The mean
            // first-passage time is a full sparse solve, so this is the cost driver of the per-power loop.
            LandscapeMixingCollator::writeStateReport(stateOut, c->getName(), power, analyzer, *spaces[i], mapTree.getMapTree());
            stateOut.flush();

            analyzer.writeEfficiencyTsvRow(effOut, c->getName(), power, "MAPtree", analyzer.efficiencyFor(analyzer.indicatorForTree(mapTree.getMapTree())));
            for (const auto& [part, trees] : mapTree.getPartitions())
                {
                std::vector<uint64_t> hashes;
                hashes.reserve(trees.size());
                for (TreeInfo* info : trees)
                    hashes.push_back(info->hash);
                analyzer.writeEfficiencyTsvRow(effOut, c->getName(), power, mapTree.partitionString(part), analyzer.efficiencyFor(analyzer.indicatorForTrees(hashes)));
                }
            analyzer.writeTsvRow(diagnosticsOut, c->getName(), power);
            diagnosticsOut.flush();
            effOut.flush();

            if (writeSmallStateFiles)
                {
                std::string prefix = settings.getOutputFileName() + "." + c->getName() + ".beta_" + powerLabel(power);
                std::cout << "   Writing small-state exact files with prefix " << prefix << "\n";
                analyzer.writeSmallStateAnalysisFiles(prefix, false, true, false);
                }
            }
        }
    std::cout << "   Markov-chain diagnostics written to " << diagnosticsFileName << "\n";
           
    if (analyticsOnly == false)
        {
        // Markov chain Monte Carlo exploration of tree space    
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
            
        std::string convergenceFileName = ".conv_Gibbs";
        std::string label = "MCMC (Gibbs)";
        Mcmc gibbs(&rng, &treeCacheNni, data, true, convergenceFileName);
        gibbs.run(label, nReps);
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

    std::cout << "   Generating " << label << " neighbors for all " << numTrees << " trees:" << std::endl;

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

std::string powerLabel(double x) {

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << x;
    std::string s = ss.str();
    for (char& c : s)
        {
        if (c == '.')
            c = 'p';
        }
    return s;
}

