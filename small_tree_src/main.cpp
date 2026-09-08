#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <system_error>
#include <iomanip>
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include "Alignment.hpp"
#include "BitSetFactory.hpp"
#include "ExhaustiveSearch.hpp"
#include "LandscapeMixingCollator.hpp"
#include "LikelihoodCalculator.hpp"
#include "MapTree.hpp"
#include "MarkovChainAnalyzer.hpp"
#include "Mcmc.hpp"
#include "Msg.hpp"
#include "RandomVariable.hpp"
#include "Threads.hpp"
#include "TreeCache.hpp"
#include "TreeLikelihoods.hpp"
#include "TreeNeighborGenerator.hpp"
#include "TreeNeighbors.hpp"
#include "TreeSpace.hpp"
#include "UserSettings.hpp"

void ensureOutputDirectory(const std::string& outputPath);
std::multimap<double, Alignment*> generateBadLandscapes(RandomVariable* rng, ThreadPool* threads, Alignment* originalAlignment, int numReplicates, int numTwists);
void generateNeighbors(TreeCache& treeCache, TreeNeighbors& generator, std::string label);
void printHeader(void);
std::string powerLabel(double x);
static void recomputeLikelihoods(TreeCache& cache, Alignment* alignment, ThreadPool* threads);



int main(int argc, char* argv[]) {

    printHeader();

    // instantiate random variable and thread pool objects
    RandomVariable rng;
    ThreadPool threads;

    // read the user settings
    UserSettings& settings = UserSettings::userSettings();
    settings.readSettings(argc, argv);
    settings.print();
    ensureOutputDirectory(settings.getOutputDirectoryName());
    bool analyticsOnly = false;
    int numTaxa = 10;
    int nReps = 50;

    // read the alignment file
    Alignment* originalAlignment = new Alignment(settings.getInputFileName());
    Alignment* data = originalAlignment;
    if (originalAlignment->getNumTaxa() > numTaxa)
        data = new Alignment(*originalAlignment, numTaxa, &rng);
    if (settings.getNumTwists() > 1)
        {
        std::multimap<double, Alignment*> twistedAlignments = generateBadLandscapes(&rng, &threads, data, 100, settings.getNumTwists());

        data = twistedAlignments.rbegin()->second;
        std::cout << "   * Selecting alignment with " << twistedAlignments.rbegin()->first << " peaks (TBR)" << std::endl;
        for (auto [key,val] : twistedAlignments)
            delete val;
        if (originalAlignment->getNumTaxa() > numTaxa)
            delete data;
        delete originalAlignment;
        std::cout << "   Completed generating twisted data matrices" << std::endl;
        return EXIT_SUCCESS;
        }
    BitSetFactory::getFactory().initialize(data->getNumTaxa());
    data->print(settings.getOutputFileName() + ".nex");
    data->summarize();
    data->compress();
        
    // integrate branch lengths out under an IID Exp() prior (Laplace) in addition to the ML fit
    LikelihoodCalculator::setComputeMarginalLikelihood(true);
    LikelihoodCalculator::setExponentialPriorRate(10.0);
    LikelihoodCalculator::setMarginalHessianMethod(1);

    // Build the canonical tree set once (in the NNI cache) and inject the trees + likelihoods into the
    // NNI2 and TBR caches. Each cache keeps its own copies because the move-set neighborhoods differ.
    TreeCache treeCacheNni("NNI");
    TreeLikelihoods treeLikelihoods(&treeCacheNni);
    ExhaustiveSearch exhaustive(data, &treeCacheNni, &threads);
    treeCacheNni.calculatePosteriorProbabilities();

    TreeCache treeCacheNni2("NNI2");
    treeCacheNni2.injectTreesAndLikelihoods(&treeCacheNni);
    TreeCache treeCacheTbr("TBR");
    treeCacheTbr.injectTreesAndLikelihoods(&treeCacheNni);

    // Generate every move-set's neighbors up front. NNI2 is built from the NNI neighborhoods, so the
    // NNI neighbors must still exist when NNI2 is generated, which is why all neighbor generation
    // happens here, before any cache is released below.
    TreeNeighborGeneratorNNI treeNeighborGeneratorNni(&treeCacheNni);
    TreeNeighbors treeNeighborsNni(&treeCacheNni, &treeNeighborGeneratorNni, data->getNumTaxa());
    generateNeighbors(treeCacheNni, treeNeighborsNni, "NNI");

    TreeNeighborGeneratorNNI2 treeNeighborGeneratorNni2(&treeCacheNni2, &treeCacheNni);
    TreeNeighbors treeNeighborsNni2(&treeCacheNni2, &treeNeighborGeneratorNni2, data->getNumTaxa());
    generateNeighbors(treeCacheNni2, treeNeighborsNni2, "NNI2");

    TreeNeighborGeneratorTBR treeNeighborGeneratorTbr(&treeCacheTbr);
    TreeNeighbors treeNeighborsTbr(&treeCacheTbr, &treeNeighborGeneratorTbr, data->getNumTaxa());
    generateNeighbors(treeCacheTbr, treeNeighborsTbr, "TBR");

    // MAP tree. Its partitions are read from the Tree objects, so build it before freeing them, and
    // snapshot the MAP hash and each split's member-tree hashes into cache-independent structures so
    // the per-move analytics below do not depend on any particular cache (or its trees) staying alive.
    MapTree mapTree(&treeCacheNni);
    uint64_t mapHash = mapTree.getMapTree();
    std::vector<std::pair<std::string, std::vector<uint64_t>>> partitionHashLists;
    for (const auto& [part, trees] : mapTree.getPartitions())
        {
        std::vector<uint64_t> hashes;
        hashes.reserve(trees.size());
        for (TreeInfo* info : trees)
            hashes.push_back(info->hash);
        partitionHashLists.emplace_back(mapTree.partitionString(part), std::move(hashes));
        }

    // Release the heavyweight Tree objects now. Neighbor generation and the MAP-tree partitions are
    // the only consumers of TreeInfo::tree; the kernel analysis and the profile MCMC read only hashes,
    // neighbour pointers, and cached likelihoods. At ten taxa these objects (held across three caches)
    // are the dominant memory cost, and freeing them here keeps the analytics phase's working set in
    // RAM instead of thrashing the page compressor.
    treeCacheNni.freeTreeObjects();
    treeCacheNni2.freeTreeObjects();
    treeCacheTbr.freeTreeObjects();

    // Output files (shared across all moves and powers; rows are self-identifying via their columns).
    std::string basinTableFileName = settings.getOutputFileName() + ".basins.tsv";
    std::ofstream basinsOut(basinTableFileName);
    if (!basinsOut)
        throw std::runtime_error("Could not open basin table file: " + basinTableFileName);
    TreeSpace::writeBasinTableHeader(basinsOut);

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

    std::vector<double> powers = { 0.0, 0.02, 0.05, 0.1, 0.2, 0.3 };
    bool writeSmallStateFiles = (data->getNumTaxa() <= 8);

    // The exact analytics for one move set: landscape, basin table, and the per-power kernel diagnostics.
    auto analyzeMove = [&](TreeCache* c, const std::string& moveName, const std::string& fileTag) 
        {
        TreeSpace space(c, moveName);
        space.characterize();
        space.printPosterior();
        space.printPosterior(settings.getOutputFileName() + "." + fileTag + ".true");
        space.writeRuggednessStatistics(settings.getOutputFileName() + "." + fileTag + ".ruggedness.tsv");
        space.writeBasinTable(basinsOut, mapHash);
        basinsOut.flush();

        for (double power : powers)
            {
            std::cout << "   Analyzing " << moveName << " with power " << power << "\n";

            c->sortNeighborsByLikelihood();
            c->cacheNeighborProposalProbabilities(power);

            MarkovChainAnalyzer analyzer(&threads, c, moveName + " (" + std::to_string(power) + ")", true);

            LandscapeMixingCollator::writeStateReport(stateOut, moveName, power, analyzer, space, mapHash);
            stateOut.flush();

            analyzer.writeEfficiencyTsvRow(effOut, moveName, power, "MAPtree",
                                           analyzer.efficiencyFor(analyzer.indicatorForTree(mapHash)));
            for (const auto& [label, hashes] : partitionHashLists)
                analyzer.writeEfficiencyTsvRow(effOut, moveName, power, label,
                                               analyzer.efficiencyFor(analyzer.indicatorForTrees(hashes)));

            analyzer.writeTsvRow(diagnosticsOut, moveName, power);
            diagnosticsOut.flush();
            effOut.flush();

            if (writeSmallStateFiles)
                {
                std::string prefix = settings.getOutputFileName() + "." + moveName + ".beta_" + powerLabel(power);
                analyzer.writeSmallStateAnalysisFiles(prefix, false, true, false);
                }
            }
        };

    // Process each move set to completion -- analytics, then its MCMC runs -- and free its cache before
    // moving on, so beyond the shared tree metadata only one move's neighbour/kernel data is ever live.

    // NNI (also carries the Metropolis-coupled and Gibbs runs, which sample the NNI cache)
    analyzeMove(&treeCacheNni, "NNI", "nni");
    if (analyticsOnly == false)
        {
        for (double power : powers)
            {
            std::string cfn = ".conv_NNI_" + std::to_string(power);
            Mcmc mcmc(&rng, &treeCacheNni, data, true, cfn);
            mcmc.run("MCMC (NNI, " + std::to_string(power) + ")", power, 0, nReps);

            cfn = ".conv_mc3_NNI_" + std::to_string(power);
            Mcmc mcmcmc(&rng, &treeCacheNni, data, true, cfn);
            mcmcmc.run("MCMCMC (NNI, " + std::to_string(power) + ")", power, 0, nReps, 4);
            }
        Mcmc gibbs(&rng, &treeCacheNni, data, true, ".conv_Gibbs");
        gibbs.run("MCMC (Gibbs)", nReps);
        }
    treeCacheNni.freeTreeCache();

    // NNI2
    analyzeMove(&treeCacheNni2, "NNI2", "nni2");
    if (analyticsOnly == false)
        {
        for (double power : powers)
            {
            std::string cfn = ".conv_NNI2_" + std::to_string(power);
            Mcmc mcmc(&rng, &treeCacheNni2, data, true, cfn);
            mcmc.run("MCMC (NNI2, " + std::to_string(power) + ")", power, 0, nReps);
            }
        }
    treeCacheNni2.freeTreeCache();

    // TBR (also carries the rTBR multiple-try runs, which sample the TBR cache)
    analyzeMove(&treeCacheTbr, "TBR", "tbr");
    if (analyticsOnly == false)
        {
        for (double power : powers)
            {
            std::string cfn = ".conv_TBR_" + std::to_string(power);
            Mcmc mcmc(&rng, &treeCacheTbr, data, true, cfn);
            mcmc.run("MCMC (TBR, " + std::to_string(power) + ")", power, 0, nReps);

            cfn = ".conv_rTBR_" + std::to_string(power);
            Mcmc rtbr(&rng, &treeCacheTbr, data, true, cfn);
            rtbr.run("MCMC (rTBR, " + std::to_string(power) + ")", power, 2 * (data->getNumTaxa() - 3), nReps);
            }
        }
    treeCacheTbr.freeTreeCache();

    basinsOut.close();
    std::cout << "   Basin barrier table written to " << basinTableFileName << "\n";
    std::cout << "   Markov-chain diagnostics written to " << diagnosticsFileName << "\n";

    // clean up
    if (originalAlignment->getNumTaxa() > numTaxa)
        delete data;
    delete originalAlignment;

    return EXIT_SUCCESS;
}

void ensureOutputDirectory(const std::string& outputDir) {

    if (outputDir.empty())
        return;

    std::filesystem::path dir(outputDir);        // the whole path is the directory; nothing is stripped

    std::error_code ec;
    if (std::filesystem::exists(dir, ec))
        {
        if (!std::filesystem::is_directory(dir, ec))
            Msg::error("Output path exists but is not a directory: " + dir.string());
        return;
        }

    if (!std::filesystem::create_directories(dir, ec) || ec)
        Msg::error("Could not create output directory '" + dir.string() + "': " + ec.message());

    std::cout << "   Created output directory " << dir.string() << std::endl;
}

std::multimap<double, Alignment*> generateBadLandscapes(RandomVariable* rng, ThreadPool* threads, Alignment* originalAlignment, int numReplicates, int numTwists) {

    std::cout << "   Generating twisted landscapes" << std::endl;
    std::cout << "   * Number of twists: " << numTwists << std::endl;
    std::cout << "   * Number of landscapes: " << numReplicates << std::endl << std::endl;
    // re-enable the marginal later, on the few winners.
    LikelihoodCalculator::setComputeMarginalLikelihood(false);

    int numTaxa = originalAlignment->getNumTaxa();

    // fix the taxon set once, so every replicate shares the same enumerated trees and TBR neighbourhoods
    Alignment* base = new Alignment(*originalAlignment, numTaxa, rng);
    BitSetFactory::getFactory().initialize(base->getNumTaxa());

    // enumerate the trees and build the TBR neighbourhoods once
    TreeCache cache("TBR");
    TreeLikelihoods treeLikelihoods(&cache);
    ExhaustiveSearch exhaustive(base, &cache, threads);            // enumerates + an initial (discarded) pass
    TreeNeighborGeneratorTBR gen(&cache);
    TreeNeighbors neighbors(&cache, &gen, base->getNumTaxa());
    generateNeighbors(cache, neighbors, "TBR");

    std::multimap<double, Alignment*> byBasins; // key: effectiveNumBasins (TBR)

    for (int i=0; i<numReplicates; i++)
        {
        Alignment* a = new Alignment(*base);
        a->twist(rng, numTwists);
        a->compress();

        recomputeLikelihoods(cache, a, threads);
        cache.calculatePosteriorProbabilities();

        TreeSpace space(&cache, "TBR");
        space.characterize();
        BasinSummary bs = space.basinSummary();

        byBasins.insert( {bs.effectiveNumBasins, a} );
        std::cout << "   [landscape screen] " << (i + 1) << "/" << numReplicates
                  << ": effectiveNumBasins=" << bs.effectiveNumBasins
                  << "  peaks(mass>0.05)=" << bs.numPeaksMassGreater05
                  << "  outsideMapBasin=" << bs.posteriorMassOutsideMapBasin << "\n";

        char temp[20];
        snprintf(temp, sizeof(temp), "%1.6lf", bs.effectiveNumBasins);
        std::string numPeaksStr(temp);
        std::string fileName = UserSettings::userSettings().getOutputFileName() + "_" + std::to_string(i+1) + "_" + numPeaksStr + ".nex";
        a->print(fileName);
        }

    delete base;
    cache.freeTreeCache();
    
    return byBasins;
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

// Recompute the profile log likelihood of every tree already in the cache under a new alignment,
// reusing the existing Tree topologies (which do not depend on the sequence data). This mirrors the
// likelihood pass of ExhaustiveSearch::enumerateAllTrees, without re-enumerating.
static void recomputeLikelihoods(TreeCache& cache, Alignment* alignment, ThreadPool* threads) {

    TreeCacheMap& tCache = cache.getCache();
    size_t numTrees = tCache.size();
    size_t maxJobs  = threads->getQueueCapacity();
    if (numTrees < maxJobs)
        maxJobs = numTrees;

    std::vector<LikelihoodCalculator> calculators;
    calculators.reserve(maxJobs);
    for (size_t i = 0; i < maxJobs; i++)
        calculators.emplace_back(alignment);

    size_t cnt = 0;
    auto harvest = [&](void) {
        threads->wait();
        for (size_t i = 0; i < cnt; i++)
            {
            calculators[i].getTreeInfo()->lnLikelihood    = calculators[i].getResult();
            calculators[i].getTreeInfo()->hasLnLikelihood = true;
            }
        cnt = 0;
    };

    for (auto& [key, val] : tCache)
        {
        if (val == nullptr || val->tree == nullptr)
            continue;
        calculators[cnt].setTree(val->tree);
        calculators[cnt].setTreeInfo(val);
        calculators[cnt].setOffset(0);
        threads->pushTask(&calculators[cnt]);
        if (++cnt == maxJobs)
            harvest();
        }
    if (cnt > 0)
        harvest();
}

