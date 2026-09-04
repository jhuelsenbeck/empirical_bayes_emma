#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <vector>
#include "LandscapeMixingCollator.hpp"
#include "MarkovChainAnalyzer.hpp"
#include "TreeSpace.hpp"



void LandscapeMixingCollator::writeStateReportHeader(std::ostream& os) {

    os << "moveType"
       << "\tpower"
       << "\tstateIndex"
       << "\ttreeHash"
       << "\tlogLikelihood"
       << "\tposteriorProbability"
       << "\tisMapTree"
       << "\tbasinPeakId"
       << "\tbasinPeakHash"
       << "\tbasinPosteriorMass"
       << "\tbasinSize"
       << "\tisLocalPeak"
       << "\tgraphDistanceToMap"
       << "\tinCredible95"
       << "\tmeanFirstPassageTimeToMap"
       << "\tmeanReturnTime"
       << "\tleaveProbability"
       << "\n";
}

void LandscapeMixingCollator::writeStateReport(std::ostream& os, const std::string& moveType, double power,
                                               const MarkovChainAnalyzer& analyzer, TreeSpace& space,
                                               uint64_t mapHash, double credibleMass) {

    // The landscape half, built once per move and cached; identical across proposal powers.
    const TreeLandscapeMap& landscape = space.landscapeByHash(mapHash, credibleMass);

    // The dynamical half, specific to this power's kernel. Mean first-passage time is one sparse
    // solve; the other two are read directly off the stationary distribution and the kernel diagonal.
    const MarkovChainAnalyzer::Vector& pi = analyzer.getPosterior();
    MarkovChainAnalyzer::Vector mfpt      = analyzer.meanFirstPassageTimesToTree(mapHash);
    MarkovChainAnalyzer::Vector leave     = analyzer.leaveProbabilities();
    const std::vector<uint64_t>& hashes   = analyzer.getStateHashes();

    Eigen::Index N = static_cast<Eigen::Index>(analyzer.numStates());
    bool haveMfpt = (mfpt.size() == N);

    double nan = std::numeric_limits<double>::quiet_NaN();

    os << std::setprecision(10);
    for (Eigen::Index i = 0; i < N; ++i)
        {
        uint64_t h = (static_cast<size_t>(i) < hashes.size()) ? hashes[static_cast<size_t>(i)] : 0;

        double   logLike     = nan;
        int      basinId     = -1;
        uint64_t basinHash   = 0;
        double   basinMass   = nan;
        int      basinSize   = 0;
        bool     isLocalPeak = false;
        int      distToMap   = -1;
        bool     inCred      = false;

        TreeLandscapeMap::const_iterator it = landscape.find(h);
        if (it != landscape.end())
            {
            const TreeLandscapeRecord& rec = it->second;
            logLike     = rec.logLikelihood;
            basinId     = rec.basinPeakId;
            basinHash   = rec.basinPeakHash;
            basinMass   = rec.basinPosteriorMass;
            basinSize   = rec.basinSize;
            isLocalPeak = rec.isLocalPeak;
            distToMap   = rec.graphDistanceToMap;
            inCred      = rec.inCredible95;
            }

        double post       = pi(i);
        double meanReturn  = (post > 0.0) ? 1.0 / post : nan;
        double mfptToMap   = haveMfpt ? mfpt(i) : nan;

        os << moveType
           << "\t" << power
           << "\t" << i
           << "\t" << h
           << "\t" << logLike
           << "\t" << post
           << "\t" << ((h == mapHash) ? 1 : 0)
           << "\t" << basinId
           << "\t" << basinHash
           << "\t" << basinMass
           << "\t" << basinSize
           << "\t" << (isLocalPeak ? 1 : 0)
           << "\t" << distToMap
           << "\t" << (inCred ? 1 : 0)
           << "\t" << mfptToMap
           << "\t" << meanReturn
           << "\t" << leave(i)
           << "\n";
        }
}
