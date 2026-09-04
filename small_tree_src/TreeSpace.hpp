#ifndef TreeSpace_hpp
#define TreeSpace_hpp

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "TreeCache.hpp"

struct TreeSpaceNode {

    uint64_t                    treeHash;
    float                       lnL;
    int                         peakId;
    TreeSpaceNode*              peakPtr;
    int                         distance;
    bool                        isPeak;
    std::set<TreeSpaceNode*>    neighbors;
};

struct PeakInfo {

    uint64_t                    treeHash;
    float                       lnL;
    float                       minLnL;
    int                         peakId;
    int                         basinSize;
};

struct LocalSummary {
    size_t              numTrees = 0;
    size_t              numEdges = 0;
    double              meanDegree = 0.0;
    double              varDegree = 0.0;
    double              meanUphillFraction = 0.0;
    double              posteriorWeightedMeanUphillFraction = 0.0;
    double              meanBestNeighborLnLDifference = 0.0;
    double              posteriorWeightedMeanBestNeighborLnLDifference = 0.0;
    double              meanNeighborAbsLnLDifference = 0.0;
    double              posteriorWeightedMeanNeighborAbsLnLDifference = 0.0;
    double              posteriorMassOfLocalPeaks = 0.0;
    size_t              numLocalPeaks = 0;
};

struct BasinSummary {
    size_t              numPeaks = 0;
    size_t              numPeaksMassGreater01 = 0;
    size_t              numPeaksMassGreater05 = 0;
    int                 largestBasinSize = 0;
    double              largestBasinMass = 0.0;
    double              basinMassEntropy = 0.0;
    double              effectiveNumBasins = 0.0;
    double              mapBasinMass = 0.0;
    double              posteriorMassOutsideMapBasin = 0.0;
    uint64_t            mapTreeHash = 0;
    uint64_t            mapPeakHash = 0;
    int                 mapPeakId = -1;
};

struct AscentSummary {
    double              meanAscentLength = 0.0;
    double              posteriorWeightedMeanAscentLength = 0.0;
    double              meanAscentLengthCredible95 = 0.0;
    double              posteriorWeightedMeanAscentLengthCredible95 = 0.0;
    int                 maxAscentLength = 0;
};

struct CredibleComponentSummary {
    size_t              credible95NumTrees = 0;
    double              credible95Mass = 0.0;
    size_t              credible95NumComponents = 0;
    size_t              credible95LargestComponentSize = 0;
    double              credible95LargestComponentMass = 0.0;
    size_t              credible95MapComponentSize = 0;
    double              credible95MapComponentMass = 0.0;
};

struct SaddleInfo {
    int                 peakId1 = -1;
    int                 peakId2 = -1;
    uint64_t            peakHash1 = 0;
    uint64_t            peakHash2 = 0;
    uint64_t            treeHash1 = 0;
    uint64_t            treeHash2 = 0;
    double              saddleLnL = -std::numeric_limits<double>::infinity();
    double              saddleProbability = 0.0;
    double              barrierFromPeak1 = 0.0;
    double              barrierFromPeak2 = 0.0;
};

struct BarrierSummary {
    size_t              basinGraphEdges = 0;
    double              meanBasinGraphDegree = 0.0;
    int                 mapBasinGraphDegree = 0;
    bool                basinGraphConnected = false;
    double              minBarrierFromMap = 0.0;
    double              meanBarrier = 0.0;
    double              posteriorWeightedMeanBarrier = 0.0;
    double              maxBarrier = 0.0;
};

struct TreeLandscapeRecord {
    double              logLikelihood = 0.0;
    double              posteriorProbability = 0.0;
    int                 basinPeakId = -1;
    uint64_t            basinPeakHash = 0;
    double              basinPosteriorMass = 0.0;
    int                 basinSize = 0;
    bool                isLocalPeak = false;
    int                 graphDistanceToMap = -1;
    bool                inCredible95 = false;
};

class Peak;
class TreeSamples;
typedef std::unordered_map<uint64_t,TreeSpaceNode*> TreeNodesMap;
typedef std::unordered_map<uint64_t,double> TreeProbMap;
typedef std::unordered_map<uint64_t,Peak*> PeakMap;
typedef std::unordered_map<uint64_t,TreeLandscapeRecord> TreeLandscapeMap;



class TreeSpace {

    public:
                                TreeSpace(void) = delete;
                                TreeSpace(TreeCache* tc, std::string st);
                               ~TreeSpace(void);
        void                    characterize(void);
        Peak*                   findPeak(uint64_t treeHash);
        Peak*                   findPeakForTreeWithHash(uint64_t treeHash);
        Peak*                   findPeakWithId(int id);
        TreeSpaceNode*          getTree(uint64_t treeHash);
        double                  getTreeProbabiity(uint64_t treeHash);

                                // A per-tree landscape record for every vertex, keyed by tree hash, so the
                                // dynamical quantities computed from the kernel (mean first-passage time,
                                // return time, leave probability) can be joined to the tree it describes.
                                // Basin identity, basin mass, credible-set membership, and the graph
                                // distance from the MAP tree are all properties of the posterior and the
                                // move-neighbour graph, so they do not depend on the proposal power and are
                                // built once and cached. The distance is a breadth-first distance from the
                                // supplied MAP tree; the record is rebuilt only if the MAP tree or credible
                                // mass changes.
        const TreeLandscapeMap& landscapeByHash(uint64_t mapHash, double credibleMass = 0.95);

                                // Per-basin barrier table. For every basin, the barrier that
                                // separates it from the MAP basin: the descent from the MAP peak to
                                // the level of the lowest pass on the easiest route back to the MAP,
                                // found by a widest-path flood of the saddle graph. Written in log
                                // likelihood units, which for a flat prior over topologies are the
                                // log-probability barrier. Barriers are a property of the posterior
                                // surface and the neighbour graph, so they do not depend on power.
        static void             writeBasinTableHeader(std::ostream& os);
        void                    writeBasinTable(std::ostream& os, uint64_t mapHash);
        int                     graphDistance(const TreeSpaceNode* a, const TreeSpaceNode* b);
        void                    printPosterior(void);
        void                    printPosterior(std::string fileName);
        void                    printPosterior(TreeSamples* samples);
        void                    writeRuggednessStatistics(std::string fileName);
    
    private:
        void                    adjacentTreeDistances(uint64_t treeHash);
        void                    calculateDistances(uint64_t treeHash);
        void                    characterizeBasins(std::unordered_map<uint64_t,int>& basins);
        void                    clearPeaks(void);
        LocalSummary            computeLocalSummary(void);
        BasinSummary            computeBasinSummary(void);
        AscentSummary           computeAscentSummary(void);
        CredibleComponentSummary computeCredible95ComponentSummary(void);
        BarrierSummary          computeBarrierSummary(std::vector<SaddleInfo>& saddleInfo);
        std::unordered_map<uint64_t,uint64_t> computePeakAssignments(void);
        std::vector<uint64_t>    credibleSet(double probability);
        TreeSpaceNode*          findBestNeighbor(TreeSpaceNode* n);
        int                     peakIdForTreeWithHash(uint64_t treeHash);
        int                     steepestAscentLength(uint64_t startTree);
        uint64_t                steepestAscent(uint64_t startTree);
        TreeCache*              treeCache;
        TreeNodesMap            treeNodes;
        PeakMap                 peaks;
        TreeProbMap             treeProbabilities;
        std::string             swapType;
        double                  averageDegree;
        double                  varianceDegree;
        TreeLandscapeMap        landscapeRecords;
        bool                    landscapeReady = false;
        uint64_t                landscapeMapHash = 0;
        double                  landscapeCredibleMass = 0.0;
};

#endif
