#ifndef TreeSpace_hpp
#define TreeSpace_hpp

#include <cstdint>
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
class Peak;
class TreeSamples;
typedef std::unordered_map<uint64_t,TreeSpaceNode*> TreeNodesMap;
typedef std::unordered_map<uint64_t,double> TreeProbMap;
typedef std::unordered_map<uint64_t,Peak*> PeakMap;



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
        int                     graphDistance(const TreeSpaceNode* a, const TreeSpaceNode* b);
        void                    printPosterior(void);
        void                    printPosterior(std::string fileName);
        void                    printPosterior(TreeSamples* samples);
        void                    writeRuggednessStatistics(std::string fileName);
    
    private:
        struct LocalSummary {
            size_t              numTrees = 0;
            size_t              numEdges = 0;
            double              meanDegree = 0.0;
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
        int                     steepestAscentLength(uint64_t startTree);
        uint64_t                steepestAscent(uint64_t startTree);
        TreeCache*              treeCache;
        TreeNodesMap            treeNodes;
        PeakMap                 peaks;
        TreeProbMap             treeProbabilities;
        std::string             swapType;
        double                  averageDegree;
};

#endif
