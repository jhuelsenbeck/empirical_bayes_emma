#ifndef LandscapeMixingCollator_hpp
#define LandscapeMixingCollator_hpp

#include <cstdint>
#include <iosfwd>
#include <string>
class MarkovChainAnalyzer;
class TreeSpace;



class LandscapeMixingCollator {

    public:
                        LandscapeMixingCollator(void) = delete;
        static void     writeStateReportHeader(std::ostream& os);
        static void     writeStateReport(std::ostream& os, const std::string& moveType, double power, const MarkovChainAnalyzer& analyzer, TreeSpace& space, uint64_t mapHash, double credibleMass = 0.95);
};

#endif
