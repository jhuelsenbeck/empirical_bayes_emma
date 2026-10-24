#ifndef UserSettings_hpp
#define UserSettings_hpp

#include <cstdint>
#include <string>



class UserSettings {

    public:
        static UserSettings&        userSettings(void)
                                        {
                                        static UserSettings userSettings;
                                        return userSettings;
                                        }
        double                      getBurnin(void) { return burnin; }
        std::string                 getChainStart(void) { return chainStart; }
        uint32_t                    getSeed(void) { return seed; }
        int                         getChainLength(void) { return chainLength; }
        bool                        getComputeMarginalLikelihoods(void) { return computeMarginalLikelihoods; }
        std::string                 getInputFileName(void) { return inputFileName; }
        int                         getNumTwists(void) { return numTwists; }
        std::string                 getOutputDirectoryName(void) { return outputDirectoryName; }
        std::string                 getOutputFileName(void) { return outputFileName; }
        int                         getPrintFrequency(void) { return printFrequency; }
        int                         getSampleFrequency(void) { return sampleFrequency; }   // not user-settable: every cycle is sampled
        double                      getTemperature(void) { return temperature; }
        void                        print(void);
        void                        readSettings(int argc, char* argv[]);
    
    private:
                                    UserSettings(void);
                                    UserSettings(UserSettings& u) { }
        void                        usage(void);
        double                      burnin;
        double                      temperature;
        int                         chainLength;
        int                         printFrequency;
        int                         sampleFrequency;
        int                         numTwists;
        bool                        computeMarginalLikelihoods;    // Laplace marginal per tree (-l); off by default
        uint32_t                    seed;                          // random number seed (-r); drawn at random if not given
        std::string                 chainStart;                    // initial trees of the MCMC replicates (-c)
        std::string                 inputFileName;
        std::string                 outputFileName;
        std::string                 outputDirectoryName;
        std::string                 outputFileNameStub;
        bool                        settingsInitialized;
        std::string                 executableName;
};

#endif
