#ifndef UserSettings_hpp
#define UserSettings_hpp

#include <string>



class UserSettings {

    public:
        static UserSettings&   userSettings(void)
                                    {
                                    static UserSettings userSettings;
                                    return userSettings;
                                    }
        double                  getBurnin(void) { return burnin; }
        int                     getChainLength(void) { return chainLength; }
        std::string             getInputFileName(void) { return inputFileName; }
        int                     getNumTwists(void) { return numTwists; }
        std::string             getOutputFileName(void) { return outputFileName; }
        int                     getPrintFrequency(void) { return printFrequency; }
        int                     getSampleFrequency(void) { return sampleFrequency; }
        double                  getTemperature(void) { return temperature; }
        void                    print(void);
        void                    readSettings(int argc, char* argv[]);
    
    private:
                                UserSettings(void);
                                UserSettings(UserSettings& u) { }
        void                    usage(void);
        double                  burnin;
        double                  temperature;
        int                     chainLength;
        int                     printFrequency;
        int                     sampleFrequency;
        int                     numTwists;
        std::string             inputFileName;
        std::string             outputFileName;
        bool                    settingsInitialized;
        std::string             executableName;
};

#endif
