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
        std::string             getInputTreeFileName(void) { return inputTreeFileName; }
        std::string             getOutputFileName(void) { return outputFileName; }
        int                     getPrintFrequency(void) { return printFrequency; }
        int                     getSampleFrequency(void) { return sampleFrequency; }
        void                    print(void);
        void                    readSettings(int argc, char* argv[]);
    
    private:
                                UserSettings(void);
                                UserSettings(UserSettings& u) { }
        void                    usage(void);
        double                  burnin;
        int                     chainLength;
        int                     printFrequency;
        int                     sampleFrequency;
        std::string             inputFileName;
        std::string             inputTreeFileName;
        std::string             outputFileName;
        bool                    settingsInitialized;
        std::string             executableName;
};

#endif
