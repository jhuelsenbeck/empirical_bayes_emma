#ifndef UserSettings_hpp
#define UserSettings_hpp

#include <string>



class UserSettings {

    public:
        static UserSettings&        userSettings(void)
                                        {
                                        static UserSettings userSettings;
                                        return userSettings;
                                        }
        double                      getBurnin(void) { return burnin; }
        bool                        getShouldAppend(void) { return appendResults; }
        std::vector<std::string>&   getTreeFiles(void) { return treeFiles; }
        std::string                 getTrueFile(void) { return trueFile; }
        std::string                 getOutputFileName(void) { return outputFileName; }
        void                        print(void);
        void                        readSettings(int argc, char* argv[]);
    
    private:
                                    UserSettings(void);
                                    UserSettings(UserSettings& u) { }
        void                        usage(void);
        double                      burnin;
        std::vector<std::string>    treeFiles;
        std::string                 trueFile;
        std::string                 outputFileName;
        bool                        appendResults;
        bool                        settingsInitialized;
        std::string                 executableName;
};

#endif
