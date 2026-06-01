#include <iostream>
#include <string>
#include <vector>
#include "Msg.hpp"
#include "UserSettings.hpp"

#undef DEBUG_MODE



UserSettings::UserSettings(void) {

    executableName = "";
    settingsInitialized = false;
    trueFile = "";
    outputFileName = "";
    burnin = 0.0;
    appendResults = false;
}

void UserSettings::readSettings(int argc, char* argv[]) {

    executableName = argv[0];

    if (settingsInitialized == true)
        {
        Msg::warning("User settings already read. Re-reading settings not allowed");
        return;
        }

    // set up a vector containing the commands
    std::vector<std::string> userCommands;
    for (int i=0; i<argc; i++)
        userCommands.push_back( argv[i] );
        
    // set default values
    
    // check the vector of commands
    if (userCommands.size() / 2 == 0)
        usage();
        
    if (userCommands.size() == 1)
        usage();
        
    // parse the user commands
    bool readingKey = true;
    std::string key = "";
    for (int i=1; i<userCommands.size(); i++)
        {
        std::string cmd = userCommands[i];
        if (readingKey == true)
            {
            key = cmd;
            readingKey = false;
            }
        else
            {
            if (key == "-t")
                treeFiles.push_back(cmd);
            else if (key == "-p")
                trueFile = cmd;
            else if (key == "-o")
                outputFileName = cmd;
            else if (key == "-b")
                burnin = stof(cmd);
            else if (key == "-a")
                {
                if (cmd[0] == 't' || cmd[0] == 'T')
                    appendResults = true;
                else
                    appendResults = false;
                }
            else
                Msg::error("Improperly formatted commands");
            readingKey = true;
            }
        }
        
    settingsInitialized = true;
}

void UserSettings::print(void) {

    std::cout << "   User settings:" << std::endl;
    for (int i=0; i<treeFiles.size(); i++)
        std::cout << "   * Tree file name (-t)             = " << treeFiles[i] << std::endl;
    std::cout << "   * True probs file name (-p)       = " << trueFile << std::endl;
    std::cout << "   * Output file name (-o)           = " << outputFileName << std::endl;
    std::cout << "   * Burn in fraction (-b)           = " << burnin << std::endl;
    std::cout << std::endl;
}

void UserSettings::usage(void) {

    std::cout << "-t        Tree file name" << std::endl;
    std::cout << "-p        True probabilities file name" << std::endl;
    std::cout << "-o        Output file name" << std::endl;
    std::cout << "-b        MCMC burn-in fraction" << std::endl;
    Msg::error("Incorrectly formatted command line input");
}
