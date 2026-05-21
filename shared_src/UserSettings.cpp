#include <iostream>
#include <string>
#include <vector>
#include "Msg.hpp"
#include "UserSettings.hpp"

#undef DEBUG_MODE



UserSettings::UserSettings(void) {

    executableName = "";
    settingsInitialized = false;
    numChains = 1;
    temperature = 0.2;
    chainLength = 1000000;
    printFrequency = 100;
    sampleFrequency = 100;
    inputFileName = "";
    outputFileName = "";
    burnin = 0.0;
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
            if (key == "-i")
                inputFileName = cmd;
            else if (key == "-o")
                outputFileName = cmd;
            else if (key == "-n")
                chainLength = stod(cmd);
            else if (key == "-m")
                numChains = stod(cmd);
            else if (key == "-t")
                temperature = stof(cmd);
            else if (key == "-d")
                burnin = stof(cmd);
            else if (key == "-p")
                printFrequency = stod(cmd);
            else if (key == "-s")
                sampleFrequency = stod(cmd);
            else
                Msg::error("Improperly formatted commands");
            readingKey = true;
            }
        }
        
    settingsInitialized = true;
}

void UserSettings::print(void) {

    std::cout << "   User settings:" << std::endl;
    std::cout << "   * Input file name (-i)            = " << inputFileName << std::endl;
    std::cout << "   * Output file name (-o)           = " << outputFileName << std::endl;
    std::cout << "   * Number of chains (-m)           = " << numChains << std::endl;
    std::cout << "   * Temperature (-t)                = " << temperature << std::endl;
    std::cout << "   * Chain length (-n)               = " << chainLength << std::endl;
    std::cout << "   * Burn in fraction (-d)           = " << burnin << std::endl;
    std::cout << "   * Print frequency (-p)            = " << printFrequency << std::endl;
    std::cout << "   * Sample frequency (-s)           = " << sampleFrequency << std::endl;
    std::cout << std::endl;
}

void UserSettings::usage(void) {

    std::cout << "-i        Input sequence file name" << std::endl;
    std::cout << "-o        Output file name" << std::endl;
    std::cout << "-m        Number of chains for MCMCMC" << std::endl;
    std::cout << "-t        Temperature parameter for MCMCMC" << std::endl;
    std::cout << "-n        MCMC chain length" << std::endl;
    std::cout << "-d        MCMC burn-in fraction" << std::endl;
    std::cout << "-p        Print to screen frequency" << std::endl;
    std::cout << "-s        Chain sample frequency" << std::endl;
    Msg::error("Incorrectly formatted command line input");
}
