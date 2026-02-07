#include <iostream>
#include <string>
#include <vector>
#include "Msg.hpp"
#include "UserSettings.hpp"

#undef DEBUG_MODE



UserSettings::UserSettings(void) {

    executableName = "";
    settingsInitialized = false;
    chainLength = 1000000;
    printFrequency = 100;
    sampleFrequency = 100;
    inputFileName = "";
    inputTreeFileName = "";
    outputFileName = "";
    calculateMarginalLikelihood = false;
    brlenLambda = 10.0;
    shapeLambda = 2.0;
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
            else if (key == "-t")
                inputTreeFileName = cmd;
            else if (key == "-o")
                outputFileName = cmd;
            else if (key == "-n")
                chainLength = stod(cmd);
            else if (key == "-d")
                burnin = stof(cmd);
            else if (key == "-p")
                printFrequency = stod(cmd);
            else if (key == "-s")
                sampleFrequency = stod(cmd);
            else if (key == "-m")
                {
                if (cmd == "true")
                    calculateMarginalLikelihood = true;
                else if (cmd == "false")
                    calculateMarginalLikelihood = false;
                else
                    Msg::error("Could not interpret command \"" + cmd + "\" for the marginal likelihood option");
                }
            else if (key == "-b")
                brlenLambda = stof(cmd);
            else if (key == "-a")
                shapeLambda = stof(cmd);
            else
                Msg::error("Improperly formatted commands");
            readingKey = true;
            }
        }
        
    settingsInitialized = true;
}

void UserSettings::print(void) {

    std::vector<std::string> codes = { "Universal Genetic Code", "Vertebrate Mitochondrial Code", "Mycoplasma Genetic Code", "Yeast Genetic Code", "Ciliate Genetic Code", "Metazoan Mitochondrial Genetic Code" };

    std::cout << "   * Executable path/name            = " << executableName << std::endl;
    std::cout << "   * Input file name (-i)            = " << inputFileName << std::endl;
    std::cout << "   * Input tree file name (-t)       = " << inputTreeFileName << std::endl;
    std::cout << "   * Output file name (-o)           = " << outputFileName << std::endl;
    std::cout << "   * Chain length (-n)               = " << chainLength << std::endl;
    std::cout << "   * Burn in fraction (-d)           = " << burnin << std::endl;
    std::cout << "   * Print frequency (-p)            = " << printFrequency << std::endl;
    std::cout << "   * Sample frequency (-s)           = " << sampleFrequency << std::endl;
    std::cout << "   * Branch length lambda (-b)       = " << brlenLambda << std::endl;
    std::cout << "   * Gamma shape lambda (-a)         = " << shapeLambda << std::endl;
}

void UserSettings::usage(void) {

    std::cout << "-i        Input sequence file name" << std::endl;
    std::cout << "-t        Input tree file name" << std::endl;
    std::cout << "-o        Output file name" << std::endl;
    std::cout << "-n        MCMC chain length" << std::endl;
    std::cout << "-d        MCMC burn-in fraction" << std::endl;
    std::cout << "-p        Print to screen frequency" << std::endl;
    std::cout << "-s        Chain sample frequency" << std::endl;
    std::cout << "-g        Number of gamma rate categories" << std::endl;
    std::cout << "-b        Parameter of exponential prior for branch lengths" << std::endl;
    std::cout << "-a        Parameter of exponential prior for gamma shape parameter for rate variation across sites" << std::endl;
    std::cout << "Example: ./Auswahl -i ape.in -o ape.out -n 10000 -g 4" << std::endl;
    Msg::error("Incorrectly formatted command line input");
}
