#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "Msg.hpp"
#include "UserSettings.hpp"

#undef DEBUG_MODE



UserSettings::UserSettings(void) {

    executableName = "";
    settingsInitialized = false;
    temperature = 0.2;
    chainLength = 1000000;
    printFrequency = 100;
    sampleFrequency = 100;
    inputFileName = "";
    outputFileName = "";
    outputDirectoryName = "";
    outputFileNameStub = "";
    burnin = 0.1;
    seed = 0;
    chainStart = "random";
    numTwists = 0;
    computeMarginalLikelihoods = false;
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
    // done on construction
    
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
            //else if (key == "-o")
            //    outputFileName = cmd;
            else if (key == "-d")
                outputDirectoryName = cmd;
            else if (key == "-f")
                outputFileNameStub = cmd;
            else if (key == "-n")
                chainLength = stod(cmd);
            else if (key == "-t")
                temperature = stof(cmd);
            else if (key == "-b")
                {
                burnin = stof(cmd);
                if (!(burnin >= 0.0 && burnin < 1.0))
                    Msg::error("The burn-in fraction (-b) must be at least 0 and less than 1");
                }
            else if (key == "-r")
                {
                long long v = std::stoll(cmd);
                if (v < 0 || v > 4294967295LL)
                    Msg::error("The random seed (-r) must be between 0 and 4294967295");
                seed = static_cast<uint32_t>(v);
                }
            else if (key == "-c")
                {
                if (cmd != "random" && cmd != "posterior" && cmd != "both")
                    Msg::error("The chain start (-c) is random, posterior, or both");
                chainStart = cmd;
                }
            else if (key == "-p")
                printFrequency = stod(cmd);
            else if (key == "-s")
                Msg::error("The -s (sample frequency) option is not supported: every cycle is sampled");
            else if (key == "-w")
                numTwists = stod(cmd);
            else if (key == "-e")
                Msg::error("The -e (exclude taxa) option is not supported");
            else if (key == "-l")
                {
                if (cmd == "yes" || cmd == "1" || cmd == "true")
                    computeMarginalLikelihoods = true;
                else if (cmd == "no" || cmd == "0" || cmd == "false")
                    computeMarginalLikelihoods = false;
                else
                    Msg::error("The -l option takes yes or no");
                }
            else
                Msg::error("Improperly formatted commands");
            readingKey = true;
            }
        }
        
    // set output file name from directory name and file stub name
    if (outputDirectoryName == "" || outputFileNameStub == "")
        Msg::error("Problem specifying output file name");
    if (outputDirectoryName.back()  == '/' && outputFileNameStub.front() != '/')
        outputFileName = outputDirectoryName + outputFileNameStub;
    else if (outputDirectoryName.back()  != '/' && outputFileNameStub.front() == '/')
        outputFileName = outputDirectoryName + outputFileNameStub;
    else 
        outputFileName = outputDirectoryName + "/" + outputFileNameStub;

    // A seed of zero (the default) asks for a random one. It is drawn here, once, so that the seed in
    // force is always a single number that is printed and can be given back with -r to repeat the run.
    while (seed == 0)
        seed = std::random_device()();
        
    settingsInitialized = true;
}

void UserSettings::print(void) {

    std::cout << "   User settings:" << std::endl;
    std::cout << "   * Input file name (-i)      = " << inputFileName << std::endl;
    std::cout << "   * Output file name (-o)     = " << getOutputFileName() << std::endl;
    std::cout << "   * Temperature (-t)          = " << temperature << std::endl;
    std::cout << "   * Chain length (-n)         = " << chainLength << std::endl;
    std::cout << "   * Burn in fraction (-b)     = " << burnin << std::endl;
    std::cout << "   * Chain start (-c)          = " << chainStart << std::endl;
    std::cout << "   * Random seed (-r)          = " << seed << std::endl;
    std::cout << "   * Print frequency (-p)      = " << printFrequency << std::endl;
    std::cout << "   * Num alignment twists (-w) = " << numTwists << std::endl;
    std::cout << "   * Laplace marginals (-l)    = " << (computeMarginalLikelihoods ? "yes" : "no") << std::endl;
    std::cout << std::endl;
}

void UserSettings::usage(void) {

    std::cout << "-i        Input sequence file name" << std::endl;
    std::cout << "-t        Temperature parameter for MCMCMC" << std::endl;
    std::cout << "-n        MCMC chain length" << std::endl;
    std::cout << "-d        Output directory" << std::endl;
    std::cout << "-f        Output file name stub" << std::endl;
    std::cout << "-b        MCMC burn-in fraction, discarded in the MCMC functional summaries (default 0.1)" << std::endl;
    std::cout << "-c        MCMC starting trees: random (uniform over trees, the default), posterior (drawn" << std::endl;
    std::cout << "          from the exact posterior), or both (random, plus the plain MCMC runs again from" << std::endl;
    std::cout << "          posterior draws)" << std::endl;
    std::cout << "-r        Random number seed (default: drawn at random and printed)" << std::endl;
    std::cout << "-p        Print to screen frequency" << std::endl;
    std::cout << "-w        Number of alignment twists (permutations)" << std::endl;
    std::cout << "-l        Laplace marginal likelihood of every tree: yes or no (default no)" << std::endl;
    Msg::error("Incorrectly formatted command line input");
}
