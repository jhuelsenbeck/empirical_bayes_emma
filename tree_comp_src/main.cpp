#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "BitSetFactory.hpp"
#include "Msg.hpp"
#include "ncl.h"
#include "Tree.hpp"
#include "UserSettings.hpp"

void addTreesToMap(std::unordered_map<uint64_t, std::pair<int,int>>& treeMap, std::vector<Tree*>& trees1, std::vector<Tree*>& trees2);
void printTreeProbabilities(std::map<uint64_t,std::pair<double,int>>& probs);
void printTreeProbabilities(std::unordered_map<uint64_t, std::pair<int,int>>& treeMap, int n1, int n2);
std::map<uint64_t,std::pair<double,int>> readTreeProbabilities(const std::string& filename);
std::vector<Tree*> readTrees(const std::string& filename, double burnFraction, int& nt);



int main(int argc, char* argv[]) {

    // get the user settings
    UserSettings& settings = UserSettings::userSettings();
    settings.readSettings(argc, argv);
    settings.print();
        
    // read the trees from the two files
    std::vector<std::string>& treeFiles = settings.getTreeFiles();
    std::string trueFile = settings.getTrueFile();
    double burnFraction = settings.getBurnin();
    
    if (treeFiles.size() == 2 && trueFile == "")
        {
        int nTaxa1 = 0, nTaxa2 = 0;
        std::vector<Tree*> trees1 = readTrees(treeFiles[0], burnFraction, nTaxa1);
        std::vector<Tree*> trees2 = readTrees(treeFiles[1], burnFraction, nTaxa2);
        if (nTaxa1 != nTaxa2)
            Msg::error("Mismatched tree sizes");

        // add the trees to the map
        std::unordered_map<uint64_t, std::pair<int,int>> treeMap;
        addTreesToMap(treeMap, trees1, trees2);
            
        // print values
        printTreeProbabilities(treeMap, (int)trees1.size(), (int)trees2.size());
        }
    else if (treeFiles.size() == 1 && trueFile != "")
        {
        int nTaxa = 0;
        std::vector<Tree*> trees = readTrees(treeFiles[0], burnFraction, nTaxa);
        std::map<uint64_t,std::pair<double,int>> probs = readTreeProbabilities(trueFile);
        
        int n = 0;
        int start = trees.size() * burnFraction;
        if (trees.size() - start < 100)
            Msg::error("Too few sampled trees");
        for (size_t i=start; i<trees.size(); i++)
            {
            uint64_t key = trees[i]->getHash();
            std::map<uint64_t,std::pair<double,int>>::iterator it = probs.find(key);
            if (it == probs.end())
                {
                std::pair<double,int> val = std::make_pair(0.0, 1);
                probs.insert( std::make_pair(key,val) );
                }
            else 
                {
                it->second.second++;
                }
            n++;
            }
            
        // print to a file
        printTreeProbabilities(probs);
        }
    else 
        Msg::error("Not clear which tree sets to compare");
        
    return EXIT_SUCCESS;
}

void addTreesToMap(std::unordered_map<uint64_t, std::pair<int,int>>& treeMap, std::vector<Tree*>& trees1, std::vector<Tree*>& trees2) {

    for (size_t i=0; i<trees1.size(); i++)
        {
        uint64_t key = trees1[i]->getHash();
        std::unordered_map<uint64_t, std::pair<int,int>>::iterator it = treeMap.find(key);
        if (it == treeMap.end())
            {
            std::pair<int,int> val = std::make_pair(1,0);
            treeMap.insert(std::make_pair(key,val));
            }
        else 
            it->second.first++;
        }
    for (size_t i=0; i<trees2.size(); i++)
        {
        uint64_t key = trees2[i]->getHash();
        std::unordered_map<uint64_t, std::pair<int,int>>::iterator it = treeMap.find(key);
        if (it == treeMap.end())
            {
            std::pair<int,int> val = std::make_pair(0,1);
            treeMap.insert(std::make_pair(key,val));
            }
        else 
            it->second.second++;
        }
}

void printTreeProbabilities(std::map<uint64_t,std::pair<double,int>>& probs) {

    UserSettings& settings = UserSettings::userSettings();
    std::string dataname = settings.getTrueFile();
    std::string filename = settings.getOutputFileName();
    bool appendResults = settings.getShouldAppend();

    std::ios_base::openmode mode = appendResults ? std::ios::app : std::ios::trunc;
    std::ofstream outStrm(filename, mode);
    
    if (appendResults == false)
        outStrm << "dataset" << '\t' << "point_index" << '\t' << "x" << '\t' << "y" << '\n';
    
    int n = 0;
    for (auto& [key,val] : probs)
        n += val.second;
        
    int i = 0;
    for (auto& [key,val] : probs)
        {
        double x = (double)val.second / n;
        if (val.first + x > 0.001)
            {
            std::cout << std::setw(20) << key << " -- " << val.first << " " << x << std::endl;
            outStrm << dataname << '\t' << ++i << '\t' << val.first << '\t' << x << '\n';
            }
        }
        
    outStrm.close();
}

void printTreeProbabilities(std::unordered_map<uint64_t, std::pair<int,int>>& treeMap, int n1, int n2) {

    std::vector<std::pair<uint64_t, std::pair<int,int>>> sortedEntries(treeMap.begin(), treeMap.end());
    std::sort(sortedEntries.begin(), sortedEntries.end(),
        [](const auto& a, const auto& b)
            {
            return (a.second.first + a.second.second) > (b.second.first + b.second.second);
            });
    for (auto& [key,val] : sortedEntries)
        {
        std::cout << std::setw(20) << key << " -- ";
        std::cout << std::fixed << std::setprecision(4) << (double)val.first / n1 << " ";
        std::cout << std::fixed << std::setprecision(4) << (double)val.second / n2 << " ";
        std::cout << std::endl;
        }
}

std::map<uint64_t,std::pair<double,int>> readTreeProbabilities(const std::string& filename) {

    std::map<uint64_t,std::pair<double,int>> probsMap;

    std::ifstream strm(filename);
    if (!strm.good())
        Msg::error("Cannot open file: " + filename);

    std::string lineString = "";
    std::string theSequence = "";
    uint64_t treeHash = 0;
    double treeProb = 0.0;
    while( getline(strm, lineString).good() )
        {
        std::istringstream linestream(lineString);
        int ch;
        std::string word = "";
        int wordNum = 0;
        treeHash = 0;
        treeProb = 0.0;
        std::string cmdString = "";
        do
            {
            word = "";
            linestream >> word;
            wordNum++;
            if (wordNum == 1)
                treeHash = static_cast<uint64_t>(std::stoull(word));
            else if (wordNum == 3)
                treeProb = std::stod(word);
            //std::cout << wordNum << " " << word << std::endl;
            } while ( (ch=linestream.get()) != EOF );
        //std::cout << treeHash << " -> " << treeProb << std::endl;
        std::pair val = std::make_pair(treeProb,0);
        probsMap.insert( std::make_pair(treeHash, val) );
        }
    
    strm.close();
    
#   if 0
    for (auto& [key,val] : probsMap)
        std::cout << key << " " << val.first << " " << val.second << std::endl;
#   endif
    
    return probsMap;
}

std::vector<Tree*> readTrees(const std::string& filename, double burnFraction, int& nt) {

    std::vector<Tree*> trees;

    MultiFormatReader reader(-1, NxsReader::WARNINGS_TO_STDERR);

    try {
        reader.ReadFilepath(filename.c_str(), MultiFormatReader::NEXUS_FORMAT);

        const int ntaxaBlocks = reader.GetNumTaxaBlocks();

        for (int i=0; i<ntaxaBlocks; i++) 
            {
            NxsTaxaBlock* taxaBlock = reader.GetTaxaBlock(i);
            const unsigned ntax = taxaBlock->GetNTax();
            
            BitSetFactory& bitFactory = BitSetFactory::getFactory();
            if (bitFactory.getIsInitialized() == false)
                bitFactory.initialize(ntax);
                
            if (nt == 0)
                nt = ntax;

            std::vector<std::string> taxonNames;
            taxonNames.reserve(ntax);
            for (unsigned n=0; n<ntax; n++)
                taxonNames.push_back(taxaBlock->GetTaxonLabel(n));

            const unsigned ntreesBlocks = reader.GetNumTreesBlocks(taxaBlock);
            for (unsigned j=0; j<ntreesBlocks; j++) 
                {
                NxsTreesBlock* treesBlock = reader.GetTreesBlock(taxaBlock, j);
                const unsigned ntrees = treesBlock->GetNumTrees();
                trees.reserve(trees.size() + ntrees);
                
                int burnNum = burnFraction * ntrees;
                for (unsigned k=0; k<ntrees; k++) 
                    {
                    if (k < burnNum)
                        continue;
                    std::string newick = treesBlock->GetTranslatedTreeDescription(k).c_str();
                    Tree* t = new Tree(newick, taxonNames, true);
                    trees.push_back(t);
                    }
                }
            }

        reader.DeleteBlocksFromFactories();
        }
    catch (...) 
        {
        reader.DeleteBlocksFromFactories();
        throw;
        }

    return trees;
}
