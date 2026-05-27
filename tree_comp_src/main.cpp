#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "BitSetFactory.hpp"
#include "Msg.hpp"
#include "Tree.hpp"
#include "ncl.h"

void addTreesToMap(std::unordered_map<uint64_t, std::pair<int,int>>& treeMap, std::vector<Tree*>& trees1, std::vector<Tree*>& trees2);
void printTreeProbabilities(std::unordered_map<uint64_t, std::pair<int,int>>& treeMap, int n1, int n2);
std::vector<Tree*> readTrees(const std::string& filename, double burnFraction, int& nt);



int main(int argc, char* argv[]) {

    // get the user settings
    std::string fileName1 = "/Users/johnh/Desktop/empirical_bayes_data/test.nex.t";
    std::string fileName2 = "/Users/johnh/Desktop/empirical_bayes_data/test.nex.tre";
    double burnFraction = 0.20;
        
    // read the trees from the two files
    int nTaxa1 = 0, nTaxa2 = 0;
    std::vector<Tree*> trees1 = readTrees(fileName1, burnFraction, nTaxa1);
    std::vector<Tree*> trees2 = readTrees(fileName2, burnFraction, nTaxa2);
    if (nTaxa1 != nTaxa2)
        Msg::error("Mismatched tree sizes");

    // add the trees to the map
    std::unordered_map<uint64_t, std::pair<int,int>> treeMap;
    addTreesToMap(treeMap, trees1, trees2);
        
    // print values
    printTreeProbabilities(treeMap, (int)trees1.size(), (int)trees2.size());
        
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
