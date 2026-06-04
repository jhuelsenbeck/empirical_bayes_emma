#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <sstream>
#include <map>
#include "ncl.h"
#include "Alignment.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "RandomVariable.hpp"
#include "Tree.hpp"
#include "UserSettings.hpp"



Alignment::Alignment(std::string fileName) : 
    matrix(nullptr), patternCount(nullptr), numTaxa(0), numSites(0), isCompressed(false)  {

    MultiFormatReader nexusReader;
    const char* fn = fileName.c_str();
    nexusReader.ReadFilepath(fn, MultiFormatReader::NEXUS_FORMAT);

    size_t num_taxaBlocks = nexusReader.GetNumTaxaBlocks();
    for (unsigned tBlck=0; tBlck<num_taxaBlocks; ++tBlck)
        {
        NxsTaxaBlock* taxaBlock = nexusReader.GetTaxaBlock(tBlck);
        std::string taxaBlockTitle          = taxaBlock->GetTitle();
        const unsigned nCharBlocks          = nexusReader.GetNumCharactersBlocks(taxaBlock);
        
        // make alignment objects
        for (unsigned cBlck=0; cBlck<nCharBlocks; cBlck++)
            {
            NxsCharactersBlock* charBlock = nexusReader.GetCharactersBlock(taxaBlock, cBlck);
            std::string charBlockTitle = taxaBlock->GetTitle();
            int dt = charBlock->GetDataType();
            if (dt == NxsCharactersBlock::dna || dt == NxsCharactersBlock::nucleotide)
                {
                createDnaMatrix(charBlock);
                }
            else if (dt == NxsCharactersBlock::rna)
                {
                //createRnaMatrix(charBlock);
                }
            else if (dt == NxsCharactersBlock::protein)
                {
                //m = createAminoAcidMatrix(charBlock);
                }
            else if (dt == NxsCharactersBlock::standard)
                {
                //m = createStandardMatrix(charBlock);
                }
            else if (dt == NxsCharactersBlock::continuous)
                {
                //m = createContinuousMatrix(charBlock);
                }
            else if (dt == NxsCharactersBlock::mixed)
                {
                //addWarning("Mixed data types are not allowed");
                }
            else
                {
                //addWarning("Unknown data type");
                }
            
        }
        
    }
    
    //print();
}

Alignment::Alignment(std::vector<std::string> tn, int nr, int nc) : 
    matrix(nullptr), patternCount(nullptr), numTaxa(0), numSites(0), isCompressed(false) {

    taxonNames = tn;
    numTaxa = (int)taxonNames.size();
    numSites = nc;

    // instantiate the character matrix
    matrix = new int*[nr];
    matrix[0] = new int[nr * numSites];
    for (size_t i=1; i<nr; i++)
        matrix[i] = matrix[i-1] + numSites;
    for (size_t i=0; i<nr; i++)
        for (size_t j=0; j<numSites; j++)
            matrix[i][j] = 0;
}

Alignment::Alignment(Alignment& a, int nt, RandomVariable* rng) :
    matrix(nullptr), patternCount(nullptr), numTaxa(0), numSites(0), isCompressed(false) {

    numTaxa = nt;
    numSites = a.numSites;

    matrix = new int*[numTaxa];
    matrix[0] = new int[numTaxa * numSites];
    for (size_t i=1; i<numTaxa; i++)
        matrix[i] = matrix[i-1] + numSites;
    for (size_t i=0; i<numTaxa; i++)
        for (size_t j=0; j<numSites; j++)
            matrix[i][j] = 0;
            
    std::vector<int> taxonIndices(a.numTaxa);
    for (int i=0; i<a.numTaxa; i++)
        taxonIndices[i] = i;
    while (taxonIndices.size() > numTaxa)
        {
        int whichIndex = (int)(rng->uniformRv()*taxonIndices.size());
        taxonIndices[whichIndex] = taxonIndices[taxonIndices.size()-1];
        taxonIndices.pop_back();
        }
    sort(taxonIndices.begin(),taxonIndices.end());
    
    for (int i=0; i<taxonIndices.size(); i++)
        taxonNames.push_back(a.taxonNames[taxonIndices[i]]);

    for (int i=0; i<numTaxa; i++)
        {
        for (int j=0; j<numSites; j++)
            matrix[i][j] = a.matrix[taxonIndices[i]][j];
        }
}

Alignment::Alignment(std::vector<std::string> tn, std::string newickString, int nc, RandomVariable* rng) : 
    matrix(nullptr), patternCount(nullptr), numTaxa(0), numSites(0), isCompressed(false) {
    
    numTaxa = (int)tn.size();
    numSites = nc;
    for (int i=0; i<numTaxa; i++)
        taxonNames.push_back(tn[i]);
    
    Tree t(newickString, taxonNames, false);
    //t.print();
    
    int** tempMatrix = new int*[t.getNumNodes()];
    tempMatrix[0] = new int[t.getNumNodes()*numSites];
    for (int i=1; i<t.getNumNodes(); i++)
        tempMatrix[i] = tempMatrix[i-1] + numSites;
    for (int i=0; i<t.getNumNodes(); i++)
        for (int j=0; j<numSites; j++)
            tempMatrix[i][j] = 0;

    std::vector<Node*>& dpSeq = t.getDownPassSequence();
    for (std::vector<Node*>::reverse_iterator it=dpSeq.rbegin(); it != dpSeq.rend(); it++)
        {
        Node* p = *it;
        if (p == t.getRoot())
            {
            for (int i=0; i<numSites; i++)
                {
                int nuc = (int)(rng->uniformRv()*4.0);
                tempMatrix[p->getIndex()][i] = nuc;
                }
            }
        else 
            {
            double rate = (double)1.0/3.0;
            
            double brlen = p->getBrlen();
            for (int i=0; i<numSites; i++)
                {
                int currentNuc = tempMatrix[p->getAncestor()->getIndex()][i];
                double pos = 0.0;
                while (pos < brlen)
                    {
                    pos += -log(rng->uniformRv()) / 1.0; // JC69 rate is 1.0
                    if (pos < brlen)
                        {
                        double u = rng->uniformRv();
                        double sum = 0.0;
                        for (int j=0; j<4; j++)
                            {
                            if (j != currentNuc)
                                {
                                sum += rate;
                                if (u < sum)
                                    {
                                    currentNuc = j;
                                    break;
                                    }
                                }
                            }
                        }
                    }
                tempMatrix[p->getIndex()][i] = currentNuc;
                }
            }
        }

    matrix = new int*[numTaxa];
    matrix[0] = new int[numTaxa * numSites];
    for (size_t i=1; i<numTaxa; i++)
        matrix[i] = matrix[i-1] + numSites;
    for (size_t i=0; i<numTaxa; i++)
        for (size_t j=0; j<numSites; j++)
            matrix[i][j] = 0;
    
    for (Node* p : dpSeq)
        {
        int nodeIdx = p->getIndex();
        if (p->getIsTip() == true)
            {
            for (int i=0; i<numSites; i++)
                matrix[nodeIdx][i] = pow(2.0,tempMatrix[nodeIdx][i]);
            }
        }
        
    delete [] tempMatrix[0];
    delete [] tempMatrix;
}

Alignment::~Alignment(void) {

    if (matrix != nullptr) 
        {
        delete [] matrix[0];
        delete [] matrix;
        }
    if (patternCount != nullptr) 
        delete [] patternCount;
}

void Alignment::compress(void) {

    if (matrix == nullptr || numTaxa == 0 || numSites == 0)
        return;
    
    if (isCompressed == true)
        return;

    // vector to store unique patterns and their counts
    std::vector<std::vector<int>> uniquePatterns;
    std::vector<int> patternCounts;
    int removedSites = 0;
    
    // process each site (column) in the original matrix
    for (int site = 0; site < numSites; site++) 
        {
        // extract the current site pattern
        std::vector<int> currentPattern(numTaxa);
        for (int taxon = 0; taxon < numTaxa; taxon++) 
            currentPattern[taxon] = matrix[taxon][site];
        
        // check if all taxa have code 15 (complete missing data)
        bool allMissing = true;
        for (int taxon = 0; taxon < numTaxa; taxon++) 
            {
            if (currentPattern[taxon] != 15) 
                {
                allMissing = false;
                break;
                }
            }
        
        if (allMissing) 
            {
            removedSites++;
            continue; // Skip this site
            }
        
        // check if this pattern already exists
        bool patternFound = false;
        for (size_t i = 0; i < uniquePatterns.size(); i++) 
            {
            bool isIdentical = true;
            for (int taxon = 0; taxon < numTaxa; taxon++) 
                {
                if (uniquePatterns[i][taxon] != currentPattern[taxon]) 
                    {
                    isIdentical = false;
                    break;
                    }
                }
            if (isIdentical) 
                {
                patternCounts[i]++;
                patternFound = true;
                break;
                }
            }
        
        // if pattern is unique, add it
        if (!patternFound) 
            {
            uniquePatterns.push_back(currentPattern);
            patternCounts.push_back(1);
            }
        }
        
    // create the compressed matrix
    int newNumSites = (int)uniquePatterns.size();
    
    // deallocate old matrix
    delete [] matrix[0];
    delete [] matrix;
    
    // deallocate old patternCount if it exists
    if (patternCount != nullptr) 
        delete [] patternCount;
    
    // allocate new compressed matrix
    matrix = new int*[numTaxa];
    matrix[0] = new int[numTaxa * newNumSites];
    for (int i = 1; i < numTaxa; i++)
        matrix[i] = matrix[i-1] + newNumSites;
    
    // allocate pattern count array
    patternCount = new int[newNumSites];
    
    // copy unique patterns to the new matrix and set pattern counts
    for (int pattern = 0; pattern < newNumSites; pattern++) 
        {
        patternCount[pattern] = patternCounts[pattern];
        for (int taxon = 0; taxon < numTaxa; taxon++) 
            matrix[taxon][pattern] = uniquePatterns[pattern][taxon];
        }
    
    // update number of sites
    int originalNumSites = numSites;
    numSites = newNumSites;
    
    std::cout << "   Data compression:" << std::endl;
    std::cout << "   * Alignment compressed from " << originalNumSites << " sites to " << numSites << " unique patterns";
    if (removedSites > 0)
        std::cout << " (" << removedSites << " all-missing sites removed)";
    std::cout << std::endl << std::endl;
    
    isCompressed = true;
}

void Alignment::concatenateTwist(RandomVariable* rng, int nTrees) {

    if (isCompressed == true)
        Msg::error("Cannot permute a compressed alignment");
        
    if (nTrees < 2)
        return;
            
    std::vector<std::vector<int>> twistMap(nTrees);
    for (int n=0; n<nTrees; n++)
        {
        twistMap[n].resize(numTaxa);
        
        std::vector<int> vals;
        vals.clear();
        for (int i=0; i<numTaxa; i++)
            vals.push_back(i);

        for (size_t i=0; i<numTaxa; i++)
            {
            int whichElement = (int)(rng->uniformRv()*vals.size());
            twistMap[n][i] = vals[whichElement];
            vals[whichElement] = vals[vals.size()-1];
            vals.pop_back();
            }
        }
    int newNumSites = numSites * nTrees;
        
    int** tempMatrix = new int*[numTaxa];
    tempMatrix[0] = new int[numTaxa * newNumSites];
    for (size_t i=1; i<numTaxa; i++)
        tempMatrix[i] = tempMatrix[i-1] + newNumSites;
    for (size_t i=0; i<numTaxa; i++)
        for (size_t j=0; j<newNumSites; j++)
            tempMatrix[i][j] = matrix[i][j];
            
    for (size_t k=0, c=0; k<nTrees; k++)
        {
        for (size_t j=0; j<numSites; j++)
            {
            for (size_t i=0; i<numTaxa; i++)
                tempMatrix[i][c] = matrix[ twistMap[k][i] ][j];
            c++;
            }
        }
    numSites = newNumSites;
    
    delete [] matrix[0];
    delete [] matrix;
   
    matrix = tempMatrix;
}

void Alignment::createDnaMatrix(NxsCharactersBlock* charblock) {
    
    if ( charblock == NULL )
        {
        //throw RbException("Trying to create an DNA matrix from a NULL pointer.");
        }
    
    // check that the character block is of the correct type
    if ( charblock->GetDataType() != NxsCharactersBlock::dna )
        {
        std::cout << "Could not read in data matrix of type DNA because the nexus files says the type is:" << std::endl;
        switch ( charblock->GetDataType() )
        {
        case 1:
            std::cout << "Standard" << std::endl;
            break;
                
        case 2:
            std::cout << "DNA" << std::endl;
            break;
                
        case 3:
            std::cout << "RNA" << std::endl;
            break;
                
        case 4:
            std::cout << "Nucleotide" << std::endl;
            break;
                
        case 5:
            std::cout << "Protein" << std::endl;
            break;
                
        case 6:
            std::cout << "Continuous" << std::endl;
            break;
                
        case 7:
            std::cout << "Codon" << std::endl;
            break;
                
        case 8:
            std::cout << "Mixed" << std::endl;
            break;
                
        default:
            std::cout << "Unknown" << std::endl;
            break;
        }
    }
    
    // get the set of characters (and the number of taxa)
    NxsUnsignedSet charset;
    for (unsigned int i=0; i<charblock->GetNumChar(); i++)
        charset.insert(i);
    
    unsigned numOrigTaxa = charblock->GetNTax();
    numTaxa = charblock->GetNTax();
    numSites = charblock->GetNumChar();
    
    // get the set of excluded characters
    NxsUnsignedSet excluded = charblock->GetExcludedIndexSet();
    
    // instantiate the character matrix
    matrix = new int*[numTaxa];
    matrix[0] = new int[numTaxa * numSites];
    for (size_t i=1; i<numTaxa; i++)
        matrix[i] = matrix[i-1] + numSites;
    for (size_t i=0; i<numTaxa; i++)
        for (size_t j=0; j<numSites; j++)
            matrix[i][j] = 0;
            
    std::cout << "   Reading DNA alignment:" << std::endl;
    std::cout << "   * Alignment has " << numTaxa << " taxa and " << numSites << " sites" << std::endl;
    std::cout << std::endl;

    // read in the data, including taxon names
    for (unsigned origTaxIndex=0; origTaxIndex<numOrigTaxa; origTaxIndex++)
        {
        // add the taxon name
        NxsString   tLabel = charblock->GetTaxonLabel(origTaxIndex);
        std::string tName  = NxsString::GetEscaped(tLabel).c_str();
        
        taxonNames.push_back(tName);
        
        std::vector<std::string> tokens;
                
        // add the sequence information for the sequence associated with the taxon
        for (NxsUnsignedSet::iterator cit = charset.begin(); cit != charset.end(); cit++)
            {
            // add the character state to the matrix
            char site = charblock->GetState(origTaxIndex, *cit);
            matrix[origTaxIndex][*cit] = nucID(site);
            }
        
        }
}

/*-------------------------------------------------------------------
|
|   GetPossibleNucs: 
|
|   This function initializes a vector, nuc[MAX_NUM_STATES]. The four elements
|   of nuc correspond to the four nucleotides in alphabetical order.
|   We are assuming that the nucCode is a binary representation of
|   the nucleotides that are consistent with the observation. For
|   example, if we observe an A, then the nucCode is 1 and the 
|   function initalizes nuc[0] = 1 and the other elements of nuc
|   to be 0.
|
|   Observation    nucCode        nuc
|        A            1           1000
|        C            2           0100
|        G            4           0010
|        T            8           0001
|        R            5           1010
|        Y           10           0101
|        M            3           1100
|        K           12           0011
|        S            6           0110
|        W            9           1001
|        H           11           1101
|        B           14           0111
|        V            7           1110
|        D           13           1011
|        N - ?       15           1111
|
-------------------------------------------------------------------*/
void Alignment::getPossibleNucs (int nucCode, int* nuc) {

	if (nucCode == 1)
		{
		nuc[0] = 1;
		nuc[1] = 0;
		nuc[2] = 0;
		nuc[3] = 0;
		}
	else if (nucCode == 2)
		{
		nuc[0] = 0;
		nuc[1] = 1;
		nuc[2] = 0;
		nuc[3] = 0;
		}
	else if (nucCode == 3)
		{
		nuc[0] = 1;
		nuc[1] = 1;
		nuc[2] = 0;
		nuc[3] = 0;
		}
	else if (nucCode == 4)
		{
		nuc[0] = 0;
		nuc[1] = 0;
		nuc[2] = 1;
		nuc[3] = 0;
		}
	else if (nucCode == 5)
		{
		nuc[0] = 1;
		nuc[1] = 0;
		nuc[2] = 1;
		nuc[3] = 0;
		}
	else if (nucCode == 6)
		{
		nuc[0] = 0;
		nuc[1] = 1;
		nuc[2] = 1;
		nuc[3] = 0;
		}
	else if (nucCode == 7)
		{
		nuc[0] = 1;
		nuc[1] = 1;
		nuc[2] = 1;
		nuc[3] = 0;
		}
	else if (nucCode == 8)
		{
		nuc[0] = 0;
		nuc[1] = 0;
		nuc[2] = 0;
		nuc[3] = 1;
		}
	else if (nucCode == 9)
		{
		nuc[0] = 1;
		nuc[1] = 0;
		nuc[2] = 0;
		nuc[3] = 1;
		}
	else if (nucCode == 10)
		{
		nuc[0] = 0;
		nuc[1] = 1;
		nuc[2] = 0;
		nuc[3] = 1;
		}
	else if (nucCode == 11)
		{
		nuc[0] = 1;
		nuc[1] = 1;
		nuc[2] = 0;
		nuc[3] = 1;
		}
	else if (nucCode == 12)
		{
		nuc[0] = 0;
		nuc[1] = 0;
		nuc[2] = 1;
		nuc[3] = 1;
		}
	else if (nucCode == 13)
		{
		nuc[0] = 1;
		nuc[1] = 0;
		nuc[2] = 1;
		nuc[3] = 1;
		}
	else if (nucCode == 14)
		{
		nuc[0] = 0;
		nuc[1] = 1;
		nuc[2] = 1;
		nuc[3] = 1;
		}
	else if (nucCode == 15)
		{
		nuc[0] = 1;
		nuc[1] = 1;
		nuc[2] = 1;
		nuc[3] = 1;
		}
	else if (nucCode == 16)
		{
		nuc[0] = 1;
		nuc[1] = 1;
		nuc[2] = 1;
		nuc[3] = 1;
		}
}

int Alignment::getTaxonIndex(std::string ns) {

	int taxonIndex = -1;
	int i = 0;
	for (std::vector<std::string>::iterator p=taxonNames.begin(); p != taxonNames.end(); p++)
		{
		if ( (*p) == ns )
			{
			taxonIndex = i;
			break;
			}
		i++;
		}
	return taxonIndex;
}

int Alignment::lengthOfLongestTaxonName(void) {

    int len = 0;
    for (int i=0,n=(int)taxonNames.size(); i<n; i++)
        {
        if (taxonNames[i].length() > len)
            len = (int)taxonNames[i].length();
        }
    return len;
}

void Alignment::listTaxa(void) {

	int i = 1;
	for (std::vector<std::string>::iterator p=taxonNames.begin(); p != taxonNames.end(); p++)
		std::cout << std::setw(4) << i++ << " -- " << (*p) << '\n';
}

/*-------------------------------------------------------------------
|
|   NucID: 
|
|   Take a character, nuc, and return an integer:
|
|       nuc        returns
|        A            1 
|        C            2     
|        G            4      
|        T U          8     
|        R            5      
|        Y           10       
|        M            3      
|        K           12   
|        S            6     
|        W            9      
|        H           11      
|        B           14     
|        V            7      
|        D           13  
|        N - ?       15       
|
-------------------------------------------------------------------*/
int Alignment::nucID(char nuc) {

	char n = nuc;
	if (nuc == 'U' || nuc == 'u')
		n = 'T';

	if (n == 'A' || n == 'a')
		return 1;
	else if (n == 'C' || n == 'c')
		return 2;
	else if (n == 'G' || n == 'g')
		return 4;
	else if (n == 'T' || n == 't')
		return 8;
	else if (n == 'R' || n == 'r')
		return 5;
	else if (n == 'Y' || n == 'y')
		return 10;
	else if (n == 'M' || n == 'm')
		return 3;
	else if (n == 'K' || n == 'k')
		return 12;
	else if (n == 'S' || n == 's')
		return 6;
	else if (n == 'W' || n == 'w')
		return 9;
	else if (n == 'H' || n == 'h')
		return 11;
	else if (n == 'B' || n == 'b')
		return 14;
	else if (n == 'V' || n == 'v')
		return 7;
	else if (n == 'D' || n == 'd')
		return 13;
	else if (n == 'N' || n == 'n')
		return 15;
	else if (n == '-')
		return 15;
	else if (n == '?')
		return 15;
	else
		return -1;
}

void Alignment::print(void) {

    int** x = matrix;
    int matrixSize = numSites;
        
    std::cout << "        ";
    for (size_t i=0; i<numTaxa; i++)
        std::cout << std::setw(3) << i;
    std::cout << '\n';
    std::cout << "------------------------";
    for (size_t i=0; i<numTaxa; i++)
        std::cout << "---";
    std::cout << '\n';
    for (size_t j=0; j<matrixSize; j++)
        {
        std::cout << std::setw(4) << j+1 << " -- ";
        for (size_t i=0; i<numTaxa; i++)
            std::cout << std::setw(3) << x[i][j];
        if (patternCount != nullptr)
            std::cout << " -- " << patternCount[j];
        std::cout << '\n';
        }
}

void Alignment::print(std::string fileName) {

    size_t longestName = 0;
    for (size_t i=0; i<numTaxa; i++)
        {
        if (taxonNames[i].length() > longestName)
            longestName = taxonNames[i].length();
        }
        
    std::ofstream strm(fileName);
    
    strm << "#NEXUS" << std::endl << std::endl;
    strm << "begin data;" << std::endl;
    strm << "   dimensions ntax=" << numTaxa << " nchar=" << numSites << ";" << std::endl;
    strm << "   format datatype=dna;" << std::endl;
    strm << "   matrix" << std::endl;
    for (size_t i=0; i<numTaxa; i++)
        {
        strm << "   " << taxonNames[i];
        for (size_t j=0; j<longestName-taxonNames[i].length() + 1; j++)
            strm << " ";
        for (size_t j=0; j<numSites; j++)
            strm << toNuc(matrix[i][j]);
        strm << std::endl;
        }
    strm << "   ;" << std::endl;
    strm << "end;" << std::endl << std::endl;
    strm << "begin mrbayes;" << std::endl;
    strm << "   lset nst=1 rates=equal;" << std::endl;
    strm << "   prset statefreqpr=fixed(0.25,0.25,0.25,0.25);" << std::endl;
    strm << "   mcmc nruns=1 nchains=1 ngen=1000000;" << std::endl;
    strm << "   sumt;" << std::endl;
    strm << "end;" << std::endl;
    
    strm.close();
}

void Alignment::summarize(void) {
    
    int nucTypes[16];
    for (int i=0; i<16; i++)
        nucTypes[i] = 0;
        
    for (int i=0; i<numTaxa; i++)
        for (int j=0; j<numSites; j++)
            nucTypes[ matrix[i][j] ]++;
            
    int sum = 0, maxNumDigits = 0;
    for (int i=0; i<16; i++)
        {
        sum += nucTypes[i];
        int nd = log((double)nucTypes[i]) / log(10.0) + 1;
        if (nd > maxNumDigits)
            maxNumDigits = nd;
        }
            
    std::vector<std::string> ids;
    ids.push_back("       ");  // 0
    ids.push_back("A      ");  // 1
    ids.push_back("C      ");  // 2
    ids.push_back("A/C    ");  // 3
    ids.push_back("G      ");  // 4
    ids.push_back("A/G    ");  // 5
    ids.push_back("C/G    ");  // 6
    ids.push_back("A/C/G  ");  // 7
    ids.push_back("T      ");  // 8
    ids.push_back("A/T    ");  // 9
    ids.push_back("C/T    ");  // 10
    ids.push_back("A/C/T  ");  // 11
    ids.push_back("G/T    ");  // 12
    ids.push_back("A/G/T  ");  // 13
    ids.push_back("C/G/T  ");  // 14
    ids.push_back("A/C/G/T");  // 15

    std::cout << "   Data summary:" << std::endl;
    std::cout << "   * Number of taxa    = " << std::setw(maxNumDigits) << numTaxa << std::endl;
    std::cout << "   * Number of sites   = " << std::setw(maxNumDigits) << numSites << std::endl;

    for (int i=1; i<16; i++)
        std::cout << "   * Number of " << ids[i] << " = " << std::setw(maxNumDigits) << nucTypes[i] << " " << std::setw(7) << std::fixed << std::setprecision(4) << ((double)nucTypes[i]/sum) * 100.0 << "\%" << std::endl;
    std::cout << std::endl;
}

char Alignment::toNuc(int charCode) {

    if (charCode == 1)
        return 'A';
    else if (charCode == 2)
        return 'C';
    else if (charCode == 4)
        return 'G';
    else if (charCode == 8)
        return 'T';
    else if (charCode == 3)
        return 'M';
    else if (charCode == 5)
        return 'R';
    else if (charCode == 6)
        return 'S';
    else if (charCode == 7)
        return 'V';
    else if (charCode == 9)
        return 'W';
    else if (charCode == 10)
        return 'Y';
    else if (charCode == 11)
        return 'H';
    else if (charCode == 12)
        return 'K';
    else if (charCode == 13)
        return 'D';
    else if (charCode == 14)
        return 'B';
    return 'N';
}

void Alignment::twist(RandomVariable* rng, int nTrees) {

    if (isCompressed == true)
        Msg::error("Cannot permute a compressed alignment");
        
    if (nTrees < 2)
        return;
            
    std::vector<std::vector<int>> twistMap(nTrees);
    for (int n=0; n<nTrees; n++)
        {
        twistMap[n].resize(numTaxa);
        
        std::vector<int> vals;
        vals.clear();
        for (int i=0; i<numTaxa; i++)
            vals.push_back(i);

        for (size_t i=0; i<numTaxa; i++)
            {
            int whichElement = (int)(rng->uniformRv()*vals.size());
            twistMap[n][i] = vals[whichElement];
            vals[whichElement] = vals[vals.size()-1];
            vals.pop_back();
            }
        }
        
    int** tempMatrix = new int*[numTaxa];
    tempMatrix[0] = new int[numTaxa * numSites];
    for (size_t i=1; i<numTaxa; i++)
        tempMatrix[i] = tempMatrix[i-1] + numSites;
    for (size_t i=0; i<numTaxa; i++)
        for (size_t j=0; j<numSites; j++)
            tempMatrix[i][j] = matrix[i][j];
            
    for (size_t j=0; j<numSites; j++)
        {
        int whichTree = (int)(rng->uniformRv()*nTrees);
        for (size_t i=0; i<numTaxa; i++)
            tempMatrix[i][j] = matrix[ twistMap[whichTree][i] ][j];
        }
    
    delete [] matrix[0];
    delete [] matrix;
   
    matrix = tempMatrix;
}
