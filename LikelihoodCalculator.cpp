#include <cmath>
#include "Alignment.hpp"
#include "ConditionalLikelihoods.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "LikelihoodCalculator.hpp"
#include "TransitionMatrix.hpp"
#include "Tree.hpp"



LikelihoodCalculator::LikelihoodCalculator(Alignment* aln) : tree(nullptr) {

    // information on the size of the tree and data
    numSites = aln->getNumSites();
    numTaxa = aln->getNumTaxa();
    numNodes = 2 * numTaxa - 2;
    
    // allocate conditional likelihoods
    condLikes = new ConditionalLikelihoods*[numNodes];
    for (int i=0; i<numTaxa; i++)
        condLikes[i] = new ConditionalLikelihoods(*aln, i);
    for (size_t i=numTaxa; i<numNodes; i++)
        condLikes[i] = new ConditionalLikelihoods(*aln);
       
    // initialize pattern count information
    patternCount = new int[numSites];
    for (int i=0; i<numSites; i++)
        patternCount[i] = aln->getPatternCount(i);
        
    // initialize the transition probabilities
    tiProbs = new TransitionMatrix[numNodes];
}

LikelihoodCalculator::~LikelihoodCalculator(void) {

    for (size_t i=0; i<numNodes; i++)
        delete condLikes[i];
    delete [] condLikes;
    delete [] patternCount;
    delete [] tiProbs;
}

double LikelihoodCalculator::calculateLFS(double* clsUp, double* clsDn, TransitionMatrix& tMat, TransitionMatrix& fMat, TransitionMatrix& sMat, double& firstDerivative, double& secondDerivative) {

    double lnL = 0.0;
    double* cD = clsDn;
    double* cU = clsUp;
    for (size_t c=0; c<numSites; c++)
        {
        double like = 0.0, first = 0.0, second = 0.0;
        for (size_t i=0; i<4; i++)
            {
            for (size_t j=0; j<4; j++)
                {
                double x = 0.25 * cD[i] * cU[j];
                like   += x * tMat(i,j);
                first  += x * fMat(i,j);
                second += x * sMat(i,j);
                }
            }

        if (like <= 0.0) 
            Msg::warning("Warning: zero or negative likelihood at site " + std::to_string(c) + " " + std::to_string(like));

        int numSitesForPattern = patternCount[c];
        lnL += numSitesForPattern * log(like);
        firstDerivative += (numSitesForPattern * first/like);
        secondDerivative += (numSitesForPattern * ((second * like - first * first)/(like * like)));
        
        cU += 4;
        cD += 4;
        }
        
    return lnL;
}

double LikelihoodCalculator::iterateBranch(Node* p) {

    double* clsUp = condLikes[p->getIndex()]->beginUp();
    double* clsDn = condLikes[p->getIndex()]->beginDn();

    TransitionMatrix& tMat = tiProbs[p->getIndex()];
    TransitionMatrix& fMat = firstDerivative;
    TransitionMatrix& sMat = secondDerivative;
 
    double lnL = 0.0;
	int numIterations = 0;
	bool converged = false;
    double v = p->getBrlen();
    double bestBrlen=0.0, bestLnL=0.0;
	while (!converged)
		{
        tMat.setBrlen(v);
        fMat.setBrlen(v);
        sMat.setBrlen(v);
        double oldV = v;
        
        double firstDerivative = 0.0;
        double secondDerivative = 0.0;
        lnL = calculateLFS(clsUp, clsDn, tMat, fMat, sMat, firstDerivative, secondDerivative);
            
        if (numIterations == 0)
            {
            bestLnL = lnL;
            bestBrlen = v;
            }
        else 
            {
            if (lnL > bestLnL)
                {
                bestLnL = lnL;
                bestBrlen = v;
                }
            }
                 
        // propose a step using Newton when possible
        double step = 0.0;
        if (std::abs(secondDerivative) >= 1e-12 && secondDerivative < 0.0)
            step = - firstDerivative / secondDerivative;
        else
            step = 0.01 * firstDerivative; // fallback small gradient step
            
        v = oldV + step;
        if (v < minBrlen)
            v = minBrlen;
        if (v > maxBrlen)
            v = maxBrlen;
        
        // objective-based convergence checks
        double dv = std::abs(v - oldV);
        if (dv < 1e-10)
            converged = true;
        if (std::abs(firstDerivative) < 1e-4)
            converged = true;
        if (numIterations >= 10)
            converged = true;
       
		numIterations++;
		}
  
    p->setBrlen(bestBrlen);
    tMat.setBrlen(bestBrlen);
    
    return lnL;
}

double LikelihoodCalculator::lnLikelihood(void) {
    
    if (tree == nullptr)
        return 0.0;

    double bestLnL = 0.0;
    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    
    tree->markUpClsAsDirty();
    tree->markDnClsAsDirty();
    tree->setAllFlags(false);
//    std::vector<double> tempLikes;
    for (size_t rep=0; rep<10; rep++)
        {
        // pass up the tree, optimizing the branch for each node
        double lnL = 0.0;
        for (std::vector<Node*>::const_reverse_iterator p = postOrder.rbegin(); p != postOrder.rend(); ++p) 
            {
            if (*p != treeRoot)
                {
                updateUpConditionalLikelihoods(*p);
                updateDnConditionalLikelihoods(*p);
                
                lnL = iterateBranch(*p);
                
                markUpClsAsDirtyFromNode(*p);
                markDnClsAsDirtyFromNode(*p);
                }
            }
            
        if (rep == 0)
            {
            bestLnL = lnL;
//            tempLikes.push_back(lnL);
            }
        else 
            {
//            tempLikes.push_back(lnL);
//            if (lnL < bestLnL && std::fabs(lnL - bestLnL) > 0.1)
//                {
//                for (size_t i=0; i<tempLikes.size(); i++)
//                    std::cout << i << " " << tempLikes[i] << std::endl;
//                tree->print();
//                Msg::error("Going downhill!");
//                }
            if (std::abs(lnL - bestLnL) < 1.0e-6)
                break;
            }
        if (bestLnL < lnL)
            bestLnL = lnL;
        //std::cout << "rep=" << rep << " -- " << lnL << " " << bestLnL << std::endl;
        }
        
    return bestLnL;
}

void LikelihoodCalculator::markDnClsAsDirtyFromNode(Node* p) {
        
    Node* q = p;
    while (q != nullptr)
        {
        q->setFlag(true);
        q = q->getAncestor();
        }
    
    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    for (Node* nde : postOrder)
        {
        if (nde->getFlag() == false)
            {
            nde->setDirtyDnCl(true);
            }
        else
            {
            nde->setDirtyDnCl(false);
            nde->setFlag(false);
            }
        }
}

void LikelihoodCalculator::markUpClsAsDirtyFromNode(Node* p) {

    Node* q = p;
    while (q != treeRoot)
        {
        if (q != p)
            q->setDirtyUpCl(true);
        q = q->getAncestor();
        }
}

void LikelihoodCalculator::setTree(Tree* t) {

    // set the pointers to the tree and the root node of the tree
    tree = t;
    treeRoot = tree->getRoot();
    treeRootDescendant = treeRoot->getFirstDescendant();
    if (treeRootDescendant->getNextSibling() != nullptr)
        Msg::error("Expecting the descendant of the root node to have no siblings");
    
    // update all of the transition probabilities based on the current branch lengths
    for (Node* p : tree->getDownPassSequence())
        {
        int idx = p->getIndex();
        double v = p->getBrlen();
        tiProbs[idx].setBrlen(v);
        }
        
    // set flags indicating all conditional likelihoods are dirty
        
    // move the conditional likelihoods from the root node (the tip with index 0)
    // to the clsDn conditional likelihoods of the single node descendant of the root
    if (treeRoot->getNumDescendants() != 1)
        Msg::error("The root node should have only one descendant");
    Node* p = treeRoot->getFirstDescendant();
    if (p == nullptr)
        Msg::error("Could not find single descendant of root node");
    p->setDirtyDnCl(false);
    double* clsR = condLikes[treeRoot->getIndex()]->beginUp();
    double* clsREnd = condLikes[treeRoot->getIndex()]->endUp();
    double* clsP = condLikes[p->getIndex()]->beginDn();
    for (; clsR != clsREnd; clsR++, clsP++)
        *clsP = *clsR;
}

void LikelihoodCalculator::updateDnConditionalLikelihoods(Node* p) {

    if (p == treeRoot || p == treeRootDescendant || p->getDirtyDnCl() == false)
        return;

    Node* sis = p->getSister();
    if (sis == nullptr)
        return;
    Node* anc = p->getAncestor();
    if (anc == nullptr)
        Msg::error("Problem calculating likelihood up to node " + std::to_string(p->getIndex()));
 
    if (sis->getDirtyUpCl() == true)
        updateUpConditionalLikelihoods(sis);
    if (anc->getDirtyDnCl() == true)
        updateDnConditionalLikelihoods(anc);

    //std::cout << "Updating Dn CL for node " << p->getIndex() << std::endl;
        
    TransitionMatrix& tiMatSis = tiProbs[sis->getIndex()];   
    TransitionMatrix& tiMatAnc = tiProbs[anc->getIndex()];        
    
    double* clP = condLikes[p->getIndex()]->beginDn();
    double* clS = condLikes[sis->getIndex()]->beginUp();
    double* clA = condLikes[anc->getIndex()]->beginDn();
    
    for (size_t c=0; c<numSites; c++)
        {
        for (size_t i=0; i<4; i++)
            {
            double sumSis = 0.0, sumAnc = 0.0;
            for (size_t j=0; j<4; j++)
                {
                sumSis += clS[j] * tiMatSis(i,j);
                sumAnc += clA[j] * tiMatAnc(i,j);
                }
            *clP = sumSis * sumAnc;
            clP++;
            }
        clS += 4;
        clA += 4;
        }
        
    p->setDirtyDnCl(false);
}

void LikelihoodCalculator::updateUpConditionalLikelihoods(Node* p) {

    if (p->getIsTip() == true || p->getDirtyUpCl() == false)
        return;
        
    Node* d0 = p->getFirstDescendant();
    Node* d1 = d0->getNextSibling();
    if (d0 == nullptr || d1 == nullptr)
        Msg::error("Cannot calculate conditional likelihoods for node " + std::to_string(p->getIndex()));
        
    if (d0->getDirtyUpCl() == true)
        updateUpConditionalLikelihoods(d0);
    if (d1->getDirtyUpCl() == true)
        updateUpConditionalLikelihoods(d1);

    //std::cout << "Updating Up CL for node " << p->getIndex() << std::endl;
        
    TransitionMatrix& dMat0 = tiProbs[d0->getIndex()];
    TransitionMatrix& dMat1 = tiProbs[d1->getIndex()];
    
    double* p_cls  = condLikes[p->getIndex()]->beginUp();
    double* d0_cls = condLikes[d0->getIndex()]->beginUp();
    double* d1_cls = condLikes[d1->getIndex()]->beginUp();
    
    for (size_t c=0; c<numSites; c++)
        {
        for (size_t i=0; i<4; i++)
            {
            double sum0 = 0.0, sum1 = 0.0;
            for (size_t j=0; j<4; j++)
                {
                sum0 += d0_cls[j] * dMat0(i,j);
                sum1 += d1_cls[j] * dMat1(i,j);
                }
            *p_cls = sum0 * sum1;
            p_cls++;
            }
        d0_cls += 4;
        d1_cls += 4;
        }
        
    p->setDirtyUpCl(false);
}

