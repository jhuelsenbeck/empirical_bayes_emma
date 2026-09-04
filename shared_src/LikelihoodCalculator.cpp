#include <cmath>
#include "Alignment.hpp"
#include "ConditionalLikelihoods.hpp"
#include "Msg.hpp"
#include "Node.hpp"
#include "LikelihoodCalculator.hpp"
#include "TransitionMatrix.hpp"
#include "Tree.hpp"
#include "TreeCache.hpp"

bool   LikelihoodCalculator::computeMarginal = false;
double LikelihoodCalculator::exponentialPriorRate = 10.0;   // IID Exp(10) prior on each branch length
int    LikelihoodCalculator::hessianMethod = 1;             // 0 = diagonal, 1 = full (finite difference)



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
            }
        else 
            {
            if (std::abs(lnL - bestLnL) < 1.0e-6)
                break;
            }
        if (bestLnL < lnL)
            bestLnL = lnL;
        }

    // Optionally integrate the branch lengths out under the prior. This is a side calculation that
    // starts from the maximum-likelihood branch lengths just found, moves them to the joint posterior
    // mode, evaluates the Laplace approximation, and restores the ML branch lengths -- so the value
    // returned here (the profile / ML log likelihood) is unaffected. The result is stored on the tree.
    if (computeMarginal == true && treeInfo != nullptr)
        {
        treeInfo->lnMarginalLikelihood = lnMarginalLikelihood();
        treeInfo->hasLnMarginalLikelihood = true;
        }

    return bestLnL;
}

double LikelihoodCalculator::mapRefineBranchStep(Node* p, double lambda) {

    // One Newton step, in theta = log(v), toward the mode of the log posterior for this branch,
    //   h(theta) = ell(v) + log lambda - lambda v + theta,     v = e^theta,
    // the last term being the Jacobian of the log transform. Working in theta sends the v -> 0
    // boundary to -infinity, so branches the prior pushes toward zero stay in the interior.
    double v = p->getBrlen();
    if (v < minBrlen)
        v = minBrlen;

    double* clsUp = condLikes[p->getIndex()]->beginUp();
    double* clsDn = condLikes[p->getIndex()]->beginDn();
    TransitionMatrix& tMat = tiProbs[p->getIndex()];
    TransitionMatrix& fMat = firstDerivative;
    TransitionMatrix& sMat = secondDerivative;
    tMat.setBrlen(v);
    fMat.setBrlen(v);
    sMat.setBrlen(v);

    double d1 = 0.0, d2 = 0.0;
    (void)calculateLFS(clsUp, clsDn, tMat, fMat, sMat, d1, d2);   // d1 = d ell/dv, d2 = d^2 ell/dv^2

    double grad = d1 * v - lambda * v + 1.0;          // dh/dtheta
    double hess = d2 * v * v + (d1 - lambda) * v;     // d^2 h/dtheta^2  (negative near the mode)

    double step = 0.0;
    if (hess < -1.0e-12)
        step = -grad / hess;                          // Newton
    else
        step = 0.01 * grad;                           // gradient fallback when not locally concave
    if (step >  2.0)
        step =  2.0;
    if (step < -2.0)
        step = -2.0;

    double vNew = std::exp(std::log(v) + step);
    if (vNew < minBrlen)
        vNew = minBrlen;
    if (vNew > maxBrlen)
        vNew = maxBrlen;

    p->setBrlen(vNew);
    tMat.setBrlen(vNew);
    return std::abs(vNew - v);
}

// Cholesky factor of a symmetric positive-definite n x n matrix (row-major), returning the log
// determinant. Returns false if the matrix is not positive definite (which, for -H, signals that the
// branch lengths are not at a proper maximum -- the caller then falls back or reports NaN).
static bool choleskyLogDet(const std::vector<double>& A, int n, double& logDet) {

    std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
    logDet = 0.0;
    for (int i = 0; i < n; ++i)
        {
        for (int j = 0; j <= i; ++j)
            {
            double sum = A[static_cast<size_t>(i) * n + j];
            for (int k = 0; k < j; ++k)
                sum -= L[static_cast<size_t>(i) * n + k] * L[static_cast<size_t>(j) * n + k];
            if (i == j)
                {
                if (sum <= 0.0)
                    return false;
                double d = std::sqrt(sum);
                L[static_cast<size_t>(i) * n + j] = d;
                logDet += 2.0 * std::log(d);
                }
            else
                {
                L[static_cast<size_t>(i) * n + j] = sum / L[static_cast<size_t>(j) * n + j];
                }
            }
        }
    return true;
}

void LikelihoodCalculator::collectBranchNodes(std::vector<Node*>& order) {

    order.clear();
    for (Node* p : tree->getDownPassSequence())
        if (p != treeRoot)
            order.push_back(p);
}

void LikelihoodCalculator::refreshAllConditionalLikelihoods(const std::vector<Node*>& order) {

    tree->markUpClsAsDirty();
    tree->markDnClsAsDirty();
    tree->setAllFlags(false);
    for (Node* nde : order)
        {
        updateUpConditionalLikelihoods(nde);
        updateDnConditionalLikelihoods(nde);
        }
}

void LikelihoodCalculator::computeThetaGradient(const std::vector<Node*>& order, double lambda, std::vector<double>& G) {

    // theta-gradient of the log posterior g: G_e = ell'_e b_e - lambda b_e + 1, at the current branches.
    refreshAllConditionalLikelihoods(order);
    G.assign(order.size(), 0.0);
    for (size_t k=0; k<order.size(); ++k)
        {
        Node* nde = order[k];
        int idx = nde->getIndex();
        double v = nde->getBrlen();
        TransitionMatrix& tMat = tiProbs[idx];
        TransitionMatrix& fMat = firstDerivative;
        TransitionMatrix& sMat = secondDerivative;
        tMat.setBrlen(v);
        fMat.setBrlen(v);
        sMat.setBrlen(v);
        double d1 = 0.0, d2 = 0.0;
        calculateLFS(condLikes[idx]->beginUp(), condLikes[idx]->beginDn(), tMat, fMat, sMat, d1, d2);
        G[k] = d1 * v - lambda * v + 1.0;
        }
}

void LikelihoodCalculator::hessianThetaFiniteDifference(const std::vector<Node*>& order, double lambda, std::vector<double>& H) {

    // Central finite differences of the theta-gradient. Done in theta = log(v), so perturbations never
    // cross the v > 0 boundary; leaves every branch length at its incoming value.
    const int d = static_cast<int>(order.size());
    const double h = 1.0e-4;
    H.assign(static_cast<size_t>(d) * d, 0.0);
    std::vector<double> Gp, Gm;

    for (int k=0; k<d; ++k)
        {
        double v0 = order[k]->getBrlen();
        double logv = std::log(v0);

        double vp = std::exp(logv + h);
        if (vp > maxBrlen) vp = maxBrlen;
        order[k]->setBrlen(vp);
        computeThetaGradient(order, lambda, Gp);

        double vm = std::exp(logv - h);
        if (vm < minBrlen) vm = minBrlen;
        order[k]->setBrlen(vm);
        computeThetaGradient(order, lambda, Gm);

        order[k]->setBrlen(v0);
        double denom = std::log(vp) - std::log(vm);       // exact spacing after any clamping
        for (int i=0; i<d; ++i)
            H[static_cast<size_t>(i) * d + k] = (Gp[i] - Gm[i]) / denom;
        }

    // symmetrize
    for (int i=0; i<d; ++i)
        for (int j=i+1; j<d; ++j)
            {
            double m = 0.5 * (H[static_cast<size_t>(i)*d+j] + H[static_cast<size_t>(j)*d+i]);
            H[static_cast<size_t>(i)*d+j] = H[static_cast<size_t>(j)*d+i] = m;
            }
}

int LikelihoodCalculator::branchHessianTheta(std::vector<double>& H) {

    // Public entry to extract the theta-space (log branch length) log-posterior Hessian at the tree's
    // current branch lengths, filled row-major (d x d); returns d. Its negative is the Gaussian
    // precision of the Laplace fit, so its inverse is the branch-length covariance.
    std::vector<Node*> order;
    collectBranchNodes(order);
    refreshAllConditionalLikelihoods(order);
    hessianThetaFiniteDifference(order, exponentialPriorRate, H);
    return static_cast<int>(order.size());
}

double LikelihoodCalculator::lnMarginalLikelihood(void) {

    // Marginal (integrated) likelihood of the topology under an IID Exp(lambda) prior on branch
    // lengths, by a Laplace approximation in theta = log(v):
    //
    //   L(tau) = integral P(D | tau, b) prod_e [ lambda e^{-lambda b_e} ] db
    //          ~ exp( g(theta*) ) (2 pi)^{d/2} det(-H)^{-1/2},
    //   g(theta) = ell(b) + sum_e [ log lambda - lambda b_e + theta_e ],   b_e = e^{theta_e}.
    //
    // theta* is the joint posterior mode; -H is the full theta-space Hessian there. The log transform
    // removes the b > 0 boundary the Exp(10) prior would otherwise run into. The Hessian is built by
    // hessianMethod: 0 = diagonal only (fast, no inter-branch curvature), 1 = full, by finite
    // differences of the analytic gradient (the inter-branch curvature retained).
    if (tree == nullptr)
        return std::numeric_limits<double>::quiet_NaN();

    const double lambda = exponentialPriorRate;
    const double LN_2PI = 1.837877066409345483560659;

    std::vector<Node*> order;
    collectBranchNodes(order);
    const int d = static_cast<int>(order.size());

    // Snapshot the ML branch lengths so the tree can be put back exactly as it was.
    std::vector<double> savedBrlen(numNodes, 0.0);
    for (Node* p : order)
        savedBrlen[p->getIndex()] = p->getBrlen();

    // 1. Refine the ML branch lengths to the joint MAP under the prior (coordinate Newton in theta).
    tree->markUpClsAsDirty();
    tree->markDnClsAsDirty();
    tree->setAllFlags(false);
    for (size_t rep=0; rep<10; ++rep)
        {
        double maxStep = 0.0;
        for (std::vector<Node*>::const_reverse_iterator p = order.rbegin(); p != order.rend(); ++p)
            {
            updateUpConditionalLikelihoods(*p);
            updateDnConditionalLikelihoods(*p);
            double s = mapRefineBranchStep(*p, lambda);
            if (s > maxStep)
                maxStep = s;
            markUpClsAsDirtyFromNode(*p);
            markDnClsAsDirtyFromNode(*p);
            }
        if (maxStep < 1.0e-8)
            break;
        }

    // 2. Height of the log posterior at the mode: ell(b*) + sum_e[ log lambda - lambda b*_e + log b*_e ].
    refreshAllConditionalLikelihoods(order);
    double lnLatMode = 0.0;
    double priorJac  = 0.0;
    for (Node* nde : order)
        {
        int idx = nde->getIndex();
        double v = nde->getBrlen();
        TransitionMatrix& tMat = tiProbs[idx];
        TransitionMatrix& fMat = firstDerivative;
        TransitionMatrix& sMat = secondDerivative;
        tMat.setBrlen(v);
        fMat.setBrlen(v);
        sMat.setBrlen(v);
        double d1 = 0.0, d2 = 0.0;
        lnLatMode = calculateLFS(condLikes[idx]->beginUp(), condLikes[idx]->beginDn(), tMat, fMat, sMat, d1, d2);
        priorJac += std::log(lambda) - lambda * v + std::log(v);
        }

    // 3. Build the theta-space Hessian and take (1/2) log det(-H). The full (finite-difference) Hessian
    //    can be indefinite for a tree whose mode sits against a near-zero branch, where the Gaussian
    //    curvature degenerates; when its Cholesky fails we fall back to the diagonal Laplace, whose
    //    -H_ee = 1 - ell''_e b*_e^2 is positive by construction, so every tree gets a finite marginal.
    double halfLogDet = 0.0;
    bool haveFull = false;

    if (hessianMethod != 0)
        {
        std::vector<double> Ht;
        hessianThetaFiniteDifference(order, lambda, Ht);
        std::vector<double> negH(static_cast<size_t>(d) * d, 0.0);
        for (size_t x=0; x<negH.size(); ++x)
            negH[x] = -Ht[x];
        double logDet = 0.0;
        if (choleskyLogDet(negH, d, logDet) == true)
            {
            halfLogDet = 0.5 * logDet;
            haveFull = true;
            }
        }

    if (haveFull == false)
        {
        // Diagonal Laplace: the chosen method when hessianMethod == 0, and the fallback whenever the
        // full Hessian is not positive definite. The finite-difference pass above leaves the branches at
        // the mode but the conditional likelihoods stale, so refresh them before reading curvatures.
        refreshAllConditionalLikelihoods(order);
        halfLogDet = 0.0;
        for (Node* nde : order)
            {
            int idx = nde->getIndex();
            double v = nde->getBrlen();
            tiProbs[idx].setBrlen(v);
            firstDerivative.setBrlen(v);
            secondDerivative.setBrlen(v);
            double d1 = 0.0, d2 = 0.0;
            calculateLFS(condLikes[idx]->beginUp(), condLikes[idx]->beginDn(), tiProbs[idx], firstDerivative, secondDerivative, d1, d2);
            double curv = 1.0 - d2 * v * v;
            if (curv < 1.0e-12) curv = 1.0e-12;
            halfLogDet += 0.5 * std::log(curv);
            }
        }

    double logMarginal = lnLatMode + priorJac + 0.5 * d * LN_2PI - halfLogDet;

    // 4. Restore the ML branch lengths.
    for (Node* p : order)
        {
        double v = savedBrlen[p->getIndex()];
        p->setBrlen(v);
        tiProbs[p->getIndex()].setBrlen(v);
        }

    return logMarginal;
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

