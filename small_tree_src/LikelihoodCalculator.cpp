#include <algorithm>
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
LikelihoodCalculator::BranchOptimizer LikelihoodCalculator::branchOptimizer = LikelihoodCalculator::BranchOptimizer::Joint;



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

    // scratch space for the joint three-branch optimizer
    c0.resize(numSites);
    cab.resize(numSites);
    cac.resize(numSites);
    cbc.resize(numSites);
    cabc.resize(numSites);
    nodeL.resize(numSites);
    nodeTrialL.resize(numSites);
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
                 
        // Propose the next branch length. Newton where the log-likelihood is locally concave; otherwise an
        // uphill step. Either way the multiplicative change in v is capped (|delta log v| <= 2, matching the
        // cap in mapRefineBranchStep) so no single iteration can leap across the admissible range. The old
        // fallback, 0.01 * firstDerivative, was unscaled -- the derivative grows with the number of sites --
        // and routinely slammed into maxBrlen, setting off the oscillation that best-point tracking then hid.
        // Near the maximum the Newton step is small and the cap never binds, so quadratic convergence is kept.
        const double maxLogStep = 2.0;
        const double maxFactor  = std::exp(maxLogStep);
        double vNew;
        if (std::abs(secondDerivative) >= 1e-12 && secondDerivative < 0.0)
            vNew = oldV - firstDerivative / secondDerivative;                       // Newton
        else
            vNew = (firstDerivative > 0.0) ? oldV * maxFactor : oldV / maxFactor;   // bounded uphill step
        if (vNew > oldV * maxFactor)                                                // cap the leap (also
            vNew = oldV * maxFactor;                                                // keeps v positive if
        if (vNew < oldV / maxFactor)                                                // Newton overshoots
            vNew = oldV / maxFactor;                                                // below zero)
        v = vNew;
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
    
    // Return the likelihood at bestBrlen -- the value the branch was just set to -- not lnL, which is
    // the likelihood at the last point Newton happened to evaluate. The two coincide only when the final
    // iterate was also the best; when the iteration oscillates or hits the cap they differ, and the
    // caller's convergence test then sees a value inconsistent with the tree it is testing.
    return bestLnL;
}

double LikelihoodCalculator::lnLikelihood(void) {

    if (tree == nullptr)
        return 0.0;

    double bestLnL = (branchOptimizer == BranchOptimizer::Joint) ? lnLikelihoodJoint() : lnLikelihoodSingleBranch();

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

// The earlier optimizer: Newton on one branch at a time, sweeping the tree until the log likelihood
// stops changing. It stalls on likelihood ridges -- where a branch shrinks toward zero and length
// can be traded between the branches meeting at a node -- creeping along them by less than the
// tolerance per sweep and stopping well short of the maximum (by 33-49 log-likelihood units in tests).
// Kept so the two optimizers can be compared; lnLikelihoodJoint() is the default.
double LikelihoodCalculator::lnLikelihoodSingleBranch(void) {

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

    return bestLnL;
}

// ---- Joint three-branch optimization (after FastTbr's JcLikelihood) ----
//
// The conditional likelihoods follow this class's convention: the "Up" vector of a node is the
// subtree below it, evaluated at the node; the "Dn" vector is everything outside that subtree,
// evaluated at the node's ancestor. The branch above node p therefore joins Up[p] and Dn[p].
//
// Branch lengths are optimized three at a time, the three branches meeting at an interior node v:
// the branch above v, whose far end is Dn[v], and the branches to v's children d0 and d1, whose far
// ends are Up[d0] and Up[d1]. Under JC69, with e = exp(-4b/3), (P(b) X)_s = sum(X)/4 + e (X_s -
// sum(X)/4), and because the deviations X_s - sum(X)/4 sum to zero the likelihood of pattern i
// collapses to
//
//      L_i = c0 + cab ea eb + cac ea ec + cbc eb ec + cabc ea eb ec,
//
// five coefficients formed once per node. A safeguarded Newton search in (ea, eb, ec) then costs one
// pass over the patterns per evaluation and needs no logarithms: steps are accepted on a lower
// bound for the gain in log likelihood (log x >= 1 - 1/x). Optimizing the three together lets
// length move along the ridges on which the single-branch optimizer stalls.
//
// A sweep visits the interior nodes from the root's descendant downward, recomputing each child's
// Dn vector on the way down and each node's Up vector on the way back, so every node is optimized
// with current conditionals; tip branches are optimized by their parent. Sweeps repeat until the
// log likelihood improves by less than 1e-6.
double LikelihoodCalculator::lnLikelihoodJoint(void) {

    const std::vector<Node*>& postOrder = tree->getDownPassSequence();
    for (Node* p : postOrder)
        {
        if (p != treeRoot)
            p->setBrlen(std::min(std::max(p->getBrlen(), minBrlen), maxBrlen));
        }

    // every Up vector from the current branch lengths; the root descendant's Dn vector is the root
    // tip's data, placed there by setTree()
    for (Node* p : postOrder)
        {
        if (p != treeRoot && p->getIsTip() == false)
            jcUp(p);
        }

    const double tolerance = 1.0e-6;
    const int    maxPasses = 100;
    double lnL = rootEdgeLnL();
    for (int pass=0; pass<maxPasses; pass++)
        {
        smoothNode(treeRootDescendant);
        double newLnL = rootEdgeLnL();
        bool done = (newLnL - lnL < tolerance);
        lnL = newLnL;
        if (done == true)
            break;
        }

    // Leave the cached transition matrices consistent with the branch lengths, and every conditional
    // likelihood marked stale, for any later calculation that uses the dirty-flag machinery.
    for (Node* p : postOrder)
        {
        if (p != treeRoot)
            tiProbs[p->getIndex()].setBrlen(p->getBrlen());
        }
    tree->markUpClsAsDirty();
    tree->markDnClsAsDirty();
    tree->setAllFlags(false);

    return lnL;
}

// Up[p] = P(b_d0) Up[d0] * P(b_d1) Up[d1], with (P(b) v)_s = (1-e)/4 sum(v) + e v_s under JC69
void LikelihoodCalculator::jcUp(Node* p) {

    Node* d0 = p->getFirstDescendant();
    Node* d1 = d0->getNextSibling();
    const double* a = condLikes[d0->getIndex()]->beginUp();
    const double* b = condLikes[d1->getIndex()]->beginUp();
    double e0 = std::exp(-4.0 * d0->getBrlen() / 3.0);
    double e1 = std::exp(-4.0 * d1->getBrlen() / 3.0);
    double q0 = 0.25 * (1.0 - e0);
    double q1 = 0.25 * (1.0 - e1);
    double* out = condLikes[p->getIndex()]->beginUp();

    for (size_t c=0; c<numSites; c++, a+=4, b+=4, out+=4)
        {
        double c0s = q0 * (a[0] + a[1] + a[2] + a[3]);
        double c1s = q1 * (b[0] + b[1] + b[2] + b[3]);
        for (int s=0; s<4; s++)
            out[s] = (c0s + e0 * a[s]) * (c1s + e1 * b[s]);
        }
}

// Dn[c] = P(b_anc) Dn[anc] * P(b_sis) Up[sis], with anc the ancestor of c and sis its sister: the
// conditional likelihood at anc of everything outside the subtree below c
void LikelihoodCalculator::jcDn(Node* c) {

    Node* anc = c->getAncestor();
    Node* sis = c->getSister();
    const double* a = condLikes[anc->getIndex()]->beginDn();
    const double* b = condLikes[sis->getIndex()]->beginUp();
    double ea = std::exp(-4.0 * anc->getBrlen() / 3.0);
    double es = std::exp(-4.0 * sis->getBrlen() / 3.0);
    double qa = 0.25 * (1.0 - ea);
    double qs = 0.25 * (1.0 - es);
    double* out = condLikes[c->getIndex()]->beginDn();

    for (size_t k=0; k<numSites; k++, a+=4, b+=4, out+=4)
        {
        double cas = qa * (a[0] + a[1] + a[2] + a[3]);
        double css = qs * (b[0] + b[1] + b[2] + b[3]);
        for (int s=0; s<4; s++)
            out[s] = (cas + ea * a[s]) * (css + es * b[s]);
        }
}

// The log likelihood across the branch above the root's descendant. With A = Up, B = Dn, S = sum_s
// A_s B_s and T = sum(A) sum(B), L_i = 1/4 sum_s A_s [(1-e)/4 sum(B) + e B_s] = T/16 + e (S - T/4) / 4.
double LikelihoodCalculator::rootEdgeLnL(void) {

    const double* a = condLikes[treeRootDescendant->getIndex()]->beginUp();
    const double* b = condLikes[treeRootDescendant->getIndex()]->beginDn();
    double e = std::exp(-4.0 * treeRootDescendant->getBrlen() / 3.0);
    double lnL = 0.0;
    for (size_t i=0; i<numSites; i++, a+=4, b+=4)
        {
        double S = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
        double T = (a[0] + a[1] + a[2] + a[3]) * (b[0] + b[1] + b[2] + b[3]);
        double L = T / 16.0 + e * (S - 0.25 * T) / 4.0;
        lnL += patternCount[i] * std::log(L);
        }
    return lnL;
}

// On entry Dn[v] and the Up vectors of v's children are current. Optimize the three branches at v,
// then every interior node below it; on return Up[v] is current again.
void LikelihoodCalculator::smoothNode(Node* v) {

    optimizeNode(v);

    Node* d0 = v->getFirstDescendant();
    Node* d1 = d0->getNextSibling();
    if (d0->getIsTip() == false)
        {
        jcDn(d0);
        smoothNode(d0);
        }
    if (d1->getIsTip() == false)
        {
        jcDn(d1);
        smoothNode(d1);
        }
    jcUp(v);
}

// The likelihood of pattern i at interior node v as a polynomial in (ea, eb, ec), the transformed
// lengths of the branch above v and the branches to its two children. With far-end vectors A =
// Dn[v], B = Up[d0], C = Up[d1], a0 = sum(A)/4 and a1_s = A_s - a0 (likewise b and c),
// L = 1/4 sum_s (a0 + ea a1_s)(b0 + eb b1_s)(c0 + ec c1_s); the terms linear in one e vanish
// because sum_s a1_s = 0.
void LikelihoodCalculator::setNodeCoefficients(Node* v) {

    Node* d0 = v->getFirstDescendant();
    Node* d1 = d0->getNextSibling();
    const double* A = condLikes[v->getIndex()]->beginDn();
    const double* B = condLikes[d0->getIndex()]->beginUp();
    const double* C = condLikes[d1->getIndex()]->beginUp();
    for (size_t i=0; i<numSites; i++, A+=4, B+=4, C+=4)
        {
        double a0 = 0.25 * (A[0] + A[1] + A[2] + A[3]);
        double b0 = 0.25 * (B[0] + B[1] + B[2] + B[3]);
        double cc = 0.25 * (C[0] + C[1] + C[2] + C[3]);
        double sab = 0.0, sac = 0.0, sbc = 0.0, sabc = 0.0;
        for (int s=0; s<4; s++)
            {
            double a1 = A[s] - a0, b1 = B[s] - b0, c1 = C[s] - cc;
            sab += a1 * b1;
            sac += a1 * c1;
            sbc += b1 * c1;
            sabc += a1 * b1 * c1;
            }
        c0[i] = a0 * b0 * cc;
        cab[i] = 0.25 * cc * sab;
        cac[i] = 0.25 * b0 * sac;
        cbc[i] = 0.25 * a0 * sbc;
        cabc[i] = 0.25 * sabc;
        }
}

// Gradient g[3] and Hessian h[3][3] (row major) of the log likelihood with respect to e = (ea, eb, ec)
// at the node whose coefficients are set. The pattern likelihoods at e are left in nodeL for nodeGain().
void LikelihoodCalculator::nodeDerivatives(const double* e, double* g, double* h) {

    const double ea = e[0], eb = e[1], ec = e[2];
    double g0 = 0.0, g1 = 0.0, g2 = 0.0;
    double h00 = 0.0, h11 = 0.0, h22 = 0.0, h01 = 0.0, h02 = 0.0, h12 = 0.0;
    for (size_t i=0; i<numSites; i++)
        {
        double xab = cab[i], xac = cac[i], xbc = cbc[i], xabc = cabc[i];
        double L  = c0[i] + xab * ea * eb + xac * ea * ec + xbc * eb * ec + xabc * ea * eb * ec;
        double La = xab * eb + xac * ec + xabc * eb * ec;
        double Lb = xab * ea + xbc * ec + xabc * ea * ec;
        double Lc = xac * ea + xbc * eb + xabc * ea * eb;
        double w = patternCount[i];
        double r = 1.0 / L;
        nodeL[i] = L;
        double da = La * r, db = Lb * r, dc = Lc * r;
        g0 += w * da;
        g1 += w * db;
        g2 += w * dc;
        h00 -= w * da * da;
        h11 -= w * db * db;
        h22 -= w * dc * dc;
        h01 += w * ((xab + xabc * ec) * r - da * db);
        h02 += w * ((xac + xabc * eb) * r - da * dc);
        h12 += w * ((xbc + xabc * ea) * r - db * dc);
        }
    g[0] = g0;
    g[1] = g1;
    g[2] = g2;
    h[0] = h00; h[1] = h01; h[2] = h02;
    h[3] = h01; h[4] = h11; h[5] = h12;
    h[6] = h02; h[7] = h12; h[8] = h22;
}

// Change in log likelihood from the point of the last nodeDerivatives() call to e. Because
// log x >= 1 - 1/x, sum w log(L'/L) >= sum w (1 - L/L'); when that bound is positive it is returned
// (the step is known to be uphill) and no logarithms are taken. Only otherwise is the exact change
// computed.
double LikelihoodCalculator::nodeGain(const double* e) {

    const double ea = e[0], eb = e[1], ec = e[2];
    double bound = 0.0;
    for (size_t i=0; i<numSites; i++)
        {
        double L = c0[i] + cab[i] * ea * eb + cac[i] * ea * ec + cbc[i] * eb * ec + cabc[i] * ea * eb * ec;
        nodeTrialL[i] = L;
        bound += patternCount[i] * (1.0 - nodeL[i] / L);
        }
    if (bound > 0.0)
        return bound;

    double gain = 0.0;
    for (size_t i=0; i<numSites; i++)
        gain += patternCount[i] * std::log(nodeTrialL[i] / nodeL[i]);
    return gain;
}

// Jointly optimize the three branches at interior node v: the branch above v and the branches to its
// two children. Projected Newton in (ea, eb, ec) on the box [eMin, eMax]^3, with backtracking so that
// every accepted step increases the log likelihood. Where the Hessian is not negative definite the
// step falls back to the diagonal (one-branch) Newton step; the log likelihood is concave in each e
// separately, so that step is always uphill.
void LikelihoodCalculator::optimizeNode(Node* v) {

    setNodeCoefficients(v);

    const double eMin = std::exp(-4.0 * maxBrlen / 3.0);
    const double eMax = std::exp(-4.0 * minBrlen / 3.0);
    Node* d0 = v->getFirstDescendant();
    Node* slot[3] = { v, d0, d0->getNextSibling() };

    double e[3], g[3], h[9], trial[3];
    for (int k=0; k<3; k++)
        e[k] = std::min(std::max(std::exp(-4.0 * slot[k]->getBrlen() / 3.0), eMin), eMax);

    nodeDerivatives(e, g, h);
    for (int iter=0; iter<50; iter++)
        {
        // coordinates held at a bound by the gradient are fixed this iteration
        bool free_[3];
        int nFree = 0;
        for (int k=0; k<3; k++)
            {
            free_[k] = !((e[k] >= eMax && g[k] > 0.0) || (e[k] <= eMin && g[k] < 0.0));
            if (free_[k] == true)
                nFree++;
            }
        if (nFree == 0)
            break;

        // Newton direction on the free coordinates: solve (-H) d = g by Cholesky
        double d[3] = { 0.0, 0.0, 0.0 };
        int idx[3], m = 0;
        for (int k=0; k<3; k++)
            if (free_[k] == true)
                idx[m++] = k;
        double A[9], L[9] = { 0.0 };
        for (int i=0; i<m; i++)
            for (int j=0; j<m; j++)
                A[3*i+j] = -h[3*idx[i]+idx[j]];
        bool pd = true;
        for (int i=0; i<m && pd == true; i++)
            {
            for (int j=0; j<=i; j++)
                {
                double sum = A[3*i+j];
                for (int k=0; k<j; k++)
                    sum -= L[3*i+k] * L[3*j+k];
                if (i == j)
                    {
                    if (sum <= 1.0e-300)
                        {
                        pd = false;
                        break;
                        }
                    L[3*i+i] = std::sqrt(sum);
                    }
                else
                    {
                    L[3*i+j] = sum / L[3*j+j];
                    }
                }
            }
        if (pd == true)
            {
            double y[3];
            for (int i=0; i<m; i++)
                {
                double sum = g[idx[i]];
                for (int k=0; k<i; k++)
                    sum -= L[3*i+k] * y[k];
                y[i] = sum / L[3*i+i];
                }
            for (int i=m-1; i>=0; i--)
                {
                double sum = y[i];
                for (int k=i+1; k<m; k++)
                    sum -= L[3*k+i] * d[idx[k]];
                d[idx[i]] = sum / L[3*i+i];
                }
            }
        else
            {
            for (int k=0; k<3; k++)
                if (free_[k] == true && h[4*k] < 0.0)
                    d[k] = -g[k] / h[4*k];
            }

        // converged when the quadratic model the step came from (the full one, or the diagonal one
        // when the Hessian is not negative definite) predicts a negligible gain
        double predicted = 0.0;
        for (int i=0; i<3; i++)
            {
            predicted += g[i] * d[i];
            if (pd == true)
                {
                for (int j=0; j<3; j++)
                    predicted += 0.5 * d[i] * h[3*i+j] * d[j];
                }
            else
                {
                predicted += 0.5 * d[i] * h[4*i] * d[i];
                }
            }
        if (predicted < 1.0e-10)
            break;

        // backtrack along the projected step until the log likelihood improves
        double t = 1.0, gain = 0.0;
        bool improved = false;
        for (int bt=0; bt<30; bt++)
            {
            for (int k=0; k<3; k++)
                trial[k] = std::min(std::max(e[k] + t * d[k], eMin), eMax);
            gain = nodeGain(trial);
            if (gain > 0.0)
                {
                improved = true;
                break;
                }
            t *= 0.25;
            }
        if (improved == false)
            break;

        double change = 0.0;
        for (int k=0; k<3; k++)
            {
            change = std::max(change, std::fabs(trial[k] - e[k]) / e[k]);
            e[k] = trial[k];
            }
        if (gain < 1.0e-10 || change < 1.0e-10)
            break;
        nodeDerivatives(e, g, h);
        }

    for (int k=0; k<3; k++)
        slot[k]->setBrlen(std::min(std::max(-0.75 * std::log(e[k]), minBrlen), maxBrlen));
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

    // The conditional-likelihood updates read the cached transition matrices (tiProbs), not the nodes'
    // branch lengths, so the two must agree before any CL is rebuilt. The finite-difference Hessian
    // perturbs a node's branch length directly; without this sync that perturbation never reaches the
    // CLs of the other branches and the inter-branch (off-diagonal) curvature is lost.
    for (Node* nde : order)
        tiProbs[nde->getIndex()].setBrlen(nde->getBrlen());

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
    //    -H_ee = -ell''_e b*_e^2 - (ell'_e - lambda) b*_e is used directly, so every tree gets a finite marginal.
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
            // -H_ee in theta space is -ell''_e b^2 - (ell'_e - lambda) b. The shortcut 1 - ell'' b^2 is
            // only equal to this where the gradient vanishes, (ell'-lambda) b = -1; it is wrong at a branch
            // clamped to minBrlen/maxBrlen -- exactly the case this fallback is invoked for.
            double curv = -(d2 * v * v + (d1 - lambda) * v);
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

