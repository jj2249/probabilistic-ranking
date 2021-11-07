import scipy.linalg
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def gibbs_sample(G, M, num_iters):
    # M is the number of players
    # number of games
    N = G.shape[0] # 1801 games, G is 1801 by 2
    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((M, 1)) # length M array with each element in its own array
    # Array that will contain skill samples
    skill_samples = np.zeros((M, num_iters))
    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(M)
    # number of iterations of Gibbs
    ones = np.ones(N)
    for i in tqdm(range(num_iters)):
        # sample performance given differences in skills and outcomes
        t = np.zeros((N, 1))
        for g in range(N): # ones sample for each game
            # skill sample for player 1 minus skill sample for player 2
            s = w[G[g, 0]] - w[G[g, 1]]  # difference in skills
            # account for random performance
            t[g] = s + np.random.randn()  # Sample performance
            # y = +1 always so reject if performance invalid
            while t[g] < 0:  # rejection step
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((M, 1))
        for p in range(M): # for each player
            # the == operator acts as a delta function for sifting through player identities
            m[p] =  np.dot(t.T, (p==G[:, 0]).astype(int) - (p==G[:,1 ]).astype(int)) # TODO: COMPLETE THIS LINE
        iS = np.zeros((M, M))  # Container for sum of precision matrices (likelihood terms)

        for j in range(M):
            for k in range(j+1):
                # use expressions in the lecture notes
                if j == k:
                    iS[j, k] = np.sum((j==G[:,0]).astype(int)) + np.sum((j==G[:,1]).astype(int))
                else:
                    iS[j, k] = -np.sum((j==G[:,0]).astype(int)*(k==G[:,1]).astype(int)) -np.sum((k==G[:,0]).astype(int)*(j==G[:,1]).astype(int))
                    # exploit symmetry
                    iS[k, j] = iS[j, k]
        # TODO: Build the iS matrix

        # Posterior precision matrix
        iSS = iS + np.diag(1. / pv)
        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(M, 1), check_finite=False)
        skill_samples[:, i] = w[:, 0]
    return skill_samples


    