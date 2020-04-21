import numpy as np
from scipy.stats import multivariate_normal

"""
Gaussian mixture model, trained using the
Expectation-Maximisation (EM) algorithm.

P.L.Green

"""


class GMM:

    def __init__(self, X, mu_init, C_init, pi_init, N_mixtures):
        """ Initialiser class method

        """

        self.X = np.vstack(X)   # Inputs always vertically stacked
        self.mu = mu_init       # Initial means of Gaussian mixture
        self.C = C_init         # Initial covariance matrices
        self.pi = pi_init       # Initial mixture proportions
        self.N_mixtures = N_mixtures      # No. components in mixture
        self.N, self.D = np.shape(self.X)  # No. data points and dimension of X
        self.EZ = np.zeros([self.N, N_mixtures])  # Initialise expected labels

    def expectation(self):
        """ The 'E' part of the EM algorithm.
        Finds the expected labels of each data point.

        """

        for n in range(self.N):
            den = 0.0
            for k in range(self.N_mixtures):
                den += self.pi[k] * multivariate_normal.pdf(self.X[n],
                                                            self.mu[k],
                                                            self.C[k])
            for k in range(self.N_mixtures):
                num = self.pi[k] * multivariate_normal.pdf(self.X[n],
                                                           self.mu[k],
                                                           self.C[k])
                self.EZ[n, k] = num/den

    def maximisation(self, X, L):
        """ The 'M' part of the EM algorithm.
        Finds the maximum likelihood parameters of our model.
        Here we use 'L' to represent labels.

        """

        for k in range(self.N_mixtures):
            Nk = np.sum(L[:, k])
            self.pi[k] = Nk / self.N

            # Note - should vectorise this next bit in the future as
            # it will be a lot faster
            self.mu[k] = 0.0
            for n in range(self.N):
                self.mu[k] += 1/Nk * L[n, k]*X[n]
            self.C[k] = np.zeros([self.D, self.D])
            for n in range(self.N):
                self.C[k] += 1/Nk * L[n, k] * (np.vstack(X[n] - self.mu[k]) *
                                               (X[n]-self.mu[k]))

    def train(self, Ni):
        """ Train Gaussian mixture model using the EM algorithm.

        """

        print('Training...')
        for i in range(Ni):
            print('Iteration', i)
            self.expectation()
            self.maximisation(self.X, self.EZ)
