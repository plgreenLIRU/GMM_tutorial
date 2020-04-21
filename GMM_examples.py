import numpy as np
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as mvn
from GMM import *
import pickle

"""
Example Gaussian Mixture Model code to accompany EM tutorial.

P.L.Green

University of Liverpool (1)
Engineering Data Analytics Ltd. (2)

(1) p.l.green@liverpool.ac.uk
(2) engineeringdataanalytics@gmail.com
"""

# Load 2D data with 3 clusters 
file = open('3_cluster_data_unsupervised.dat', 'rb')
X, N = pickle.load(file)
file.close()

# Create and train GMM object (our code)
mu = []
C = []
for i in range(3):
    mu.append(np.random.randn(2))
    C.append(np.eye(2))
pi = np.array([1/3, 1/3, 1/3])
gmm = GMM(X=X, mu_init=mu, C_init=C, pi_init=pi, N_mixtures=3)
gmm.train(Ni=10)

# Print results
print('\n### Our code ###')
for k in range(3):
    print('Mean', k+1, ' = ', gmm.mu[k])
    print('Covariance matrix', k+1, ' = ', gmm.C[k])
    print('Mixture proportion', k+1, ' = ', gmm.pi[k], '\n')

# Plot results
r1 = np.linspace(np.min(gmm.X[:, 0]), np.max(gmm.X[:, 1]), 100)
r2 = np.linspace(np.min(gmm.X[:, 1]), np.max(gmm.X[:, 1]), 100)
x_r1, x_r2 = np.meshgrid(r1, r2)
pos = np.empty(x_r1.shape + (2, ))
pos[:, :, 0] = x_r1
pos[:, :, 1] = x_r2
for k in range(gmm.N_mixtures):
    p = multivariate_normal(gmm.mu[k], gmm.C[k])
    plt.contour(x_r1, x_r2, p.pdf(pos))
plt.plot(gmm.X[:, 0], gmm.X[:, 1], 'o',
         markerfacecolor='red',
         markeredgecolor='black')

plt.show()
