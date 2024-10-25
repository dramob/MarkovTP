# est_empiriques.py

import numpy as np

def est_empiriques(X, Y, cl1, cl2):
    p1 = np.mean(X == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[X == cl1])
    m2 = np.mean(Y[X == cl2])
    sig1 = np.std(Y[X == cl1])
    sig2 = np.std(Y[X == cl2])
    return p1, p2, m1, sig1, m2, sig2