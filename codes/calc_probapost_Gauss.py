

from scipy.stats import norm
import numpy as np

def calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2):
    f1 = p1 * norm.pdf(Y, m1, sig1)
    f2 = p2 * norm.pdf(Y, m2, sig2)
    total = f1 + f2
    # Éviter la division par zéro
    total[total == 0] = 1e-10
    Ppost_class1 = f1 / total
    Ppost_class2 = f2 / total
    Ppost = np.stack((Ppost_class1, Ppost_class2), axis=-1)
    return Ppost