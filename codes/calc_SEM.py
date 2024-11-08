import numpy as np

from calc_probapost_Gauss import calc_probapost_Gauss


def tirage_apost(Ppost, cl1, cl2, m, n):
    X_post = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            X_post[i, j] = cl1 if np.random.rand() < Ppost[i, j, 0] else cl2
    return X_post


from est_empiriques import est_empiriques


def calc_SEM(Y, p10, p20, m10, sig10, m20, sig20, nb_iter):
    p1, p2 = p10, p20
    m1, sig1 = m10, sig10
    m2, sig2 = m20, sig20
    m, n = Y.shape
    for _ in range(nb_iter):
        # E-step: calculer les probabilités a posteriori
        Ppost = calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2)
        # S-step: simuler une réalisation de X suivant la loi a posteriori
        X_post = tirage_apost(Ppost, 0, 1, m, n)
        # M-step: estimer les paramètres à partir de la réalisation simulée
        p1, p2, m1, sig1, m2, sig2 = est_empiriques(X_post, Y, 0, 1)
    return p1, p2, m1, sig1, m2, sig2
