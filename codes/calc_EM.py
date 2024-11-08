import numpy as np

from calc_probapost_Gauss import calc_probapost_Gauss


def calc_EM(Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM):
    p1, p2 = p10, p20
    m1, sig1 = m10, sig10
    m2, sig2 = m20, sig20
    for _ in range(nb_iterEM):
        # E-step
        Ppost = calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2)
        # M-step
        sum_Ppost_class1 = np.sum(Ppost[:, :, 0])
        sum_Ppost_class2 = np.sum(Ppost[:, :, 1])
        p1 = sum_Ppost_class1 / (sum_Ppost_class1 + sum_Ppost_class2)
        p2 = 1 - p1
        m1 = np.sum(Ppost[:, :, 0] * Y) / sum_Ppost_class1
        m2 = np.sum(Ppost[:, :, 1] * Y) / sum_Ppost_class2
        sig1 = np.sqrt(np.sum(Ppost[:, :, 0] * (Y - m1) ** 2) / sum_Ppost_class1)
        sig2 = np.sqrt(np.sum(Ppost[:, :, 1] * (Y - m2) ** 2) / sum_Ppost_class2)
    return p1, p2, m1, sig1, m2, sig2
