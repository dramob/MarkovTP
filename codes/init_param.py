import numpy as np
from sklearn.cluster import KMeans


def init_param(Y, iter_KM=10):
    m, n = Y.shape
    Y_flat = Y.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=iter_KM).fit(Y_flat)
    labels = kmeans.labels_.reshape(m, n)
    # Assign classes based on means
    m1_cluster = np.mean(Y[labels == 0])
    m2_cluster = np.mean(Y[labels == 1])
    if m1_cluster < m2_cluster:
        cl1, cl2 = 0, 1
    else:
        cl1, cl2 = 1, 0
    # Estimate parameters
    p1 = np.mean(labels == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[labels == cl1])
    m2 = np.mean(Y[labels == cl2])
    sig1 = np.std(Y[labels == cl1])
    sig2 = np.std(Y[labels == cl2])
    return p1, p2, m1, sig1, m2, sig2
