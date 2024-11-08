# EM_Gibbs_segmentation.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
import os

def lit_image(chemin_image):
    from PIL import Image
    image = Image.open(chemin_image).convert('L')
    return np.array(image)

def affiche_image(titre, image):
    plt.figure()
    plt.title(titre)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def identif_classes(X):
    classes = np.unique(X)
    cl1, cl2 = classes[0], classes[1]
    return cl1, cl2

def bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2):
    Y = np.zeros_like(X, dtype=float)
    Y[X == cl1] = np.random.normal(m1, sig1, size=np.sum(X == cl1))
    Y[X == cl2] = np.random.normal(m2, sig2, size=np.sum(X == cl2))
    return Y

def taux_erreur(X_estime, X_true):
    return np.mean(X_estime != X_true)

def nouvelle_image(Y):
    # Ajoute une bordure de 1 pixel autour de Y
    m, n = Y.shape
    Ytrans = np.zeros((m + 2, n + 2))
    Ytrans[1:-1, 1:-1] = Y
    return Ytrans

def calc_proba_champs(alpha):
    # Calcule les probabilités de transition pour le champ de Markov
    proba = np.zeros(5)
    for k in range(5):
        proba[k] = np.exp(alpha * k)
    proba = proba / np.sum(proba)
    return proba

def MPM_proba_gauss(Ytrans, classes, m1, sig1, m2, sig2, proba, nb_iter, nb_simu):
    m, n = Ytrans.shape
    X_simus = np.zeros((nb_simu, m, n))
    for simu in range(nb_simu):
        X = np.random.choice(classes, size=(m, n))
        for iter in range(nb_iter):
            for i in range(1, m - 1):
                for j in range(1, n - 1):
                    voisins = [X[i-1,j], X[i+1,j], X[i,j-1], X[i,j+1]]
                    nb_voisins_meme_classe = np.sum(voisins == classes[0])
                    proba_classe1 = proba[nb_voisins_meme_classe] * norm.pdf(Ytrans[i,j], m1, sig1)
                    proba_classe2 = proba[4 - nb_voisins_meme_classe] * norm.pdf(Ytrans[i,j], m2, sig2)
                    P_total = proba_classe1 + proba_classe2
                    p1 = proba_classe1 / P_total
                    X[i,j] = np.random.choice(classes, p=[p1, 1 - p1])
        X_simus[simu] = X
    # Estimation MPM
    X_seg_trans = np.mean(X_simus, axis=0)
    X_seg_trans = np.where(X_seg_trans >= 0.5, classes[0], classes[1])
    return X_seg_trans

def init_param_EM(Y):
    # Initialisation avec KMeans
    Y_flat = Y.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(Y_flat)
    labels = kmeans.labels_.reshape(Y.shape)
    classe_values = np.unique(labels)
    cl1_label, cl2_label = classe_values[0], classe_values[1]

    # Calcul des paramètres initiaux
    p1 = np.mean(labels == cl1_label)
    p2 = 1 - p1
    m1 = np.mean(Y[labels == cl1_label])
    m2 = np.mean(Y[labels == cl2_label])
    sig1 = np.std(Y[labels == cl1_label])
    sig2 = np.std(Y[labels == cl2_label])

    alpha = 1  # Paramètre du champ
    proba = calc_proba_champs(alpha)

    return proba, m1, sig1, m2, sig2

def EM_gibbsien_Gauss(Y, classes, m1, sig1, m2, sig2, proba, nb_iter_Gibbs_EM, nb_simu_EM):
    m, n = Y.shape
    Post = np.zeros((m, n, 2))
    N = np.zeros((5, 2))  # N_{k,j}

    for simu in range(nb_simu_EM):
        X = np.random.choice(classes, size=(m+2, n+2))  # Avec bords pour Ytrans
        Ytrans = nouvelle_image(Y)

        for iter in range(nb_iter_Gibbs_EM):
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    voisins = [X[i-1,j], X[i+1,j], X[i,j-1], X[i,j+1]]
                    nb_voisins_classe1 = np.sum(voisins == classes[0])
                    proba_classe1 = proba[nb_voisins_classe1] * norm.pdf(Ytrans[i,j], m1, sig1)
                    proba_classe2 = proba[4 - nb_voisins_classe1] * norm.pdf(Ytrans[i,j], m2, sig2)
                    P_total = proba_classe1 + proba_classe2
                    p1 = proba_classe1 / P_total
                    X[i,j] = np.random.choice(classes, p=[p1, 1 - p1])

        X_estime = X[1:-1, 1:-1]  # Retirer les bords
        for i in range(m):
            for j in range(n):
                if X_estime[i,j] == classes[0]:
                    Post[i,j,0] += 1
                else:
                    Post[i,j,1] += 1

                # Comptage pour N_{k,j}
                voisins = []
                if i > 0:
                    voisins.append(X_estime[i-1,j])
                if i < m -1:
                    voisins.append(X_estime[i+1,j])
                if j > 0:
                    voisins.append(X_estime[i,j-1])
                if j < n -1:
                    voisins.append(X_estime[i,j+1])
                nb_voisins_classe1 = np.sum(np.array(voisins) == classes[0])
                N[nb_voisins_classe1, 0] += (X_estime[i,j] == classes[0])
                N[nb_voisins_classe1, 1] += (X_estime[i,j] == classes[1])

    Post = Post / nb_simu_EM  # Normaliser pour obtenir des probabilités
    # Mise à jour de proba à partir de N
    proba = N[:,0] / (N[:,0] + N[:,1])
    proba = proba / np.sum(proba)  # Normaliser

    return proba, Post

def estim_param_bruit_gauss_EM(Y, classes, Post):
    m1 = np.sum(Post[:,:,0] * Y) / np.sum(Post[:,:,0])
    m2 = np.sum(Post[:,:,1] * Y) / np.sum(Post[:,:,1])
    sig1 = np.sqrt(np.sum(Post[:,:,0] * (Y - m1)**2) / np.sum(Post[:,:,0]))
    sig2 = np.sqrt(np.sum(Post[:,:,1] * (Y - m2)**2) / np.sum(Post[:,:,1]))
    return m1, sig1, m2, sig2

def EM_gauss(Y, classes, m1, sig1, m2, sig2, proba, nb_iter_EM, nb_iter_Gibbs_EM, nb_simu_EM):
    for iter in range(nb_iter_EM):
        print(f'Itération EM {iter+1}/{nb_iter_EM}')
        proba, Post = EM_gibbsien_Gauss(Y, classes, m1, sig1, m2, sig2, proba, nb_iter_Gibbs_EM, nb_simu_EM)
        m1, sig1, m2, sig2 = estim_param_bruit_gauss_EM(Y, classes, Post)
        print(f'Paramètres estimés : m1={m1:.2f}, sig1={sig1:.2f}, m2={m2:.2f}, sig2={sig2:.2f}')
    return proba, m1, sig1, m2, sig2

def taux_erreur_segmentation(X_seg, X_true, classes):
    erreur1 = np.mean(X_seg != X_true)
    # Inversion des labels
    X_seg_inverted = np.where(X_seg == classes[0], classes[1], classes[0])
    erreur2 = np.mean(X_seg_inverted != X_true)
    erreur = min(erreur1, erreur2)
    if erreur == erreur2:
        X_seg = X_seg_inverted
    return erreur, X_seg

if __name__ == "__main__":
    chemin_image = './images_BW/beee2.bmp'
    X = lit_image(chemin_image)
    affiche_image('Image originale', X)

    cl1, cl2 = identif_classes(X)
    classes = [cl1, cl2]

    # Bruitage
    m1_true, sig1_true = 1, 1
    m2_true, sig2_true = 4, 1
    Y = bruit_gauss(X, cl1, cl2, m1_true, sig1_true, m2_true, sig2_true)
    affiche_image('Image bruitée', Y)

    # Initialisation des paramètres
    proba, m1_est, sig1_est, m2_est, sig2_est = init_param_EM(Y)

    nb_iter_EM = 5
    nb_iter_Gibbs_EM = 5
    nb_simu_EM = 3

    proba, m1_est, sig1_est, m2_est, sig2_est = EM_gauss(Y, classes, m1_est, sig1_est, m2_est, sig2_est, proba, nb_iter_EM, nb_iter_Gibbs_EM, nb_simu_EM)

    # Segmentation finale
    Ytrans = nouvelle_image(Y)
    nb_iter = 10
    nb_simu = 5
    X_seg_trans = MPM_proba_gauss(Ytrans, classes, m1_est, sig1_est, m2_est, sig2_est, proba, nb_iter, nb_simu)
    X_seg = X_seg_trans[1:-1,1:-1]

    affiche_image('Image segmentée - EM Gibbs', X_seg)

    # Calcul du taux d'erreur avec inversion des labels si nécessaire
    erreur, X_seg_corrected = taux_erreur_segmentation(X_seg, X, classes)
    print(f'Taux d\'erreur de segmentation : {erreur * 100:.2f}%')

    # Sauvegarde des outputs
    if not os.path.exists('question25'):
        os.makedirs('question25')
    plt.imsave('question25/image_bruitee.png', Y, cmap='gray')
    plt.imsave('question25/image_segmentee.png', X_seg_corrected, cmap='gray')