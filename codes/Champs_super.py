# Champs_super.py

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans


def lit_image(chemin_image):
    from PIL import Image

    image = Image.open(chemin_image).convert("L")
    return np.array(image)


def affiche_image(titre, image):
    plt.figure()
    plt.title(titre)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
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
                    voisins = [X[i - 1, j], X[i + 1, j], X[i, j - 1], X[i, j + 1]]
                    nb_voisins_meme_classe = np.sum(voisins == classes[0])
                    proba_classe1 = proba[nb_voisins_meme_classe] * norm.pdf(
                        Ytrans[i, j], m1, sig1
                    )
                    proba_classe2 = proba[4 - nb_voisins_meme_classe] * norm.pdf(
                        Ytrans[i, j], m2, sig2
                    )
                    P = proba_classe1 + proba_classe2
                    p1 = proba_classe1 / P
                    X[i, j] = np.random.choice(classes, p=[p1, 1 - p1])
        X_simus[simu] = X
    # Estimation MPM
    X_seg_trans = np.mean(X_simus, axis=0)
    X_seg_trans = np.where(X_seg_trans >= 0.5, classes[0], classes[1])
    return X_seg_trans


if __name__ == "__main__":
    # Lecture de l'image
    chemin_image = "./images_BW/beee2.bmp"
    X = lit_image(chemin_image)
    affiche_image("Image originale", X)

    # Identification des classes
    cl1, cl2 = identif_classes(X)
    classes = [cl1, cl2]

    # Ajout de bruit
    m1_true, sig1_true = 1, 1
    m2_true, sig2_true = 4, 1
    Y = bruit_gauss(X, cl1, cl2, m1_true, sig1_true, m2_true, sig2_true)
    affiche_image("Image bruitée", Y)

    # Initialisation des paramètres
    Ytrans = nouvelle_image(Y)

    alpha = 1  # paramètre du champ
    proba = calc_proba_champs(alpha)

    nb_iter = 10
    nb_simu = 5

    # Segmentation
    X_seg_trans = MPM_proba_gauss(
        Ytrans, classes, m1_true, sig1_true, m2_true, sig2_true, proba, nb_iter, nb_simu
    )

    # Retirer les bords ajoutés
    X_seg = X_seg_trans[1:-1, 1:-1]

    affiche_image("Image segmentée", X_seg)

    # Calcul du taux d'erreur
    erreur = taux_erreur(X_seg, X)
    print(f"Taux d'erreur de segmentation : {erreur * 100:.2f}%")

    # Sauvegarde des outputs
    if not os.path.exists("question19"):
        os.makedirs("question19")
    plt.imsave("question19/image_originale.png", X, cmap="gray")
    plt.imsave("question19/image_bruitee.png", Y, cmap="gray")
    plt.imsave("question19/image_segmentee.png", X_seg, cmap="gray")
