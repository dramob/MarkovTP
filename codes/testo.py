import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Fonction pour lire une image en niveaux de gris et la transformer en une matrice

def lit_image(chemin_image):
    if not os.path.exists(chemin_image):
        raise ValueError(f"Erreur : le chemin {chemin_image} n'existe pas.")
    image = cv.imread(chemin_image)
    if image is None:
        raise ValueError(f"Erreur : l'image au chemin {chemin_image} n'a pas pu être lue.")
    # Conversion en niveaux de gris si ce n'est pas déjà le cas
    image_gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return image_gris

# Fonction pour afficher une image avec un titre (utilise matplotlib)

def affiche_image(titre, image):
    plt.imshow(image, cmap='gray')
    plt.title(titre)
    plt.axis('off')
    plt.show()

# Fonction pour identifier les deux classes dans une image d'origine

def identif_classes(X):
    classes = np.unique(X)
    print(f"Classes trouvées : {classes}")  # Affichage des classes trouvées
    if len(classes) < 2:
        raise ValueError("L'image ne contient pas suffisamment de classes distinctes.")
    elif len(classes) > 2:
        # Binarisation pour réduire à deux classes
        _, X_binaire = cv.threshold(X, 127, 255, cv.THRESH_BINARY)
        classes = np.unique(X_binaire)
        print(f"Après binarisation, classes : {classes}")
        return classes[0], classes[1]
    return classes[0], classes[1]

# Fonction pour ajouter un bruit gaussien à l'image

def bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2):
    Y = np.copy(X)
    bruit_cl1 = np.random.normal(m1, sig1, X.shape)
    bruit_cl2 = np.random.normal(m2, sig2, X.shape)
    Y[X == cl1] = X[X == cl1] + bruit_cl1[X == cl1]
    Y[X == cl2] = X[X == cl2] + bruit_cl2[X == cl2]
    return np.clip(Y, 0, 255).astype(np.uint8)

# Fonction pour estimer la probabilité a priori des classes

def calc_probaprio(X, cl1, cl2):
    p1 = np.sum(X == cl1) / X.size
    p2 = np.sum(X == cl2) / X.size
    return p1, p2

# Fonction pour appliquer la segmentation selon le critère MPM

def MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    X_seg = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            likelihood_cl1 = p1 * (1 / (np.sqrt(2 * np.pi) * sig1)) * np.exp(-0.5 * ((Y[i, j] - m1) ** 2) / (sig1 ** 2))
            likelihood_cl2 = p2 * (1 / (np.sqrt(2 * np.pi) * sig2)) * np.exp(-0.5 * ((Y[i, j] - m2) ** 2) / (sig2 ** 2))
            X_seg[i, j] = cl1 if likelihood_cl1 >= likelihood_cl2 else cl2
    return X_seg

# Fonction pour calculer le taux d'erreur entre deux images

def taux_erreur(A, B):
    if A.shape != B.shape:
        raise ValueError("Les images doivent avoir la même dimension pour calculer le taux d'erreur.")
    erreur = np.sum(A != B) / A.size
    print(f"Taux d'erreur : {erreur * 100:.2f}%")
    return erreur

# Script aveugle_super_2classes.py
if __name__ == "__main__":
    # Lecture de l'image
    chemin = "./images_BW/alfa2.bmp"  # Remplacer par le bon chemin
    try:
        image = lit_image(chemin)
        affiche_image("Image originale", image)

        # Identification des classes
        cl1, cl2 = identif_classes(image)

        # Ajout de bruit gaussien à l'image
        m1, sig1, m2, sig2 = 1, 1, 4, 1
        image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)
        affiche_image("Image bruitée", image_bruitee)

        # Calcul des probabilités a priori
        p1, p2 = calc_probaprio(image, cl1, cl2)

        # Segmentation par le critère MPM
        image_segmentee = MPM_Gauss(image_bruitee, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
        affiche_image("Image segmentée (MPM)", image_segmentee)

        # Calcul du taux d'erreur entre l'image originale et l'image segmentée
        taux_erreur(image, image_segmentee)
    except ValueError as e:
        print(e)

