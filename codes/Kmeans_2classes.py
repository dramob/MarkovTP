import os
import numpy as np
import cv2 as cv


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# Fonction pour afficher et sauvegarder une image avec un titre (utilise matplotlib)

def affiche_image(titre, image, save_path=None):
    plt.imshow(image, cmap='gray')
    plt.title(titre)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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



def kmeans_segmentation(image_bruitee, n_clusters=2):
    # Transformation de l'image en un vecteur de pixels
    pixels = image_bruitee.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    # Transformation des labels en image
    segmented_image = kmeans.labels_.reshape(image_bruitee.shape)
    return segmented_image * (255 // (n_clusters - 1))

# Fonction pour calculer le taux d'erreur entre deux images

def taux_erreur(A, B):
    if A.shape != B.shape:
        raise ValueError("Les images doivent avoir la même dimension pour calculer le taux d'erreur.")
    erreur = np.sum(A != B) / A.size
    print(f"Taux d'erreur : {erreur * 100:.2f}%")
    return erreur

if __name__ == "__main__":

    chemin = "./images_BW/alfa2.bmp"  # Remplacer par le bon chemin
    try:
        image = lit_image(chemin)
        affiche_image("Image originale", image, save_path="image_originale.png")
  
        # Identification des classes
        cl1, cl2 = identif_classes(image)

        # Ajout de bruit gaussien à l'image
        m1, sig1, m2, sig2 = 1, 1, 4, 1
        image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)
        affiche_image("Image bruitée", image_bruitee, save_path="image_bruitee.png")
        
        # Segmentation par K-means
        image_segmentee = kmeans_segmentation(image_bruitee)
        affiche_image("Image segmentée (K-means)", image_segmentee, save_path="image_segmentee_kmeans.png")

        # Calcul du taux d'erreur entre l'image originale et l'image segmentée
        taux_erreur(image, image_segmentee)
    except ValueError as e:
        print(e)