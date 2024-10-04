import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Fonction pour lire une image en niveaux de gris et la transformer en une matrice

def lit_image(chemin_image):
    image = cv.imread(chemin_image)
    if image is None:
        raise ValueError(f"Erreur : l'image au chemin {chemin_image} n'a pas pu être lue.")
    # Conversion en niveaux de gris si ce n'est pas déjà le cas
    image_gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return image_gris

# Fonction pour afficher une image avec un titre

def affiche_image(titre, image):
    cv.imshow(titre, image)
    cv.waitKey(0)  # Attendre une touche pour fermer la fenêtre
    cv.destroyAllWindows()

# Fonction pour identifier les deux classes dans une image d'origine

def identif_classes(X):
    classes = np.unique(X)
    if len(classes) != 2:
        raise ValueError("L'image ne contient pas exactement deux classes.")
    return classes[0], classes[1]

# Fonction pour ajouter un bruit gaussien à l'image

def bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2):
    Y = np.copy(X)
    bruit_cl1 = np.random.normal(m1, sig1, X.shape)
    bruit_cl2 = np.random.normal(m2, sig2, X.shape)
    Y[X == cl1] = X[X == cl1] + bruit_cl1[X == cl1]
    Y[X == cl2] = X[X == cl2] + bruit_cl2[X == cl2]
    return np.clip(Y, 0, 255).astype(np.uint8)

# Script C_un_debut.py
if __name__ == "__main__":
    # Lecture de l'image
    chemin = "./images_BW/chemin_vers_image_a_deux_classes.png"  # Remplacer par le bon chemin
    image = lit_image(chemin)
    affiche_image("Image originale", image)

    # Identification des classes
    cl1, cl2 = identif_classes(image)

    # Ajout de bruit gaussien à l'image
    m1, sig1, m2, sig2 = 1, 1, 4, 1
    image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)
    affiche_image("Image bruitée", image_bruitee)

    # Création d'un tableau récapitulatif sur 5 images pour lesquelles on utilisera à chaque fois 3 bruits différents
    images_chemins = [
        "./images_BW/3096.bmp",
        "./images_BW/15088.bmp",
        "./images_BW/35008.bmp",
        "./images_BW/69020.bmp",
]  
    bruits = [(1, 1, 4, 1), (1, 1, 2, 1), (1, 1, 1, 9)]

    fig, axes = plt.subplots(len(images_chemins), len(bruits) + 1, figsize=(15, 10))
    for i, chemin_image in enumerate(images_chemins):
        image = lit_image(chemin_image)
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title("Originale")
        axes[i, 0].axis('off')
        
        for j, (m1, sig1, m2, sig2) in enumerate(bruits):
            image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)
            axes[i, j + 1].imshow(image_bruitee, cmap='gray')
            axes[i, j + 1].set_title(f"Bruit: (m1,s1)=({m1},{sig1}), (m2,s2)=({m2},{sig2})")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()