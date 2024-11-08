import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

# Créer les dossiers pour sauvegarder les résultats
output_folder = "part8"
output_outputs_folder = "part8_outputs"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_outputs_folder, exist_ok=True)


# Fonctions de lecture et affichage d'image
def lit_image(chemin_image):
    image = plt.imread(chemin_image)
    if image.ndim == 3:
        image = image[:, :, 0]  # Convertir en niveaux de gris si nécessaire
    return image


def affiche_image(titre, image, save_path=None):
    plt.imshow(image, cmap="gray")
    plt.title(titre)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2):
    Y = np.zeros_like(X, dtype=float)
    Y[X == cl1] = np.random.normal(m1, sig1, size=(X == cl1).sum())
    Y[X == cl2] = np.random.normal(m2, sig2, size=(X == cl2).sum())
    return Y


def calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2):
    f1 = p1 * norm.pdf(Y, m1, sig1)
    f2 = p2 * norm.pdf(Y, m2, sig2)
    total = f1 + f2

    # Eviter les divisions par zéro
    total[total == 0] = np.finfo(
        float
    ).eps  # Remplacer les valeurs nulles par un epsilon très petit

    Ppost_class1 = f1 / total
    Ppost_class2 = f2 / total
    Ppost = np.stack((Ppost_class1, Ppost_class2), axis=-1)
    return Ppost


def est_empiriques(X, Y, cl1, cl2):
    p1 = np.mean(X == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[X == cl1])
    m2 = np.mean(Y[X == cl2])
    sig1 = np.std(Y[X == cl1])
    sig2 = np.std(Y[X == cl2])
    return p1, p2, m1, sig1, m2, sig2


def calc_SEM(Y, p10, p20, m10, sig10, m20, sig20, nb_iter):
    p1, p2 = p10, p20
    m1, sig1 = m10, sig10
    m2, sig2 = m20, sig20

    for _ in range(nb_iter):
        # E-step: Calculer Ppost
        Ppost = calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2)

        # Tirage de X_post suivant la loi a posteriori
        X_post = tirage_apost(Ppost, 0, 1, *Y.shape)

        # M-step: Estimer les nouveaux paramètres à partir de X_post
        p1, p2, m1, sig1, m2, sig2 = est_empiriques(X_post, Y, 0, 1)

    return p1, p2, m1, sig1, m2, sig2


def tirage_apost(Ppost, cl1, cl2, m, n):
    X_post = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            if np.random.rand() < Ppost[i, j, 0]:
                X_post[i, j] = cl1
            else:
                X_post[i, j] = cl2

    return X_post


if __name__ == "__main__":
    # Lire une image réelle
    chemin_image_reelle = "./images_reelles/3096.bmp"  # Choisir une image réelle
    Y = lit_image(chemin_image_reelle)

    # Initialisation des paramètres avec les meilleures estimations précédentes
    p10, p20, m10, sig10, m20, sig20 = (
        0.5,
        0.5,
        1,
        1,
        4,
        1,
    )  # Paramètres pour SEM basés sur les meilleures performances

    # Estimation des paramètres avec SEM
    p1_sem, p2_sem, m1_sem, sig1_sem, m2_sem, sig2_sem = calc_SEM(
        Y, p10, p20, m10, sig10, m20, sig20, nb_iter=10
    )

    # Segmenter l'image
    Ppost = calc_probapost_Gauss(Y, p1_sem, p2_sem, m1_sem, sig1_sem, m2_sem, sig2_sem)
    X_seg = np.where(Ppost[:, :, 0] > Ppost[:, :, 1], 0, 1)  # Classes 0 et 1

    # Afficher l'image segmentée
    affiche_image(
        "Image Segmentée",
        X_seg,
        save_path=os.path.join(output_folder, "image_segmentee_reelle.png"),
    )

    # Sauvegarder également l'image segmentée
    plt.imsave(
        os.path.join(output_folder, "image_segmentee_reelle_saved.png"),
        X_seg,
        cmap="gray",
    )
