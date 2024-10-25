import numpy as np
import matplotlib.pyplot as plt
from Kmeans_2classes import lit_image, identif_classes, bruit_gauss, affiche_image
from calc_SEM import calc_SEM
from init_param import init_param
from aveugle_non_super_2classes import MPM_Gauss
from Kmeans_2classes import taux_erreur

if __name__ == "__main__":
    # Lire et afficher l'image originale
    chemin_image = '/workspaces/MarkovTP/codes/images_BW/beee2.bmp'
    X = lit_image(chemin_image)
    affiche_image('Image Originale', X)

    # Identifier les classes
    cl1, cl2 = identif_classes(X)

    # Ajouter du bruit gaussien
    m1_true, sig1_true = 1, 1
    m2_true, sig2_true = 4, 1
    Y = bruit_gauss(X, cl1, cl2, m1_true, sig1_true, m2_true, sig2_true)
    affiche_image('Image Bruitée', Y)

    # Initialiser les paramètres en utilisant init_param
    p10, p20, m10, sig10, m20, sig20 = init_param(Y, iter_KM=10)

    # Tester l'estimation SEM
    nb_iter_SEM = 10
    p1_est_SEM, p2_est_SEM, m1_est_SEM, sig1_est_SEM, m2_est_SEM, sig2_est_SEM = calc_SEM(
        Y, p10, p20, m10, sig10, m20, sig20, nb_iter_SEM)

    # Segmentation avec les paramètres estimés par SEM
    X_seg_SEM = MPM_Gauss(Y, cl1, cl2, p1_est_SEM, p2_est_SEM, m1_est_SEM, sig1_est_SEM, m2_est_SEM, sig2_est_SEM)
    affiche_image('Image Segmentée (SEM)', X_seg_SEM)

    # Calculer le taux d'erreur
    taux_SEM = taux_erreur(X, X_seg_SEM)
    print(f"Taux d'erreur de segmentation avec SEM : {taux_SEM:.2f}%")
