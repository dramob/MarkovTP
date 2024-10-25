import numpy as np
import cv2 as cv
from scipy.stats import norm
from Kmeans_2classes import lit_image, identif_classes, bruit_gauss, taux_erreur, affiche_image
from init_param import init_param
from calc_EM import calc_EM

def calc_probaprio(X, cl1, cl2):
    p1 = np.mean(X == cl1)
    p2 = 1 - p1
    return p1, p2

def MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    f1 = p1 * norm.pdf(Y, m1, sig1)
    f2 = p2 * norm.pdf(Y, m2, sig2)
    X_seg = np.where(f1 >= f2, cl1, cl2)
    return X_seg

if __name__ == "__main__":
    # Lire et afficher l'image originale
    chemin_image = './images_BW/beee2.bmp'
    X = lit_image(chemin_image)
    affiche_image('Image Originale', X)
    
    # Identifier les classes
    cl1, cl2 = identif_classes(X)
    
    # Ajouter du bruit gaussien
    m1_true, sig1_true = 1, 1
    m2_true, sig2_true = 4, 1
    Y = bruit_gauss(X, cl1, cl2, m1_true, sig1_true, m2_true, sig2_true)
    affiche_image('Image Bruitée', Y)
    
    # Oublier tous les paramètres
    p10, p20, m10, sig10, m20, sig20 = init_param(Y, iter_KM=10)

    # Estimation des paramètres par EM
    nb_iterEM = 10
    p1, p2, m1, sig1, m2, sig2 = calc_EM(Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM)

    # Segmentation aveugle non supervisée
    X_seg = MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    affiche_image('Image Segmentée', X_seg)

    # Calcul du taux d'erreur
    taux = taux_erreur(X, X_seg)
    print(f"Taux d'erreur de segmentation : {taux:.2f}%")
