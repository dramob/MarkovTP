# aveugle_non_super_2classes.py

import numpy as np
import cv2 as cv
from scipy.stats import norm
from Kmeans_2classes import lit_image, identif_classes, bruit_gauss, taux_erreur, affiche_image

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
    
    # On oublie tout ce que l’on sait sur les paramètres %%%
    p1 = 0
    p2 = 0
    m1 = 0
    m2 = 0
    sig1 = 0
    sig2 = 0
    
    # Le reste du code sera complété dans les tâches suivantes