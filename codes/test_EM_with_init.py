import numpy as np
import matplotlib.pyplot as plt
from Kmeans_2classes import lit_image, identif_classes, bruit_gauss, affiche_image
from calc_EM import calc_EM
from init_param import init_param

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

    # Initialiser les paramètres en utilisant init_param
    p10, p20, m10, sig10, m20, sig20 = init_param(Y, iter_KM=10)
    
    # Nombre d'itérations de l'algorithme EM
    nb_iterEM = 10

    # Exécuter l'algorithme EM
    p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est, p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list = calc_EM(
        Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM)

    # Tracer l'évolution des paramètres
    plt.figure()
    plt.plot(p1_list, label='p1')
    plt.plot(p2_list, label='p2')
    plt.legend()
    plt.title('Évolution des probabilités a priori')
    plt.show()

    plt.figure()
    plt.plot(m1_list, label='m1')
    plt.plot(m2_list, label='m2')
    plt.legend()
    plt.title('Évolution des moyennes')
    plt.show()

    plt.figure()
    plt.plot(sig1_list, label='sigma1')
    plt.plot(sig2_list, label='sigma2')
    plt.legend()
    plt.title('Évolution des écarts-types')
    plt.show()
