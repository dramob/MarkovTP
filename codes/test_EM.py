# test_EM.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aveugle_non_super_2classes import lit_image, identif_classes, bruit_gauss, affiche_image
from calc_probapost_Gauss import calc_probapost_Gauss
from calc_EM import calc_EM
from init_param import init_param

def save_results_csv(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

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
    
    # Définir le nombre d'itérations EM
    nb_iterEM = 10
    
    # Exécuter l'algorithme EM
    p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est, p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list = calc_EM(
        Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM)
    
    # Créer un DataFrame pour stocker les résultats
    results = {
        'Iteration': np.arange(1, nb_iterEM + 1),
        'p1': p1_list,
        'p2': p2_list,
        'm1': m1_list,
        'm2': m2_list,
        'sig1': sig1_list,
        'sig2': sig2_list
    }
    
    df_results = pd.DataFrame(results)
    
    # Sauvegarder les résultats dans un fichier CSV
    df_results.to_csv('em_results.csv', index=False)
    print("Les résultats EM ont été sauvegardés dans 'em_results.csv'.")
    
    # Sauvegarder les figures
    # Figure 1: Probabilités a priori
    fig1, ax1 = plt.subplots()
    ax1.plot(p1_list, label='p1')
    ax1.plot(p2_list, label='p2')
    ax1.legend()
    ax1.set_title('Évolution des probabilités a priori')
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Probabilité')
    save_plot(fig1, 'em_probabilities.png')
    print("La figure 'em_probabilities.png' a été sauvegardée.")
    
    # Figure 2: Moyennes
    fig2, ax2 = plt.subplots()
    ax2.plot(m1_list, label='m1')
    ax2.plot(m2_list, label='m2')
    ax2.legend()
    ax2.set_title('Évolution des moyennes')
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Moyenne')
    save_plot(fig2, 'em_means.png')
    print("La figure 'em_means.png' a été sauvegardée.")
    
    # Figure 3: Écarts-types
    fig3, ax3 = plt.subplots()
    ax3.plot(sig1_list, label='sigma1')
    ax3.plot(sig2_list, label='sigma2')
    ax3.legend()
    ax3.set_title('Évolution des écarts-types')
    ax3.set_xlabel('Itération')
    ax3.set_ylabel('Écart-type')
    save_plot(fig3, 'em_std_devs.png')
    print("La figure 'em_std_devs.png' a été sauvegardée.")
    
    # Optionnel: Afficher les figures
    plt.show()