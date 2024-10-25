import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import pandas as pd

# Créer le dossier pour sauvegarder les résultats
output_folder = 'part3'
os.makedirs(output_folder, exist_ok=True)

# Question 7 à 9: Fonctions de l'algorithme EM
def lit_image(chemin_image):
    image = plt.imread(chemin_image)
    if image.ndim == 3:
        image = image[:, :, 0]  # Convertir en niveaux de gris si nécessaire
    return image

def affiche_image(titre, image, save_path=None):
    plt.imshow(image, cmap='gray')
    plt.title(titre)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def identif_classes(X):
    classes = np.unique(X)
    return classes[0], classes[1]

def bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2):
    Y = np.zeros_like(X, dtype=float)
    Y[X == cl1] = np.random.normal(m1, sig1, size=(X == cl1).sum())
    Y[X == cl2] = np.random.normal(m2, sig2, size=(X == cl2).sum())
    return Y

def calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2):
    f1 = p1 * norm.pdf(Y, m1, sig1)
    f2 = p2 * norm.pdf(Y, m2, sig2)
    total = f1 + f2
    Ppost_class1 = f1 / total
    Ppost_class2 = f2 / total
    Ppost = np.stack((Ppost_class1, Ppost_class2), axis=-1)
    return Ppost

def calc_EM(Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM):
    p1, p2 = p10, p20
    m1, sig1 = m10, sig10
    m2, sig2 = m20, sig20

    p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list = [], [], [], [], [], []

    for _ in range(nb_iterEM):
        # E-step
        Ppost = calc_probapost_Gauss(Y, p1, p2, m1, sig1, m2, sig2)

        # M-step
        sum_Ppost_class1 = np.sum(Ppost[:, :, 0])
        sum_Ppost_class2 = np.sum(Ppost[:, :, 1])
        p1 = sum_Ppost_class1 / (sum_Ppost_class1 + sum_Ppost_class2)
        p2 = 1 - p1
        m1 = np.sum(Ppost[:, :, 0] * Y) / sum_Ppost_class1
        m2 = np.sum(Ppost[:, :, 1] * Y) / sum_Ppost_class2
        sig1 = np.sqrt(np.sum(Ppost[:, :, 0] * (Y - m1) ** 2) / sum_Ppost_class1)
        sig2 = np.sqrt(np.sum(Ppost[:, :, 1] * (Y - m2) ** 2) / sum_Ppost_class2)

        # Stocker les valeurs pour les graphiques
        p1_list.append(p1)
        p2_list.append(p2)
        m1_list.append(m1)
        m2_list.append(m2)
        sig1_list.append(sig1)
        sig2_list.append(sig2)

    return p1, p2, m1, sig1, m2, sig2, p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list

if __name__ == "__main__":
    # Lire et afficher l'image originale
    chemin_image = './images_BW/beee2.bmp'
    X = lit_image(chemin_image)
    affiche_image('Image Originale', X, save_path=os.path.join(output_folder, 'image_originale_test_em.png'))

    # Identifier les classes
    cl1, cl2 = identif_classes(X)

    # Ajouter du bruit gaussien
    m1_true, sig1_true = 1, 1
    m2_true, sig2_true = 4, 1
    Y = bruit_gauss(X, cl1, cl2, m1_true, sig1_true, m2_true, sig2_true)
    affiche_image('Image Bruitée', Y, save_path=os.path.join(output_folder, 'image_bruitee_test_em.png'))

    # Initialiser les paramètres avec les vraies valeurs
    p1_true, p2_true = np.mean(X == cl1), np.mean(X == cl2)
    p10, p20 = p1_true, p2_true
    m10, sig10 = m1_true, sig1_true
    m20, sig20 = m2_true, sig2_true

    # Nombre d'itérations EM
    nb_iterEM = 10

    # Exécuter EM
    p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est, p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list = calc_EM(
        Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM)

    # Sauvegarder les résultats dans un fichier CSV
    results_df = pd.DataFrame({
        'Iteration': range(1, nb_iterEM + 1),
        'p1': p1_list,
        'p2': p2_list,
        'm1': m1_list,
        'm2': m2_list,
        'sig1': sig1_list,
        'sig2': sig2_list
    })
    results_csv_path = os.path.join(output_folder, 'em_results_test_em.csv')
    results_df.to_csv(results_csv_path, index=False)

    # Afficher et sauvegarder les graphiques de l'évolution des paramètres
    plt.figure()
    plt.plot(p1_list, label='p1')
    plt.plot(p2_list, label='p2')
    plt.legend()
    plt.title('Évolution des probabilités a priori')
    plt.savefig(os.path.join(output_folder, 'evolution_probabilites_test_em.png'), bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(m1_list, label='m1')
    plt.plot(m2_list, label='m2')
    plt.legend()
    plt.title('Évolution des moyennes')
    plt.savefig(os.path.join(output_folder, 'evolution_moyennes_test_em.png'), bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(sig1_list, label='sigma1')
    plt.plot(sig2_list, label='sigma2')
    plt.legend()
    plt.title('Évolution des écarts-types')
    plt.savefig(os.path.join(output_folder, 'evolution_ecarts_types_test_em.png'), bbox_inches='tight')
    plt.show()
