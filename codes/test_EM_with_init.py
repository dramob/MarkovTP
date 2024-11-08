import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import pandas as pd
from sklearn.cluster import KMeans

# Créer les dossiers pour sauvegarder les résultats
output_folder = 'part5'
output_outputs_folder = 'part5_output'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_outputs_folder, exist_ok=True)

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

# Question 12: Fonction est_empiriques pour estimer les paramètres de la loi du couple (X, Y)
def est_empiriques(X, Y, cl1, cl2):
    p1 = np.mean(X == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[X == cl1])
    m2 = np.mean(Y[X == cl2])
    sig1 = np.std(Y[X == cl1])
    sig2 = np.std(Y[X == cl2])
    return p1, p2, m1, sig1, m2, sig2

# Question 12: Fonction init_param pour initialiser les paramètres uniquement à partir de Y
def init_param(Y, iter_KM=10):
    Y_flat = Y.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=iter_KM).fit(Y_flat)
    labels = kmeans.labels_.reshape(Y.shape)

    # Identifier les classes
    m1_cluster = np.mean(Y[labels == 0])
    m2_cluster = np.mean(Y[labels == 1])
    if m1_cluster < m2_cluster:
        cl1, cl2 = 0, 1
    else:
        cl1, cl2 = 1, 0

    # Estimer les paramètres
    p1 = np.mean(labels == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[labels == cl1])
    m2 = np.mean(Y[labels == cl2])
    sig1 = np.std(Y[labels == cl1])
    sig2 = np.std(Y[labels == cl2])
    return p1, p2, m1, sig1, m2, sig2

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

    # Question 11: Tester avec différentes initialisations
    initializations = [
        {'p1': 0.5, 'p2': 0.5, 'm1': m1_true, 'sig1': sig1_true, 'm2': m2_true, 'sig2': sig2_true},
        {'p1': 0.3, 'p2': 0.7, 'm1': m1_true + 1, 'sig1': sig1_true + 1, 'm2': m2_true - 1, 'sig2': sig2_true + 1},
        {'p1': 0.1, 'p2': 0.9, 'm1': m1_true + 3, 'sig1': sig1_true + 2, 'm2': m2_true - 2, 'sig2': sig2_true + 2},
    ]

    for idx, init in enumerate(initializations):
        p10, p20 = init['p1'], init['p2']
        m10, sig10 = init['m1'], init['sig1']
        m20, sig20 = init['m2'], init['sig2']

        # Exécuter EM
        p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est, p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list = calc_EM(
            Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM=10)

        # Sauvegarder les résultats dans un fichier CSV
        results_df = pd.DataFrame({
            'Iteration': range(1, 11),
            'p1': p1_list,
            'p2': p2_list,
            'm1': m1_list,
            'm2': m2_list,
            'sig1': sig1_list,
            'sig2': sig2_list
        })
        results_csv_path = os.path.join(output_outputs_folder, f'em_results_initialization_{idx + 1}.csv')
        results_df.to_csv(results_csv_path, index=False)

        # Afficher et sauvegarder les graphiques de l'évolution des paramètres
        plt.figure()
        plt.plot(p1_list, label='p1')
        plt.plot(p2_list, label='p2')
        plt.legend()
        plt.title(f'Évolution des probabilités a priori - Init {idx + 1}')
        plt.savefig(os.path.join(output_outputs_folder, f'evolution_probabilites_init_{idx + 1}.png'), bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(m1_list, label='m1')
        plt.plot(m2_list, label='m2')
        plt.legend()
        plt.title(f'Évolution des moyennes - Init {idx + 1}')
        plt.savefig(os.path.join(output_outputs_folder, f'evolution_moyennes_init_{idx + 1}.png'), bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(sig1_list, label='sigma1')
        plt.plot(sig2_list, label='sigma2')
        plt.legend()
        plt.title(f'Évolution des écarts-types - Init {idx + 1}')
        plt.savefig(os.path.join(output_outputs_folder, f'evolution_ecarts_types_init_{idx + 1}.png'), bbox_inches='tight')
        plt.show()

    # Question 13: Initialisation automatique avec init_param
    p10, p20, m10, sig10, m20, sig20 = init_param(Y, iter_KM=10)

    # Exécuter EM avec initialisation automatique
    p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est, p1_list, p2_list, m1_list, m2_list, sig1_list, sig2_list = calc_EM(
        Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM=10)

    # Sauvegarder les résultats dans un fichier CSV
    results_df = pd.DataFrame({
        'Iteration': range(1, 11),
        'p1': p1_list,
        'p2': p2_list,
        'm1': m1_list,
        'm2': m2_list,
        'sig1': sig1_list,
        'sig2': sig2_list
    })
    results_csv_path = os.path.join(output_outputs_folder, 'em_results_init_param.csv')
    results_df.to_csv(results_csv_path, index=False)

    # Afficher et sauvegarder les graphiques de l'évolution des paramètres
    plt.figure()
    plt.plot(p1_list, label='p1')
    plt.plot(p2_list, label='p2')
    plt.legend()
    plt.title('Évolution des probabilités a priori - Init Param')
    plt.savefig(os.path.join(output_outputs_folder, 'evolution_probabilites_init_param.png'), bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(m1_list, label='m1')
    plt.plot(m2_list, label='m2')
    plt.legend()
    plt.title('Évolution des moyennes - Init Param')
    plt.savefig(os.path.join(output_outputs_folder, 'evolution_moyennes_init_param.png'), bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(sig1_list, label='sigma1')
    plt.plot(sig2_list, label='sigma2')
    plt.legend()
    plt.title('Évolution des écarts-types - Init Param')
    plt.savefig(os.path.join(output_outputs_folder, 'evolution_ecarts_types_init_param.png'), bbox_inches='tight')
    plt.show()
