import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import pandas as pd
from sklearn.cluster import KMeans

# Créer les dossiers pour sauvegarder les résultats
output_folder = 'part4'
output_outputs_folder = 'part4_out'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_outputs_folder, exist_ok=True)

# Fonctions de lecture et affichage d'image
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

# Fonction pour calculer le taux d'erreur de segmentation
def taux_erreur(X, X_seg):
    return np.mean(X != X_seg)

# Fonction pour initialiser les paramètres à partir de Y
def est_empiriques(X, Y, cl1, cl2):
    p1 = np.mean(X == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[X == cl1])
    m2 = np.mean(Y[X == cl2])
    sig1 = np.std(Y[X == cl1])
    sig2 = np.std(Y[X == cl2])
    return p1, p2, m1, sig1, m2, sig2

def init_param(Y, iter_KM=10):
    Y_flat = Y.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=iter_KM).fit(Y_flat)
    labels = kmeans.labels_.reshape(Y.shape)

    m1_cluster = np.mean(Y[labels == 0])
    m2_cluster = np.mean(Y[labels == 1])
    if m1_cluster < m2_cluster:
        cl1, cl2 = 0, 1
    else:
        cl1, cl2 = 1, 0

    p1 = np.mean(labels == cl1)
    p2 = 1 - p1
    m1 = np.mean(Y[labels == cl1])
    m2 = np.mean(Y[labels == cl2])
    sig1 = np.std(Y[labels == cl1])
    sig2 = np.std(Y[labels == cl2])
    return p1, p2, m1, sig1, m2, sig2

if __name__ == "__main__":
    # Images à utiliser
    image_paths = ['./images_BW/beee2.bmp', './images_BW/image2.bmp', './images_BW/image3.bmp',
                   './images_BW/image4.bmp', './images_BW/image5.bmp']
    
    bruits = [
        (1, 1, 4, 1),  # N(1, 1) - N(4, 1)
        (1, 1, 2, 1),  # N(1, 1) - N(2, 1)
        (1, 1, 1, 9)   # N(1, 1) - N(1, 9)
    ]

    # Tableau pour les taux d'erreur
    error_rates = []

    # Pour chaque image
    for image_path in image_paths:
        X = lit_image(image_path)
        cl1, cl2 = identif_classes(X)

        # Pour chaque bruit
        for m1, sig1, m2, sig2 in bruits:
            # Répéter pour 100 classifications
            taux_erreur_moyen = []
            for _ in range(100):
                Y = bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2)
                
                p10, p20, m10, sig10, m20, sig20 = est_empiriques(X, Y, cl1, cl2)
                p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est, _, _, _, _, _, _ = calc_EM(
                    Y, p10, p20, m10, sig10, m20, sig20, nb_iterEM=10)

                # Estimer les classes segmentées
                Ppost = calc_probapost_Gauss(Y, p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est)
                X_seg = np.where(Ppost[:, :, 0] > Ppost[:, :, 1], cl1, cl2)

                # Calculer le taux d'erreur
                error_rate = taux_erreur(X, X_seg)
                taux_erreur_moyen.append(error_rate)

            # Calculer la moyenne des taux d'erreur pour ce bruit et cette image
            error_rates.append({
                'Image': os.path.basename(image_path),
                'Bruit': f'N({m1}, {sig1}) - N({m2}, {sig2})',
                'Taux d\'Erreur': np.mean(taux_erreur_moyen)
            })

    # Présenter les taux moyens d'erreur de segmentation
    results_df = pd.DataFrame(error_rates)
    results_csv_path = os.path.join(output_folder, 'taux_erreur_segmentation.csv')
    results_df.to_csv(results_csv_path, index=False)

    print("Taux d'erreur de segmentation :")
    print(results_df)

    # Commentaires sur les résultats
    # Vous pouvez ajouter ici des commentaires sur les résultats obtenus.
