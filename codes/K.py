import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Kmeans_2classes import lit_image, identif_classes, bruit_gauss, kmeans_segmentation, taux_erreur

if __name__ == "__main__":
    # Chemins des images à tester
    images_chemins = [
        "./images_BW/alfa2.bmp",
        "./images_BW/beee2.bmp",
        "./images_BW/cible2.bmp",
        "./images_BW/city2.bmp",
        "./images_BW/veau2.bmp"
    ]

    # Paramètres des bruits
    bruits = [
        {'m1': 1, 'sig1': 1, 'm2': 4, 'sig2': 1},
        {'m1': 1, 'sig1': 1, 'm2': 2, 'sig2': 1},
        {'m1': 1, 'sig1': 1, 'm2': 1, 'sig2': 9}
    ]
    n_classifications = 100

    # Initialiser une liste pour stocker les résultats
    results = []

    for i, chemin_image in enumerate(images_chemins):
        try:
            image = lit_image(chemin_image)
            cl1, cl2 = identif_classes(image)

            for j, bruit_params in enumerate(bruits):
                m1 = bruit_params['m1']
                sig1 = bruit_params['sig1']
                m2 = bruit_params['m2']
                sig2 = bruit_params['sig2']

                erreurs = []
                for _ in range(n_classifications):
                    # Ajout de bruit gaussien à l'image
                    image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)

                    # Segmentation par K-means
                    image_segmentee = kmeans_segmentation(image_bruitee)

                    # Calcul du taux d'erreur entre l'image originale et l'image segmentée
                    erreur = taux_erreur(image, image_segmentee)
                    erreurs.append(erreur)

                # Calcul des statistiques pour cette configuration
                taux_erreur_moyen = np.mean(erreurs)
                ecart_type_erreur = np.std(erreurs)
                
                # Ajouter les résultats à la liste
                results.append({
                    'Image': os.path.basename(chemin_image),
                    'Bruit': f"m1={m1}, sig1={sig1}, m2={m2}, sig2={sig2}",
                    'Taux d\'erreur moyen (%)': taux_erreur_moyen * 100,
                    'Écart-type (%)': ecart_type_erreur * 100
                })

                print(f"Image: {chemin_image}, Bruit {j + 1}: Taux d'erreur moyen = {taux_erreur_moyen * 100:.2f}%")

        except ValueError as e:
            print(e)

    # Créer un DataFrame pandas pour stocker les résultats
    df_results = pd.DataFrame(results)

    # Afficher le tableau récapitulatif des taux d'erreurs moyens
    print("\nTableau récapitulatif des taux d'erreurs moyens:")
    print(df_results)

    # Sauvegarder le DataFrame dans un fichier CSV
    df_results.to_csv('resultats_segmentation.csv', index=False)
    print("\nLes résultats ont été sauvegardés dans 'resultats_segmentation.csv'.")