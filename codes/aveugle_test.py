import numpy as np
import pandas as pd

from testo import (
    MPM_Gauss,
    bruit_gauss,
    calc_probaprio,
    identif_classes,
    lit_image,
    taux_erreur,
)

if __name__ == "__main__":
    images_chemins = [
        "./images_BW/alfa2.bmp",
        "./images_BW/beee2.bmp",
        "./images_BW/cible2.bmp",
        "./images_réelles/15088.bmp",
        "./images_réelles/3096.bmp",
    ]

    # Paramètres des bruits
    bruits = [
        {"m1": 1, "sig1": 1, "m2": 4, "sig2": 1},
        {"m1": 1, "sig1": 1, "m2": 2, "sig2": 1},
        {"m1": 1, "sig1": 1, "m2": 1, "sig2": 9},
    ]
    n_classifications = 100

    # Initialiser une liste pour stocker les résultats
    results = []

    for i, chemin_image in enumerate(images_chemins):
        try:
            image = lit_image(chemin_image)
            cl1, cl2 = identif_classes(image)

            for j, bruit_params in enumerate(bruits):
                m1 = bruit_params["m1"]
                sig1 = bruit_params["sig1"]
                m2 = bruit_params["m2"]
                sig2 = bruit_params["sig2"]

                erreurs = []
                for _ in range(n_classifications):
                    # Ajout de bruit gaussien à l'image
                    image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)

                    # Calcul des probabilités a priori
                    p1, p2 = calc_probaprio(image, cl1, cl2)

                    # Segmentation par MPM Gaussien
                    image_segmentee = MPM_Gauss(
                        image_bruitee, cl1, cl2, p1, p2, m1, sig1, m2, sig2
                    )

                    # Calcul du taux d'erreur entre l'image originale et l'image segmentée
                    erreur = taux_erreur(image, image_segmentee)
                    erreurs.append(erreur)

                # Calcul des statistiques pour cette configuration
                taux_erreur_moyen = np.mean(erreurs)
                ecart_type_erreur = np.std(erreurs)

                # Ajouter les résultats à la liste
                results.append(
                    {
                        "Image": chemin_image,
                        "Bruit": f"m1={m1}, sig1={sig1}, m2={m2}, sig2={sig2}",
                        "Taux d'erreur moyen (%)": taux_erreur_moyen * 100,
                        "Écart-type (%)": ecart_type_erreur * 100,
                    }
                )
                print(
                    f"Image: {chemin_image}, Bruit {j + 1}: Taux d'erreur moyen = {taux_erreur_moyen * 100:.2f}%"
                )
        except ValueError as e:
            print(e)

    # Créer un DataFrame pandas pour stocker les résultats
    df_results = pd.DataFrame(results)

    # Afficher le tableau récapitulatif des taux d'erreurs moyens
    print("\nTableau récapitulatif des taux d'erreurs moyens:")
    print(df_results)

    # Sauvegarder le DataFrame dans un fichier CSV
    df_results.to_csv("resultats_aveugle_supervise.csv", index=False)
    print("\nLes résultats ont été sauvegardés dans 'resultats_aveugle_supervise.csv'.")
