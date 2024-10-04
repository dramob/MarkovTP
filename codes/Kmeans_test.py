import os
import numpy as np
import matplotlib.pyplot as plt
from Kmeans_2classes import lit_image, identif_classes, bruit_gauss, kmeans_segmentation, taux_erreur

if __name__ == "__main__":
    # Chemins des images à tester
    images_chemins = [
        "./images_BW/alfa2.bmp",
        "./images_BW/beee2.bmp",
        "./images_BW/cible2.bmp",
        "./images_reelles/15088.bmp",
        "./images_reelles/3096.bmp"
    ]

    # Paramètres des bruits
    bruits = [(1, 1, 4, 1), (1, 1, 2, 1), (1, 1, 1, 9)]
    n_classifications = 100

    # Tableau récapitulatif des taux d'erreurs moyens
    tableau_erreurs = np.zeros((len(images_chemins), len(bruits)))

    for i, chemin_image in enumerate(images_chemins):
        try:
            image = lit_image(chemin_image)
            cl1, cl2 = identif_classes(image)

            for j, (m1, sig1, m2, sig2) in enumerate(bruits):
                erreurs = []
                for _ in range(n_classifications):
                    # Ajout de bruit gaussien à l'image
                    image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)

                    # Segmentation par K-means
                    image_segmentee = kmeans_segmentation(image_bruitee)

                    # Calcul du taux d'erreur entre l'image originale et l'image segmentée
                    erreur = taux_erreur(image, image_segmentee)
                    erreurs.append(erreur)

                # Calcul du taux d'erreur moyen pour cette configuration
                taux_erreur_moyen = np.mean(erreurs)
                tableau_erreurs[i, j] = taux_erreur_moyen
                print(f"Image: {chemin_image}, Bruit {j + 1}: Taux d'erreur moyen = {taux_erreur_moyen * 100:.2f}%")
        except ValueError as e:
            print(e)

    # Affichage du tableau récapitulatif des taux d'erreurs moyens
    print("\nTableau récapitulatif des taux d'erreurs moyens:")
    for i, chemin_image in enumerate(images_chemins):
        print(f"Image: {chemin_image}")
        for j, (m1, sig1, m2, sig2) in enumerate(bruits):
            print(f"  Bruit {j + 1} (m1={m1}, sig1={sig1}, m2={m2}, sig2={sig2}): Taux d'erreur moyen = {tableau_erreurs[i, j] * 100:.2f}%")