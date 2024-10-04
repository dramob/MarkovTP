
import numpy as np
from testo import lit_image, identif_classes, bruit_gauss, calc_probaprio, MPM_Gauss, taux_erreur

if __name__ == "__main__":
    images_chemins = [
        "./images_BW/alfa2.bmp",
        "./images_BW/beee2.bmp",
        "./images_BW/cible2.bmp",
        "./images_réelles/15088.bmp",
        "./images_réelles/3096.bmp"
    ]

 
    bruits = [(1, 1, 4, 1), (1, 1, 2, 1), (1, 1, 1, 9)]
    n_classifications = 100
    tableau_erreurs = np.zeros((len(images_chemins), len(bruits)))

    for i, chemin_image in enumerate(images_chemins):
        try:
            image = lit_image(chemin_image)
            cl1, cl2 = identif_classes(image)

            for j, (m1, sig1, m2, sig2) in enumerate(bruits):
                erreurs = []
                for _ in range(n_classifications):
                   
                    image_bruitee = bruit_gauss(image, cl1, cl2, m1, sig1, m2, sig2)

             
                    p1, p2 = calc_probaprio(image, cl1, cl2)

                   
                    image_segmentee = MPM_Gauss(image_bruitee, cl1, cl2, p1, p2, m1, sig1, m2, sig2)

                    #
                    erreur = taux_erreur(image, image_segmentee)
                    erreurs.append(erreur)

               
                taux_erreur_moyen = np.mean(erreurs)
                tableau_erreurs[i, j] = taux_erreur_moyen
        except ValueError as e:
            print(e)

   
    print("\nTableau récapitulatif des taux d'erreurs moyens:")
    for i, chemin_image in enumerate(images_chemins):
        print(f"Image: {chemin_image}")
        for j, (m1, sig1, m2, sig2) in enumerate(bruits):
            print(f"  Bruit {j + 1} (m1={m1}, sig1={sig1}, m2={m2}, sig2={sig2}): Taux d'erreur moyen = {tableau_erreurs[i, j] * 100:.2f}%")