import numpy as np

def Grille2D_to_Vecteur(Grille):
    return Grille.flatten().tolist()

grille = [[1,2],[3,4]]
print(Grille2D_to_Vecteur(grille))