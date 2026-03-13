import numpy as np
import matplotlib.pyplot as plt

## Constantes ##
L = 100 # Longueur d'un côté [m]
t = 100 # temps de simulation [s]
nx= 10  # nombre de points en x
ny= 10  # nombre de points en y
h= L/nx # distance entre les points spatiaux [m]
dt=1  # distance entre les points temporaux [s]
nt=t/dt # nombre de points temporaux

rho = 1000 # densite volumique [kg/m3]
kappa = 2.2e9 # bulk modulus [Pa]
c = np.sqrt(kappa/rho) # vitesse du son dans l'eau [m/s]
alpha = {5:3.24e-5, # dictionnaire des coefficients alpha selon la frequence [Np/m]
         10:8.94e-5,
         15:1.78e-4,
         20:2.91e-4,
         25:4.21e-4,
         30:5.60e-4} 
gamma = {key: value * c for key, value in alpha.items()} # dictionnaire des constantes d'atténuation selon la fréquence [s^-1]

## conditions aux frontières ##
d1,d2,d3 = 0,0,0 # x L
e1,e2,e3 = 0,0,0 # y = 0
f1,f2,f3 = 0,0,0 # y = L

u0 = np.zeros((nx,ny), dtype=np.float64).flatten()
u = np.zeros((nx,ny), dtype=np.float64).flatten()
u_next = np.zeros((nx,ny), dtype=np.float64).flatten()
A = np.zeros((nx,ny), dtype=np.float64)

def PML(nx,ny,epaisseur,puissance,gamma):

    grille = np.ones((nx,ny))
    lignes, cols = np.indices((nx,ny))
    dist = np.minimum.reduce([lignes, cols, nx - 1 - lignes, ny - 1 - cols]) #matrice des distances par rapport aux bords
    mask = dist < epaisseur # selectionne les elements du PML selon l'épaisseur
    grille[mask] = (epaisseur - dist[mask]) #modifie les elements de la grille selon leur distance par rapport au edge
    grille = gamma*grille**puissance #construction finale de la grille des gammas

    return grille

print(PML(nx,ny,3,1,0.025))