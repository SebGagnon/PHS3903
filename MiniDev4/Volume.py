# PHS3903 - Projet de simulation
# Mini-devoir 1
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate
from scipy import stats

# Paramètres géométriques
R = 1.0 # Rayon de la sphère (m)

# Paramètres généraux de simulation
D_val = [3,4,5,6]  # Nombre de dimensions
Ntot_val = 100 * 2 ** (np.arange(0, 5, 1))  # Nombre de points par essai
Ness = 100 # Nombre d'essais par simulation
a = 1  # Dimension de la boîte cubique dans laquelle les points aléatoires seront générés
VDth = []
# Boucle sur le nombre de simulations
ND = len(D_val)
NNtot = len(Ntot_val)

V = np.zeros((ND, NNtot+1))  # Volumes calculés pour chaque série d'essais
Vlist = V.tolist()
for i in np.arange(0, 4):
    Vlist[i][0] = [D_val[i]," Dimensions"]


for d in range(0, ND):
    D = D_val[d]  # Dimension
    Vtot = 2 ** D # Volume du domaine
    Vth = np.pi ** (D/2)/(sp.special.gamma(D/2+1)) # Volume théorique
    VDth.append(Vth)
    for n in range(0, NNtot):
        Ntot = Ntot_val[n]  # Nombre de points

        Vind = np.zeros(Ness) # Volumes calculés pour chaque essai individuel
        
        erreur = np.zeros(Ness)
        for k in range(0,Ness): # Boucle sur les essais
            # Génération des nombres aléatoires (distribution uniforme)
            np.random.seed() # Initialise le générateur de nombres pseudo-aléatoires afin de ne pas toujours produire la même séquence à l'ouverture de Python...
            pts = a * np.random.uniform(low=-1, high=1, size=(Ntot, D)) # Coordonnées des points
            Nint = 0

            for x in pts:
                Apts=0
                for i in np.arange(0, D, 1):
            # Calcul du volume
                    Apts += x[i]**2

                if Apts <= 1:
                    Nint += 1 # Nombre de points à l'intérieur
            
            
            Vind[k] = Nint / Ntot * Vtot # Volume calculé pour cet essai
        ecart_type = np.abs( - Vth)
        incertitude_relative = ecart_type * 100 /(Ness**0.5 * np.mean(Vind))
        Vlist[d][n+1] = [np.mean(Vind), incertitude_relative] # Volume moyenné sur l'ensemble des essais
header = []
header.append("Dimensions")
for i in Ntot_val:
    header.append(f"N = {i}")
header.append("Valeur théorique")
j=0
Table = []
for m in Vlist:
    vecteur = []
    h = 0
    for value in m:
        if isinstance(value[1], str):
            vecteur.append(f"{value[0]:5g} {value[1]}") 
        else:    
            vecteur.append(f"{value[0]:5g}{u"\u00B1"}{value[1]:3g}%")   
    Table.append(vecteur) 
for i in Table:
    i.append(VDth[j])
    j+=1

print(tabulate(Table, headers = header, tablefmt="github"))
