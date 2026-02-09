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
V3 = []
V6 = []
D = (np.arange(3, 7, 1))
Vth = np.pi ** (D/2)/(sp.special.gamma(D/2+1)) # Volume théorique
for d in range(0, ND):
    D = D_val[d]  # Dimension
    Vtot = 2 ** D # Volume du domaine

    VDth = Vth
    for n in range(0, NNtot):
        Ntot = Ntot_val[n]  # Nombre de points

        Vind = np.zeros(Ness) # Volumes calculés pour chaque essai individuel
        
        erreur = np.zeros(Ness)
        for k in range(0,Ness): # Boucle sur les essais
            # Génération des nombres aléatoires (distribution uniforme)
            np.random.seed() # Initialise le générateur de nombres pseudo-aléatoires afin de ne pas toujours produire la même séquence à l'ouverture de Python...
            pts = a * np.random.uniform(low=-1, high=1, size=(Ntot, D)) # Coordonnées des points

            Nint = np.sum(np.sum(pts**2, axis=1) <= 1) 
            
            Vind[k] = Nint / Ntot * Vtot # Volume calculé pour cet essai
        ecart_type = np.std(Vind, ddof=1)
        incertitude_relative = ecart_type * 100 /(Ness**0.5 * np.mean(Vind))
        Vlist[d][n+1] = [np.mean(Vind), incertitude_relative] # Volume moyenné sur l'ensemble des essais
        if D == 3:
            V3.append(np.mean(Vind))
        if D == 6:
            V6.append(np.mean(Vind))

erreur3D = np.abs(np.array(V3)-Vth[0])/Vth[0]
erreur6D = np.abs(np.array(V6)-Vth[3])/Vth[3]
print(erreur3D)
print(erreur6D)
logN = np.log(Ntot_val)
logE3 = np.log(erreur3D)
logE6 = np.log(erreur6D)

# Ajustement linéaire
p3, b3 = np.polyfit(logN, logE3, 1)
p6, b6 = np.polyfit(logN, logE6, 1)

print(f"Exposant p pour D=3  : {p3:.3f}")
print(f"Exposant p pour D=6  : {p6:.3f}")

plt.loglog(np.array(Ntot_val), erreur3D, 'o-', label=f"D=3")
plt.loglog(np.array(Ntot_val), erreur6D, 's-', label=f"D=6")

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

plt.xlabel("Ntot")
plt.ylabel("Erreur théorique")
plt.legend()
plt.grid()
plt.show()
