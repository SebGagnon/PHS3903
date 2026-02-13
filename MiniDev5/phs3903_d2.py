# PHS3903 - Projet de simulation
# Mini-devoir 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats

# Paramètres physiques du problème
g = 9.81     # Champ gravitationnel (m²/s)
m = 1.000    # Masse du pendule (kg)
L = 1.000    # Longueur du câble (m)
beta = 0.1   # Constante d'amortissement (1/s)

# Conditions initiales
theta0 = np.pi/6     # Position initiale (rad)
omega0 = 5           # Vitesse inititale (rad/s)

# Paramètres généraux de simulation
tf = 10             # Temps final (s)
dt0 = 0.01          # Pas de temps le plus élevé (s)

# Boucle sur le nombre de simulations
K = 5                                       # Nombre de simulations
dt_val = [dt0, dt0/2, dt0/4, dt0/8, dt0/16] # Vecteur des pas de temps pour chaque simulation
thetaf = np.zeros(K)                        # Vecteur des positions finales pour chaque simulation

for k in range(0,K):
# Paramètres spécifiques de la simulation
    dt = dt_val[k]               # Pas de temps de la simulation
    N = int(tf/dt)               # Nombre d'itérations

# Initialisation
    t = np.arange(0, tf + dt, dt)  # Vecteur des valeurs t_n
    theta = np.zeros(N + 1)        # Vecteur des valeurs theta_n
    theta[0] = theta0

    # Calcul de theta1 (formule (5))
    theta[1] = theta0 \
               + (1 - beta*dt/2)*omega0*dt \
               - (g/(2*L))*dt**2*np.sin(theta0)

# Exécution
    for n in range(1, N):
        theta[n+1] = (
            4*theta[n]
            - (2 - beta*dt)*theta[n-1]
            - (2*g/L)*dt**2*np.sin(theta[n])
        ) / (2 + beta*dt)

    thetaf[k] = theta[-1]  # Position au temps final tf

# Graphique pour le plus grand pas de temps
plt.plot(t, theta)
plt.xlabel("Temps (s)")
plt.ylabel("θ (rad)")
plt.title("Pendule amorti")
plt.grid()
plt.show()

# Tableau des résultats
for k in range(K):
    print("dt =", dt_val[k], " -> theta(tf) =", thetaf[k])
