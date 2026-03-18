import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Constantes ##
L = 100  # Longueur d'un côté du domaine [m]
t = 1  # temps total de simulation [s]
nx = 300  # nombre de points en x
ny = 300  # nombre de points en y
h = L / nx  # distance entre les points spatiaux [m]

rho = 1000  # densite volumique [kg/m3]
kappa = 2.2e9  # bulk modulus [Pa]
c = np.sqrt(kappa / rho)  # vitesse du son dans l'eau [m/s]

alpha = {5: 3.24e-5,   # dictionnaire des coefficients alpha selon la fréquence [Np/m]
         10: 8.94e-5,
         15: 1.78e-4,
         20: 2.91e-4,
         25: 4.21e-4,
         30: 5.60e-4}

gamma_dict = {key: value * c for key, value in alpha.items()}  # dictionnaire des constantes d'atténuation selon la fréquence [s^-1]

## Pas de temps ##
dt_max = h / (c * np.sqrt(2))  # pas de temps maximal pour respecter la condition CFL en 2D
dt = 0.9 * dt_max  # pas de temps choisi avec une marge de sécurité
nt = int(t / dt)  # nombre de points temporels

print(f"vitesse de l'onde = {c:.2f} m/s")
print(f"distance entre les points = {h:.4f} m")
print(f"durée totale de la simul = {dt:.2e} s")
print(f"nt = {nt}")

## conditions aux frontières ##
c1, c2, c3 = 0, 0, 0 
d1, d2, d3 = 0, 0, 0  
e1, e2, e3 = 0, 0, 0  
f1, f2, f3 = 0, 0, 0  

def PML(nx, ny, epaisseur, puissance, gamma_max):
    grille = np.zeros((nx, ny))  # grille de gamma initialisée à 0 dans tout le domaine
    lignes, cols = np.indices((nx, ny))  # matrices contenant les indices de chaque point
    dist = np.minimum.reduce([lignes, cols, nx - 1 - lignes, ny - 1 - cols])  # matrice des distances par rapport aux bords
    mask = dist < epaisseur  # sélectionne les éléments appartenant à la couche absorbante
    s = (epaisseur - dist[mask]) / epaisseur  # distance normalisée au bord entre 0 et 1
    grille[mask] = gamma_max * s**puissance  # profil progressif de gamma dans la couche absorbante

    return grille

def grille_vers_vecteur(grille):  # prend une grille 2D et retourne un vecteur en partant du bas à gauche
    return np.flipud(grille).flatten()

def vecteur_vers_grille(vecteur, nx, ny):  # prend un vecteur et reconstruit la grille 2D correspondante
    return np.flipud(vecteur.reshape(nx, ny))

def pulse_gaussien_module(nx, ny, x0=None, y0=None, sigma=10, w=0.4, A0=1):
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')  
    r = np.sqrt((x - x0)**2 + (y - y0)**2)  # distance par rapport au point source

    pulse = A0 * np.exp(-(r**2) / (2 * sigma**2)) * np.cos(w * r)  # modulation en fonction de r
    return pulse

#Conditions initiales
u0 = pulse_gaussien_module(nx, ny, sigma=8, w=0.5, A0=1.0)  #pulse

u_nm1 = u0.copy()  # champ au temps n-1
u_n = u0.copy()  # champ au temps n
u_np1 = np.zeros((nx, ny))  # champ au temps n+1

#PML
epaisseur_pml = 75  # épaisseur 
gamma_max = 20000  # valeur maximale de gamma sur les bords finaux
gamma = PML(nx, ny, epaisseur_pml, puissance=3, gamma_max=gamma_max)  

#Plot du PML initial
plt.figure(figsize=(6, 5))
plt.imshow(gamma.T, origin="lower", cmap="inferno")
plt.colorbar(label=r"$\gamma(x,y)$")
plt.title("Couche absorbante")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

frames = []  # liste des images conservées pour l'animation
pas_sauvegarde = max(1, nt // 120)  # intervalle entre deux sauvegardes d'image

# Loop
for n in range(nt):
    lap = np.zeros_like(u_n)  # initialisation du laplacien
    lap[1:-1, 1:-1] = (
        u_n[2:, 1:-1] + u_n[:-2, 1:-1]
        + u_n[1:-1, 2:] + u_n[1:-1, :-2]
        - 4 * u_n[1:-1, 1:-1]
    ) / h**2  # calcul du laplacien par différences centrées

    a = gamma * dt / 2  # coefficient local d'amortissement

    u_np1[1:-1, 1:-1] = (
        2 * u_n[1:-1, 1:-1]
        - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
        + c**2 * dt**2 * lap[1:-1, 1:-1]
    ) / (1 + a[1:-1, 1:-1])  # mise à jour explicite de l'équation d'onde amortie

    ## Frontières
    u_np1[0, :] = 0  # bord gauche
    u_np1[-1, :] = 0  # bord droit
    u_np1[:, 0] = 0  # bord bas
    u_np1[:, -1] = 0  # bord haut

    if n % pas_sauvegarde == 0:
        frames.append(u_n.copy())  # sauvegarde du champ pour l'animation

    u_nm1[:, :] = u_n  #  n devient n-1
    u_n[:, :] = u_np1  # n+1 devient n

# Calcul de l'énergie résiudelle pour tester l'efficacité du PML
energie_initiale = np.sum(u0**2)  # Énergie initiale
energie_finale = np.sum(u_n**2)  # Énergie finale

pourcentage_residuel = 100 * energie_finale / energie_initiale  # pourcentage d'énergie restante dans le domaine

print(f"Énergie résiduelle dans le domaine : {pourcentage_residuel:.6f} %")

# Animation
fig, ax = plt.subplots(figsize=(7, 6))
img = ax.imshow(
    frames[0].T,
    origin="lower",
    cmap="seismic",
    extent=[0, L, 0, L],
    animated=True
)
plt.colorbar(img, ax=ax, label="Amplitude")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

def maj(k):
    img.set_array(frames[k].T)
    
    return [img]

ani = FuncAnimation(fig, maj, frames=len(frames), interval=40, blit=True)
plt.show()