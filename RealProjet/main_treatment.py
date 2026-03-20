import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm
## Constantes ##
L = 300  # Longueur d'un côté du domaine [m]
t = 0.2  # temps total de simulation [s]
nx = 300  # nombre de points en x
ny = 300  # nombre de points en y
h = L / nx  # distance entre les points spatiaux [m]

Rho_eau = 1000  # densite volumique [kg/m3]
Kappa_eau = 2.2e9  # bulk modulus [Pa]
Gamma_eau = 0

Rho_acier = 7850
Kappa_acier = 1.6e11
Gamma_acier = 8.1e10

c_max = np.sqrt((Kappa_acier + 4/3 * Gamma_acier)/ Rho_acier) # vitesse du son dans l'eau [m/s]
c_eau = np.sqrt( Kappa_eau/Rho_eau )
C_grid =  c_eau * np.ones((nx-2, ny-2), dtype=float); C_grid[195:205, 195:205] = c_max

alpha = {5: 3.24e-5,   # dictionnaire des coefficients alpha selon la fréquence [Np/m]
         10: 8.94e-5,
         15: 1.78e-4,
         20: 2.91e-4,
         25: 4.21e-4,
         30: 5.60e-4}

gamma_dict = {key: value * c_eau for key, value in alpha.items()}  # dictionnaire des constantes d'atténuation selon la fréquence [s^-1]

## Pas de temps ##
dt_max = h / (c_max * np.sqrt(2))  # pas de temps maximal pour respecter la condition CFL en 2D
dt = 0.9 * dt_max  # pas de temps choisi avec une marge de sécurité
nt = int(t / dt)  # nombre de points temporels

print(f"vitesse de l'onde = {c_max:.2f} m/s")
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

def pulse_gaussien_module(nx, ny, x0=None, y0=None, sigma=10, w=1, A0=1):
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
gamma_max = 20000
u0 = pulse_gaussien_module(nx, ny, sigma=5, w=0.5, A0=1.0)  #pulse

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
pas_sauvegarde = max(1, nt // 500)  # intervalle entre deux sauvegardes d'image

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
        + C_grid**2 * dt**2 * lap[1:-1, 1:-1]
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


vmax = np.max(np.abs(frames))
linthresh = vmax * 1e-4

fig, ax = plt.subplots(figsize=(7, 6))

img = ax.imshow(
    frames[0].T,
    origin="lower",
    cmap="rainbow",
    extent=[0, L, 0, L],
    animated=True,
    norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
)

plt.colorbar(img, ax=ax, label="Amplitude (symlog)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"t = 0.000 s")

def maj(k):
    img.set_array(frames[k].T)
    ax.set_title(f"t = {k * pas_sauvegarde * dt:.4f} s")
    return [img]

ani = FuncAnimation(fig, maj, frames=len(frames), interval=1, blit=False)
plt.tight_layout()
plt.show()
