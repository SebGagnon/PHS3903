import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Constantes ##
L = 100  # Longueur d'un côté du domaine [m]
t = 0.1  # temps total de simulation [s]
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
print(f"pas de temps = {dt:.2e} s")
print(f"nt = {nt}")

## conditions aux frontières ##
c1, c2, c3 = 0, 0, 0
d1, d2, d3 = 0, 0, 0
e1, e2, e3 = 0, 0, 0
f1, f2, f3 = 0, 0, 0

def PML(nx, ny, epaisseur, puissance, gamma_max):
    grille = np.zeros((nx, ny))  # grille de gamma initialisée à 0 dans tout le domaine
    lignes, cols = np.indices((nx, ny))  # matrices contenant les indices de chaque point
    dist = np.minimum.reduce([lignes, cols, nx - 1 - lignes, ny - 1 - cols])  # distance au bord
    mask = dist < epaisseur  # points qui appartiennent à la couche absorbante
    s = (epaisseur - dist[mask]) / epaisseur  # distance normalisée entre 0 et 1
    grille[mask] = gamma_max * s**puissance  # profil progressif de gamma
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
    r = np.sqrt((x - x0)**2 + (y - y0)**2)  # distance au point source

    pulse = A0 * np.exp(-(r**2) / (2 * sigma**2)) * np.cos(w * r)  # modulation en fonction de r
    return pulse

def simuler_avec_pml(epaisseur_pml, puissance_pml, gamma_max_pml, sauvegarder_frames=False):
    # Conditions initiales
    u0 = pulse_gaussien_module(nx, ny, sigma=10, w=0.5, A0=1.0)

    u_nm1 = u0.copy()  # champ au temps n-1
    u_n = u0.copy()    # champ au temps n
    u_np1 = np.zeros((nx, ny))  # champ au temps n+1

    # Sponge layer
    gamma = PML(nx, ny, epaisseur_pml, puissance_pml, gamma_max_pml)

    # Liste des images pour animation si on le veut
    frames = []
    pas_sauvegarde = max(1, nt // 600)

    # Boucle temporelle
    for n in range(nt):
        lap = np.zeros_like(u_n)
        lap[1:-1, 1:-1] = (
            u_n[2:, 1:-1] + u_n[:-2, 1:-1]
            + u_n[1:-1, 2:] + u_n[1:-1, :-2]
            - 4 * u_n[1:-1, 1:-1]
        ) / h**2  # laplacien centré

        a = gamma * dt / 2  # coefficient local d'amortissement

        u_np1[1:-1, 1:-1] = (
            2 * u_n[1:-1, 1:-1]
            - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
            + c**2 * dt**2 * lap[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])  # mise à jour explicite

        ## Frontières
        u_np1[0, :] = 0    # bord gauche
        u_np1[-1, :] = 0   # bord droit
        u_np1[:, 0] = 0    # bord bas
        u_np1[:, -1] = 0   # bord haut

        if sauvegarder_frames and n % pas_sauvegarde == 0:
            frames.append(u_n.copy())

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    # Énergie résiduelle
    energie_initiale = np.sum(u0**2)
    energie_finale = np.sum(u_n**2)
    pourcentage_residuel = 100 * energie_finale / energie_initiale

    return pourcentage_residuel, energie_finale, energie_initiale, gamma, u_n, frames

# PARAMÈTRES À TESTER


liste_epaisseurs = [75]
liste_puissances = [1, 2, 3, 4, 5]
liste_gamma_max = [500,1000,2000, 5000, 10000, 20000, 40000]

resultats = []

print("\nDébut du balayage des paramètres du sponge layer...\n")

for epaisseur in liste_epaisseurs:
    for puissance in liste_puissances:
        for gamma_max in liste_gamma_max:
            residuel, Ef, Ei, gamma, u_final, frames = simuler_avec_pml(
                epaisseur_pml=epaisseur,
                puissance_pml=puissance,
                gamma_max_pml=gamma_max,
                sauvegarder_frames=False
            )

            resultats.append({
                "epaisseur": epaisseur,
                "puissance": puissance,
                "gamma_max": gamma_max,
                "energie_residuelle_pct": residuel
            })

            print(
                f"epaisseur = {epaisseur:3d}, "
                f"puissance = {puissance:1d}, "
                f"gamma_max = {gamma_max:6d}  -->  "
                f"E résiduelle = {residuel:.6f} %"
            )

# ======================================================================
# TRI DES MEILLEURS RÉSULTATS
# ======================================================================

resultats_tries = sorted(resultats, key=lambda x: x["energie_residuelle_pct"])

print("\nMeilleurs paramètres trouvés :")
for i in range(min(10, len(resultats_tries))):
    r = resultats_tries[i]
    print(
        f"{i+1:2d}) epaisseur = {r['epaisseur']}, "
        f"puissance = {r['puissance']}, "
        f"gamma_max = {r['gamma_max']} "
        f"--> E résiduelle = {r['energie_residuelle_pct']:.6f} %"
    )

meilleur = resultats_tries[0]
print("\nParamètres optimaux retenus :")
print(meilleur)

# ======================================================================
# RE-SIMULATION AVEC LES MEILLEURS PARAMÈTRES
# ======================================================================

residuel_opt, Ef_opt, Ei_opt, gamma_opt, u_final_opt, frames_opt = simuler_avec_pml(
    epaisseur_pml=meilleur["epaisseur"],
    puissance_pml=meilleur["puissance"],
    gamma_max_pml=meilleur["gamma_max"],
    sauvegarder_frames=False
)

print(f"\nÉnergie résiduelle optimale : {residuel_opt:.6f} %")


plt.figure(figsize=(6, 5))
plt.imshow(gamma_opt.T, origin="lower", cmap="inferno")
plt.colorbar(label=r"$\gamma(x,y)$")
plt.title("Couche absorbante optimale")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()



# fig, ax = plt.subplots(figsize=(7, 6))
# img = ax.imshow(
#     frames_opt[0].T,
#     origin="lower",
#     cmap="seismic",
#     extent=[0, L, 0, L],
#     animated=True
# )
# plt.colorbar(img, ax=ax, label="Amplitude")
# ax.set_xlabel("x [m]")
# ax.set_ylabel("y [m]")

# def maj(k):
#     img.set_array(frames_opt[k].T)
#     return [img]

# ani = FuncAnimation(fig, maj, frames=len(frames_opt), interval=1, blit=True)
# plt.show()



# 1) Effet de l'épaisseur pour différentes puissances, gamma_max fixé
gamma_ref = meilleur["gamma_max"]

plt.figure(figsize=(8, 5))
for puissance in liste_puissances:
    x = []
    y = []
    for r in resultats:
        if r["puissance"] == puissance and r["gamma_max"] == gamma_ref:
            x.append(r["epaisseur"])
            y.append(r["energie_residuelle_pct"])

    ordre = np.argsort(x)
    x = np.array(x)[ordre]
    y = np.array(y)[ordre]
    plt.plot(x, y, marker='o', label=f"puissance = {puissance}")

plt.xlabel("Épaisseur du sponge layer")
plt.ylabel("Énergie résiduelle [%]")
plt.title(f"Énergie résiduelle vs épaisseur (gamma_max = {gamma_ref})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2) Effet de gamma_max pour différentes puissances, épaisseur fixée
epaisseur_ref = meilleur["epaisseur"]

plt.figure(figsize=(8, 5))
for puissance in liste_puissances:
    x = []
    y = []
    for r in resultats:
        if r["puissance"] == puissance and r["epaisseur"] == epaisseur_ref:
            x.append(r["gamma_max"])
            y.append(r["energie_residuelle_pct"])

    ordre = np.argsort(x)
    x = np.array(x)[ordre]
    y = np.array(y)[ordre]
    plt.plot(x, y, marker='o', label=f"puissance = {puissance}")

plt.xlabel(r"$\gamma_{max}$")
plt.ylabel("Énergie résiduelle [%]")
plt.title(f"Énergie résiduelle vs gamma_max (épaisseur = {epaisseur_ref})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 3) Effet de la puissance pour différentes épaisseurs, gamma_max fixé
plt.figure(figsize=(8, 5))
for epaisseur in liste_epaisseurs:
    x = []
    y = []
    for r in resultats:
        if r["epaisseur"] == epaisseur and r["gamma_max"] == gamma_ref:
            x.append(r["puissance"])
            y.append(r["energie_residuelle_pct"])

    ordre = np.argsort(x)
    x = np.array(x)[ordre]
    y = np.array(y)[ordre]
    plt.plot(x, y, marker='o', label=f"epaisseur = {epaisseur}")

plt.xlabel("Puissance de la courbe")
plt.ylabel("Énergie résiduelle [%]")
plt.title(f"Énergie résiduelle vs puissance (gamma_max = {gamma_ref})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 4) Carte 2D pour une puissance donnée : épaisseur vs gamma_max
puissance_ref = meilleur["puissance"]

mat = np.full((len(liste_epaisseurs), len(liste_gamma_max)), np.nan)

for i, epaisseur in enumerate(liste_epaisseurs):
    for j, gamma_max in enumerate(liste_gamma_max):
        for r in resultats:
            if (
                r["epaisseur"] == epaisseur
                and r["gamma_max"] == gamma_max
                and r["puissance"] == puissance_ref
            ):
                mat[i, j] = r["energie_residuelle_pct"]

plt.figure(figsize=(8, 5))
plt.imshow(mat, origin="lower", aspect="auto")
plt.colorbar(label="Énergie résiduelle [%]")
plt.xticks(range(len(liste_gamma_max)), liste_gamma_max)
plt.yticks(range(len(liste_epaisseurs)), liste_epaisseurs)
plt.xlabel(r"$\gamma_{max}$")
plt.ylabel("Épaisseur")
plt.title(f"Carte d'énergie résiduelle (puissance = {puissance_ref})")
plt.tight_layout()
plt.show()