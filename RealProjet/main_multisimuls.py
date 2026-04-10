import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.signal import hilbert

# Le maillage de être au moins plus petit que la longueur d'onde sur 2

## Constantes ##
L = 1000  # Longueur d'un côté du domaine [m]
t = 4  # temps total de simulation [s]
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

# Définition des pulses : (idx_capteur_emetteur, temps_emission)
pulses_config = [
    (0, 0.0),   # Capteur 1 émet à t=0.00s
    (1, 0.8),   # Capteur 1 émet à t=0.80s
    (2, 1.6),   # Capteur 2 émet à 
    (0, 2.4),   # Capteur 3 émet à 
    (1, 3.2),   # Capteur 3 émet à t=0.30s
]
t0_vect = np.array([p[1] for p in pulses_config])

print(f"vitesse de l'onde maximale = {c_max:.2f} m/s")
print(f"distance entre les points = {h:.4f} m")
print(f"durée totale de la simul = {dt:.2e} s")
print(f"nt = {nt}")

# Conditions aux frontières 
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

def pulse_gaussien_module(nx, ny, x0=None, y0=None, sigma= 10 , w = 0.4, A0 = 1 ):
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')  
    r = np.sqrt((x - x0)**2 + (y - y0)**2)  # distance par rapport au point source

    pulse = A0 * np.exp(-(r**2) / (2 * sigma**2)) * np.cos(w * r)  # modulation en fonction de r
    return pulse

def source_temporelle_gaussienne(nx, ny, t, x0=None, y0=None, sigma=10, w=2*np.pi*5000, A0 = 1):
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    r2 = (x - x0)**2 + (y - y0)**2

    source = A0 * np.exp(-r2 / (2 * sigma**2)) * np.cos(w * t)
    return source

def show_pml(show_PML):
    if show_PML == True:

        #Plot du PML initial
        plt.figure(figsize=(6, 5))
        plt.imshow(gamma.T, origin="lower", cmap="inferno")
        plt.colorbar(label=r"$\gamma(x,y)$")
        plt.title("Couche absorbante")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

def simulation_sonar(position_capteurs, position_bateau, taille_bateau, pulses_config, vx_bat=0, vy_bat=0, show_animation = True, show_capteurs = False):

    vx_noeuds = vx_bat / h
    vy_noeuds = vy_bat / h

    u0    = np.zeros((nx, ny))
    u_nm1 = u0.copy()
    u_n   = u0.copy()
    u_np1 = np.zeros((nx, ny))

    cx1, cy1 = position_capteurs[0]
    cx2, cy2 = position_capteurs[1]
    cx3, cy3 = position_capteurs[2]

    frames          = []
    frames_pos_bateau = []
    pas_sauvegarde  = max(1, nt // 120)

    steps  = nt
    t_hist  = np.zeros(steps)
    p1_hist = np.zeros(steps)
    p2_hist = np.zeros(steps)
    p3_hist = np.zeros(steps)

    # Construction des indices de pulse
    pulse_indices = []
    for idx_emetteur, t0 in pulses_config:
        pulse_indices.append({
            "pos": position_capteurs[idx_emetteur],
            "n0" : int(t0 / dt),
            "idx_emetteur": idx_emetteur
        })

    for n in range(nt):

        # --- Position du bateau à l'instant n (en noeuds) ---
        dx = int(np.floor(n * dt * vx_noeuds))
        dy = int(np.floor(n * dt * vy_noeuds))

        x_bat = position_bateau[0] + dx
        y_bat = position_bateau[1] + dy

        # Grille de vitesse mise à jour à chaque pas
        C_grid = c_eau * np.ones((nx-2, ny-2), dtype=float)
        C_grid[x_bat - taille_bateau : x_bat + taille_bateau,
               y_bat - taille_bateau : y_bat + taille_bateau] = c_max

        lap = np.zeros_like(u_n)
        lap[1:-1, 1:-1] = (
            4 * (
                u_n[2:, 1:-1] + u_n[:-2, 1:-1]
                + u_n[1:-1, 2:] + u_n[1:-1, :-2]
            )
            + (
                u_n[2:, 2:] + u_n[2:, :-2]
                + u_n[:-2, 2:] + u_n[:-2, :-2]
            )
            - 20 * u_n[1:-1, 1:-1]
        ) / (6 * h**2)

        a = gamma * dt / 2

        u_np1[1:-1, 1:-1] = (
            2 * u_n[1:-1, 1:-1]
            - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
            + C_grid**2 * dt**2 * lap[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])

        for pulse in pulse_indices:
            if n == pulse["n0"]:
                sx, sy = pulse["pos"]
                array_pulse = pulse_gaussien_module(nx, ny, x0=sx, y0=sy)
                u_n   = u_n   + array_pulse
                u_np1 = u_np1 + array_pulse

        u_np1[0, :]  = 0
        u_np1[-1, :] = 0
        u_np1[:, 0]  = 0
        u_np1[:, -1] = 0

        t_hist[n]  = n * dt
        p1_hist[n] = u_n[cx1, cy1]
        p2_hist[n] = u_n[cx2, cy2]
        p3_hist[n] = u_n[cx3, cy3]

        if n % pas_sauvegarde == 0:
            frames.append(u_n.copy())
            frames_pos_bateau.append((x_bat, y_bat))  # ← ajout

        u_nm1[:, :] = u_n
        u_n[:, :]   = u_np1

    intensité_capteurs = np.array([p1_hist, p2_hist, p3_hist])
    # Calcul de l'énergie résiudelle pour tester l'efficacité du PML
    energie_initiale = np.sum(u0**2)  # Énergie initiale
    energie_finale = np.sum(u_n**2)  # Énergie finale

    # pourcentage_residuel = 100 * energie_finale / energie_initiale  # pourcentage d'énergie restante dans le domaine

    # print(f"Énergie résiduelle dans le domaine : {pourcentage_residuel:.6f} %")à

    if show_animation:
        afficher_animation(frames, frames_pos_bateau, taille_bateau, position_capteurs, L, h)

    if show_capteurs == True:
        
        plt.figure(figsize=(8,5))
        plt.plot(t_hist, p1_hist, label="Capteur 1")
        plt.plot(t_hist, p2_hist, label="Capteur 2")
        plt.plot(t_hist, p3_hist, label="Capteur 3")
        plt.xlabel("Temps [s]")
        plt.ylabel("Amplitude")
        plt.title("Signaux enregistrés par les capteurs")
        plt.legend()
        plt.grid()
        plt.show()
    

    return intensité_capteurs, t_hist

def afficher_animation(frames, frames_pos_bateau, taille_bateau, position_capteurs, L, h=h):

    capteurs_x = [pos[0] * h for pos in position_capteurs]  
    capteurs_y = [pos[1] * h for pos in position_capteurs]  

    fig, ax = plt.subplots(figsize=(7, 6))

    img = ax.imshow(
        frames[0].T,
        origin="lower",
        cmap="seismic",
        extent=[0, L, 0, L],
        animated=True,)
    
    plt.colorbar(img, ax=ax, label="Amplitude")

    # --- Capteurs ---
    couleurs_capteurs = ["red", "blue", "green"]
    for i, (cx, cy) in enumerate(zip(capteurs_x, capteurs_y)):
        ax.scatter(cx, cy, color=couleurs_capteurs[i], marker="^", s=80, zorder=5, label=f"C{i+1}")

    # --- Bateau (rectangle dynamique) ---
    x_bat0, y_bat0 = frames_pos_bateau[0]
    taille_m = taille_bateau * h

    bateau_rect = plt.Rectangle(
        (y_bat0 * h - taille_m, x_bat0 * h - taille_m),  # coin bas-gauche
        2 * taille_m, 2 * taille_m,
        linewidth=2, edgecolor="black", facecolor="brown", alpha=0.7,
        animated=True, label="Bateau"
    )
    ax.add_patch(bateau_rect)

    # --- Trajectoire ---
    traj_x = [pos[1] * h for pos in frames_pos_bateau]
    traj_y = [pos[0] * h for pos in frames_pos_bateau]
    traj_line, = ax.plot([], [], color="black", linestyle="--", linewidth=1, alpha=0.5, label="Trajectoire")

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    def maj(k):
        # Champ
        img.set_array(frames[k].T)

        # Position bateau
        x_bat, y_bat = frames_pos_bateau[k]
        bateau_rect.set_xy((y_bat * h - taille_m, x_bat * h - taille_m))

        # Trajectoire jusqu'à maintenant
        traj_line.set_data(traj_x[:k+1], traj_y[:k+1])

        # Titre avec le temps
        ax.set_title(f"t = {k * (t / len(frames)):.3f} s")

        return [img, bateau_rect, traj_line]

    ani = FuncAnimation(fig, maj, frames=len(frames), interval=40, blit=True)
    plt.tight_layout()
    plt.show()

def get_plages_pulses(pulses_config, t_total):

    # Si c'est un array simple de temps (ancien format)
    if np.ndim(pulses_config[0]) == 0:
        t0_list = list(pulses_config)
        plages  = []
        for i, t0 in enumerate(t0_list):
            t_debut = t0
            t_fin   = t0_list[i + 1] if i + 1 < len(t0_list) else t_total
            plages.append((i, t_debut, t_fin))  # idx_emit = i par défaut
        return plages

    # Si c'est une liste de tuples (idx_emit, t0) — nouveau format
    t0_list = [p[1] for p in pulses_config]
    plages  = []
    for i, (idx_emit, t0) in enumerate(pulses_config):
        t_debut = t0
        t_fin   = t0_list[i + 1] if i + 1 < len(pulses_config) else t_total
        plages.append((idx_emit, t_debut, t_fin))
    return plages

def max_enveloppe_plage(signal, t_hist, t_debut_plage, t_fin_plage, seuil_montee= 0.4):
    """
    Trouve le maximum de l'enveloppe dans une plage temporelle donnée,
    puis remonte au début du front montant.

    t_debut_plage : temps de début de la plage [s] (= t0 du pulse émis)
    t_fin_plage   : temps de fin de la plage [s]  (= t0 du prochain pulse, ou t_fin si dernier)
    seuil_montee  : fraction du pic pour définir le début du front montant
    """

    # --- Sélection des indices dans la plage ---
    mask = (t_hist >= t_debut_plage) & (t_hist < t_fin_plage)

    if not np.any(mask):
        return None, None, np.abs(hilbert(signal))

    signal_plage = signal[mask]
    t_hist_plage = t_hist[mask]

    # --- Enveloppe sur la plage ---
    enveloppe_plage = np.abs(hilbert(signal_plage))

    # --- Maximum dans la plage ---
    idx_pic_local = np.argmax(enveloppe_plage)
    amp_pic       = enveloppe_plage[idx_pic_local]

    # --- Remontée vers le début du front montant ---
    seuil_debut = seuil_montee * amp_pic
    idx_debut   = idx_pic_local
    while idx_debut > 0 and enveloppe_plage[idx_debut] > seuil_debut:
        idx_debut -= 1

    t_debut  = t_hist_plage[idx_debut]
    amp_debut = enveloppe_plage[idx_debut]

    # Enveloppe complète pour l'affichage
    enveloppe_full = np.abs(hilbert(signal))

    return t_debut, amp_debut, enveloppe_full

def show_echo(intensité_echo, t_hist, pulses_config, Echo=True):
    """
    Affiche les échos par plage temporelle (un sous-graphique par pulse).
    """
    if not Echo:
        return

    plages        = get_plages_pulses(pulses_config, t)
    n_pulses      = len(plages)
    couleurs      = ["red", "blue", "green"]
    labels        = ["Capteur 1", "Capteur 2", "Capteur 3"]

    fig, axes = plt.subplots(n_pulses, 1, figsize=(10, 4 * n_pulses), sharex=False)

    # Si un seul pulse, axes n'est pas une liste
    if n_pulses == 1:
        axes = [axes]

    for k, (idx_emit, t_debut, t_fin) in enumerate(plages):
        ax = axes[k]

        # Masque temporel pour la plage
        mask = (t_hist >= t_debut) & (t_hist < t_fin)

        for i, signal in enumerate(intensité_echo):

            # Signal complet sur la plage
            ax.plot(t_hist[mask], signal[mask],
                    color=couleurs[i], alpha=0.4)

            # Enveloppe + pic sur la plage
            t_max, amp_max, enveloppe_full = max_enveloppe_plage(
                signal, t_hist, t_debut, t_fin
            )
            ax.plot(t_hist[mask], enveloppe_full[mask],
                    color=couleurs[i], linestyle="--", label=labels[i])

            if t_max is not None:
                ax.scatter(t_max, amp_max, color=couleurs[i], s=80, zorder=5)
                ax.axvline(t_max, color=couleurs[i], linestyle=":", alpha=0.7)

        # Ligne d'émission du pulse
        ax.axvline(t_debut, color="black", linestyle="-", linewidth=1.5,
                   label=f"Émission C{idx_emit+1} (t={t_debut:.2f}s)")

        ax.set_xlim(t_debut, t_fin)
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Pulse {k+1} — Émetteur C{idx_emit+1} | [{t_debut:.2f}s, {t_fin:.2f}s]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle("Échos par plage de pulse", fontsize=13)
    plt.tight_layout()
    plt.show()

def multilateration_un_pulse(intensite_echos, t_hist, position_capteurs, c_eau, idx_emetteur=0, t_debut_plage=None, t_fin_plage=None):

    intensite_echos          = np.array(intensite_echos)
    position_capteurs_noeuds = np.array(position_capteurs)
    n_capteurs               = len(position_capteurs)

    # -------- DETECTION DES TEMPS D'ECHO --------
    temps_echo = []
    for i, signal in enumerate(intensite_echos):
        if t_debut_plage is not None and t_fin_plage is not None:
            t_max, amp_max, enveloppe = max_enveloppe_plage(signal, t_hist, t_debut_plage, t_fin_plage)

        if t_max is None:
            raise ValueError(f"Pas de pic détecté pour capteur {i+1}")
        temps_echo.append(t_max)

    temps_echo = np.array(temps_echo)

    # t0 du pulse émis
    t0_emit = t_debut_plage if t_debut_plage is not None else 0.0

    # -------- DISTANCES (toujours indexées 0,1,2 = capteurs) --------
    # Émetteur : aller-retour
    d_emetteur = c_eau * (temps_echo[idx_emetteur] - t0_emit) / 2 / h

    # Récepteurs : trajet total - aller
    indices_recepteurs = [i for i in range(n_capteurs) if i != idx_emetteur]
    d_recepteurs = {
        i: c_eau * (temps_echo[i] - t0_emit) / h - d_emetteur
        for i in indices_recepteurs
    }

    # Assemblage distances indexées par capteur (toujours taille n_capteurs)
    distances = np.zeros(n_capteurs)
    distances[idx_emetteur] = d_emetteur
    for i in indices_recepteurs:
        distances[i] = d_recepteurs[i]

    # -------- RÉSIDUS --------
    def residuals(x):
        res = [np.linalg.norm(x - position_capteurs_noeuds[idx_emetteur]) - d_emetteur]
        for i in indices_recepteurs:
            res.append(np.linalg.norm(x - position_capteurs_noeuds[i]) - d_recepteurs[i])
        return res

    x0  = np.mean(position_capteurs_noeuds, axis=0)
    sol = least_squares(residuals, x0)

    return sol.x, temps_echo, distances, idx_emetteur  # ← on retourne aussi idx_emetteur

def calc_position_reelle_impact(position_reelle, t0_emit, d_emetteur, vx_bat, vy_bat, h, c_eau):
    t_aller  = d_emetteur * h / c_eau   # temps aller [s]
    t_impact = t0_emit + t_aller

    dx_impact = vx_bat * t_impact  # [m]
    dy_impact = vy_bat * t_impact  # [m]

    position_impact_m = np.array(position_reelle) * h + np.array([dx_impact, dy_impact])

    return position_impact_m, t_impact

def plot_localisation_echo(position_capteurs, position_estimee, position_reelle, distances, taille_bateau, t0_emit, d_emetteur, vx_bat, vy_bat, L=L, h=h):
    """
    position_reelle : position initiale du bateau [noeuds]
    t0_emit         : temps d'émission du pulse [s]
    d_emetteur      : distance émetteur → bateau [noeuds]
    vx_bat, vy_bat  : vitesses du bateau [m/s]
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    n_capteurs      = position_capteurs.shape[0]
    couleurs_capteurs = ["red", "blue", "green"]

    # --- Temps d'impact : t0 + temps aller ---
    t_aller  = d_emetteur * h / c_eau   # distance [m] / vitesse [m/s]
    t_impact = t0_emit + t_aller

    # --- Position réelle au moment de l'impact ---
    dx_impact = (vx_bat / h) * t_impact  # noeuds
    dy_impact = (vy_bat / h) * t_impact  # noeuds
    position_impact = np.array(position_reelle) + np.array([dx_impact, dy_impact])

    # --- Conversion maillage → mètres ---
    capteurs_m  = np.array(position_capteurs) * h
    estimee_m   = np.array(position_estimee)  * h
    reelle_m    = np.array(position_reelle)   * h
    impact_m    = position_impact             * h
    distances_m = np.array(distances)         * h

    taille_m = taille_bateau * h

    # --- Position initiale (fantôme) ---
    x_r, y_r = reelle_m
    bateau_initial = plt.Rectangle(
        (y_r - taille_m / 2, x_r - taille_m / 2),
        taille_m, taille_m,
        color="grey", alpha=0.3, linestyle="--",
        label="Position initiale (t=0)", zorder=3
    )
    ax.add_patch(bateau_initial)

    # --- Position au moment de l'impact ---
    x_i, y_i = impact_m
    bateau_impact = plt.Rectangle(
        (y_i - taille_m / 2, x_i - taille_m / 2),
        taille_m, taille_m,
        color="black", alpha=0.6,
        label=f"Position à l'impact (t={t_impact:.3f} s)", zorder=4
    )
    ax.add_patch(bateau_impact)

    # --- Trajectoire initiale → impact ---
    ax.annotate("", xy=(y_i, x_i), xytext=(y_r, x_r),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5, linestyle="dashed"))

    # --- Position estimée ---
    x_e, y_e = estimee_m
    ax.scatter(y_e, x_e, marker="*", color="gold", s=250, zorder=5,
               edgecolors="black", linewidths=0.8, label="Position estimée (MC)")

    # --- Capteurs + cercles + lignes ---
    for i in range(n_capteurs):
        x_c, y_c = capteurs_m[i]
        couleur   = couleurs_capteurs[i]

        ax.scatter(y_c, x_c, color=couleur, s=100, zorder=4, marker="^")
        ax.text(y_c + 0.5, x_c + 0.5, f"C{i+1}", color=couleur, fontsize=10, fontweight="bold")

        cercle = plt.Circle(
            (y_c, x_c), radius=distances_m[i],
            color=couleur, fill=False, linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"C{i+1} — d = {distances_m[i]:.1f} m"
        )
        ax.add_patch(cercle)

        ax.plot([y_c, y_e], [x_c, x_e],
                color=couleur, linestyle="-", linewidth=1.2, alpha=0.6)

        mid_x = (y_c + y_e) / 2
        mid_y = (x_c + x_e) / 2
        ax.text(mid_x, mid_y, f"{distances_m[i]:.1f} m", color=couleur,
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    # --- Erreur par rapport à la position d'impact ---
    erreur = np.linalg.norm(estimee_m - impact_m)
    ax.set_title(f"Localisation par multilatération\n"
                 f"Erreur vs impact : {erreur:.2f} m  |  t_impact = {t_impact:.3f} s", fontsize=13)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

def generer_position_bateau(position_capteurs, nx, ny, epaisseur_pml, distance_min_capteur, taille_bateau):
    """
    position_capteurs    : array[N_capteurs, 2] en indices de maillage
    nx, ny               : taille du maillage
    epaisseur_pml        : épaisseur de la couche PML en indices
    distance_min_capteur : distance minimale aux capteurs en indices
    taille_bateau        : demi-taille du bateau en indices
    """

    # Zone valide : à l'intérieur du PML + marge pour le bateau
    marge = epaisseur_pml + taille_bateau
    x_min, x_max = marge, nx - marge
    y_min, y_max = marge, ny - marge

    position_capteurs = np.array(position_capteurs)

    while True:
        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)

        # Vérification distance minimale à chaque capteur
        distances = np.linalg.norm(position_capteurs - np.array([x, y]), axis = 0)

        if np.all(distances >= distance_min_capteur):
            return [x, y]

def plot_positions_estimees(position_capteurs, positions_estimees, positions_reelles_impact, t_impacts, t0_par_pulse, taille_bateau, vx_bat=0, vy_bat=0, L=L, h=h):

    fig, ax = plt.subplots(figsize=(9, 9))
    n_capteurs         = position_capteurs.shape[0]
    couleurs_capteurs  = ["red", "blue", "green"]
    capteurs_m         = np.array(position_capteurs) * h
    positions_estimees = np.array(positions_estimees)
    positions_m        = positions_estimees * h
    taille_m           = taille_bateau * h

    # --- Trajectoire réelle ---
    t_fine   = np.linspace(min(t_impacts), max(t_impacts), 200)
    x0_m, y0_m = np.array(positions_reelles_impact[0]) - vx_bat * t_impacts[0], \
                 np.array(positions_reelles_impact[0]) - vy_bat * t_impacts[0]
    # On reconstitue depuis la position initiale en mètres
    pos_init_m = np.array(position_capteurs[0]) * 0  # juste pour clarté
    traj_x = [r[0] for r in positions_reelles_impact]
    traj_y = [r[1] for r in positions_reelles_impact]
    ax.plot(traj_y, traj_x, color="black", linestyle="--", linewidth=1.2,
            alpha=0.5, label="Trajectoire réelle", zorder=2)

    # --- Position réelle à chaque impact ---
    for k, (pos_r, t_imp) in enumerate(zip(positions_reelles_impact, t_impacts)):
        x_r, y_r = pos_r
        bateau = plt.Rectangle(
            (y_r - taille_m / 2, x_r - taille_m / 2),
            taille_m, taille_m,
            color="black", alpha=0.15, zorder=3,
            label="Position réelle à l'impact" if k == 0 else "_nolegend_"
        )
        ax.add_patch(bateau)
        ax.text(y_r + 0.5, x_r + 0.5, f"R{k+1}\nt={t_imp:.2f}s", color="black", fontsize=7, alpha=0.7)

    # --- Capteurs ---
    for i in range(n_capteurs):
        x_c, y_c = capteurs_m[i]
        ax.scatter(y_c, x_c, color=couleurs_capteurs[i], s=100, zorder=5, marker="^")
        ax.text(y_c + 0.5, x_c + 0.5, f"C{i+1}", color=couleurs_capteurs[i], fontsize=10, fontweight="bold")
        
    # --- Positions estimées ---
    cmap    = plt.cm.plasma
    colors  = [cmap(k / max(len(positions_estimees) - 1, 1)) for k in range(len(positions_estimees))]
    erreurs = []

    for k, (pos_m, pos_r, couleur, t0) in enumerate(zip(positions_m, positions_reelles_impact, colors, t0_par_pulse)):
        x_e, y_e = pos_m
        x_r, y_r = pos_r
        erreur    = np.linalg.norm(pos_m - pos_r)
        erreurs.append(erreur)

        ax.scatter(y_e, x_e, marker="*", color=couleur, s=250, zorder=6,
                   edgecolors="black", linewidths=0.8,
                   label=f"P{k+1} | t={t0:.2f}s | err={erreur:.1f}m")
        ax.text(y_e + 0.5, x_e + 0.5, f"P{k+1}", color=couleur, fontsize=9, fontweight="bold")

        # Ligne erreur réelle → estimée
        ax.plot([y_r, y_e], [x_r, x_e],
                color=couleur, linestyle=":", linewidth=1.0, alpha=0.7)

    # --- Trajectoire estimée ---
    if len(positions_m) >= 2:
        ax.plot(positions_m[:, 1], positions_m[:, 0],
                color="orange", linestyle="-", linewidth=1.5,
                alpha=0.8, label="Trajectoire estimée", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(t0_par_pulse), vmax=max(t0_par_pulse)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Temps d'émission [s]", shrink=0.6)

    erreur_moy = np.mean(erreurs)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_aspect("equal")
    ax.set_title(f"Évolution des positions estimées\nErreur moyenne : {erreur_moy:.2f} m", fontsize=13)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

def estimer_vitesse(positions_estimees, t0_par_pulse, h):

    positions_estimees = np.array(positions_estimees)
    t0_par_pulse       = np.array(t0_par_pulse)

    if len(positions_estimees) < 2:
        print("⚠ Pas assez de positions pour estimer la vitesse.")
        return None, None

    # Positions en mètres
    positions_m = positions_estimees * h  # [N_pulses, 2]

    # Régression linéaire sur x et y séparément
    # x(t) = vx * t + x0
    # y(t) = vy * t + y0
    A = np.column_stack([t0_par_pulse, np.ones(len(t0_par_pulse))])  # matrice [N, 2]

    # Moindres carrés : [vx, x0] et [vy, y0]
    sol_x, _, _, _ = np.linalg.lstsq(A, positions_m[:, 0], rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, positions_m[:, 1], rcond=None)

    vx_estime, x0_estime = sol_x
    vy_estime, y0_estime = sol_y

    vitesse_estimee = np.array([vx_estime, vy_estime])
    norme_vitesse   = np.linalg.norm(vitesse_estimee)

    print(f"\n{'='*45}")
    print(f"  Estimation de la vitesse du bateau")
    print(f"{'='*45}")
    print(f"  vx estimé : {vx_estime:.2f} m/s")
    print(f"  vy estimé : {vy_estime:.2f} m/s")
    print(f"  ||v||     : {norme_vitesse:.2f} m/s")
    print(f"{'='*45}")

    # --- Graphique ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t_fine = np.linspace(t0_par_pulse[0], t0_par_pulse[-1], 100)

    for ax, coord, sol, label in zip(
        axes,
        [positions_m[:, 0], positions_m[:, 1]],
        [sol_x, sol_y],
        ["x [m]", "y [m]"]
    ):
        v_est, p0_est = sol
        ax.scatter(t0_par_pulse, coord, color="blue", zorder=5, label="Positions estimées")
        ax.plot(t_fine, v_est * t_fine + p0_est, color="red", linestyle="--", label=f"Régression : v = {v_est:.2f} m/s")
        ax.set_xlabel("Temps d'émission [s]")
        ax.set_ylabel(label)
        ax.set_title(f"Régression linéaire — {label}")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.suptitle("Estimation de la vitesse par régression linéaire", fontsize=13)
    plt.tight_layout()
    plt.show()

    return vitesse_estimee, (x0_estime, y0_estime)

# Paramètre du PML
epaisseur_pml = 75  # épaisseur 
gamma_max = 20000  # valeur maximale de gamma sur les bords finaux
gamma = PML(nx, ny, epaisseur_pml, puissance=3, gamma_max=gamma_max)  

# Graphique du PML
show_pml(False)

# Position et taille du bateau (pour le moment c'est juste un carré)
position_bateau = [150, 150] # ( x, y ) en position du noeud. Ne sert seulement si position_aléatoire = False
taille_bateau = 6 # En nombre de noeuds
vx_bat = 50 #m/s
vy_bat = 0 #m/s


# Si on veut générer une position aléatoire du bateau
distance_min_capteur = 25 # Distance minimale entre le capteur et le bateau (sinon ça detecte pas l'echo)
n_simulations        = 20 # Nombre de simulation (ne sert seulement si position_aléatoire = False)

# Positions des capteurs (en position du noeud)
sensor1_pos = (85, 215)
sensor2_pos = (85, 75)
sensor3_pos = (225, 75)
position_capteurs = np.array([sensor1_pos, sensor2_pos, sensor3_pos])
plages = get_plages_pulses(t0_vect, t)

position_aléatoire = False
if position_aléatoire == True:
    #erreurs_centre, erreurs_bord = run_simulations(n_simulations, position_capteurs, distance_min_capteur, taille_bateau, nx, ny, epaisseur_pml, c_eau)
    joe = "biden"


else : # Cette option sert si on veut juste faire une seule simulation
    
    # Sert à trouver l'intensité de l'echo (on pourrait optimiser quand on va faire pleins de simulations pour ne pas refaire celle sans bateau)

    intensité_avec_bateau, t_hist = simulation_sonar(position_capteurs, position_bateau, taille_bateau, pulses_config, vx_bat=vx_bat, vy_bat=vy_bat)
    intensité_sans_bateau, t_hist = simulation_sonar(position_capteurs, [5, 5], taille_bateau, pulses_config, vx_bat=0, vy_bat=0)
    intensité_echo = intensité_avec_bateau - intensité_sans_bateau

    # Graphique de l'echo
    show_echo(intensité_echo, t_hist, pulses_config)

    plages = get_plages_pulses(pulses_config, t)

    positions_estimees  = []
    distances_par_pulse = []
    t0_par_pulse        = []
    plot_schema = True

    positions_reelles_impact = [] 
    t_impacts                = []  

    for k, (idx_emit, t_debut_plage, t_fin_plage) in enumerate(plages):

        pos, temps_echo, distances, idx_emit_retour = multilateration_un_pulse(
            intensité_echo, t_hist, position_capteurs, c_eau,
            idx_emetteur  = idx_emit,
            t_debut_plage = t_debut_plage,
            t_fin_plage   = t_fin_plage
        )

        # Position réelle au moment de l'impact
        position_impact_m, t_impact = calc_position_reelle_impact(
            position_bateau,
            t0_emit    = t_debut_plage,
            d_emetteur = distances[idx_emit_retour],
            vx_bat     = vx_bat,
            vy_bat     = vy_bat,
            h          = h,
            c_eau      = c_eau
        )

        positions_reelles_impact.append(position_impact_m)
        t_impacts.append(t_impact)
        positions_estimees.append(pos)
        distances_par_pulse.append(distances)
        t0_par_pulse.append(t_debut_plage)

        erreur = np.linalg.norm(np.array(pos) * h - position_impact_m)
        print(f"  Position réelle à l'impact : {position_impact_m} m  (t_impact={t_impact:.3f}s)")
        print(f"  Position estimée           : {np.array(pos) * h} m")
        print(f"  Erreur                     : {erreur:.2f} m")

        if plot_schema:
            plot_localisation_echo(
                position_capteurs, pos, position_bateau, distances,
                taille_bateau,
                t0_emit    = t_debut_plage,
                d_emetteur = distances[idx_emit_retour],
                vx_bat     = vx_bat,
                vy_bat     = vy_bat
            )

    positions_estimees = np.array(positions_estimees)

    plot_positions_estimees(
        position_capteurs, positions_estimees, positions_reelles_impact,
        t_impacts, t0_par_pulse, taille_bateau,
        vx_bat=vx_bat, vy_bat=vy_bat)

    # Moyenne
    pos_moyenne = np.mean(positions_estimees, axis=0)
    erreur_moyenne = np.linalg.norm(pos_moyenne - np.array(position_bateau))
    print(f"\n--- Moyenne des {len(positions_estimees)} estimées ---")
    print(f"  Position moyenne : {pos_moyenne * h} m")
    print(f"  Erreur moyenne   : {erreur_moyenne * h:.2f} m")

    # Estimation de la vitesse
    if len(positions_estimees) >= 2:
        vitesse_estimee, position_initiale = estimer_vitesse(positions_estimees, t0_par_pulse, h)

        # Comparaison avec la vraie vitesse
        if vitesse_estimee is not None:
            vrai_v     = np.array([vx_bat, vy_bat])
            erreur_v   = np.linalg.norm(vitesse_estimee - vrai_v)
            print(f"  Vitesse réelle    : vx={vx_bat:.2f} m/s, vy={vy_bat:.2f} m/s, ||v||={np.linalg.norm(vrai_v):.2f} m/s")
            print(f"  Erreur sur vitesse: {erreur_v:.2f} m/s")

