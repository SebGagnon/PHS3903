import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.signal import hilbert

# Le maillage de être au moins plus petit que la longueur d'onde sur 2

## Constantes ##
L = 100  # Longueur d'un côté du domaine [m]
t = 0.3  # temps total de simulation [s]
nx = 100  # nombre de points en x
ny = 100  # nombre de points en y
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

# Temps des pulses (modifiable à un seul endroit)
t0_1 = 0.0
t0_2 = 0.1
t0_3 = 0.2

# Vecteur des temps
t0_vect = np.array([t0_1, t0_2, t0_3])

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

def pulse_gaussien_module(nx, ny, x0=None, y0=None, sigma= 10 , w = 0.4, A0= 1 ):
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')  
    r = np.sqrt((x - x0)**2 + (y - y0)**2)  # distance par rapport au point source

    pulse = A0 * np.exp(-(r**2) / (2 * sigma**2)) * np.cos(w * r)  # modulation en fonction de r
    return pulse

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

def simulation_sonar(position_capteurs, position_bateau, taille_bateau, show_animation = True, show_capteurs=False):

    #Conditions initiales
    C_grid =  c_eau * np.ones((nx-2, ny-2), dtype=float);  C_grid[ (position_bateau[0] - taille_bateau) : (position_bateau[0] + taille_bateau), (position_bateau[1] - taille_bateau) : (position_bateau[1] + taille_bateau)] = c_max
    #u0 = pulse_gaussien_module(nx, ny, x0 = 225, y0 = 150)  #pulse
    u0 = np.zeros((nx, ny)) 
    u_nm1 = u0.copy()  # champ au temps n-1
    u_n = u0.copy()  # champ au temps n
    u_np1 = np.zeros((nx, ny))  # champ au temps n+1


    pulses = [
        {"pos": sensor1_pos, "t0": t0_1},
        {"pos": sensor2_pos, "t0": t0_2},
        {"pos": sensor3_pos, "t0": t0_3},
    ]

    cx1, cy1 = sensor1_pos
    cx2, cy2 = sensor2_pos
    cx3, cy3 = sensor3_pos
    frames = []  # liste des images conservées pour l'animation
    pas_sauvegarde = max(1, nt // 120)  # intervalle entre deux sauvegardes d'image



    # buffers capteurs
    steps = nt
    t_hist  = np.zeros(steps)
    p1_hist = np.zeros(steps)
    p2_hist = np.zeros(steps)
    p3_hist = np.zeros(steps)

    pulse_indices = []

    for p in pulses:
        pulse_indices.append({
            "pos": p["pos"],
            "n0": int(p["t0"] / dt)
        })

    temps = 0
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

        # Détection des pulses
        for pulse in pulse_indices:
            if n == pulse["n0"]:
                sx, sy = pulse["pos"]
                array_pulse = pulse_gaussien_module(nx, ny, x0 = sx, y0 = sy)

                u_n = u_n + array_pulse
                u_np1 = u_np1 + array_pulse


        ## Frontières
        u_np1[0, :] = 0  # bord gauche
        u_np1[-1, :] = 0  # bord droit
        u_np1[:, 0] = 0  # bord bas
        u_np1[:, -1] = 0  # bord haut


        # Enregistrement capteurs
        t_hist[n]  = n*dt
        p1_hist[n] = u_n[cx1, cy1]
        p2_hist[n] = u_n[cx2, cy2]
        p3_hist[n] = u_n[cx3, cy3]

        if n % pas_sauvegarde == 0:
            frames.append(u_n.copy())  # sauvegarde du champ pour l'animation

        u_nm1[:, :] = u_n  #  n devient n-1
        u_n[:, :] = u_np1  # n+1 devient n
        temps = temps = dt

    intensité_capteurs = np.array([p1_hist, p2_hist, p3_hist])
    # Calcul de l'énergie résiudelle pour tester l'efficacité du PML
    energie_initiale = np.sum(u0**2)  # Énergie initiale
    energie_finale = np.sum(u_n**2)  # Énergie finale

    # pourcentage_residuel = 100 * energie_finale / energie_initiale  # pourcentage d'énergie restante dans le domaine

    # print(f"Énergie résiduelle dans le domaine : {pourcentage_residuel:.6f} %")


    # Affichage des capteurs
    capteurs_x = [cx1 * h, cx2 * h, cx3 * h]
    capteurs_y = [cy1 * h, cy2 * h, cy3 * h]
    if show_animation == True :
        # Animation
        fig, ax = plt.subplots(figsize=(7, 6))
        img = ax.imshow(
            frames[0].T,
            origin="lower",
            cmap="seismic",
            extent=[0, L, 0, L],
            animated=True
        )

        # Capteurs avec couleurs différentes
        scatter1 = ax.scatter(capteurs_x[0], capteurs_y[0], c="red", marker="o", s=80, label="C1")
        scatter2 = ax.scatter(capteurs_x[1], capteurs_y[1], c="blue", marker="o", s=80, label="C2")
        scatter3 = ax.scatter(capteurs_x[2], capteurs_y[2], c="green", marker="o", s=80, label="C3")

        ax.legend()
        plt.colorbar(img, ax=ax, label="Amplitude")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        def maj(k):
            img.set_array(frames[k].T)
            
            return [img, scatter1, scatter2, scatter3]

        ani = FuncAnimation(fig, maj, frames=len(frames), interval=40, blit=True)
        plt.show()

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

def max_signaux(intensite_capteurs, t_hist):
    """
    intensite_capteurs : array [N_capteurs, N_samples]
    """

    temps_max = []
    amplitudes_max = []

    for signal in intensite_capteurs:
        idx = np.argmax(signal)
        temps_max.append(t_hist[idx])
        amplitudes_max.append(signal[idx])

    return np.array(temps_max), np.array(amplitudes_max)

def max_enveloppe_ordre(signal, t_hist, ordre, seuil_ratio=0.10, dt_min=t/8, seuil_montee=0.5):
    """
    ordre : numéro du pic (1,2,3...)
    seuil_ratio : seuil amplitude relatif
    dt_min : temps minimal entre deux pics (en secondes)
    """

    dt = t_hist[1] - t_hist[0]

    # 🔹 Enveloppe
    enveloppe = np.abs(hilbert(signal))

    # 🔹 Seuil amplitude
    seuil = seuil_ratio * np.max(enveloppe)

    # 🔹 Seuil temporel → converti en indices
    distance_min = int(dt_min / dt)

    # 🔹 Détection des pics 
    peaks, _ = find_peaks(
        enveloppe,
        height=seuil,
        distance=distance_min,
        prominence=seuil * 0.5
    )

    # 🔹 Sécurité
    if len(peaks) < ordre:
        return None, None, enveloppe
    
    idx_pic = peaks[ordre - 1]
    amp_pic = enveloppe[idx_pic]

    # --- Remontée vers le début du front montant ---
    seuil_debut = seuil_montee * amp_pic  # ex: 10% de l'amplitude du pic

    # On remonte depuis le pic jusqu'à ce que l'enveloppe tombe sous le seuil
    idx_debut = idx_pic
    while idx_debut > 0 and enveloppe[idx_debut] > seuil_debut:
        idx_debut -= 1

    t_debut = t_hist[idx_debut]
    amp_debut = enveloppe[idx_debut]

    return t_debut, amp_debut, enveloppe

def show_echo(intensité_echo, t_hist, Echo=False):
    if Echo:
        plt.figure(figsize=(8,5))

        couleurs = ["red", "blue", "green"]
        labels = ["Capteur 1", "Capteur 2", "Capteur 3"]

        for i, signal in enumerate(intensité_echo):

            ordre = i + 1  

            t_max, amp_max, enveloppe = max_enveloppe_ordre(signal, t_hist, ordre)

            # signal + enveloppe
            plt.plot(t_hist, signal, color=couleurs[i], alpha=0.4)
            plt.plot(t_hist, enveloppe, color=couleurs[i], linestyle="--", label=labels[i])

            if t_max is not None:
                plt.scatter(t_max, amp_max, color=couleurs[i], s=80)
                plt.axvline(t_max, color=couleurs[i], linestyle=":")

        plt.xlabel("Temps [s]")
        plt.ylabel("Amplitude")
        plt.title("Échos avec pics ordonnés par capteur")
        plt.legend()
        plt.grid()
        plt.show()

def multilateration(intensite_capteurs, t_hist, position_capteurs, c_eau):

    intensite_capteurs = np.array(intensite_capteurs)
    position_capteurs = np.array(position_capteurs) 

    temps_echo = []

    # -------- DETECTION DES TEMPS --------
    for i, signal in enumerate(intensite_capteurs):
        ordre = i + 1
        t_max, amp_max, enveloppe = max_enveloppe_ordre(signal, t_hist, ordre)

        if t_max is None:
            raise ValueError(f"Pas assez de pics pour capteur {i+1}")

        temps_echo.append(t_max)

    temps_echo = np.array(temps_echo)

    # -------- DISTANCES en noeuds --------
    distances = c_eau * (temps_echo - t0_vect) / 2 / h

    # -------- RÉSIDUS : différence entre distance réelle et rayon du cercle --------
    def residuals(x):
        return [
            np.linalg.norm(x - position_capteurs[i]) - distances[i]
            for i in range(len(distances))
        ]

    # -------- ESTIMATION INITIALE en mètres --------
    x0 = np.mean(position_capteurs, axis = 0)

    # -------- OPTIMISATION --------
    sol = least_squares(residuals, x0)
    position_estimee = sol.x  # en mètres

    return position_estimee, temps_echo

def plot_localisation_echo(position_capteurs, position_estimee, position_reelle, t0_vect, temps_echo, c_eau, taille_bateau, L=100, h=h):

    fig, ax = plt.subplots(figsize=(8, 8))
    n_capteurs = position_capteurs.shape[0]
    couleurs_capteurs = ["red", "blue", "green"]

    # --- Conversion maillage → mètres ---
    capteurs_m = position_capteurs * h          # shape [N, 2]
    estimee_m  = np.array(position_estimee) * h
    reelle_m   = np.array(position_reelle)  * h

    # --- Objet réel ---
    x_r, y_r = reelle_m
    bateau_reel = plt.Rectangle(
        (y_r - taille_bateau / 2, x_r - taille_bateau / 2),
        taille_bateau, taille_bateau,
        color="black", alpha=0.6, label="Position réelle", zorder=4
    )
    ax.add_patch(bateau_reel)

    # --- Position estimée ---
    x_e, y_e = estimee_m
    ax.scatter(y_e, x_e, marker="*", color="gold", s=250, zorder=5,
               edgecolors="black", linewidths=0.8, label="Position estimée (MC)")

    # --- Capteurs + cercles + lignes ---
    for i in range(n_capteurs):
        x_c, y_c = capteurs_m[i]
        couleur = couleurs_capteurs[i]

        ax.scatter(y_c, x_c, color=couleur, s=100, zorder=4, marker="^")
        ax.text(y_c + 0.5, x_c + 0.5, f"C{i+1}", color=couleur, fontsize=10, fontweight="bold")

        # Distance en mètres
        distance = c_eau * (temps_echo[i] - t0_vect[i]) / 2

        # Cercle de distance centré sur le capteur (en mètres)
        cercle = plt.Circle(
            (y_c, x_c), radius=distance,
            color=couleur, fill=False, linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"C{i+1} — d = {distance:.1f} m"
        )
        ax.add_patch(cercle)

        # Ligne capteur → estimée
        ax.plot([y_c, y_e], [x_c, x_e],
                color=couleur, linestyle="-", linewidth=1.2, alpha=0.6)

        # Annotation distance
        mid_x = (y_c + y_e) / 2
        mid_y = (x_c + x_e) / 2
        ax.text(mid_x, mid_y, f"{distance:.1f} m", color=couleur,
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    # --- Erreur ---
    erreur = np.linalg.norm(estimee_m - reelle_m)
    ax.set_title(f"Localisation par multilatération\nErreur : {erreur:.2f} m", fontsize=13)

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

def run_simulations(n_simulations, position_capteurs, distance_min_capteur, taille_bateau, nx, ny, epaisseur_pml, c_eau):
    """
    n_simulations        : nombre de simulations à effectuer
    position_capteurs    : array[N_capteurs, 2] en indices de maillage
    distance_min_capteur : distance minimale entre capteur et bateau (indices)
    taille_bateau        : demi-taille du bateau (indices)
    """

    erreurs_centre = []
    erreurs_bord   = []

    # Simulation sans bateau (faite une seule fois)
    intensité_sans_bateau, t_hist = simulation_sonar(position_capteurs, [5, 5], taille_bateau, show_animation=False)

    for i in range(n_simulations):

        print(f"\n--- Simulation {i+1}/{n_simulations} ---")

        # Position aléatoire du bateau
        position_bateau = generer_position_bateau(position_capteurs, nx, ny, epaisseur_pml, distance_min_capteur,taille_bateau)

        # Simulation avec bateau
        intensité_avec_bateau, t_hist = simulation_sonar(
            position_capteurs, position_bateau, taille_bateau, show_animation=False
        )

        intensité_echo = intensité_avec_bateau - intensité_sans_bateau
        show_echo(intensité_echo, t_hist)

        # Multilatération
        try:
            pos_estimee, temps_echo = multilateration(intensité_echo, t_hist, position_capteurs, c_eau)
        except ValueError as e:
            print(f"  ⚠ Simulation ignorée : {e}")
            continue

        # Positions en mètres
        reelle = np.array(position_bateau)
        estimee = np.array(pos_estimee)

        # Erreur centre → centre
        erreur_centre = np.linalg.norm(estimee - reelle)

        # Erreur bord → bord (on soustrait la taille du bateau en mètres)
        erreur_bord = max(0.0, erreur_centre - taille_bateau)

        erreurs_centre.append(erreur_centre)
        erreurs_bord.append(erreur_bord)

        print(f"  Position réelle    : {position_bateau}  ({reelle*h} )")
        print(f"  Position estimée   : {estimee} ({estimee*h} ) m")
        print(f"  Erreur centre      : {erreur_centre*h:.2f} m")
        print(f"  Erreur bord        : {erreur_bord*h:.2f} m")

    erreurs_centre = np.array(erreurs_centre) * h
    erreurs_bord   = np.array(erreurs_bord) * h

    print(f"\n{'='*45}")
    print(f"  Résultats sur {len(erreurs_centre)} simulations valides")
    print(f"{'='*45}")
    print(f"  Erreur moyenne centre : {np.mean(erreurs_centre):.2f} m  (±{np.std(erreurs_centre):.2f} m)")
    print(f"  Erreur moyenne bord   : {np.mean(erreurs_bord):.2f} m  (±{np.std(erreurs_bord):.2f} m) pour un bateau de {taille_bateau*h:.2f} m")
    print(f"{'='*45}")

    return erreurs_centre, erreurs_bord


# Paramètre du PML
epaisseur_pml = 75  # épaisseur 
gamma_max = 20000  # valeur maximale de gamma sur les bords finaux
gamma = PML(nx, ny, epaisseur_pml, puissance=3, gamma_max=gamma_max)  

# Graphique du PML
show_pml(False)

# Position et taille du bateau (pour le moment c'est juste un carré)
position_bateau = [95, 75] # ( x, y ) en position du noeud. Ne sert seulement si position_aléatoire = False
taille_bateau = 4 # En nombre de noeuds

# Si on veut générer une position aléatoire du bateau
distance_min_capteur = 25 # Distance minimale entre le capteur et le bateau (sinon ça detecte pas l'echo)
n_simulations        = 20 # Nombre de simulation (ne sert seuelement si position_aléatoire = False)

# Positions des capteurs (en position du noeud)
sensor1_pos = (75, 225)
sensor2_pos = (75, 75)
sensor3_pos = (225, 75)
position_capteurs = np.array([sensor1_pos, sensor2_pos, sensor3_pos])

position_aléatoire = True
if position_aléatoire == True:
    erreurs_centre, erreurs_bord = run_simulations(n_simulations, position_capteurs, distance_min_capteur, taille_bateau, nx, ny, epaisseur_pml, c_eau)


else : # Cette option sert si on veut juste faire une seule simulation
    
    # Sert à trouver l'intensité de l'echo (on pourrait optimiser quand on va faire pleins de simulations pour ne pas refaire celle sans bateau)
    intensité_avec_bateau, t_hist = simulation_sonar(position_capteurs, position_bateau, taille_bateau)
    intensité_sans_bateau, t_hist = simulation_sonar(position_capteurs, [5,5], taille_bateau) # Le bateau dans le pml comme si il était pas là
    intensité_echo = intensité_avec_bateau - intensité_sans_bateau

    # Graphique de l'echo
    show_echo(False)

    # Trouver la position de l'objet avec la multilatération
    pos, temps_echo = multilateration(intensité_echo, position_capteurs, c_eau)
    print("Position réelle", position_bateau, "et il a un rayon de ", taille_bateau)
    print("Position estimée :", pos)

    # Afficher le graphique de la position estimée
    plot_shema = True
    if plot_shema == True: 
        plot_localisation_echo(position_capteurs, pos, position_bateau, t0_vect, temps_echo, c_eau, taille_bateau)







