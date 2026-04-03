import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from scipy import signal
import matplotlib.patches as patches
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import math

## Constantes ##
L = 400  # Longueur d'un côté du domaine [m]
t = 1.2  # temps total de simulation [s]
nx = 600  # nombre de points en x
ny = 600  # nombre de points en y
h = L / nx  # distance entre les points spatiaux [m]
positionbat = [[260,310], [290,310]]
Rho_eau = 1000  # densite volumique [kg/m3]
Kappa_eau = 2.2e9  # bulk modulus [Pa]
Gamma_eau = 0
vx_bat = 11 #m/s
vy_bat = 5 #m/s

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



print(f"vitesse de l'onde = {c_max:.2f} m/s")
print(f"distance entre les points = {h:.4f} m")
print(f"durée totale de la simul = {dt:.2e} s")
print(f"nt = {nt}")

# Positions des capteurs
sensor1_pos = (75, 75)
sensor2_pos = (100, 75)
sensor3_pos = (75, 100)
t1 = 0
t2 = 0.4
t3 = 0.8

# Dynamique selon dt
n0_1 = int(t1 / dt)   # pulse 1 at t0=0
n0_2 = int(t2 / dt)   # pulse 2 at t0=0.1
n0_3 = int(t3 / dt)   # pulse 3 at t0=0.2
pulse_len = n0_2 - n0_1  # samples per pulse window
nlist = [n0_1, n0_2, n0_3]
print(nlist)

#
pulses = [
    {"pos": sensor1_pos, "t0": t1},
    {"pos": sensor2_pos, "t0": t2},
    {"pos": sensor3_pos, "t0": t3},
]

cx1, cy1 = sensor1_pos
cx2, cy2 = sensor2_pos
cx3, cy3 = sensor3_pos
cx123 = [cx1,cx2,cx3]
cy123 = [cy1,cy2,cy3]

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

def pulse_gaussien_module(nx, ny, x0=None, y0=None, sigma= 10 , w = 0.4, A0= 1 ):
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')  
    r = np.sqrt((x - x0)**2 + (y - y0)**2)  # distance par rapport au point source

    pulse = A0 * np.exp(-(r**2) / (2 * sigma**2)) * np.cos(w * r)  # modulation en fonction de r
    return pulse


#Conditions initiales
#u0 = pulse_gaussien_module(nx, ny, x0 = 225, y0 = 150)  #pulse


#PML
epaisseur_pml = 75  # épaisseur 
gamma_max = 20000  # valeur maximale de gamma sur les bords finaux
gamma = PML(nx, ny, epaisseur_pml, puissance=3, gamma_max=gamma_max)  

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


def apply_threshold(s, threshold=0.0006):
    s = s.copy()
    s[np.abs(s) < threshold ] = 0
    return s


def gaussian_smooth(s, sigma=84):
    return gaussian_filter1d(s, sigma=sigma)

def square_wave_filter(s, width=200):
    kernel = np.ones(width)
    return np.convolve(s, kernel, mode='same')


def corr_delay(s1, s2, threshold=0.0003, 
                 smooth='gaussian', sigma = 84, width=200):
    s1 = apply_threshold(s1, threshold)
    s2 = apply_threshold(s2, threshold)
    
    if smooth == 'gaussian':
        s1 = gaussian_smooth(s1, sigma=sigma)
        s2 = gaussian_smooth(s2, sigma=sigma)
    elif smooth == 'square':
        s1 = square_wave_filter(s1, width=width)
        s2 = square_wave_filter(s2, width=width)

    lags = signal.correlation_lags(len(s1), len(s2), mode='full')
    r = lags[np.argmax(signal.correlate(s1, s2, mode='full'))]
    return r

def calc_sigma_test(Filtered_data, cx123, cy123, C123):
    error=0
    gerror=0    
    for sigma in [30,60,90,120,150]:
        gerror = 0
        for i in range(3):
            rth = np.sqrt(
                min(((cx123[i] - np.array(positionbat[0])) * L / nx)**2) +
                min(((cy123[i] - np.array(positionbat[1])) * L / nx)**2))

            grayon = np.abs(c_eau * dt / 2 * corr_delay(
                np.abs(C123[i]), np.abs(Filtered_data[i][i]),
                smooth='gaussian', sigma=sigma))
            #if 100 * np.abs(grayon - rth) / (3 * rth) < 50:
            gerror += 100 * np.abs(grayon - rth) / (3 * rth)
        print(f"{grayon:2f}")    
        print(f"sigma = {sigma} : {gerror:.2f}%") 
    #return 84

def calc_dist(Filtered_data, cx123, cy123, C123):
    error=0
    gerror=0
    for i in range(3):
        grayon = np.abs(c_eau * dt / 2 * corr_delay(
                np.abs(C123[i]), np.abs(Filtered_data[i][i]),
                smooth='gaussian', sigma=84))
        grth = np.sqrt(
                min(((cx123[i] - np.array(positionbat[0])) * L / nx)**2) +
                min(((cy123[i] - np.array(positionbat[1])) * L / nx)**2))

        rayon = np.abs(c_eau * dt / 2 * corr_delay(
                np.abs(C123[i]), np.abs(Filtered_data[i][i]),
                smooth='rien'))
        rth = np.sqrt(
                min(((cx123[i] - np.array(positionbat[0])) * L / nx)**2) +
                min(((cy123[i] - np.array(positionbat[1])) * L / nx)**2))
        gerror += 100 * np.abs(grayon - grth) / (3 * rth)
        error += 100 * np.abs(rayon - rth) / (3 * rth)

        print(grayon)
        print(f'{grth}\n')
        print(rayon)
        print(f'{rth}\n')
    print(f"{gerror}\n")
    print(f"{error}\n")
    #return 84

def optimisation_filtre(cx123, cy123, C123):
    lrayon = []
    sigs = []
    phi = (1 + np.sqrt(5)) / 2

    for n in range(20):
        print(f"{100*n/6:.2f}%")
        positionbat = [[100+n*5, 110+n*5], [100, 110]]
        Filtered_data, fp0, fp1, fp2 = simulation(positionbat=positionbat)

        def calc_error(sigma):
            error = 0
            ath = 0
            for i in range(3):
                rayon = np.abs(c_eau * dt / 2 * corr_delay(
                    np.abs(C123[i]), np.abs(Filtered_data[i][i]),
                    smooth='gaussian', sigma=sigma))
                rth = np.sqrt(
                    min(((cx123[i] - np.array(positionbat[0])) * L / nx)**2) +
                    min(((cy123[i] - np.array(positionbat[1])) * L / nx)**2))
                error += 100 * np.abs(rayon - rth) / (3 * rth)
                ath += rth / 3
            return error, ath  # scalar error, average rth

        # Reset interval for each boat position
        a, b = 1, 600
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        perror = float('inf')
        best_sigma = c

        for iteration in range(80):
            e_c, _ = calc_error(c)  # unpack tuple, ignore ath
            e_d, _ = calc_error(d)

            if e_c < e_d:           # now comparing scalars ✓
                b = d
                best_sigma = c
                perror = min(perror, e_c)
            else:
                a = c
                best_sigma = d
                perror = min(perror, e_d)

            c = b - (b - a) / phi
            d = a + (b - a) / phi

        _, ath = calc_error(best_sigma)  # get ath at best sigma
        print(f"best_sigma={best_sigma:.2f}  error={perror:.2f}%  rth={ath:.2f}m")
        lrayon.append(ath)
        sigs.append(best_sigma)

    plt.figure(figsize=(8, 5))
    plt.plot(lrayon, sigs, marker='o', markersize=3)
    plt.xlabel("Distance [m]")
    plt.ylabel("Meilleur sigma")
    plt.title("Sigma optimal vs distance")
    plt.grid()
    plt.show()
    print(lrayon)
    print(sigs)
    return sigs



frames = []  # liste des images conservées pour l'animation
pas_sauvegarde = max(1, nt // 100)  # intervalle entre deux sauvegardes d'image

def pulse_canvas():    
    u0 = np.zeros((nx, ny)) 
    u_nm1 = u0.copy()  # champ au temps n-1
    u_n = u0.copy()  # champ au temps n
    u_np1 = np.zeros((nx, ny))  # champ au temps n+1

    C_grid_canvas = c_eau * np.ones((nx-2, ny-2), dtype=float)  # pas d'inclusion
    pulse_indices = []

    for p in pulses:
        pulse_indices.append({
            "pos": p["pos"],
            "n0": int(p["t0"] / dt)
        })


    steps = nt
    u0_c = np.zeros((nx, ny))
    u_nm1_c = u0_c.copy()
    u_n_c = u0_c.copy()
    u_np1_c = np.zeros((nx, ny))

    c11_hist = np.zeros(steps)
    c22_hist = np.zeros(steps)
    c33_hist = np.zeros(steps)
    
    for n in range(nt):
        lap_c = np.zeros_like(u_n_c)
        lap_c[1:-1, 1:-1] = (
        4 * (
            u_n_c[2:, 1:-1] + u_n_c[:-2, 1:-1]
            + u_n_c[1:-1, 2:] + u_n_c[1:-1, :-2]
        )
        + (
            u_n_c[2:, 2:] + u_n_c[2:, :-2]
            + u_n_c[:-2, 2:] + u_n_c[:-2, :-2]
        )
        - 20 * u_n_c[1:-1, 1:-1]
        ) / (6 * h**2)
 

        a_c = gamma * dt / 2

        u_np1_c[1:-1, 1:-1] = (
            2 * u_n_c[1:-1, 1:-1]
            - (1 - a_c[1:-1, 1:-1]) * u_nm1_c[1:-1, 1:-1]
            + C_grid_canvas**2 * dt**2 * lap_c[1:-1, 1:-1]
        ) / (1 + a_c[1:-1, 1:-1])

        for pulse in pulse_indices:
            if n == pulse["n0"]:
                sx, sy = pulse["pos"]
                array_pulse = pulse_gaussien_module(nx, ny, x0=sx, y0=sy)
                u_n_c   = u_n_c   + array_pulse
                u_np1_c = u_np1_c + array_pulse
    
        u_np1_c[0, :] = 0
        u_np1_c[-1, :] = 0
        u_np1_c[:, 0] = 0
        u_np1_c[:, -1] = 0
    
        c11_hist[n] = u_n_c[cx1, cy1]
        c22_hist[n] = u_n_c[cx2, cy2]
        c33_hist[n] = u_n_c[cx3, cy3]

        u_nm1_c[:, :] = u_n_c
        u_n_c[:, :]   = u_np1_c
 
    c11, c12, c13 = c11_hist[n0_1:n0_2], c11_hist[n0_2:n0_3], c11_hist[n0_3:]
    c21, c22, c23 = c22_hist[n0_1:n0_2], c22_hist[n0_2:n0_3], c22_hist[n0_3:]
    c31, c32, c33 = c33_hist[n0_1:n0_2], c33_hist[n0_2:n0_3], c33_hist[n0_3:]
    clist = [[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]]
    C123 = [c11, c22, c33]
    return C123, clist 
C123, clist = pulse_canvas()


def simulation(positionbat = positionbat, clist = clist, vx_bat = 0, vy_bat = 0, show_ani = False):
    vx_bateau = vx_bat/h
    vy_bateau = vy_bat/h
    
    # buffers capteurs
    u0 = np.zeros((nx, ny)) 
    u_nm1 = u0.copy()  # champ au temps n-1
    u_n = u0.copy()  # champ au temps n
    u_np1 = np.zeros((nx, ny))  # champ au temps n+1


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
        x_fact = math.floor(n*dt*vx_bateau)
        y_fact = math.floor(n*dt*vy_bateau)
        C_grid =  c_eau * np.ones((nx-2, ny-2), dtype=float); C_grid[positionbat[0][0]+x_fact:positionbat[0][1]+x_fact, positionbat[1][0]+y_fact:positionbat[1][1]+y_fact] = c_max

        lap = np.zeros_like(u_n)  # initialisation du laplacien
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
        ) / (6 * h**2)   # calcul du laplacien par différences centrées

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
        if show_ani == True:
            if n % pas_sauvegarde == 0:
                frames.append(u_n.copy())  # sauvegarde du champ pour l'animation

        u_nm1[:, :] = u_n  #  n devient n-1
        u_n[:, :] = u_np1  # n+1 devient n
        temps = temps = dt

    plist = [[p1_hist[n0_1:n0_2], p1_hist[n0_2:n0_3], p1_hist[n0_3:]],
            [p2_hist[n0_1:n0_2], p2_hist[n0_2:n0_3], p2_hist[n0_3:]],
            [p3_hist[n0_1:n0_2], p3_hist[n0_2:n0_3], p3_hist[n0_3:]]]
    Filtered_data = [[plist[i][j] - clist[i][j] for j in range(3)] for i in range(3)]
    
    fp0 = np.concatenate((Filtered_data[0][0],Filtered_data[0][1],Filtered_data[0][2]))
    fp1 = np.concatenate((Filtered_data[1][0],Filtered_data[1][1],Filtered_data[1][2]))
    fp2 = np.concatenate((Filtered_data[2][0],Filtered_data[2][1],Filtered_data[2][2]))
    
    if show_ani == True:
        affichage_simul(frames)
    return Filtered_data, fp0, fp1, fp2





def affichage_simul(frames):
# Affichage des capteurs
    capteurs_x = [cx1 * h, cx2 * h, cx3 * h]
    capteurs_y = [cy1 * h, cy2 * h, cy3 * h]

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

# Red box around dx=190:210, dy=190:210
    rect = patches.Rectangle(
        (positionbat[0][0]*h, positionbat[1][0]*h),        # (x, y) bottom-left corner
        width=(positionbat[0][1]-positionbat[0][0])*h,          # 210 - 190
        height=(positionbat[1][1]-positionbat[1][0])*h,         # 210 - 190
        linewidth=1,
        edgecolor="brown",
        facecolor="none",
        animated=True# transparent fill
    )
    ax.add_patch(rect)
    ax.legend()
    plt.colorbar(img, ax=ax, label="Amplitude")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    def maj(k):
        img.set_array(frames[k].T)
        return [img, scatter1, scatter2, scatter3, rect]  # add rect here
    print(f"Axis limits: x={ax.get_xlim()},  y={ax.get_ylim()}")
    ani = FuncAnimation(fig, maj, frames=len(frames), interval=20, blit=True)
    plt.show()

    

def graph_capteur(show):
    if show == True:
        
        plt.figure(figsize=(8,5))
        plt.plot(t_hist[:], apply_threshold(np.abs(fp0)), label="Capteur 1")
        plt.plot(t_hist[:], apply_threshold(np.abs(fp1)), label="Capteur 2")
        plt.plot(t_hist[:], apply_threshold(np.abs(fp2)), label="Capteur 3")

        plt.xlabel("Temps [s]")
        plt.ylabel("Amplitude")
        plt.title("Signaux enregistrés par les capteurs")
        plt.legend()
        plt.grid()
        plt.show()

simulation(positionbat, vx_bat = vx_bat, vy_bat = vy_bat, clist = clist, show_ani = True )
calc_sigma_test(Filtered_data, cx123, cy123, C123)
#optimisation_filtre(cx123, cy123, C123)
show_PML = False
show_pml(show_PML)
show_capteurs = False
graph_capteur(show_capteurs)



def graph_sigmae():
    sigmas = [55.38, 55.75, 55.52, 52.35, 56.58, 60.44, 55.73, 54.10, 50.23, 46.84,
          39.88, 18.96, 20.97, 22.06, 12.77, 18.46, 12.50, 21.50, 14.23, 24.49,
          25.87, 26.69, 27.96, 31.09, 32.84, 35.08, 40.63, 44.95, 50.38, 56.25,
          59.38, 64.34, 68.55, 73.08, 77.73, 82.31, 87.10, 88.13, 89.46, 94.31,
          93.91, 96.75, 98.18, 98.50, 101.98, 106.61, 109.17, 115.06, 116.99, 121.11,
          126.49, 128.24, 130.81, 136.33, 137.62, 140.12, 145.50, 147.83, 150.10, 152.71,
          157.79, 160.44, 165.08, 166.83, 170.35, 173.21, 176.30, 178.52, 180.81]

    errors = [13.05, 10.89, 9.35, 8.06, 6.90, 5.86, 5.19, 4.79, 4.31, 3.82,
          3.36, 2.56, 2.18, 1.78, 0.53, 0.43, 0.43, 0.35, 0.35, 0.23,
          0.16, 0.10, 0.06, 0.06, 0.08, 0.11, 0.13, 0.12, 0.13, 0.15,
          0.15, 0.14, 0.15, 0.14, 0.15, 0.14, 0.15, 0.13, 0.13, 0.13,
          0.13, 0.12, 0.11, 0.13, 0.14, 0.16, 0.20, 0.21, 0.24, 0.25,
          0.28, 0.30, 0.32, 0.33, 0.35, 0.36, 0.38, 0.39, 0.39, 0.40,
          0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.32, 0.30]

    rths = [9.48, 10.51, 11.66, 12.92, 14.28, 15.70, 17.16, 18.67, 20.20, 21.76,
        23.33, 24.91, 26.51, 28.12, 29.73, 31.35, 32.97, 34.60, 36.23, 37.87,
        39.51, 41.15, 42.79, 44.43, 46.08, 47.73, 49.38, 51.03, 52.68, 54.33,
        55.98, 57.64, 59.29, 60.95, 62.60, 64.26, 65.92, 67.57, 69.23, 70.89,
        72.55, 74.21, 75.87, 77.53, 79.19, 80.85, 82.51, 84.17, 85.83, 87.49,
        89.15, 90.82, 92.48, 94.14, 95.80, 97.46, 99.13, 100.79, 102.45, 104.11,
        105.78, 107.44, 109.10, 110.77, 112.43, 114.09, 115.76, 117.42, 119.09]






    plt.figure(figsize=(8, 5))
    plt.plot(rths, errors, marker='o', markersize=3)
    plt.xlabel("Distance [m]")
    plt.ylabel("Erreur")
    plt.title("Sigma optimal vs distance")
    plt.grid()
    plt.show()

#graph_sigmae()


