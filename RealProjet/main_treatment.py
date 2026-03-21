import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from scipy import signal
import matplotlib.patches as patches

def load_signals_from_csv(filename):
    t_hist = []
    p1_hist = []
    p2_hist = []
    p3_hist = []
    sensor_positions = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        
        section = None
        for row in reader:
            if not row:  # blank row = section separator
                continue
            
            if row[0] == "time":  # signals header
                section = "signals"
                continue
            
            if row[0] == "sensor":  # positions header
                section = "positions"
                continue
            
            if section == "signals":
                t_hist.append(float(row[0]))
                p1_hist.append(float(row[1]))
                p2_hist.append(float(row[2]))
                p3_hist.append(float(row[3]))
            
            elif section == "positions":
                sensor_positions.append((float(row[1]), float(row[2])))

    # Convert to numpy arrays
    t_hist  = np.array(t_hist)
    p1_hist = np.array(p1_hist)
    p2_hist = np.array(p2_hist)
    p3_hist = np.array(p3_hist)

    #sensor1_pos = sensor_positions[0]
    #sensor2_pos = sensor_positions[1]
    #sensor3_pos = sensor_positions[2]

    return t_hist, p1_hist, p2_hist, p3_hist#, sensor1_pos, sensor2_pos, sensor3_pos


# --- Load ---
t_hist, p1_hist, p2_hist, p3_hist = \
    load_signals_from_csv("canvas.csv")
c11, c12, c13 = p1_hist[:2753], p1_hist[2754:5507], p1_hist[5508:-1]
c21, c22, c23 = p2_hist[:2753], p2_hist[2754:5507], p2_hist[5508:-1]
c31, c32, c33 = p3_hist[:2753], p3_hist[2754:5507], p3_hist[5508:-1]


# --- Verify ---
# Le maillage de être au moins plus petit que la longueur d'onde sur 2

## Constantes ##
L = 100  # Longueur d'un côté du domaine [m]
t = 0.3  # temps total de simulation [s]
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
C_grid =  c_eau * np.ones((nx-2, ny-2), dtype=float); C_grid[190:210, 190:210] = c_max

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
sensor1_pos = (225, 150)
sensor2_pos = (150, 75)
sensor3_pos = (225, 75)

pulses = [
    {"pos": sensor1_pos, "t0": 0},
    {"pos": sensor2_pos, "t0": 0.1},
    {"pos": sensor3_pos, "t0": 0.2},
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
u0 = np.zeros((nx, ny)) 
u_nm1 = u0.copy()  # champ au temps n-1
u_n = u0.copy()  # champ au temps n
u_np1 = np.zeros((nx, ny))  # champ au temps n+1


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


frames = []  # liste des images conservées pour l'animation
pas_sauvegarde = max(1, nt // 500)  # intervalle entre deux sauvegardes d'image

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


# Calcul de l'énergie résiudelle pour tester l'efficacité du PML
energie_initiale = np.sum(u0**2)  # Énergie initiale
energie_finale = np.sum(u_n**2)  # Énergie finale

pourcentage_residuel = 100 * energie_finale / energie_initiale  # pourcentage d'énergie restante dans le domaine

print(f"Énergie résiduelle dans le domaine : {pourcentage_residuel:.6f} %")


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
    (190*h, 190*h),        # (x, y) bottom-left corner
    width=20*h,          # 210 - 190
    height=20*h,         # 210 - 190
    linewidth=2,
    edgecolor="red",
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

p11, p12, p13 = p1_hist[:2753], p1_hist[2754:5507], p1_hist[5508:-1]
p21, p22, p23 = p2_hist[:2753], p2_hist[2754:5507], p2_hist[5508:-1]
p31, p32, p33 = p3_hist[:2753], p3_hist[2754:5507], p3_hist[5508:-1]
c333 = [c13,c23,c33] 
Filtered_data = [[p11-c11, p12-c12, p13-c13], 
                 [p21-c21, p22-c22, p23-c23], 
                 [p31-c31, p32-c32, p33-c33]]
fp = []
j = 0
for n in Filtered_data:
    fp.append([])
    for i in n:
        for v in i:
            fp[j].append(v)
    j += 1
        
def graph_capteur(show):
    if show == True:
        plt.figure(figsize=(8,5))
        plt.plot(t_hist[:-3], fp[0], label="Capteur 1")
        plt.plot(t_hist[:-3], fp[1], label="Capteur 2")
        plt.plot(t_hist[:-3], fp[2], label="Capteur 3")
        plt.xlabel("Temps [s]")
        plt.ylabel("Amplitude")
        plt.title("Signaux enregistrés par les capteurs")
        plt.legend()
        plt.grid()
        plt.show()

show_PML = False
show_pml(show_PML)
show_capteurs = True
graph_capteur(show_capteurs)


for val in range(3):
    # Time lag in x: sensor 3 vs sensor 2
    corr = signal.correlate(Filtered_data[2][val], Filtered_data[1][val], mode='full')
    lags = signal.correlation_lags(len(Filtered_data[2][val]), len(Filtered_data[1][val]), mode='full')
    time_lags = lags * dt
    time_x = time_lags[np.argmax(corr)]
    print("Best lag (samples):", lags[np.argmax(corr)])
    print("Best lag (seconds):", time_x)

    # Time lag in y: sensor 3 vs sensor 1
    corr = signal.correlate(Filtered_data[2][val], Filtered_data[0][val], mode='full')
    lags = signal.correlation_lags(len(Filtered_data[2][val]), len(Filtered_data[0][val]), mode='full')
    time_lags = lags * dt
    time_y = time_lags[np.argmax(corr)]
    print("Best lag (samples):", lags[np.argmax(corr)])
    print("Best lag (seconds):", time_y)

    # Total travel time: filtered signal vs canvas reference
    corr = signal.correlate(Filtered_data[2][val], c333[val], mode='full')
    lags = signal.correlation_lags(len(Filtered_data[2][val]), len(c333[val]), mode='full')
    time_lags = lags * dt
    total_time = time_lags[np.argmax(corr)]
    ray = c_eau * total_time / 2

    x_cord = 75 + ray * time_x / (np.sqrt(time_x**2 + time_y**2))
    y_cord = ray * time_y / (np.sqrt(time_x**2 + time_y**2)) + 25 
    print(f"Position du bateau {x_cord, y_cord}")
