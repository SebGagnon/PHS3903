import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from scipy import signal
import matplotlib.patches as patches
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import math
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
from functools import partial

## Constantes ##
L = 400  # Longueur d'un côté du domaine [m]
t = 1.2  # temps total de simulation [s]
nx = 600  # nombre de points en x
ny = 600  # nombre de points en y
h = L / nx  # distance entre les points spatiaux [m]
positionbat = [[300,320], [410,430]]
Rho_eau = 1000  # densite volumique [kg/m3]
Kappa_eau = 2.2e9  # bulk modulus [Pa]
Gamma_eau = 0
vx_bat = 0 #m/s
vy_bat = 0 #m/s

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
sensor2_pos = (200, 75)
sensor3_pos = (75, 200)
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

def apply_threshold(s, threshold=0.005):
    s = s.copy()
    s[np.abs(s) < threshold ] = 0
    return s

def gaussian_smooth(s, sigma=84):
    return gaussian_filter1d(s, sigma=sigma)

def corr_delay(s1, s2, threshold=0.0003, 
                 smooth='gaussian', sigma = 84, width=200):
    s1 = apply_threshold(s1, threshold)
    s2 = apply_threshold(s2, threshold)
    
    if smooth == 'gaussian':
        s1 = gaussian_smooth(s1, sigma=sigma)
        s2 = gaussian_smooth(s2, sigma=sigma)

    lags = signal.correlation_lags(len(s1), len(s2), mode='full')
    r = lags[np.argmax(signal.correlate(s1, s2, mode='full'))]
    return r

def corr_delay_optimized(s1, s2, sigma=84, threshold=0.0003):
    """Optimized correlation delay for repeated calls"""
    s1 = apply_threshold(s1, threshold)
    s2 = apply_threshold(s2, threshold)
    
    if sigma > 0:
        s1 = gaussian_filter1d(s1, sigma=sigma)
        s2 = gaussian_filter1d(s2, sigma=sigma)
    
    # Direct correlation without lags calculation for speed
    correlation = signal.correlate(s1, s2, mode='full')
    delay_idx = np.argmax(np.abs(correlation)) - len(s1) + 1
    return delay_idx

def simulation_original(positionbat, clist, vx_bat=0, vy_bat=0, show_ani=False):
    """YOUR EXACT original simulation - no changes"""
    vx_bateau = vx_bat/h
    vy_bateau = vy_bat/h
    
    # buffers capteurs
    u0 = np.zeros((nx, ny)) 
    u_nm1 = u0.copy()  # champ au temps n-1
    u_n = u0.copy()  # champ au temps n
    u_np1 = np.zeros((nx, ny))  # champ au temps n+1

    steps = nt
    t_hist = np.zeros(steps)
    p1_hist = np.zeros(steps)
    p2_hist = np.zeros(steps)
    p3_hist = np.zeros(steps)

    pulse_indices = []

    for p in pulses:
        pulse_indices.append({
            "pos": p["pos"],
            "n0": int(p["t0"] / dt)
        })

    frames = [] if show_ani else None
    pas_sauvegarde = max(1, nt // 100)
    
    for n in range(nt):
        x_fact = math.floor(n*dt*vx_bateau)
        y_fact = math.floor(n*dt*vy_bateau)
        C_grid = c_eau * np.ones((nx-2, ny-2), dtype=float)
        C_grid[positionbat[0][0]+x_fact:positionbat[0][1]+x_fact, 
               positionbat[1][0]+y_fact:positionbat[1][1]+y_fact] = c_max

        lap = np.zeros_like(u_n)
        lap[1:-1, 1:-1] = (
            4 * (u_n[2:, 1:-1] + u_n[:-2, 1:-1] + u_n[1:-1, 2:] + u_n[1:-1, :-2])
            + (u_n[2:, 2:] + u_n[2:, :-2] + u_n[:-2, 2:] + u_n[:-2, :-2])
            - 20 * u_n[1:-1, 1:-1]
        ) / (6 * h**2)

        a = gamma * dt / 2

        u_np1[1:-1, 1:-1] = (
            2 * u_n[1:-1, 1:-1]
            - (1 - a[1:-1, 1:-1]) * u_nm1[1:-1, 1:-1]
            + C_grid**2 * dt**2 * lap[1:-1, 1:-1]
        ) / (1 + a[1:-1, 1:-1])

        # Détection des pulses
        for pulse in pulse_indices:
            if n == pulse["n0"]:
                sx, sy = pulse["pos"]
                array_pulse = pulse_gaussien_module(nx, ny, x0=sx, y0=sy)
                u_n = u_n + array_pulse
                u_np1 = u_np1 + array_pulse

        ## Frontières
        u_np1[0, :] = 0
        u_np1[-1, :] = 0
        u_np1[:, 0] = 0
        u_np1[:, -1] = 0

        # Enregistrement capteurs
        t_hist[n] = n*dt
        p1_hist[n] = u_n[cx1, cy1]
        p2_hist[n] = u_n[cx2, cy2]
        p3_hist[n] = u_n[cx3, cy3]
        
        if show_ani and n % pas_sauvegarde == 0:
            frames.append(u_n.copy())

        u_nm1[:, :] = u_n
        u_n[:, :] = u_np1

    plist = [[p1_hist[n0_1:n0_2], p1_hist[n0_2:n0_3], p1_hist[n0_3:]],
             [p2_hist[n0_1:n0_2], p2_hist[n0_2:n0_3], p2_hist[n0_3:]],
             [p3_hist[n0_1:n0_2], p3_hist[n0_2:n0_3], p3_hist[n0_3:]]]
    
    Filtered_data = [[plist[i][j] - clist[i][j] for j in range(3)] for i in range(3)]
    
    fp0 = np.concatenate((Filtered_data[0][0], Filtered_data[0][1], Filtered_data[0][2]))
    fp1 = np.concatenate((Filtered_data[1][0], Filtered_data[1][1], Filtered_data[1][2]))
    fp2 = np.concatenate((Filtered_data[2][0], Filtered_data[2][1], Filtered_data[2][2]))
    
    if show_ani:
        affichage_simul(frames)
    
    return Filtered_data, fp0, fp1, fp2, t_hist

def process_single_position(n, m, cx123, cy123, C123, clist):
    """Process one position pair - for parallel execution"""
    print(f"Processing position ({n},{m})")
    
    positionbat = [[130 + n*15, 140 + n*15], [130 + m*15, 140 + m*15]]
    
    # Call your original simulation
    Filtered_data, fp0, fp1, fp2, t_hist = simulation_original(
        positionbat=positionbat, clist=clist, show_ani=False
    )
    
    # Calculate distances - DISTANCE MINIMALE au rectangle (point le plus proche)
    # Pour le capteur 0 (cx123[0], cy123[0])
    x_capteur = cx123[0]
    y_capteur = cy123[0]
    
    # Distance en x : 0 si le capteur est à l'intérieur du rectangle en x
    xmin, xmax = positionbat[0][0], positionbat[0][1]
    ymin, ymax = positionbat[1][0], positionbat[1][1]
    
    if x_capteur < xmin:
        dx = xmin - x_capteur
    elif x_capteur > xmax:
        dx = x_capteur - xmax
    else:
        dx = 0
    
    if y_capteur < ymin:
        dy = ymin - y_capteur
    elif y_capteur > ymax:
        dy = y_capteur - ymax
    else:
        dy = 0
    
    # Conversion en mètres
    rx = dx * (L / nx)
    ry = dy * (L / nx)
    rth = np.sqrt(rx**2 + ry**2)
    
    # ... reste du code inchangé ...
    # Golden section search optimized
    phi = (1 + np.sqrt(5)) / 2
    a, b = 1.0, 600.0
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    
    perror = float('inf')
    best_sigma = c
    
    # Pre-extract signals for faster access
    C0 = np.abs(C123[0])
    fd0 = np.abs(Filtered_data[0][0])
    const = c_eau * dt / 2
    
    # Cache for computed errors
    error_cache = {}
    
    def get_error(sigma):
        if sigma not in error_cache:
            delay = corr_delay_optimized(C0, fd0, sigma=sigma)
            rayon = np.abs(const * delay)
            error_cache[sigma] = 100 * np.abs(rayon - rth) / (rth + 1e-10)
        return error_cache[sigma]
    
    for iteration in range(60):
        error_c = get_error(c)
        error_d = get_error(d)
        
        if error_c < error_d:
            b = d
            best_sigma = c
            if error_c < perror:
                perror = error_c
        else:
            a = c
            best_sigma = d
            if error_d < perror:
                perror = error_d
        
        c = b - (b - a) / phi
        d = a + (b - a) / phi
    
    print(f"best_sigma={best_sigma:.2f}  error={perror:.2f}%  rth={rth:.2f}m")
    
    return {
        'n': n, 'm': m,
        'rx': rx, 'ry': ry,
        'rth': rth,
        'sigma': best_sigma,
        'error': perror
    }

def optimisation_filtre_optimized(cx123, cy123, C123, clist, y=5, n_jobs=6):
    """Optimized version of optimisation_filtre with parallel processing"""
    print(f"Starting optimization with {y}x{y} positions using {n_jobs} cores")
    print("Estimated time: 1-2 hours")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_position, 
                          cx123=cx123, cy123=cy123, 
                          C123=C123, clist=clist)
    
    # Generate all (n,m) pairs
    n_values = range(y)
    m_values = range(y)
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_func)(n, m) for n in n_values for m in m_values
    )
    
    # Extract results
    lrx = [r['rx'] for r in results]
    lry = [r['ry'] for r in results]
    lrayon = [r['rth'] for r in results]
    sigs = [r['sigma'] for r in results]
    errors = [r['error'] for r in results]
    
    # Plot 2D results (original)
    plt.figure(figsize=(8, 5))
    plt.plot(lrayon, sigs, marker='o', markersize=3)
    plt.xlabel("Distance [m]")
    plt.ylabel("Meilleur sigma")
    plt.title("Sigma optimal vs distance")
    plt.grid()
    plt.show()
    
    print("Optimization complete!")
    return sigs, lrayon, lrx, lry, errors

def pulse_canvas():    
    u0 = np.zeros((nx, ny)) 
    u_nm1 = u0.copy()
    u_n = u0.copy()
    u_np1 = np.zeros((nx, ny))
    
    C_grid_canvas = c_eau * np.ones((nx-2, ny-2), dtype=float)
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
                u_n_c = u_n_c + array_pulse
                u_np1_c = u_np1_c + array_pulse
        
        u_np1_c[0, :] = 0
        u_np1_c[-1, :] = 0
        u_np1_c[:, 0] = 0
        u_np1_c[:, -1] = 0
        
        c11_hist[n] = u_n_c[cx1, cy1]
        c22_hist[n] = u_n_c[cx2, cy2]
        c33_hist[n] = u_n_c[cx3, cy3]
        
        u_nm1_c[:, :] = u_n_c
        u_n_c[:, :] = u_np1_c
    
    c11, c12, c13 = c11_hist[n0_1:n0_2], c11_hist[n0_2:n0_3], c11_hist[n0_3:]
    c21, c22, c23 = c22_hist[n0_1:n0_2], c22_hist[n0_2:n0_3], c22_hist[n0_3:]
    c31, c32, c33 = c33_hist[n0_1:n0_2], c33_hist[n0_2:n0_3], c33_hist[n0_3:]
    clist = [[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]]
    C123 = [c11, c22, c33]
    return C123, clist

def affichage_simul(frames):
    """Original animation function - kept as is"""
    capteurs_x = [cx1 * h, cx2 * h, cx3 * h]
    capteurs_y = [cy1 * h, cy2 * h, cy3 * h]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(
        frames[0].T,
        origin="lower",
        cmap="seismic",
        extent=[0, L, 0, L],
        animated=True
    )
    
    scatter1 = ax.scatter(capteurs_x[0], capteurs_y[0], c="red", marker="o", s=80, label="C1")
    scatter2 = ax.scatter(capteurs_x[1], capteurs_y[1], c="blue", marker="o", s=80, label="C2")
    scatter3 = ax.scatter(capteurs_x[2], capteurs_y[2], c="green", marker="o", s=80, label="C3")
    
    rect = patches.Rectangle(
        (positionbat[0][0]*h, positionbat[1][0]*h),
        width=(positionbat[0][1]-positionbat[0][0])*h,
        height=(positionbat[1][1]-positionbat[1][0])*h,
        linewidth=1,
        edgecolor="brown",
        facecolor="none",
        animated=True
    )
    ax.add_patch(rect)
    ax.legend()
    plt.colorbar(img, ax=ax, label="Amplitude")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    def maj(k):
        img.set_array(frames[k].T)
        return [img, scatter1, scatter2, scatter3, rect]
    
    ani = FuncAnimation(fig, maj, frames=len(frames), interval=20, blit=True)
    plt.show()

def plot_3d_results(lrx, lry, sigs, errors=None):
    """Create 3D plot of optimization results - works for any grid size"""
    
    n_points = len(lrx)
    grid_size = int(np.sqrt(n_points))
    is_square_grid = (grid_size * grid_size == n_points)
    
    if is_square_grid:
        # C'est une grille carrée parfaite
        X = np.array(lrx).reshape(grid_size, grid_size)
        Y = np.array(lry).reshape(grid_size, grid_size)
        Z = np.array(sigs).reshape(grid_size, grid_size)
        if errors is not None:
            E = np.array(errors).reshape(grid_size, grid_size)
    else:
        # Pas une grille parfaite, on fait du scatter 3D
        X = np.array(lrx)
        Y = np.array(lry)
        Z = np.array(sigs)
        E = np.array(errors) if errors is not None else None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Surface or scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    if is_square_grid:
        surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.scatter(X, Y, Z, c='red', s=10, alpha=0.5)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=20, label='Sigma')
    else:
        scatter1 = ax1.scatter(X, Y, Z, c=Z, cmap='viridis', s=30, alpha=0.7)
        fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, label='Sigma')
    ax1.set_xlabel('lrx (X position)')
    ax1.set_ylabel('lry (Y position)')
    ax1.set_zlabel('Sigma')
    ax1.set_title('Sigma optimal values')
    
    # Plot 2: Scatter plot colored by sigma
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(lrx, lry, sigs, c=sigs, cmap='plasma', s=30, alpha=0.7)
    ax2.set_xlabel('lrx')
    ax2.set_ylabel('lry')
    ax2.set_zlabel('Sigma')
    ax2.set_title('Sigma values (scatter)')
    fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, label='Sigma')
    
    # Plot 3: Error plot
    if errors is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        if is_square_grid and errors is not None:
            surf3 = ax3.plot_surface(X, Y, E, cmap='hot', alpha=0.8, edgecolor='none')
            fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=20, label='Error (%)')
        else:
            scatter3 = ax3.scatter(lrx, lry, errors, c=errors, cmap='hot', s=30)
            fig.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20, label='Error (%)')
        ax3.set_xlabel('lrx')
        ax3.set_ylabel('lry')
        ax3.set_zlabel('Error (%)')
        ax3.set_title('Optimization error')
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("="*50)
    print("Starting optimized optimization_filtre")
    print("="*50)
    
    # Generate reference signals
    print("\nGenerating reference signals...")
    C123, clist = pulse_canvas()
    
    # Run optimized optimization
    print("\nRunning optimization...")
    sigs, lrayon, lrx, lry, errors = optimisation_filtre_optimized(
        cx123, cy123, C123, clist, 
        y=20,        # Grid size (20x20 = 400 positions)
        n_jobs=6     # Number of parallel processes (adjust based on your CPU)
    )
    
    # Create 3D plots of results
    print("\nCreating 3D visualization...")
    plot_3d_results(lrx, lry, sigs, errors)
    
    # Optional: Show PML
    show_pml(False)
    
    print("\nDone!")
    print(f"Processed {len(sigs)} positions")
    print(f"Sigma range: {min(sigs):.2f} - {max(sigs):.2f}")
    print(f"Error range: {min(errors):.2f}% - {max(errors):.2f}%")
