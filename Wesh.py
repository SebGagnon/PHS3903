import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# FDTD 2D : u_tt + 2*gamma*u_t = c^2 * Laplacian(u) + source
# + couche éponge + obstacle interne rigide
# + Soustraction du signal sans obstacle pour isoler les réflexions
# ------------------------------------------------------------

def build_sponge_gamma(nx, ny, gamma0, sponge_width, sponge_strength, power=4):
    gamma = gamma0 * np.ones((nx, ny), dtype=np.float64)
    if sponge_width <= 0:
        return gamma

    ix = np.minimum(np.arange(nx), np.arange(nx)[::-1])
    iy = np.minimum(np.arange(ny), np.arange(ny)[::-1])
    dist_to_edge = np.minimum(ix[:, None], iy[None, :])  # (nx, ny)

    mask = dist_to_edge < sponge_width
    t = np.zeros_like(gamma)
    t[mask] = (1.0 - dist_to_edge[mask] / sponge_width)
    gamma += sponge_strength * (t ** power)
    return gamma


def laplacian_2d(u, dx, dy):
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    return lap


def gauss(t, t0, sigma):
    x = (t - t0) / sigma
    return np.exp(-0.5 * x*x)


def obstacle_mask_circle(nx, ny, center, radius):
    cx, cy = center
    X, Y = np.ogrid[:nx, :ny]
    return (X - cx)**2 + (Y - cy)**2 <= radius**2


def apply_rigid_obstacle_neumann(u_next, mask):
    """
    Mur rigide pour la PRESSION : dp/dn = 0.
    """
    if mask is None or not np.any(mask):
        return

    outside = ~mask

    up_val    = np.roll(u_next, -1, axis=0); up_ok    = np.roll(outside, -1, axis=0)
    down_val  = np.roll(u_next,  1, axis=0); down_ok  = np.roll(outside,  1, axis=0)
    left_val  = np.roll(u_next, -1, axis=1); left_ok  = np.roll(outside, -1, axis=1)
    right_val = np.roll(u_next,  1, axis=1); right_ok = np.roll(outside,  1, axis=1)

    num = (up_val * up_ok + down_val * down_ok + left_val * left_ok + right_val * right_ok)
    den = (up_ok.astype(np.float64) + down_ok.astype(np.float64)
           + left_ok.astype(np.float64) + right_ok.astype(np.float64))
    den = np.maximum(den, 1.0)

    u_next[mask] = (num / den)[mask]


def solve_damped_wave_2d_with_subtraction(
    nx=200, ny=200, dx=1.0, dy=1.0,
    c=1.5, gamma0=0.005,
    sponge_width=50, sponge_strength=1.5,
    steps=1500, dt=None,
    dt_fixed=True, c_ref=2.0,
    
    # Sources
    pulses=None,
    
    # Capteurs (mêmes positions que dans la simu sans obstacle)
    sensor1_pos=(60, 60),
    sensor2_pos=(60, 120),
    sensor3_pos=(75, 60),
    
    bc="neumann",
    plot_every=8,
    
    # Obstacle
    obstacle_mask=None,
    
    # Fichier de référence (sans obstacle)
    reference_file="sans_obstacle_3sources.npy"
):
    """
    Version avec obstacle ET soustraction du signal sans obstacle
    """
    
    # Charger les données de référence (sans obstacle)
    print(f"Chargement des données de référence: {reference_file}")
    try:
        ref_data = np.load(reference_file)
        t_ref = ref_data[:, 0]
        p1_ref = ref_data[:, 1]
        p2_ref = ref_data[:, 2]
        p3_ref = ref_data[:, 3]
        print("Données de référence chargées avec succès")
    except FileNotFoundError:
        print(f"ERREUR: Fichier {reference_file} non trouvé!")
        print("Veuillez d'abord exécuter la simulation sans obstacle")
        return None, None, None, None, None, None, None, None

    # --- dt ---
    if dt is None:
        c_use = c_ref if dt_fixed else c
        dt = 0.95 / (c_use * np.sqrt(1.0/dx**2 + 1.0/dy**2))

    cfl = c * dt * np.sqrt(1.0/dx**2 + 1.0/dy**2)
    print(f"dt = {dt:.3e} | CFL = {cfl:.3f} (doit être < 1)")

    gamma = build_sponge_gamma(nx, ny, gamma0, sponge_width, sponge_strength, power=2)

    u_prev = np.zeros((nx, ny), dtype=np.float64)
    u      = np.zeros((nx, ny), dtype=np.float64)
    u_next = np.zeros((nx, ny), dtype=np.float64)

    # Positions des capteurs
    cx1, cy1 = sensor1_pos
    cx2, cy2 = sensor2_pos
    cx3, cy3 = sensor3_pos

    c2dt2 = (c * dt) ** 2
    dt2 = dt * dt

    denom = (1.0 + 2.0 * gamma * dt)
    a_prev = (1.0 - 2.0 * gamma * dt)

    # buffers capteurs (avec obstacle)
    t_hist  = np.zeros(steps, dtype=np.float64)
    p1_hist = np.zeros(steps, dtype=np.float64)
    p2_hist = np.zeros(steps, dtype=np.float64)
    p3_hist = np.zeros(steps, dtype=np.float64)

    # -------- Affichage ----------
    plt.figure(figsize=(12, 10))
    
    # Sous-plot 1: Champ avec obstacle
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(u, origin="lower", cmap="seismic", vmin=-0.5, vmax=0.5)
    plt.colorbar(im1, shrink=0.8)
    
    # Afficher l'obstacle
    if obstacle_mask is not None and np.any(obstacle_mask):
        xs, ys = np.where(obstacle_mask)
        plt.scatter(ys, xs, s=1, c="k", alpha=0.5, label="Obstacle")
    
    plt.plot(cy1, cx1, 'ro', markersize=6, label="Capteur 1")
    plt.plot(cy2, cx2, 'bo', markersize=6, label="Capteur 2")
    plt.plot(cy3, cx3, 'go', markersize=6, label="Capteur 3")
    plt.legend(loc="upper right")
    plt.title("Avec obstacle")

    plt.tight_layout()

    # Boucle principale
    for n in range(steps):
        t = n * dt

        lap = laplacian_2d(u, dx, dy)

        # Source
        s = np.zeros_like(u)
        if pulses is not None:
            for pulse in pulses:
                x, y = pulse["pos"]
                s[x, y] += pulse["amp"] * gauss(t, pulse["t0"], pulse["sigma"])

        # Update
        u_next = (2.0*u - a_prev*u_prev + c2dt2*lap + dt2*s) / denom

        # BC externes
        if bc.lower() == "dirichlet":
            u_next[0, :]  = 0.0; u_next[-1, :] = 0.0
            u_next[:, 0]  = 0.0; u_next[:, -1] = 0.0
        elif bc.lower() == "neumann":
            u_next[0, :]  = u_next[1, :]; u_next[-1, :] = u_next[-2, :]
            u_next[:, 0]  = u_next[:, 1]; u_next[:, -1] = u_next[:, -2]

        # Obstacle rigide (Neumann interne)
        apply_rigid_obstacle_neumann(u_next, obstacle_mask)

        # Stabilisation de l'obstacle
        if obstacle_mask is not None and np.any(obstacle_mask):
            u[obstacle_mask] = u_next[obstacle_mask]
            u_prev[obstacle_mask] = u_next[obstacle_mask]

        # Avance en temps
        u_prev, u = u, u_next

        # Enregistrement capteurs
        t_hist[n]  = t
        p1_hist[n] = u[cx1, cy1]
        p2_hist[n] = u[cx2, cy2]
        p3_hist[n] = u[cx3, cy3]

        # Affichage
        if (n % plot_every) == 0:
            plt.subplot(2, 2, 1)
            im1.set_data(u)
            plt.title(f"Avec obstacle - t = {t:.3f}s")
            
            plt.pause(0.001)

    plt.show()

    # -------- SOUSTRACTION ----------
    # S'assurer que les vecteurs temps correspondent
    # (interpolation si nécessaire - ici on suppose qu'ils sont identiques)
    print("\n--- Soustraction du signal sans obstacle ---")
    
    # Calcul des réflexions seules
    reflexions1 = p1_hist - p1_ref[:steps]
    reflexions2 = p2_hist - p2_ref[:steps]
    reflexions3 = p3_hist - p3_ref[:steps]
    
    # Visualisation complète
    plt.figure(figsize=(14, 10))
    
    # 1. Signal avec obstacle
    plt.subplot(3, 2, 1)
    plt.plot(t_hist, p1_hist, 'r-', label="Capteur 1", lw=1)
    plt.plot(t_hist, p2_hist, 'b-', label="Capteur 2", lw=1)
    plt.plot(t_hist, p3_hist, 'g-', label="Capteur 3", lw=1)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression")
    plt.title("AVEC obstacle (pulse direct + réflexions)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Signal sans obstacle (référence)
    plt.subplot(3, 2, 2)
    plt.plot(t_ref[:steps], p1_ref[:steps], 'r--', label="Capteur 1 réf", lw=1, alpha=0.7)
    plt.plot(t_ref[:steps], p2_ref[:steps], 'b--', label="Capteur 2 réf", lw=1, alpha=0.7)
    plt.plot(t_ref[:steps], p3_ref[:steps], 'g--', label="Capteur 3 réf", lw=1, alpha=0.7)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression")
    plt.title("SANS obstacle (pulse direct seul)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Réflexions seules (après soustraction)
    plt.subplot(3, 2, 3)
    plt.plot(t_hist, reflexions1, 'r-', label="Réflexions 1", lw=1)
    plt.plot(t_hist, reflexions2, 'b-', label="Réflexions 2", lw=1)
    plt.plot(t_hist, reflexions3, 'g-', label="Réflexions 3", lw=1)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression")
    plt.title("RÉFLEXIONS SEULES (avec obstacle - sans obstacle)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Comparaison capteur 1
    plt.subplot(3, 2, 4)
    plt.plot(t_hist, p1_hist, 'r-', label="Avec obstacle", lw=1, alpha=0.7)
    plt.plot(t_ref[:steps], p1_ref[:steps], 'k--', label="Sans obstacle", lw=1, alpha=0.7)
    plt.plot(t_hist, reflexions1, 'b-', label="Différence", lw=1.5)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression")
    plt.title("Capteur 1 - Comparaison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 5. Zoom sur une zone d'intérêt
    plt.subplot(3, 2, 5)
    # Trouver où les réflexions commencent (après le pulse direct)
    idx_debut = int(50 / dt)  # à ajuster selon votre cas
    plt.plot(t_hist[idx_debut:], reflexions1[idx_debut:], 'r-', label="Réflexions 1", lw=1)
    plt.plot(t_hist[idx_debut:], reflexions2[idx_debut:], 'b-', label="Réflexions 2", lw=1)
    plt.plot(t_hist[idx_debut:], reflexions3[idx_debut:], 'g-', label="Réflexions 3", lw=1)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression")
    plt.title("Zoom sur les réflexions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Énergie des réflexions
    plt.subplot(3, 2, 6)
    energie1 = np.cumsum(reflexions1**2)
    energie2 = np.cumsum(reflexions2**2)
    energie3 = np.cumsum(reflexions3**2)
    plt.plot(t_hist, energie1, 'r-', label="Énergie 1", lw=1)
    plt.plot(t_hist, energie2, 'b-', label="Énergie 2", lw=1)
    plt.plot(t_hist, energie3, 'g-', label="Énergie 3", lw=1)
    plt.xlabel("Temps (s)")
    plt.ylabel("Énergie cumulée")
    plt.title("Énergie des réflexions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Sauvegarde des résultats
    print("\n--- Sauvegarde des données ---")
    
    # Sauvegarder les réflexions
    reflexions_data = np.column_stack((t_hist, reflexions1, reflexions2, reflexions3))
    header = "Temps(s)\tReflexions1\tReflexions2\tReflexions3"
    np.savetxt("reflexions_avec_obstacle.txt", reflexions_data, header=header, delimiter='\t')
    np.save("reflexions_avec_obstacle.npy", reflexions_data)
    print("Réflexions sauvegardées: reflexions_avec_obstacle.txt/.npy")
    
    # Statistiques
    print("\n--- Statistiques des réflexions ---")
    print(f"Réflexions 1 - min: {reflexions1.min():.3e}, max: {reflexions1.max():.3e}")
    print(f"Réflexions 2 - min: {reflexions2.min():.3e}, max: {reflexions2.max():.3e}")
    print(f"Réflexions 3 - min: {reflexions3.min():.3e}, max: {reflexions3.max():.3e}")
    
    # Rapport signal/bruit approximatif
    snr1 = 20 * np.log10(np.std(reflexions1) / np.std(p1_ref[:steps]) + 1e-10)
    print(f"\nSNR approximatif - Capteur 1: {snr1:.1f} dB")
    
    return t_hist, p1_hist, p2_hist, p3_hist, reflexions1, reflexions2, reflexions3


if __name__ == "__main__":

    nx, ny = 200, 200

    # Obstacle (disque au centre)
    mask_obj = obstacle_mask_circle(nx, ny, center=(100, 100), radius=15)
    
    # Pulses (identiques à la simulation sans obstacle)
    pulses = [
        {"pos": (60, 60),   "t0": 5.0,   "sigma": 2, "amp": 5.0},
        {"pos": (60, 120),  "t0": 120.0, "sigma": 2, "amp": 5.0},
        {"pos": (75, 60),   "t0": 250.0, "sigma": 2, "amp": 5.0},
    ]

    # Lancer la simulation AVEC obstacle et soustraction
    results = solve_damped_wave_2d_with_subtraction(
        nx=nx, ny=ny,
        dx=1.0, dy=1.0,
        c=1.50,
        dt=None,
        dt_fixed=True,
        c_ref=2.0,
        gamma0=0.005,
        sponge_width=50,
        sponge_strength=1.5,
        steps=1500,
        
        pulses=pulses,
        
        sensor1_pos=(60, 60),
        sensor2_pos=(60, 120),
        sensor3_pos=(75, 60),
        
        bc="neumann",
        plot_every=8,
        
        obstacle_mask=mask_obj,
        
        reference_file="sans_obstacle_3sources.npy"  # Fichier généré précédemment
    )
    
    t, p1, p2, p3, r1, r2, r3 = results
    
    if t is not None:
        print("\nSimulation avec soustraction terminée avec succès!")
        print("Les réflexions pures ont été isolées et sauvegardées.")
