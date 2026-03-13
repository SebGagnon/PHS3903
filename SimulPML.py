import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# FDTD 2D : u_tt + 2*gamma*u_t = c^2 * Laplacian(u) + source
# + couche éponge (gamma augmente près des bords)
# + 1 source (multi-pulses, chaque pulse à une position différente)
# + 3 capteurs (enregistrement u(t)) + graphe après la simu
# + obstacle interne rigide (Neumann interne : dp/dn = 0 pour la pression)
# + affichage SANS inversion de champ (pas de .T)
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

def gaussian_pulse_train(t, times, sigma, amps=None):
    times = np.asarray(times, dtype=np.float64)
    if amps is None:
        amps = np.ones_like(times)
    else:
        amps = np.asarray(amps, dtype=np.float64)

    s = 0.0
    for t0, a in zip(times, amps):
        s += a * gauss(t, t0, sigma)
    return s


# ---------------------------
# Obstacle (masque)
# ---------------------------
def obstacle_mask_circle(nx, ny, center, radius):
    cx, cy = center
    X, Y = np.ogrid[:nx, :ny]
    return (X - cx)**2 + (Y - cy)**2 <= radius**2

def obstacle_mask_rect(nx, ny, x0, x1, y0, y1):
    m = np.zeros((nx, ny), dtype=bool)
    m[x0:x1, y0:y1] = True
    return m


def apply_rigid_obstacle_neumann(u_next, mask):
    """
    Mur rigide pour la PRESSION : dp/dn = 0.
    Implémentation simple: dans l'obstacle, on copie la moyenne des voisins
    situés à l'extérieur (approx de gradient normal nul à la frontière).
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


def solve_damped_wave_2d(
    nx=400, ny=400, dx=1.0, dy=1.0,
    c=0.05, gamma0=0.002,
    sponge_width=25, sponge_strength=0.08,
    steps=1500, dt=None,

    # contrôle du pas de temps
    dt_fixed=True,
    c_ref=1.5,

    # Source 1 "classique" (fallback si pulses=None)
    pulse_times_1=None,
    pulse_sigma_1=0.02,
    pulse_amps_1=None,
    source_amp_1=1.0,
    src1_pos=None,

    # NOUVEAU : liste de pulses chacun avec sa position
    # pulses = [{"t0":..., "sigma":..., "amp":..., "pos":(x,y)}, ...]
    pulses=None,

    # capteurs
    sensor1_pos=None,
    sensor2_pos=None,
    sensor3_pos=None,

    # BC externes
    bc="dirichlet",

    # obstacle rigide
    obstacle_mask=None,

    plot_every=10
):
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

    # Position source fallback
    if src1_pos is None:
        sx1, sy1 = nx // 2, ny // 2
    else:
        sx1, sy1 = src1_pos

    # Capteurs
    if sensor1_pos is None:
        cx1, cy1 = nx // 2, ny // 4
    else:
        cx1, cy1 = sensor1_pos

    if sensor2_pos is None:
        cx2, cy2 = nx // 2, (3 * ny) // 4
    else:
        cx2, cy2 = sensor2_pos

    if sensor3_pos is None:
        cx3, cy3 = nx // 2, (4 * ny) // 4
    else:
        cx3, cy3 = sensor3_pos

    # pulses fallback (ancien mode)
    if pulse_times_1 is None:
        pulse_times_1 = [0.4, 0.9, 1.35, 2.1, 3.0]

    c2dt2 = (c * dt) ** 2
    dt2 = dt * dt

    denom = (1.0 + 2.0 * gamma * dt)
    a_prev = (1.0 - 2.0 * gamma * dt)

    # buffers capteurs
    t_hist  = np.zeros(steps, dtype=np.float64)
    p1_hist = np.zeros(steps, dtype=np.float64)
    p2_hist = np.zeros(steps, dtype=np.float64)
    p3_hist = np.zeros(steps, dtype=np.float64)

    # -------- Affichage SANS inversion du champ ----------
    plt.figure(figsize=(7, 6))
    im = plt.imshow(u, origin="lower", cmap="seismic", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, shrink=0.8)

    # obstacle (en points noirs)
    if obstacle_mask is not None and np.any(obstacle_mask):
        xs, ys = np.where(obstacle_mask)   # x = rows, y = cols
        plt.scatter(ys, xs, s=1, c="k", alpha=0.35, label="Obstacle rigide")

    # marqueurs capteurs : imshow(u) => (x,y) s'affiche en (col=y, row=x)
    plt.plot(cy1, cx1, 'ko', markersize=6, label="Capteur 1")
    plt.plot(cy2, cx2, 'go', markersize=6, label="Capteur 2")
    plt.plot(cy3, cx3, 'ro', markersize=6, label="Capteur 3")
    plt.legend(loc="upper right")

    plt.title("p(x,y,t) — onde 2D amortie | 1 source + obstacle rigide")
    plt.tight_layout()

    for n in range(steps):
        t = n * dt

        lap = laplacian_2d(u, dx, dy)

        # -------- Source (1 seule) ----------
        s = np.zeros_like(u)

        if pulses is not None:
            # mode "pulses positionnés"
            for pulse in pulses:
                x, y = pulse["pos"]
                s[x, y] += pulse["amp"] * gauss(t, pulse["t0"], pulse["sigma"])
        else:
            # mode "train" (ancien)
            s[sx1, sy1] += source_amp_1 * gaussian_pulse_train(
                t=t, times=pulse_times_1, sigma=pulse_sigma_1, amps=pulse_amps_1
            )

        # Update
        u_next = (2.0*u - a_prev*u_prev + c2dt2*lap + dt2*s) / denom

        # BC externes
        if bc.lower() == "dirichlet":
            u_next[0, :]  = 0.0
            u_next[-1, :] = 0.0
            u_next[:, 0]  = 0.0
            u_next[:, -1] = 0.0
        elif bc.lower() == "neumann":
            u_next[0, :]  = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0]  = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        else:
            raise ValueError("bc doit être 'dirichlet' ou 'neumann'")

        # Obstacle rigide (Neumann interne)
        apply_rigid_obstacle_neumann(u_next, obstacle_mask)

        # Stabilise l’intérieur de l’obstacle (évite “mémoire” numérique)
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
            im.set_data(u)
            plt.title(f"p(x,y,t) — t = {t:.3f} s | dt = {dt:.3e} | c={c}")
            plt.pause(0.001)

    plt.show()

    # Graphe capteurs après la simulation
    plt.figure(figsize=(8, 4))
    plt.plot(t_hist, p1_hist, label=f"Capteur 1 ({cx1},{cy1})", lw=0.7)
    plt.plot(t_hist, p2_hist, label=f"Capteur 2 ({cx2},{cy2})", lw=0.7)
    plt.plot(t_hist, p3_hist, label=f"Capteur 3 ({cx3},{cy3})", lw=0.7)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression p")
    plt.title("Pression mesurée aux capteurs vs temps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    nx, ny = 200, 200

    # Obstacle (ex : disque)
    mask_obj = obstacle_mask_circle(nx, ny, center=(110, 100), radius=9)
    # Alternative rectangle:
    # mask_obj = obstacle_mask_rect(nx, ny, x0=90, x1=130, y0=70, y1=90)

    # Pulses à différentes positions (UNE seule source logique)
    pulses = [
        {"pos": (80, 60),   "t0": 1.0,   "sigma": 2, "amp": 5.0},
    ]

    solve_damped_wave_2d(
        nx=nx, ny=ny,
        dx=1.0, dy=1.0,

        c=1.50,
        dt=None,
        dt_fixed=True,
        c_ref=2.0,

        gamma0=0.005,
        sponge_width=50,
        sponge_strength=1.5,
        steps=500,

        # (fallback source "train" si pulses=None)
        src1_pos=(60, 60),
        pulse_times_1=[1, 100, 200],
        pulse_sigma_1=2,
        source_amp_1=4.0,

        # NOUVEAU
        pulses=pulses,

        # capteurs
        sensor1_pos=(95, 95),
        sensor2_pos=(60, 120),
        sensor3_pos=(120, 60),

        bc="neumann",
        plot_every=8,

        obstacle_mask=mask_obj,
    )