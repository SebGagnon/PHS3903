import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# FDTD 2D : u_tt + 2*gamma*u_t = c^2 * Laplacian(u) + source
# + couche éponge (gamma augmente près des bords)
# + 2 sources (trains d'impulsions gaussiennes)
# + 2 capteurs (enregistrement u(t)) + graphe après la simu
# + marqueurs sur la carte pour les capteurs
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
    t[mask] = (1.0 - dist_to_edge[mask] / sponge_width)  # 0->1 vers le bord
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


def solve_damped_wave_2d(
    nx=400, ny=400, dx=1.0, dy=1.0,
    c=1.0, gamma0=0.002,
    sponge_width=25, sponge_strength=0.08,
    steps=1500, dt=None,

    # Source 1
    pulse_times_1=None,
    pulse_sigma_1=0.02,
    pulse_amps_1=None,
    source_amp_1=1.0,

    # Source 2
    pulse_times_2=None,
    pulse_sigma_2=0.02,
    pulse_amps_2=None,
    source_amp_2=1.0,

    # positions sources
    src1_pos=None,
    src2_pos=None,

    # capteurs (positions)
    sensor1_pos=None,
    sensor2_pos=None,

    bc="dirichlet",
    plot_every=10
):
    # dt CFL
    if dt is None:
        dt = 0.95 / (c * np.sqrt(1.0/dx**2 + 1.0/dy**2))

    gamma = build_sponge_gamma(nx, ny, gamma0, sponge_width, sponge_strength, power=2)

    u_prev = np.zeros((nx, ny), dtype=np.float64)
    u      = np.zeros((nx, ny), dtype=np.float64)
    u_next = np.zeros((nx, ny), dtype=np.float64)

    # Positions sources
    if src1_pos is None:
        sx1, sy1 = nx // 2, ny // 2
    else:
        sx1, sy1 = src1_pos

    if src2_pos is None:
        sx2, sy2 = nx // 2, ny // 2
    else:
        sx2, sy2 = src2_pos

    # Positions capteurs
    if sensor1_pos is None:
        cx1, cy1 = nx // 2, ny // 4
    else:
        cx1, cy1 = sensor1_pos

    if sensor2_pos is None:
        cx2, cy2 = nx // 2, (3 * ny) // 4
    else:
        cx2, cy2 = sensor2_pos

    # Sources par défaut (secondes)
    if pulse_times_1 is None:
        pulse_times_1 = [0.4, 0.9, 1.35, 2.1, 3.0]
    if pulse_times_2 is None:
        pulse_times_2 = [0.6, 1.1, 1.8, 2.6, 3.3]

    c2dt2 = (c * dt) ** 2
    dt2 = dt * dt

    denom = (1.0 + 2.0 * gamma * dt)
    a_prev = (1.0 - 2.0 * gamma * dt)

    # buffers capteurs
    t_hist  = np.zeros(steps, dtype=np.float64)
    p1_hist = np.zeros(steps, dtype=np.float64)
    p2_hist = np.zeros(steps, dtype=np.float64)

    # Affichage champ
    plt.figure(figsize=(7, 6))
    im = plt.imshow(u.T, origin="lower", cmap="seismic", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, shrink=0.8)

    # Marqueurs capteurs
    # IMPORTANT: on affiche u.T => axes inversés, donc plot(y, x)
    m1, = plt.plot(cy1, cx1, 'ko', markersize=6, label="Capteur 1")
    m2, = plt.plot(cy2, cx2, 'go', markersize=6, label="Capteur 2")

    plt.legend(loc="upper right")
    plt.title("u(x,y,t) — onde 2D amortie (FDTD) | 2 sources")
    plt.tight_layout()

    for n in range(steps):
        t = n * dt

        lap = laplacian_2d(u, dx, dy)

        # Deux sources ponctuelles
        s = np.zeros_like(u)
        s[sx1, sy1] += source_amp_1 * gaussian_pulse_train(
            t=t, times=pulse_times_1, sigma=pulse_sigma_1, amps=pulse_amps_1
        )
        s[sx2, sy2] += source_amp_2 * gaussian_pulse_train(
            t=t, times=pulse_times_2, sigma=pulse_sigma_2, amps=pulse_amps_2
        )

        # Update
        u_next = (2.0*u - a_prev*u_prev + c2dt2*lap + dt2*s) / denom

        # Conditions aux limites
        if bc.lower() == "dirichlet":
            u_next[0, :] = 0.0
            u_next[-1, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
        elif bc.lower() == "neumann":
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        else:
            raise ValueError("bc doit être 'dirichlet' ou 'neumann'")

        # Avance en temps
        u_prev, u = u, u_next

        # Enregistrement capteurs ("pression" = u au point)
        t_hist[n] = t
        p1_hist[n] = u[cx1, cy1]
        p2_hist[n] = u[cx2, cy2]

        # Affichage
        if (n % plot_every) == 0:
            im.set_data(u.T)
            plt.title(f"u(x,y,t) — t = {t:.3f} s | dt = {dt:.3e}")
            plt.pause(0.001)

    plt.show()

    # Graphe capteurs après la simulation
    plt.figure(figsize=(8, 4))
    plt.plot(t_hist, p1_hist, label=f"Capteur 1 ({cx1},{cy1})", lw=0.5)
    plt.plot(t_hist, p2_hist, label=f"Capteur 2 ({cx2},{cy2})", lw= 0.5)
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression (u)")
    plt.title("Pression mesurée aux capteurs vs temps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    solve_damped_wave_2d(
        nx=200, ny=200,
        dx=1.0, dy=1.0,
        c=1.5,
        gamma0=0.0025,
        sponge_width=30,
        sponge_strength=0.30,
        steps=1000,

        # sources
        src1_pos=(60, 60),
        pulse_times_1=[10, 50, 100, 150, 200],
        pulse_sigma_1=1,
        pulse_amps_1=[1.0, 0.8, 1.2, 1.0, 0.7, 1.0],
        source_amp_1=4.0,

        src2_pos=(140, 140),
        pulse_times_2=[10, 50, 100, 150, 200],
        pulse_sigma_2=2,
        pulse_amps_2=[1.0, 0.8, 1.2, 1.0, 0.7, 1.0],
        source_amp_2=3.0,

        # capteurs
        sensor1_pos=(75, 75),
        sensor2_pos=(60, 140),

        bc="neumann",
        plot_every=8
    )