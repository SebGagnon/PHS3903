import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Staggered-grid 2D acoustics + canonical CPML (sigma, kappa, alpha)
# + material loss matching: u_tt + 2*gamma*u_t = c^2 ∇²u
# (implemented via velocity damping loss_v = 2*gamma)
#
# Time source: multiple Gaussian pulses (pulse train)
# Visualization: FIXED color scale (no dynamic clim updates)
# ------------------------------------------------------------

# Grid
nx, ny = 220, 220
dx = dy = 1.0

# Medium
c = 3.0
rho = 0.01
K = rho * c**2

# Time step (CFL)
dt = 0.45 * dx / (5 * np.sqrt(2))
steps = 5000

# Material loss (everywhere)
gamma = 0.01


# Fields (staggered)
p  = np.zeros((nx, ny), dtype=np.float64)       # centers
vx = np.zeros((nx + 1, ny), dtype=np.float64)   # x-faces
vy = np.zeros((nx, ny + 1), dtype=np.float64)   # y-faces

# ---------------- CPML profiles ----------------
pml = 30
m = 3                 # polynomial order
sigma_max = 2.5       # strength
kappa_max = 6.0       # typical 2..10
alpha_max = 0.15      # helps low-freq absorption (0.01..0.3 typical)

def pml_profiles_1d(n, pml, sigma_max, kappa_max, alpha_max, m=3):
    sigma = np.zeros(n, dtype=np.float64)
    kappa = np.ones(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d = min(i, n - 1 - i)
        if d < pml:
            x = (pml - d) / pml        # 0..1 into PML
            sigma[i] = sigma_max * (x ** m)
            kappa[i] = 1.0 + (kappa_max - 1.0) * (x ** m)
            alpha[i] = alpha_max * (1.0 - x)    # max at interface -> 0 at boundary
    return sigma, kappa, alpha

def cpml_coeff(sigma, kappa, alpha, dt):
    inv_kappa = 1.0 / kappa
    b = np.exp(-(sigma * inv_kappa + alpha) * dt)
    denom = (sigma + kappa * alpha)
    a = np.zeros_like(sigma)
    mask = denom > 0
    a[mask] = (sigma[mask] * (b[mask] - 1.0) / denom[mask]) * inv_kappa[mask]
    return a, b, inv_kappa

# 1D profiles for staggered locations
sx_p,  kx_p,  ax_p  = pml_profiles_1d(nx,     pml, sigma_max, kappa_max, alpha_max, m)
sy_p,  ky_p,  ay_p  = pml_profiles_1d(ny,     pml, sigma_max, kappa_max, alpha_max, m)
sx_vx, kx_vx, ax_vx = pml_profiles_1d(nx + 1, pml, sigma_max, kappa_max, alpha_max, m)
sy_vy, ky_vy, ay_vy = pml_profiles_1d(ny + 1, pml, sigma_max, kappa_max, alpha_max, m)

# Broadcast to 2D (match field shapes)
sigx_p = np.repeat(sx_p[:, None],  ny, axis=1)
kapx_p = np.repeat(kx_p[:, None],  ny, axis=1)
alpx_p = np.repeat(ax_p[:, None],  ny, axis=1)

sigy_p = np.repeat(sy_p[None, :],  nx, axis=0)
kapy_p = np.repeat(ky_p[None, :],  nx, axis=0)
alpy_p = np.repeat(ay_p[None, :],  nx, axis=0)

sigx_vx = np.repeat(sx_vx[:, None], ny, axis=1)
kapx_vx = np.repeat(kx_vx[:, None], ny, axis=1)
alpx_vx = np.repeat(ax_vx[:, None], ny, axis=1)

sigy_vy = np.repeat(sy_vy[None, :], nx, axis=0)
kapy_vy = np.repeat(ky_vy[None, :], nx, axis=0)
alpy_vy = np.repeat(ay_vy[None, :], nx, axis=0)

# CPML coefficients for each derivative location
a_vx, b_vx, invkap_vx = cpml_coeff(sigx_vx, kapx_vx, alpx_vx, dt)  # dp/dx at vx
a_vy, b_vy, invkap_vy = cpml_coeff(sigy_vy, kapy_vy, alpy_vy, dt)  # dp/dy at vy
a_px, b_px, invkap_px = cpml_coeff(sigx_p,  kapx_p,  alpx_p,  dt)  # dvx/dx at p
a_py, b_py, invkap_py = cpml_coeff(sigy_p,  kapy_p,  alpy_p,  dt)  # dvy/dy at p

# CPML memory variables
psi_vx = np.zeros_like(vx)  # for dp/dx
psi_vy = np.zeros_like(vy)  # for dp/dy
psi_px = np.zeros_like(p)   # for dvx/dx
psi_py = np.zeros_like(p)   # for dvy/dy

# ---------------- Source setup: multiple Gaussian pulses in time ----------------
x0, y0 = nx // 2, ny // 2

A_src = 3.0          # amplitude of each pulse
tau = 25 * dt        # pulse duration (std dev in time). Bigger => longer pulses

# Pulse centers in time (seconds). Example: 5 pulses separated by 120 steps
pulse_times = [100*dt, 460*dt]

def gaussian_pulse_train(t, centers, tau):
    # Sum of Gaussians in time
    s = 0.0
    inv2tau2 = 1.0 / (2.0 * tau * tau)
    for tc in centers:
        dtc = t - tc
        s += np.exp(-dtc*dtc * inv2tau2)
    return s

# ---------------- Initial condition (quiet start) ----------------
# Start from zero everywhere; drive with pulses instead.
# (If you want a nonzero initial field too, you can add it here.)

# ---------------- Plot (FIXED color scale) ----------------
# Choose a fixed display scale. Rule of thumb: ~A_src (or a bit bigger).
A0 = 0.05 * A_src

plt.ion()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(p, origin="lower", cmap="RdBu_r", vmin=-A0, vmax=A0)
plt.colorbar(im, ax=ax)


for n in range(steps):
    # ---- vx update (dp/dx lives on vx[1:nx, :])
    dpdx = (p[1:, :] - p[:-1, :]) / dx                 # (nx-1, ny)
    psi_vx[1:nx, :] = b_vx[1:nx, :] * psi_vx[1:nx, :] + a_vx[1:nx, :] * dpdx
    dpdx_hat = invkap_vx[1:nx, :] * dpdx + psi_vx[1:nx, :]

    vx[1:nx, :] += dt * (
        -(1.0 / rho) * dpdx_hat
        - (sigx_vx[1:nx, :] + alpx_vx[1:nx, :]) * vx[1:nx, :]
        
    )

    # ---- vy update (dp/dy lives on vy[:, 1:ny])
    dpdy = (p[:, 1:] - p[:, :-1]) / dy                 # (nx, ny-1)
    psi_vy[:, 1:ny] = b_vy[:, 1:ny] * psi_vy[:, 1:ny] + a_vy[:, 1:ny] * dpdy
    dpdy_hat = invkap_vy[:, 1:ny] * dpdy + psi_vy[:, 1:ny]

    vy[:, 1:ny] += dt * (
        -(1.0 / rho) * dpdy_hat
        - (sigy_vy[:, 1:ny] + alpy_vy[:, 1:ny]) * vy[:, 1:ny]
        
    )

    # ---- p update
    dvxdx = (vx[1:, :] - vx[:-1, :]) / dx              # (nx, ny)
    dvydy = (vy[:, 1:] - vy[:, :-1]) / dy              # (nx, ny)

    psi_px = b_px * psi_px + a_px * dvxdx
    psi_py = b_py * psi_py + a_py * dvydy

    div_hat = (invkap_px * dvxdx + psi_px) + (invkap_py * dvydy + psi_py)

    p += dt * (
        -K * div_hat
        - (sigx_p + alpx_p + sigy_p + alpy_p) * p
    )

    # ---- Inject multiple Gaussian pulses in time at (x0, y0)
    t = n * dt
    p[x0, y0] += A_src * gaussian_pulse_train(t, pulse_times, tau)

    # Visualize (fixed scale)
    if n % 5 == 0:
        im.set_data(p)
        
        plt.pause(0.001)

plt.ioff()
plt.show()