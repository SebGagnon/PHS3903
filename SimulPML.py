import numpy as np
import matplotlib.pyplot as plt


def setup_pml(n, pml_width=50, sigma_max=0.5):
    sigmax = np.zeros((n, n))
    sigmay = np.zeros((n, n))

    idx = np.arange(n)

    
    left = np.clip(pml_width - idx, 0, pml_width)
    right = np.clip(pml_width - (n - 1 - idx), 0, pml_width)

    sigma_1d = sigma_max * (left + right) / pml_width

    sigmax[:] = sigma_1d[:, None]
    sigmay[:] = sigma_1d[None, :]

    return sigmax, sigmay



n = 256
c = 2.0
tmax = 1500

xmin, xmax = -50, 50
x1d = np.linspace(xmin, xmax, n)
dx = x1d[1] - x1d[0]

x, y = np.meshgrid(x1d, x1d)

phi = np.exp(-(x**2 + y**2)/2) / (2*np.pi)

sigmax, sigmay = setup_pml(n)

dt = 0.25 * dx / c

s_xplusy = (sigmax + sigmay) / c**2
s_xtimesy = (sigmax * sigmay) / c**2

psi = np.zeros((n, n))
u_now = np.zeros((n, n))
vx = np.zeros((n, n))
vy = np.zeros((n, n))

plt.figure(figsize=(7, 6))

im = plt.imshow(
    u_now,
    extent=[xmin, xmax, xmin, xmax],
    origin='lower',
    cmap='RdBu',
    vmin=-0.2,
    vmax=0.2,
    aspect='equal'
)

plt.colorbar(label="Amplitude")
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)
plt.xlabel("x")
plt.ylabel("y")


plt.tight_layout()
plt.ion()
plt.show()


for t in range(tmax):

    dudx, dudy = np.gradient(u_now, dx)

    vx += dt * (dudx - vx * sigmax)
    vy += dt * (dudy - vy * sigmay)

    dvxdx, _ = np.gradient(vx, dx)
    _, dvydy = np.gradient(vy, dx)

    psi += dt * (
        sigmay * dvxdx
        + sigmax * dvydy
        - s_xtimesy * u_now
        + np.cos(dt * t) * phi
    )

    u_now += dt * c**2 * (
        dvxdx
        + dvydy
        - s_xplusy * u_now
        + psi
    )

    
    if t % 10 == 0:
        im.set_data(u_now)
        plt.title(f"t = {t}")
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()
