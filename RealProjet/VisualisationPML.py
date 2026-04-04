import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

with open("testPML.pkl", "rb") as f:
    testPML = pickle.load(f)


epaisseurs = sorted(set(i for i, j in testPML.keys()))
gammas = sorted(set(j for i, j in testPML.keys()))

Z = np.zeros((len(epaisseurs), len(gammas)))

for a, i in enumerate(epaisseurs):
    for b, j in enumerate(gammas):
        Z[a, b] = testPML[(i, j)]

plt.figure(figsize=(10, 6))
im = plt.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=[min(gammas), max(gammas), min(epaisseurs), max(epaisseurs)],
    cmap='inferno'
)

plt.colorbar(im, label="Energie residuelle")
plt.xlabel("gamma max")
plt.ylabel("epaisseur PML")
plt.tight_layout()
plt.show()
Z_log = np.where(Z <= 0, 1e-12, Z)

plt.figure(figsize=(10, 6))
im = plt.imshow(
    Z_log,
    origin="lower",
    aspect="auto",
    extent=[min(gammas), max(gammas), min(epaisseurs), max(epaisseurs)],
    norm=LogNorm(vmin=Z_log.min(), vmax=Z_log.max()),
    cmap='inferno'
)

plt.colorbar(im, label="Energie residuelle (log)")
plt.xlabel("gamma max")
plt.ylabel("epaisseur PML")
plt.tight_layout()
plt.show()