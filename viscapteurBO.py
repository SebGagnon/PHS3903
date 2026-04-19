import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

with open("bo_capteurs_new.pkl", "rb") as f:
    data = pickle.load(f)

historique = data["historique"]
capteurs_best = data["capteurs_best"]
couts_train = np.array(data["couts_train"])
res_fun = data["res_fun"]
res_x = data["res_x"]

# Domaine
try:
    L = params["L"]
    ratio_pml = params["epaisseur_pml_ratio"]
except NameError:
    L = 1000.0
    ratio_pml = 0.2

epaisseur_pml = ratio_pml * L
xmin_phys = epaisseur_pml
xmax_phys = L - epaisseur_pml

# Extraire les positions testées
x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
valide = []

for entry in historique:
    x = entry["x"]
    x1.append(x[0]); y1.append(x[1])
    x2.append(x[2]); y2.append(x[3])
    x3.append(x[4]); y3.append(x[5])
    valide.append(entry["valide"])

x1 = np.array(x1)
y1 = np.array(y1)
x2 = np.array(x2)
y2 = np.array(y2)
x3 = np.array(x3)
y3 = np.array(y3)
valide = np.array(valide)

# Meilleur coût cumulé
best_so_far = np.minimum.accumulate(couts_train)

print("Nombre d'essais :", len(couts_train))
print("Coût min :", np.min(couts_train))
print("Coût max :", np.max(couts_train))
print("Coût moyen :", np.mean(couts_train))
print("Meilleur coût final :", res_fun)
print("Meilleurs capteurs :", capteurs_best)

# -----------------------------
# 0) 3 meilleurs et 3 pires essais valides
# -----------------------------
indices_valides = [i for i, entry in enumerate(historique) if entry["valide"]]
hist_valides = [historique[i] for i in indices_valides]

hist_valides_tries = sorted(hist_valides, key=lambda e: e["cout"])

top3 = hist_valides_tries[:3]
worst3 = hist_valides_tries[-3:]

print("\n==============================")
print("3 MEILLEURS ESSAIS VALIDES")
print("==============================")
for rang, entry in enumerate(top3, start=1):
    x = entry["x"]
    cout = entry["cout"]
    print(f"\n#{rang}  coût = {cout:.6f}")
    print(f"  C1 = ({x[0]:.2f}, {x[1]:.2f}) m")
    print(f"  C2 = ({x[2]:.2f}, {x[3]:.2f}) m")
    print(f"  C3 = ({x[4]:.2f}, {x[5]:.2f}) m")

print("\n==============================")
print("3 PIRES ESSAIS VALIDES")
print("==============================")
for rang, entry in enumerate(reversed(worst3), start=1):
    x = entry["x"]
    cout = entry["cout"]
    print(f"\n#{rang}  coût = {cout:.6f}")
    print(f"  C1 = ({x[0]:.2f}, {x[1]:.2f}) m")
    print(f"  C2 = ({x[2]:.2f}, {x[3]:.2f}) m")
    print(f"  C3 = ({x[4]:.2f}, {x[5]:.2f}) m")

# -----------------------------
# 1) Évolution du coût
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(couts_train, marker="o", label="Coût")
plt.plot(best_so_far, linewidth=2, label="Meilleur coût à date")
plt.xlabel("Évaluation")
plt.ylabel("Coût")
plt.title("Évolution du coût d'entraînement")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 2) Même chose en échelle log
# -----------------------------
plt.figure(figsize=(9, 4))
plt.semilogy(couts_train, marker="o", label="Coût")
plt.semilogy(best_so_far, linewidth=2, label="Meilleur coût à date")
plt.xlabel("Évaluation")
plt.ylabel("Coût (log)")
plt.title("Évolution du coût d'entraînement (log)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Histogramme des coûts
# -----------------------------
plt.figure(figsize=(7, 4))
plt.hist(couts_train, bins=15)
plt.xlabel("Coût")
plt.ylabel("Nombre d'essais")
plt.title("Distribution des coûts")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 4) Positions des capteurs testés
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

labels = ["Capteur 1", "Capteur 2", "Capteur 3"]
coords = [(x1, y1), (x2, y2), (x3, y3)]
best_coords = capteurs_best

# Échelle couleur log pour mieux voir les différences
vmin = max(np.min(couts_train[couts_train > 0]), 1e-12) if np.any(couts_train > 0) else 1e-12
vmax = np.max(couts_train)

for ax, (xs, ys), (xb, yb), label in zip(axes, coords, best_coords, labels):
    sc = ax.scatter(
        xs, ys,
        c=couts_train,
        s=60,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    ax.scatter(xb, yb, marker="*", s=250, label="Meilleur")

    # Contour domaine
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], linewidth=1.5)

    # Contour zone physique hors PML
    ax.plot(
        [xmin_phys, xmax_phys, xmax_phys, xmin_phys, xmin_phys],
        [xmin_phys, xmin_phys, xmax_phys, xmax_phys, xmin_phys],
        linewidth=1.5,
        linestyle="--"
    )

    ax.set_title(label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

cbar = plt.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_label("Coût (log)")

plt.suptitle("Positions testées des capteurs")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Afficher seulement les essais valides
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

for ax, (xs, ys), (xb, yb), label in zip(axes, coords, best_coords, labels):
    sc = ax.scatter(
        xs[valide], ys[valide],
        c=couts_train[valide],
        s=60,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    ax.scatter(xb, yb, marker="*", s=250, label="Meilleur")

    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], linewidth=1.5)
    ax.plot(
        [xmin_phys, xmax_phys, xmax_phys, xmin_phys, xmin_phys],
        [xmin_phys, xmin_phys, xmax_phys, xmax_phys, xmin_phys],
        linewidth=1.5,
        linestyle="--"
    )

    ax.set_title(label + " (essais valides)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

cbar = plt.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_label("Coût (log)")

plt.suptitle("Positions testées des capteurs — essais valides seulement")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Visualisation des 6 meilleures combinaisons valides
# -----------------------------
top6 = hist_valides_tries[:3]

fig, axes = plt.subplots(1, 3, figsize=(14, 9), sharex=True, sharey=True)

def trace_configuration(ax, entry, rang):
    x = entry["x"]
    cout = entry["cout"]

    pts_x = [x[0], x[2], x[4]]
    pts_y = [x[1], x[3], x[5]]

    ax.scatter(pts_x, pts_y, s=120)

    for i, (xp, yp) in enumerate(zip(pts_x, pts_y), start=1):
        ax.text(xp + 8, yp + 8, f"C{i}", fontsize=10, weight="bold")

    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], linewidth=1.5)
    ax.plot(
        [xmin_phys, xmax_phys, xmax_phys, xmin_phys, xmin_phys],
        [xmin_phys, xmin_phys, xmax_phys, xmax_phys, xmin_phys],
        linewidth=1.5,
        linestyle="--"
    )

    ax.set_title(f"{rang}- Erreur moyenne sur la position = {cout:.4f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True)

for ax, entry, rang in zip(axes.ravel(), top6, range(1, 7)):
    trace_configuration(ax, entry, rang)

plt.tight_layout()
plt.show()