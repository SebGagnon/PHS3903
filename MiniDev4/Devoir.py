import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tabulate import tabulate

D_val = [3,4,5,6]
Ntot_val = 100 * 2 ** np.arange(0,5)
Ness = 100

VDth = []
results = []

for D in D_val:
    Vtot = 2**D
    Vth = np.pi**(D/2)/sp.special.gamma(D/2+1)
    VDth.append(Vth)

    row = [f"{D} D"]
    
    for Ntot in Ntot_val:
        Vind = np.zeros(Ness)

        for k in range(Ness):
            pts = np.random.uniform(-1,1,(Ntot,D))
            r2 = np.sum(pts**2, axis=1)   # ||x||^2 vectorisé
            Nint = np.sum(r2 <= 1)

            Vind[k] = Nint/Ntot * Vtot

        moyenne = np.mean(Vind)
        incertitude_relative = (np.std(Vind, ddof=1)/np.sqrt(Ness) )/moyenne*100

        row.append(f"{moyenne:.5f} ± {incertitude_relative:.2f}%")

    row.append(f"{Vth:.5f}")
    results.append(row)

headers = ["Dimensions"] + [f"N={n}" for n in Ntot_val] + ["Valeur théorique"]
print(tabulate(results, headers=headers, tablefmt="github"))
plt.figure()

for D in D_val:
    if D == 3 or D == 6:
        erreurs = []
        for Ntot in Ntot_val:
            Vtot = 2**D
            p = VDth[D_val.index(D)] / Vtot
            sigma = Vtot*np.sqrt(p*(1-p)/Ntot)
            erreurs.append(sigma)

        plt.loglog(Ntot_val, erreurs, 'o-', label=f"D={D}")
        pente = []
        for i in range(0, 4):
            pente.append(np.log(erreurs[i+1])-np.log(erreurs[i])/(np.log(Ntot_val[i+1])-np.log(Ntot_val[i])))
            
        print(f"La pente p du log de l'erreur en {D}D est de {np.mean(np.array(pente))}")
plt.xlabel("Ntot")
plt.ylabel("Erreur théorique")
plt.legend()
plt.grid()
plt.show()

