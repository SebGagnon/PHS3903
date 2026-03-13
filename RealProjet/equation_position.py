import numpy as np

# rayons

r1 = t1*c
r2 = t2*c
r3= t3*c

# vecteurs plans orthogonaux
x = np.array([1, 0])
y = np.array([0, 1])

# position du bateau par rapport au point 0 de la base orthogonale

px = r2**2-r1**2-np.linalg.norm(x)**2/(2*np.linalg.norm(x))
py = r3**2-r1**2-np.linalg.norm(y)**2/(2*np.linalg.norm(y))
