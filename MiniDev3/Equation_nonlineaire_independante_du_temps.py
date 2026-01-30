import numpy as np
import matplotlib.pyplot as plt

# Équation différentielle: d^2 u/dx^2=g(x) sur x=(a,b)
# Conditions aux limites générales:
# x=a: c1*du/dx+c2*u+c3=-si*T^4
# x=b: d1*du/dx+d2*u+d3=0

# Équation de transfert de chaleur d^2 T/dx^2=-S(x)/k sur x=(0,L)
# dans un mur d'isolation thermique
L=0.3; #[m] ; Épaisseur du mur

k=0.85; #[W/(m*K)]; La conductivité thermique de la brique
si=5.67e-8; #constante de Stefan-Boltzmann

To=293; #oK

# Condition radiative à x=0 (face externe du mur): -k*dT/dx=-si*(T^4-To^4)
# !!! Condition radiative est implementée SEULEMENT sur la face externe du mur !!!
c1=-k; c2=0; c3=-si*To**4;
# Condition de Neumann à x=L (face interne du mur): dT/dx=0 - flux net de chaleur est 0
d1=1; d2=0; d3=0;

#(N+1) nœuds dans la maille
# Nmax=10000 pour 1G de mémoire

#Nar1=np.array([100]); #dx=3mm
#Nar1= np.concatenate((np.arange(2, 11, 1), np.arange(20, 110, 10), np.arange(200, 1100, 100), np.arange(2000, 6000, 1000))); # Matrice pleine
#Nar=np.zeros(2*Nar1.size,dtype=Nar1.dtype);
#Nar[np.arange(0,2*Nar1.size-1,2)]=Nar1.copy();
#Nar[np.arange(0,2*Nar1.size-1,2)+1]=2*Nar1.copy();

N=100;
print('Nombre de points sur la grille=', N)
dx = L/N;  # Pas de discrétisation
x = np.linspace(0, L, N+1);
T = To*np.ones(N+1, dtype=np.double);# Pour commencer les itérations

S=np.zeros(N+1,dtype=np.double);
A = np.zeros((N+1, N+1), dtype=np.double);
J = np.zeros((N+1, N+1), dtype=np.double);
b = np.zeros(N+1, dtype=np.double);
dT = np.zeros(N+1, dtype=np.double);
F = np.zeros(N+1, dtype=np.double);

# Sourse volumique de chaleur q[W/m^3] d'épaisseur dL
# La source est intégrée dans la partie intérieure du mur
dL = 0.1;
q = 2000;  # W/m^3
S = q*np.exp(-((x-L)/dL)**2);

Err=[];

# matrice pleine
A=np.diag(-2*np.ones(N+1),0)+np.diag(np.ones(N),-1)+np.diag(np.ones(N),1); 
A[0,0]=2*c2*dx-3*c1;A[0,1]=4*c1;A[0,2]=-c1;
A[N,N]=3*d1+2*d2*dx;A[N,N-1]=-4*d1;A[N,N-2]=d1;

plt.figure(1)  
plt.plot(x,T,'r')

ci=-1;flag=1;tol=1e-12;

while flag:    
    ci = ci+1;

    b=S/k*dx**2; b[0]=T[0]**4*2*dx*si+2*c3*dx; b[N]=2*d3*dx;
    
    F=A@T+b;

    Err.append(sum(np.abs(F))/(N+1));
    print('Étape=', ci+1, '   ;   Err=', Err[ci])
    if (Err[ci]<tol):
        flag=0;
        
    J=A.copy();J[0,0]=A[0,0]+T[0]**3*8*dx*si;
    dT=-np.linalg.solve(J,F);
    T=T+dT;

    plt.plot(x,T,'b')
    
Tmax=T.max();
print('q=',q,'Tmax=',Tmax)    

plt.axis([x[0], x[-1], To, Tmax])
plt.title('Température (x)')
plt.xlabel('x [m]')    
plt.ylabel('$T_{eq}$(x) [$^o$C]')
plt.show()

plt.figure(2)
plt.plot(np.arange(1, ci + 2),Err,'-or');
plt.xscale('log')
plt.yscale('log')
plt.title('Erreur (N)')
plt.xlabel('dx [m]')
plt.ylabel('Err(dx)=|$T_{max}$(dx)-$T_{max}$(dx/2)|')
plt.show()
