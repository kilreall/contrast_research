import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit

# ------------------------
# Constants
# ------------------------
c = 3e8
mRb = 1.46e-25
h_ = 1.054571817e-34
kb = 1.38e-23
g = 9.81459
lam = 780e-9
keff = 2*2*np.pi/lam
v_s = keff*h_/mRb/2
ty = 10e-6
T = 1510e-6
W0 = np.pi/2/ty
Wg = 5e6
We = 3.5e6
D  = 1e9
tb = 15128e-6
v0z = tb*g
dw0 = keff*(v0z + v_s) # тут может быть минус
rw = 7.5e-3
xc = 0
yc = 0

# Chirp
a1 = 25.010e6
a2 = 25.865e6
na = 100
a_range = np.linspace(a1, a2, na)

# Monte Carlo parameters
T_K = 5.5e-6
v_spread = np.sqrt(kb*T_K/mRb)
s_spread = 0.6e-3
M = 10000
np.random.seed(42)

# ------------------------
# Gaussian Monte Carlo sampling
# ------------------------
vz0_m = np.random.normal(loc=v0z, scale=v_spread, size=M)
vx0_m = np.random.normal(loc=0.0, scale=v_spread, size=M)
vy0_m = np.random.normal(loc=0.0, scale=v_spread, size=M)
X = np.random.normal(loc=xc, scale=s_spread, size=M)
Y = np.random.normal(loc=yc, scale=s_spread, size=M)

# ------------------------
# Fast vectorized functions
# ------------------------
@njit
def Er(r):
    return np.exp(-r**2 / rw**2)*1

@njit
def dw2(t0, a, vz):
    vz = vz + v_s
    return dw0 + 2*np.pi*a*t0 - keff*vz

@njit
def fR2(t0, z, vz, a, Dt, ph, vz0):
    return -keff*z + dw0*t0 + np.pi*a*t0**2 + ph

@njit
def RiM2_single(t0, z, vz, Dt, a, ph, vz0, r):
    W0_loc = W0*Er(r)**2
    Wg_loc = Wg*Er(r)
    We_loc = We*Er(r)
    dtn = Wg_loc**2/D - We_loc**2/D - dw2(t0,a,vz)
    WR = np.sqrt(W0_loc**2 + dtn**2)
    laser_phase = fR2(t0,z,vz,a,Dt,ph,vz0)
    ph_p = np.exp(1j*(Wg_loc**2/D + We_loc**2/D + dw2(t0,a,vz))*Dt/2)
    ph_m = np.exp(1j*(Wg_loc**2/D + We_loc**2/D - dw2(t0,a,vz))*Dt/2)
    M = np.zeros((2,2), dtype=np.complex128)
    M[0,0] = (np.cos(WR*Dt/2)+1j*dtn/WR*np.sin(WR*Dt/2))*ph_p
    M[0,1] = (1j*W0_loc/WR*np.sin(WR*Dt/2)*np.exp(1j*laser_phase)*ph_p)
    M[1,0] = (1j*W0_loc/WR*np.sin(WR*Dt/2)*np.exp(-1j*laser_phase)*ph_m)
    M[1,1] = (np.cos(WR*Dt/2)-1j*dtn/WR*np.sin(WR*Dt/2))*ph_m
    return M

# ------------------------
# Fully vectorized interference
# ------------------------
@njit
def TS_Int_vectorized_full(a_range, vz0_m, vx0_m, vy0_m, X, Y):
    Pa = np.zeros((2, na), dtype=np.float64)
    
    c0 = np.array([1+0j,0+0j], dtype=np.complex128)
    
    for m in range(M):
        vz0 = vz0_m[m]
        vx0 = vx0_m[m]
        vy0 = vy0_m[m]
        x0 = X[m]
        y0 = Y[m]
        r0 = np.sqrt(x0**2 + y0**2)
        
        X1 = vx0*(T+ty) + x0
        Y1 = vy0*(T+ty) + y0
        r1 = np.sqrt(X1**2 + Y1**2)
        X2 = X1 + vx0*(T+2*ty)
        Y2 = Y1 + vy0*(T+2*ty)
        r2 = np.sqrt(X2**2 + Y2**2)
        
        z2I  = (vz0 + 2*v_s)*T + g*T**2/2
        vz2I = vz0 + 2*v_s + g*T
        z2II  = vz0*T + g*T**2/2
        vz2II = vz0 + g*T
        z3II = z2II + (vz2II+2*v_s)*T + g*T*T/2
        vz3I = vz2I - 2*v_s + g*T
        
        for j in range(na):
            a = a_range[j]
            # Первый импульс
            M1 = RiM2_single(0,0,vz0,ty,a,0,vz0,r0)
            c1 = M1 @ c0
            
            # Второй импульс
            c1i = np.array([0+0j, c1[1]], dtype=np.complex128)
            c1ii= np.array([c1[0],0+0j], dtype=np.complex128)
            M2i  = RiM2_single(T,z2I,vz2II,2*ty,a,0,vz0,r1)
            M2ii = RiM2_single(T,z2II,vz2II,2*ty,a,0,vz0,r1)
            c2 = M2i @ c1i + M2ii @ c1ii
            
            # Третий импульс
            M3 = RiM2_single(2*T, z3II, vz3I, ty, a,0,vz0,r2)
            c3 = M3 @ c2
            
            Pa[:,j] += np.abs(c3)/M
    return Pa

# ------------------------
# Fit function
# ------------------------
def sinss(x, A, w, ph, s, al):
    return A*np.sin(w*x+ph) + s + al*(x-a_range[0])

# ------------------------
# Plot and fit
# ------------------------
def show_result_vectorized_full():
    Pa = TS_Int_vectorized_full(a_range, vz0_m, vx0_m, vy0_m, X, Y)
    P2 = Pa[1]
    
    plt.plot(a_range, P2)
    plt.title("Interference")
    plt.xlabel("Chirp rate")
    plt.ylabel("Population")
    
    initial_guess = [(np.max(P2)-np.min(P2))/2, 2*np.pi*T*T, 0, (np.max(P2)-np.min(P2))/2, 35e-10]
    par, _ = curve_fit(sinss, a_range, P2, p0=initial_guess)
    A, w, ph, s, al = par
    V = abs(A)/s
    print("Swing:", np.max(P2)-np.min(P2))
    print("Contrast:", V)
    
    plt.show()

# ------------------------
# Run
# ------------------------
show_result_vectorized_full()