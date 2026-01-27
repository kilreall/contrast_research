import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit, prange

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
T = 5e-3
W0 = np.pi/2/ty
Wg = 5e6*0
We = 3.5e6*0
D  = 1e9
tb = 15128e-6
v0z = tb*g
dw0 = keff*(v0z + v_s)
rw = 7.5e-3
xc = 0.0
yc = 0.0

# Chirp
a0 = keff*g/(2*np.pi)
a1 = a0 - 1/(4*T*T)
a2 = a0 + 2/(T*T)
na = 100
a_range = np.linspace(a1, a2, na)

# Monte Carlo
T_K = 5.5e-6
v_spread = np.sqrt(kb*T_K/mRb)
s_spread = 0.6e-3
M = 100000
np.random.seed(42)

vz0_m = np.random.normal(v0z, v_spread, M)
vx0_m = np.random.normal(0.0, v_spread, M)
vy0_m = np.random.normal(0.0, v_spread, M)
X = np.random.normal(xc, s_spread, M)
Y = np.random.normal(yc, s_spread, M)

# ------------------------
# Physics helpers
# ------------------------

@njit
def Pe(vz):
    return W0**2/(W0**2 + dw2(0.0, 0.0, vz)**2)*0.0 + 1.0

@njit
def Er(r):
    return np.exp(-r*r/(rw*rw))

@njit
def dw2(t0, a, vz):
    return dw0 + 2*np.pi*a*t0 - keff*(vz + v_s)

@njit
def fR2(t0, z, vz, a, Dt, ph, vz0):
    return -keff*z + dw0*t0 + np.pi*a*t0*t0 + ph

# ------------------------
# RiM without matrices (exact physics)
# ------------------------
@njit(cache=True, fastmath=True)
def RiM2_apply(c0, c1, t0, z, vz, Dt, a, ph, vz0, r):

    Er_loc = Er(r)
    W0_loc = W0 * Er_loc * Er_loc
    Wg_loc = Wg * Er_loc
    We_loc = We * Er_loc

    dtn = Wg_loc*Wg_loc/D - We_loc*We_loc/D - dw2(t0, a, vz)
    WR = np.sqrt(W0_loc*W0_loc + dtn*dtn)

    laser_phase = fR2(t0, z, vz, a, Dt, ph, vz0)

    ph_p = np.exp(1j*(Wg_loc*Wg_loc/D + We_loc*We_loc/D + dw2(t0,a,vz))*Dt*0.5)
    ph_m = np.exp(1j*(Wg_loc*Wg_loc/D + We_loc*We_loc/D - dw2(t0,a,vz))*Dt*0.5)

    s = np.sin(WR*Dt*0.5)
    c = np.cos(WR*Dt*0.5)

    a00 = (c + 1j*dtn/WR*s)*ph_p
    a01 = (1j*W0_loc/WR*s*np.exp(1j*laser_phase))*ph_p
    a10 = (1j*W0_loc/WR*s*np.exp(-1j*laser_phase))*ph_m
    a11 = (c - 1j*dtn/WR*s)*ph_m

    return (
        a00*c0 + a01*c1,
        a10*c0 + a11*c1
    )

# ------------------------
# Optimized interference (same interface!)
# ------------------------
@njit(cache=True, parallel=True, fastmath=True)
def TS_Int_vectorized_full(a_range, vz0_m, vx0_m, vy0_m, X, Y):

    Pa = np.zeros((2, na))

    for m in prange(M):

        vz0 = vz0_m[m]
        vx0 = vx0_m[m]
        vy0 = vy0_m[m]
        x0 = X[m]
        y0 = Y[m]

        r0 = np.sqrt(x0*x0 + y0*y0)

        X1 = vx0*(T+ty) + x0
        Y1 = vy0*(T+ty) + y0
        r1 = np.sqrt(X1*X1 + Y1*Y1)

        X2 = X1 + vx0*(T+2*ty)
        Y2 = Y1 + vy0*(T+2*ty)
        r2 = np.sqrt(X2*X2 + Y2*Y2)

        z2I  = (vz0 + 2*v_s)*T + 0.5*g*T*T
        vz2I = vz0 + 2*v_s + g*T
        z2II = vz0*T + 0.5*g*T*T
        vz2II = vz0 + g*T
        z3II = z2II + (vz2II+2*v_s)*T + 0.5*g*T*T
        vz3I = vz2I - 2*v_s + g*T

        weight = Pe(vz0) / M

        for j in range(na):

            a = a_range[j]

            c0 = 1.0 + 0j
            c1 = 0.0 + 0j

            c0, c1 = RiM2_apply(c0, c1, 0.0, 0.0, vz0, ty, a, 0.0, vz0, r0)

            c1i0, c1i1 = 0.0+0j, c1
            c1ii0, c1ii1 = c0, 0.0+0j

            ci0, ci1   = RiM2_apply(c1i0,  c1i1,  T, z2I,  vz2II, 2*ty, a, 0.0, vz0, r1)
            cii0,cii1  = RiM2_apply(c1ii0, c1ii1, T, z2II, vz2II, 2*ty, a, 0.0, vz0, r1)

            c0 = ci0 + cii0
            c1 = ci1 + cii1

            c0, c1 = RiM2_apply(c0, c1, 2*T, z3II, vz3I, ty, a, 0.0, vz0, r2)

            Pa[0, j] += (c0.real*c0.real + c0.imag*c0.imag) * weight
            Pa[1, j] += (c1.real*c1.real + c1.imag*c1.imag) * weight

    return Pa

# ------------------------
# Fit function (НЕ ТРОНУТА)
# ------------------------
def sinss(x, A, w, ph, s, al):
    return A*np.sin(w*x+ph) + s + al*(x-a_range[0])

# ------------------------
# Plot and fit (НЕ ТРОНУТ)
# ------------------------
def show_result_vectorized_full():
    Pa = TS_Int_vectorized_full(a_range, vz0_m, vx0_m, vy0_m, X, Y)
    P2 = Pa[1]

    plt.plot(a_range, P2)
    plt.title("Interference")
    plt.xlabel("Chirp rate")
    plt.ylabel("Population")

    try:
        initial_guess = [(np.max(P2)-np.min(P2))/2,
                         2*np.pi*T*T,
                         0,
                         (np.max(P2)-np.min(P2))/2,
                         35e-10]
        par, _ = curve_fit(sinss, a_range, P2, p0=initial_guess)
        A, w, ph, s, al = par
        V = abs(A)/s
        print("Swing:", np.max(P2)-np.min(P2))
        print("Contrast:", V)
    except:
        V = (np.max(P2)-np.min(P2))/(np.max(P2)+np.min(P2))

    plt.show()

# ------------------------
# Run
# ------------------------
show_result_vectorized_full()