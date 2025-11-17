# test population from impulse duration with v_s ver 1

import numpy as np 
import matplotlib.pyplot as plt

def dw1(t0, a, vz):
    return dw0 + 2*np.pi*a*t0 - keff*vz


def RiM(t0, r, vz, f, Dt, a, dw): # Rabi impulse Matrix (r(t0))

    # dw1 = dw(t0, a) 
    # dw2 = dw1 - keff*vz # отстройка в СО атома c учётом его скорости
    WR = np.sqrt(We**2+(dw-dwAC)**2)

    M = np.zeros((2, 2), dtype=complex)

    M[0,0] = np.cos(WR*Dt/2)-1j*(dw-dwAC)/WR*np.sin(WR*Dt/2)
    M[0,1] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(-1j*(dw*t0+fR(r, dw, f, t0)))
    M[1,0] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(1j*(dw*t0+fR(r, dw, f, t0)))
    M[1,1] = np.cos(WR*Dt/2)+1j*(dw-dwAC)/WR*np.sin(WR*Dt/2)

    M = M * np.exp(-1j*(WgAC+WeAC-dw)*Dt/2)

    return M

def test(ty):
    
    ty_r = np.linspace(0, 4*ty, 100)
    P2 = []
    for tyi in ty_r:
        c0 = np.array([1,0], dtype=complex)
        c = RiM(c0, 0, 0, tyi, 0, 0, 0, 0)
        P = np.abs(c)**2
        P2.append(np.sum(P))

    plt.plot(ty_r/2/ty, P2)
    plt.xlabel("Impulse duration, [pi impulse]")
    plt.ylabel("Population")
    plt.show()

# constants
c = 3e8
mRb = 1.46e-25
h_ = 1.054571817e-34
kb = 1.38e-23
Y = 2*np.pi*6.06e6
lam = 780e-9
keff = 2*2*np.pi/lam
w0 = 6.8*1e9*2*np.pi # частота сверхтонкого перехода
v_s = keff*h_/mRb # половина переданной фотономи скорости

# experimental parameters
ty = 20e-6 # pi/2 impulse duration
a1 = 25.025e6 # start chirp
a2 = 25.225e6 # end chirp
na = 505# chirp points
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
We = np.pi/2/ty # Rabi freq 78539
T = 5e-3 # between impulse
dw0 = 0. # start laser detuning
v0z = 0. # начальное смещение по вертикальной скорости
n = 1000 # количество рассчётных точек
dwAC, WgAC, WeAC = 0., 0., 0. # dynamic start AC shifts
g = 9.81459


print(keff*h_*keff/mRb)
test(ty)