# test population from impulse duration with v_s ver 1

import numpy as np 
import matplotlib.pyplot as plt

def dw(t0, a, vz):
    vz = vz+v_s
    return dw0 + 2*np.pi*a*t0 - keff*vz

def fR(t0, r, dw, ph):
    return -keff * r + (w0 + dw) * t0 + ph

def RiM(t0, r, ph, Dt, dw):

    dtn = (Wg**2 - We**2) / D - dw
    WR = np.sqrt(W0**2 + dtn**2)

    laser_phase = fR(t0, r, dw, ph)

    # динамические фазы
    ph_p  = np.exp(1j * (Wg**2/D + We**2/D + dw) * Dt/2)
    ph_m = np.exp(1j * (Wg**2/D + We**2/D - dw) * Dt/2)

    M = np.zeros((2, 2), dtype=complex)

    # ВАЖНО: фазовый множитель умножает всю амплитуду, везде стоят скобки!
    M[0,0] = (np.cos(WR*Dt/2) + 1j*dtn/WR*np.sin(WR*Dt/2)) * ph_p
    M[0,1] = (1j * W0/WR * np.sin(WR*Dt/2)
              * np.exp(1j * laser_phase)
              * ph_p)

    M[1,0] = (1j * W0/WR * np.sin(WR*Dt/2)
              * np.exp(-1j * laser_phase)
              * ph_m)

    M[1,1] = (np.cos(WR*Dt/2) - 1j*dtn/WR*np.sin(WR*Dt/2)) * ph_m

    return M

def test(ty):
    
    ty_r = np.linspace(0, 4*ty, 100)
    P2 = []
    for tyi in ty_r:
        c0 = np.array([1,0], dtype=complex)
        c = RiM(0, 0, 0, tyi, dw(0, 0, -v_s))@c0
        P = np.abs(c)**2
        P2.append(P[1])

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
v_s = keff*h_/mRb/2 # половина переданной фотонами скорости

# experimental parameters
ty = 20e-6 # pi/2 impulse duration
a1 = 25.025e6 # start chirp
a2 = 25.225e6 # end chirp
na = 505# chirp points
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
W0= np.pi/2/ty # Rabi freq 78539
T = 5e-3 # between impulse
dw0 = 0. # start laser detuning
v0z = 0. # начальное смещение по вертикальной скорости
n = 1000 # количество рассчётных точек
Wg, We = 5e6, 3.5e6 # dynamic start AC shifts
D = 1e9
g = 9.81459


test(ty)