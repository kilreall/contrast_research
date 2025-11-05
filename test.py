import numpy as np
import matplotlib.pyplot as plt

def fR(r, dw, f, t0): # laser phase
    return -keff*r+(w0+dw)*t0+f # тут Dt? не t0

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

def test():
    c0 = np.array([1,0], dtype=complex)
    dw = np.linspace(-10*We, +10*We, 500)
    P = []
    for dwi in dw:

        ci = RiM(10, 0, 0, 0, 2*ty, 0, dwi)@c0
        Pi = np.abs(ci)**2
        Pi = Pi/np.sum(Pi)
        P.append(Pi[1])

    P = np.array(P)
    plt.plot(dw, P)
    plt.show()


# constants
c = 3e8
mRb = 1.46e-25
h_ = 1e-34
kb = 1.38e-23
Y = 2*np.pi*6.06e6
lam = 780e-9
keff = 2*2*np.pi/lam
w0 = 6.8*1e9*2*np.pi # частота сверхтонкого перехода


# experimental parameters
ty = 20e-6 # pi/2 impulse duration
a1 = 25.025e6 # start chirp
a2 = 25.225e6 # end chirp
na = 500 # chirp points
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
We = np.pi/2/ty # Rabi freq 78539
T = 3e-3 # between impulse
dw0 = 0. # start laser detuning
v0z = 0. # начальное смещение по вертикальной скорости
n = 1000 # количество рассчётных точек
dwAC, WgAC, WeAC = 0., 0., 0. # dynamic start AC shifts
g = 9.8

test()