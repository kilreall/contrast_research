import numpy as np 
import matplotlib.pyplot as plt


def maxwell_vz(vz, v0z):      # Функция распределения Максвелла для v_z
    prefactor = np.sqrt(mRb / (2 * np.pi * kb * T))
    exponent = - (mRb * (vz-v0z)**2) / (2 * kb * T)
    return prefactor * np.exp(exponent)



def dw(t, a, vz):
    return dw0 + 2*np.pi*a*t - keff*vz

def fR(r, dw, f, Dt): # laser phase
    return -keff*r+(w0+dw)*Dt+f

def RiM(t0, r, f, Dt, dw): # Rabi impulse Matrix (r(t0))

    We = np.sqrt(WR**2-(dw-dwAC)**2)

    M = np.array([[0,0],[0,0]])

    M[0,0] = np.cos(WR*Dt/2)-1j*(dw-dwAC)/WR*np.sin(WR*Dt/2)
    M[0,1] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(-1j*(dw*t0+fR(r, dw, f, Dt)))
    M[1,0] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(1j*(dw*t0+fR(r, dw, f, Dt)))
    M[1,1] = np.cos(WR*Dt/2)+1j*(dw-dwAC)/WR*np.sin(WR*Dt/2)

    M = M * np.exp(-1j*(WgAC+WeAC-dw)*Dt/2)

    return M

def interference(a, vz):

    M1 = RiM(0, 0, 0, ty, dw(0, a, vz))
    z = vz*T+g*T**2/2
    vz = vz + g*T

    M2 = RiM(T, z, 0, 2*ty, dw(T, a, vz))
    z = z + vz*T+g*T**2/2
    vz = vz + g*T

    M3 = RiM(2*T, z, 0, ty, dw(2*T, a, vz))

    M = M3@M2@M1

    c = M@[1,0]

    return c*c.conjugate()

def tempdist(a):

    vz_d = np.linspace(-Dv+v0z, +Dv+v0z, 100) # скоростной диапазон
    Pvz = []
    for vz in vz_d:
        Pvz.append(interference(a, vz))

    return 1

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
g = 9.8
ty = 20e-6 # pi/2 impulse duration
Dw = 1/ty # Raman pulse width
Dv = Dw*c/(keff*c) # cutted speed width
WR = np.pi/2/ty # Rabi freq
T = 7e-3 # between impulse
dw0 = 0 # start laser detuning
v0z = 0 # начальное смещение по вертикальной скорости
n = 1000 # количество рассчётных точек
dwAC, WgAC, WeAC = 0, 0, 0 # dynamic start AC shifts

