# test population frim detuning with vs

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dw1(t, a, vz):

    return dw0 + 2*np.pi*a*t - keff*vz

def RiM1f(t, c, a, z_1, z_2, vz_1, vz_2, ph_1, ph_2, dw): # Rabi impulse Matrix 1 first # индекс_i отвечает за состояние

    M = np.zeros((2, 2), dtype=complex)

    # M[0,0] = 0
    # M[0,1] = We*np.exp(1j*(dw*t-keff/2*z_1+ph_1))
    # M[1,0] = We*np.exp(-1j*(dw*t+keff/2*z_2+ph_2))
    # M[1,1] = 0

    phi_1 = dw0*t + np.pi*a*t*t - keff*vz_1*t - keff/2*z_1 + ph_1+dw*t
    phi_2 = dw0*t + np.pi*a*t*t - keff*vz_2*t + keff/2*z_2 + ph_2+dw*t



    M[0,1] = We * np.exp(1j * phi_1)
    M[1,0] = We * np.exp(-1j * phi_2)

    M *= -1j/2 # тут вопрос

    return M@c

def RiM1s(c0, t, Dt, a, z_1, z_2, vz_1, vz_2, ph_1, ph_2, dw): # Rabi impulse Matrix 1 solving

    t_span = (t, t+Dt)
    t_eval = np.linspace(t, t+Dt, 1000)

    sol = solve_ivp(lambda t, c: RiM1f(t, c, a, z_1, z_2, vz_1, vz_2, ph_1, ph_2, dw), t_span, c0, t_eval=t_eval, method='RK45')

    # Извлекаем результаты
    t = sol.t
    c1 = sol.y[0]  # c1(t)
    c2 = sol.y[1]  # c2(t)
    c1 = c1[-1]
    c2 = c2[-1]

    return np.array([c1, c2], dtype=complex)

def test():
    c0 = np.array([1,0], dtype=complex)
    dw = np.linspace(-10*We, +10*We, 500)
    P = []
    for dwi in dw:

        ci = RiM1s(c0, 0, 2*ty, 0, 0, 0, 0, 0, 0, 0, dwi)
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