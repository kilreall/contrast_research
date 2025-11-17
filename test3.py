# test population from impulse duration with v_s ver 1

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def dw1(t0, a, vz):

    return dw0 + 2*np.pi*a*t0 - keff*vz

def RiM1f(t, c, a, t0, z, vz_1, ph_1, ph_2): # Rabi impulse Matrix 1 first # индекс_i отвечает за состояние

    M = np.zeros((2, 2), dtype=complex)

    # M[0,1] = We*np.exp(1j*(dw1(t0, a, vz_1)*t-keff/2*z_1+ph_1))
    # M[1,0] = We*np.exp(-1j*(dw1(t0, a, vz_2)*t+keff/2*z_2+ph_2))

    # phi_1 = dw0*t + np.pi*a*t*t - keff*vz_1*t - keff/2*z_1 + ph_1
    # phi_2 = dw0*t + np.pi*a*t*t - keff*vz_2*t + keff/2*z_2 + ph_2
    phi_1 = (dw0 + 2*np.pi*a*t0 -keff*vz_1-2.5*keff*v_s)*t - keff/2*z + ph_1
    phi_2 = (dw0 + 2*np.pi*a*t0 -keff*vz_1+0.5*keff*v_s)*t + keff/2*z + ph_2

    M[0,1] = We * np.exp(1j * phi_1) 
    M[1,0] = We * np.exp(-1j * phi_2)

    M *= -1j/2 # тут вопрос

    return M@c

def RiM1s(c0, a, t0, Dt, z, vz_1, ph_1, ph_2):

    # переводим комплексное состояние в вещественный вектор
    y0 = np.concatenate([c0.real, c0.imag])

    # определяем реальную систему уравнений
    def fun_real(t, y):
        c = y[:2] + 1j * y[2:]                # восстанавливаем комплекс
        dc = RiM1f(t, c, a, t0, z, vz_1, ph_1, ph_2)
        return np.concatenate([dc.real, dc.imag])  # возвращаем real + imag

    # решаем
    sol = solve_ivp(fun_real, (t0, t0 + Dt), y0, t_eval=None,
                    method='DOP853', rtol=1e-8, atol=1e-10)

    # финальное комплексное состояние
    c_end = sol.y[:2, -1] + 1j * sol.y[2:, -1]

    return c_end

def test(ty):
    
    ty_r = np.linspace(0, 4*ty, 100)
    P2 = []
    for tyi in ty_r:
        c0 = np.array([1,0], dtype=complex)
        c = RiM1s(c0, 0, 0, tyi, 0, 0, 0, 0)
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