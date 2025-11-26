# 3D for freq width

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


def F(t0, t, Dt):
    sg = Dt/2.355
    R = np.exp(-(t-(t0+Dt/2))**2/2/sg**2)
    return R

def Rim3M(t, c, z, vz, t0, Dt, a, ph, vz0):
    
    dpt = dw0*t + np.pi*a*t**2 - keff*z # doppler

    M = np.zeros((3,3), dtype = complex)

    M[0,2] = -1j*Wg*np.exp(-1j*D*t)

    M[1,2] = -1j*We*np.exp(-1j*(D*t + dpt))

    M[2,0] = 1j*Wg*np.exp(1j*(D*t))

    M[2,1] = -1j*We*np.exp(1j*(D*t + dpt))

    return M@c

def RiM3(c0, t0, z, vz, Dt, a, ph, vz0):
    # c0 теперь массив из 3 комплексных амплитуд: [c1, c2, c3]
    # Переводим комплексное состояние в вещественный вектор длины 6
    y0 = np.concatenate([c0.real, c0.imag])  # [real(c1), real(c2), real(c3), imag(c1), imag(c2), imag(c3)]

    # Определяем реальную систему уравнений
    def fun_real(t, y):
        c = y[:3] + 1j * y[3:]  # Восстанавливаем комплексный массив [c1, c2, c3]
        dc = Rim3M(t, c, z, vz, t0, Dt, a, ph, vz0)  # dc — массив из 3 комплексных производных
        return np.concatenate([dc.real, dc.imag])  # Возвращаем [real(dc1), real(dc2), real(dc3), imag(dc1), imag(dc2), imag(dc3)]

    # Решаем ОДУ
    sol = solve_ivp(fun_real, (t0, t0 + Dt), y0, t_eval=None,
                    method='BDF')

    # Финальное комплексное состояние
    c_end = sol.y[:3, -1] + 1j * sol.y[3:, -1]  # [c1_end, c2_end, c3_end]

    return c_end

def test(ty):
    ty_r = np.linspace(0, 4*ty, 2)
    P2 = []
    for tyi in ty_r:
        c0 = np.array([1,0,0], dtype=complex)
        c = RiM3(c0, 0, 0, 0, tyi, 0, 0, 0)
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
v_s = keff*h_/mRb/2 # переданная фотонами половина скорость

# experimental parameters
g = 9.81459
ty = 20e-6 # pi/2 impulse duration
# a1 = 500000. # start chirp # 210 mks
# a2 = 50500000. # # end chirp # 210 mks
a1 = 25.050e6 # start chirp # 210 mks
a2 = 25.225e6 # # end chirp # 210 mks
na = 100# chirp points
a_range = np.linspace(a1, a2, na)
x0 = a_range[0]
nT = 300
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
W0 = np.pi/2/ty # Rabi freq 78539
T = 5e-3 # between impulse
v0z = 0 #-v_s # начальное смещение по вертикальной скорости
dw0 = 0 # start laser detuning
n = 1000 # количество рассчётных точек
Wg, We = 5e6, 3.5e6 # Rabi freq for one photon transition
D = 1e9
T_K = 5.5e-6
v_spread = 2*np.sqrt(3*kb*T_K/mRb)


print(keff*h_/mRb*keff/W0/2)
#print(keff*g/2/np.pi, "a0")
print(keff*g/2/np.pi*1e-6)
print(np.sqrt(3*kb*T_K/mRb)*1e3)
print(dw0)

test(ty)