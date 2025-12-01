import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def sinss(x, A, w, ph, s, al):
    return A*np.sin(w*x+ph) + s + al*(x-x0)

def f_vz(vz, v0z, T_K):      # Функция распределения Максвелла для v_z
    prefactor = np.sqrt(mRb / (2 * np.pi * kb * T_K))
    exponent = - (mRb * (vz-v0z)**2) / (2 * kb * T_K)
    return prefactor * np.exp(exponent)

def F(t0, t, Dt):
    sg = Dt/2.355
    R = np.exp(-(t-(t0+Dt/2))**2/2/sg**2)
    return 1

def Rim3M(t, c, z, vz, t0, Dt, a, ph, vz0):
    
    detPh = -keff*z + dw0*t0 + np.pi*a*t0**2 + ph - keff*(vz + vz + g*(t-t0) + 2*v_s)/2*(t-t0)*0

    M = np.zeros((2,2), dtype = complex)

    M[0,0] = 1j * (Wg*F(t0, t, Dt))**2/D
    M[0,1] = 1j * (W0*F(t0, t, Dt)**2)/2 * np.exp(1j*detPh)
    M[1,0] = 1j * (W0*F(t0, t, Dt)**2)/2 * np.exp(-1j*detPh)
    M[1,1] = 1j * (We*F(t0, t, Dt))**2/D

    return M@c

def RiM3(c0, t0, z, vz, Dt, a, ph, vz0):

    # переводим комплексное состояние в вещественный вектор
    y0 = np.concatenate([c0.real, c0.imag])

    # определяем реальную систему уравнений
    def fun_real(t, y):
        c = y[:2] + 1j * y[2:]                # восстанавливаем комплекс
        dc = Rim3M(t, c, z, vz, t0, Dt, a, ph, vz0)
        return np.concatenate([dc.real, dc.imag])  # возвращаем real + imag

    # решаем
    sol = solve_ivp(fun_real, (t0, t0 + Dt), y0, t_eval=None,
                    method='DOP853', rtol=1e-8, atol=1e-10)

    # финальное комплексное состояние
    c_end = sol.y[:2, -1] + 1j * sol.y[2:, -1]

    return c_end


def interference2(a, vz0):

    c0 = np.array([1,0], dtype=complex)
    c1 = RiM3(c0, 0, 0, vz0, ty, a, 0, vz0)

    z2I = (vz0 + 2*v_s)*T + g*T**2/2
    vz2I = vz0 + 2*v_s + g*T

    z2II = vz0*T + g*T**2/2
    vz2II = vz0 + g*T

    c1i = np.array([0, c1[1]], dtype=complex)
    c1ii = np.array([c1[0], 0], dtype=complex)

    c2i = RiM3(c1i, T, z2I, vz2II, 2*ty, a, 0, vz0)
    
    c2ii = RiM3(c1ii, T, z2II, vz2II, 2*ty, a, 0, vz0)


    c2 = np.array([c2i[0], c2ii[1]], dtype=complex)

    z3I = z2I + (vz2I-2*v_s)*T+g*T*T/2
    vz3I = vz2I - 2*v_s + g*T

    z3II = z2II + (vz2II + 2*v_s)*T+g*T**2/2
    vz3II = vz2II + 2*v_s + g*T

    c3 = RiM3(c2, 2*T, z3II, vz3I, ty, a, 0, vz0) # переход для разных путей считается одновременно
    #c3i = RiM2(2*T, z3I, vz3I, ty, a, 0, vz0)@np.array([c2i[0], 0], dtype=complex)

    #c3ii = RiM2(2*T, z3II, vz3I, ty, a, 0, vz0)@np.array([0, c2ii[1]], dtype=complex)

    #c3 = c3i + c3ii

    P3 = np.abs(c3)**2

    return P3


def chirp2(vz0):
    P1 = []
    P2 = []
    for a in a_range:
        Pa = interference2(a, vz0)
        P1.append(Pa[0])
        P2.append(Pa[1])

    P1 = np.array(P1)
    P2 = np.array(P2)

    # plt.plot(a_range, P2)
    # plt.title("interference")
    # plt.xlabel("chirp rate")
    # plt.ylabel("Population")
    # print(np.max(P2)- np.min(P2), "swing")
    # print((np.max(P2)- np.min(P2))/(np.max(P2) + np.min(P2)), "contrast")

    # plt.show()

    return np.array([P1, P2])

def T_Int():

    vz0_m = np.linspace(-6*v_spread, 6*v_spread, nT)
    dvz_s = 12*v_spread/(nT-1)
    Pa = np.zeros((2, na))
    for vz0 in vz0_m:
        Pa += chirp2(vz0)*f_vz(vz0, v0z, T_K)*dvz_s

    return Pa

def show_result():

    Pa = T_Int()
    P1 = Pa[0]
    P2 = Pa[1]
    plt.plot(a_range, P2)
    plt.title("interference")
    plt.xlabel("chirp rate")
    plt.ylabel("Population")

    initial_guess = [(np.max(P2) - np.min(P2))/2, 2*np.pi*T*T, 0, (np.max(P2) - np.min(P2))/2, 35e-10] 
    par, cov = curve_fit(sinss, a_range, P2, p0=initial_guess)
    A, w, ph, s, al = par
    V = abs(A)/s

    print(np.max(P2)- np.min(P2), "swing")
    print(V, "contrast")

    plt.show()

    return 1


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
W0 = np.pi/2/ty*1.2 # Rabi freq 78539
T = 5e-3 # between impulse
v0z = 15128e-6*g #-v_s # начальное смещение по вертикальной скорости
dw0 = keff*(v0z + v_s)*1 # start laser detuning
n = 1000 # количество рассчётных точек
Wg, We = 5e6, 3.5e6 # dynamic start AC shifts
D = 1e9
T_K = 5.5e-6
v_spread = 2*np.sqrt(3*kb*T_K/mRb)


print(keff*h_/mRb*keff/W0/2)
#print(keff*g/2/np.pi, "a0")
print(keff*g/2/np.pi*1e-6)
print(np.sqrt(3*kb*T_K/mRb)*1e3)
print(dw0)

show_result()