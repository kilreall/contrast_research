# check interference model using diffirential equation

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def sinss(x, A, w, ph, s, al):
    return A*np.sin(w*x+ph) + s + al*(x-x0)


def Rim3_s(t, c, t0, z, a, ph):

    c_1r, c_1i, c_2r, c_2i = c
    c_1 = c_1r + 1j*c_1i
    c_2 = c_2r + 1j*c_2i

    phase = dw0*t - keff*z + np.pi*a*t + ph

    M11 = 1j * np.abs(Wg)**2 /D 
    M12 = 1j * W0/2 * np.exp(1j*phase)
    M21 = 1j * W0/2 * np.exp(-1j*phase)
    M22 = 1j * np.abs(We)**2/D


    # Производные
    dc_1 = M11 * c_1 + M12 * c_2
    dc_2 = M21 * c_1 + M22 * c_2

    # Возвращаем реальные и мнимые части производных
    return [dc_1.real, dc_1.imag, dc_2.real, dc_2.imag]

def RiM3(t0, c_1, c_2, z, Dt, a, ph):

    c = [c_1.real, c_1.imag, c_2.real, c_2.imag]

    # Интегрирование ОДУ от t0 до t0 + Dt
    t_span = (t0, t0 + Dt)
    sol = solve_ivp(Rim3_s, t_span, c, args=(t0, z, a, ph),
                    method='RK45', rtol=1e-6, atol=1e-8)

    # Извлечение финальных комплексных амплитуд
    c_1_final = sol.y[0][-1] + 1j * sol.y[1][-1]
    c_2_final = sol.y[2][-1] + 1j * sol.y[3][-1]

    return c_1_final, c_2_final

def interference2(a, vz0):

    c0 = np.array([1,0], dtype=complex)
    c1 = RiM3(0, c0[0], c0[1], 0, ty, a, 0)

    z2I = (vz0 + 2*v_s)*T + g*T**2/2
    vz2I = vz0 + 2*v_s + g*T

    z2II = vz0*T + g*T**2/2
    vz2II = vz0 + g*T

    c1i = np.array([0, c1[1]], dtype=complex)
    c1ii = np.array([c1[0], 0], dtype=complex)

    c2i = RiM3(T, c1i[0], c1i[1], z2I, 2*ty, a, 0)
    
    c2ii = RiM3(T, c1ii[0], c1ii[1], z2II, 2*ty, a, 0)


    c2 = np.array([c2i[0], c2ii[1]], dtype=complex)

    z3I = z2I + (vz2I-2*v_s)*T+g*T*T/2
    vz3I = vz2I - 2*v_s + g*T

    z3II = z2II + (vz2II + 2*v_s)*T+g*T**2/2
    vz3II = vz2II + 2*v_s + g*T

    c3 = RiM3(2*T, c2[0], c2[1], z3II, ty, a, 0) # переход для разных путей считается одновременно
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


def show_result():

    Pa = chirp2(-v_s)
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
v_s = keff*h_/mRb/2 # переданная фотонами половина скорости

# experimental parameters
g = 9.81459
ty = 0.25e-6 # pi/2 impulse duration
# a1 = 500000. # start chirp # 210 mks
# a2 = 50500000. # # end chirp # 210 mks
a1 = 25.1655e6 # start chirp # 210 mks
a2 = 25.165650e6 # # end chirp # 210 mks
na = 100# chirp points
a_range = np.linspace(a1, a2, na)
x0 = a_range[0]
nT = 300
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
W0 = np.pi/2/ty # Rabi freq 78539
T = 120e-3 # between impulse
v0z = 15128e-6*g*0 - v_s # начальное смещение по вертикальной скорости
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
