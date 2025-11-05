import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def maxwell_vz(vz, v0z):      # Функция распределения Максвелла для v_z
    prefactor = np.sqrt(mRb / (2 * np.pi * kb * T))
    exponent = - (mRb * (vz-v0z)**2) / (2 * kb * T)
    return prefactor * np.exp(exponent)


def dw1(t0, a, vz):

    return dw0 + 2*np.pi*a*t0 - keff*vz

def RiM1f(t, c, a, t0, z_1, z_2, vz_1, vz_2, ph_1, ph_2): # Rabi impulse Matrix 1 first # индекс_i отвечает за состояние

    M = np.zeros((2, 2), dtype=complex)

    # M[0,1] = We*np.exp(1j*(dw1(t0, a, vz_1)*t-keff/2*z_1+ph_1))
    # M[1,0] = We*np.exp(-1j*(dw1(t0, a, vz_2)*t+keff/2*z_2+ph_2))

    phi_1 = dw0*t + np.pi*a*t*t - keff*vz_1*t - keff/2*z_1 + ph_1
    phi_2 = dw0*t + np.pi*a*t*t - keff*vz_2*t + keff/2*z_2 + ph_2



    M[0,1] = We * np.exp(1j * phi_1)
    M[1,0] = We * np.exp(-1j * phi_2)

    M *= -1j/2 # тут вопрос

    return M@c

def RiM1s(c0, a, t0, Dt, z_1, z_2, vz_1, vz_2, ph_1, ph_2):

    # переводим комплексное состояние в вещественный вектор
    y0 = np.concatenate([c0.real, c0.imag])

    # определяем реальную систему уравнений
    def fun_real(t, y):
        c = y[:2] + 1j * y[2:]                # восстанавливаем комплекс
        dc = RiM1f(t, c, a, t0, z_1, z_2, vz_1, vz_2, ph_1, ph_2)
        return np.concatenate([dc.real, dc.imag])  # возвращаем real + imag

    # решаем
    sol = solve_ivp(fun_real, (t0, t0 + Dt), y0, t_eval=None,
                    method='DOP853', rtol=1e-8, atol=1e-10)

    # финальное комплексное состояние
    c_end = sol.y[:2, -1] + 1j * sol.y[2:, -1]

    return c_end

def interference1(a, vz0):

    c0 = np.array([1,0], dtype=complex)
    c1 = RiM1s(c0, a, t0=0, Dt=ty, z_1=0, z_2=0, vz_1=vz0, vz_2=vz0+v_s, ph_1=0, ph_2=0)

    z2I = (vz0 + v_s)*T + g*T**2/2
    vz2I = vz0 + v_s + g*T

    z2II = vz0*T + g*T**2/2
    vz2II = vz0 + g*T

    c2i = np.array([0, c1[1]], dtype=complex)
    c2ii = np.array([c1[0], 0], dtype=complex)
    c2i = RiM1s(c2i, a, t0=T, Dt=2*ty, z_1=z2I, z_2=z2I, vz_1=vz2II, vz_2=vz2I, ph_1=0, ph_2=0)
    #c2i = c2i/np.sqrt(np.sum(np.abs(c2i)**2))*np.abs(c1[1])
    c2ii = RiM1s(c2ii, a, t0=T, Dt=2*ty, z_1=z2II, z_2=z2II, vz_1=vz2II, vz_2=vz2I, ph_1=0, ph_2=0)
    #c2ii = c2ii/np.sqrt(np.sum(np.abs(c2ii)**2))*np.abs(c1[0])

    c2 = np.array([c2i[0], c2ii[1]], dtype=complex)

    z3I = z2I + (vz2I-v_s)*T+g*T*T/2
    vz3I = vz2I - v_s + g*T

    z3II = z2II + (vz2II+v_s)*T+g*T**2/2
    vz3II = vz2II + v_s + g*T

    c3 = RiM1s(c2, a, t0=2*T, Dt=ty, z_1=z3I, z_2=z3II, vz_1=vz3I, vz_2=vz3II, ph_1=0, ph_2=0)
    #c3 = c3/np.sqrt(np.sum(np.abs(c3)**2))*np.sqrt(np.sum(np.abs(c2)**2))
    
    P3 = np.abs(c3)**2
    # cf = np.array([c3[0], c3[1], c2i[1], c2ii[0]], dtype=complex)
    # nr = np.abs(cf)**2
    # nr = np.sum(nr)
    # P3 = P3/nr

    return P3


# def dw(t, a): # отстройка в СО атома без учёта его скорости

#     return dw0 + 2*np.pi*a*t

# def fR(r, dw, f, t0): # laser phase
#     return -keff*r+(w0+dw)*t0+f # тут Dt? не t0

# def RiM(t0, r, vz, f, Dt, a): # Rabi impulse Matrix (r(t0))

#     dw1 = dw(t0, a) 
#     dw2 = dw1 - keff*vz # отстройка в СО атома c учётом его скорости
#     WR = np.sqrt(We**2+(dw2-dwAC)**2)

#     M = np.zeros((2, 2), dtype=complex)

#     M[0,0] = np.cos(WR*Dt/2)-1j*(dw2-dwAC)/WR*np.sin(WR*Dt/2)
#     M[0,1] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(-1j*(dw2*t0+fR(r, dw1, f, t0)))
#     M[1,0] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(1j*(dw2*t0+fR(r, dw1, f, t0)))
#     M[1,1] = np.cos(WR*Dt/2)+1j*(dw2-dwAC)/WR*np.sin(WR*Dt/2)

#     M = M * np.exp(-1j*(WgAC+WeAC-dw2)*Dt/2)

#     return M

# def interference(a, vz0):
    
#     c0 = np.array([1,0])
#     M1 = RiM(0, 0, vz0, 0, ty, a)
#     c1 = M1@c0

#     z2I = (vz0 + keff*h_/mRb)*T + g*T**2/2
#     vz2I = vz0 + keff*h_/mRb + g*T

#     z2II = vz0*T + g*T**2/2
#     vz2II = vz0 + g*T

#     M2I = RiM(T, z2I,  vz2I, 0, 2*ty, a)
#     c2I = M2I@np.array([0,c1[1]])


#     M2II = RiM(T, z2II, vz2II, 0, 2*ty, a)
#     c2II = M2II@np.array([c1[0],0])

#     c2 = c2I + c2II

#     z3I = z2I + (vz2I-keff*h_/mRb)*T+g*T*T/2
#     vz3I = vz2I - keff*h_/mRb + g*T

#     z3II = z2II + (vz2II+keff*h_/mRb)*T+g*T**2/2
#     vz3II = vz2I + keff*h_/mRb + g*T

#     M3I = RiM(2*T, z3I, vz3I, 0, ty, a)
#     c3I = M3I@np.array([c2[0],0])

#     M3II = RiM(2*T, z3II, vz3II, 0, ty, a)
#     c3II = M3II@np.array([0,c2[1]])

#     c3 = c3I + c3II
#     P3 = np.abs(c3)**2
#     #P3 = P3/np.sum(P3)

#     return P3

def tempdist(a):

    vz_d = np.linspace(-Dv+v0z, +Dv+v0z, 100) # скоростной диапазон
    Pvz = []
    for vz in vz_d:
        Pvz.append(interference1(a, vz))

    return 1

def chirp(a1, a2, na):
    vz0 = 0
    P1 = []
    P2 = []
    a_range = np.linspace(a1, a2, na)
    for a in a_range:
        Pa = interference1(a, vz0)
        P1.append(Pa[0])
        P2.append(Pa[1])

    P1 = np.array(P1)
    P2 = np.array(P2)

    plt.plot(a_range, P2)
    plt.title("interference")
    plt.xlabel("chirp rate")
    plt.ylabel("Population")
    print(np.max(P2)- np.min(P2), "swing")
    print((np.max(P2)- np.min(P2))/(np.max(P2) + np.min(P2)), "contrast")

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
v_s = keff*h_/mRb # переданная фотоном скорость

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

#print(keff*h_/mRb*keff/We)
print(keff*g/2/np.pi, "a0")
chirp(a1, a2, na)

