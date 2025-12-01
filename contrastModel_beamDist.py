import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Функция зависимости E(r) для гауссового пучка
def Er(r):
    return np.exp(-r**2 / rw**2)

def sinss(x, A, w, ph, s, al):
    return A*np.sin(w*x+ph) + s + al*(x-x0)

def f_vz(vz, v0z):      # Функция распределения Максвелла для v_z
    prefactor = np.sqrt(mRb / (2 * np.pi * kb * T_K))
    exponent = - (mRb * (vz-v0z)**2) / (2 * kb * T_K)
    return prefactor * np.exp(exponent)


def f_r(x, sc):
    return 1/s_spread*np.exp(-(x-sc)**2/s_spread**2)


def dw2(t0, a, vz):
    vz = vz+v_s
    return dw0 + 2*np.pi*a*t0 - keff*vz

def fR2(t0, z, vz, a, Dt, ph, vz0):
    return  -keff*z + dw0*t0 + np.pi*a*t0**2 + ph

def RiM2(t0, z, vz, Dt, a, ph, vz0, r):

    W0, Wg, We = W0*Er(r)**2, Wg*Er(r), We*Er(r)

    dtn = Wg**2/D - We**2/D - dw2(t0, a, vz)
    WR = np.sqrt(W0**2 + dtn**2)

    laser_phase = fR2(t0, z, vz, a, Dt, ph, vz0)

    # динамические фазы
    ph_p  = np.exp(1j * (Wg**2/D + We**2/D + dw2(t0, a, vz)) * Dt/2)
    ph_m = np.exp(1j * (Wg**2/D + We**2/D - dw2(t0, a, vz)) * Dt/2)

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

def interference2(a, vz0, vx0, vy0, x0, y0):

    r = np.sqrt(x0**2 + y0**2)
    c0 = np.array([1,0], dtype=complex)
    c1 = RiM2(0, 0, vz0, ty, a, 0, vz0, r)@c0

    z2I = (vz0 + 2*v_s)*T + g*T**2/2
    vz2I = vz0 + 2*v_s + g*T

    z2II = vz0*T + g*T**2/2
    vz2II = vz0 + g*T

    c1i = np.array([0, c1[1]], dtype=complex)
    c1ii = np.array([c1[0], 0], dtype=complex)

    x = vx0*(T + ty) + x0
    y = vy0*(T + ty) + y0
    r = np.sqrt(x**2 + y**2)

    c2i = RiM2(T, z2I, vz2II, 2*ty, a, 0, vz0, r)@c1i
    
    c2ii = RiM2(T, z2II, vz2II, 2*ty, a, 0, vz0, r)@c1ii


    c2 = np.array([c2i[0], c2ii[1]], dtype=complex)

    z3I = z2I + (vz2I-2*v_s)*T+g*T*T/2
    vz3I = vz2I - 2*v_s + g*T

    z3II = z2II + (vz2II + 2*v_s)*T+g*T**2/2
    vz3II = vz2II + 2*v_s + g*T

    x = x + vx0*(T + 2*ty)
    y = y + vy0*(T + 2*ty)
    r = np.sqrt(x**2 + y**2)

    c3 = RiM2(2*T, z3II, vz3I, ty, a, 0, vz0, r)@c2 # переход для разных путей считается одновременно
    #c3i = RiM2(2*T, z3I, vz3I, ty, a, 0, vz0)@np.array([c2i[0], 0], dtype=complex)

    #c3ii = RiM2(2*T, z3II, vz3I, ty, a, 0, vz0)@np.array([0, c2ii[1]], dtype=complex)

    #c3 = c3i + c3ii

    P3 = np.abs(c3)**2


    return P3

def chirp2(vz0, vx0, vy0, x0, y0):
    P1 = []
    P2 = []
    for a in a_range:
        Pa = interference2(a, vz0, vz0, vy0, x0, y0)
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

def TS_Int():

    vz0_m = np.linspace(-4*v_spread + v0z, 4*v_spread + v0z, nT)
    vx0_m = np.linspace(-4*v_spread, 4*v_spread, nT)
    vy0_m = np.linspace(-4*v_spread, 4*v_spread, nT)
    
    x0_m = np.linspace(-4*s_spread + xc, 4*s_spread + xc, ns)
    y0_m = np.linspace(-4*s_spread + yc, 4*s_spread + yc, ns)

    dvz_s = 8*v_spread/(nT-1)
    dvx_s = 8*v_spread/(nT-1)
    dvy_s = 8*v_spread/(nT-1)

    dx_s = 8*s_spread/(ns-1)
    dy_s = 8*s_spread/(ns-1)

    Pa = np.zeros((2, na))
    for vz0 in vz0_m:
        for vx0 in vy0_m:
            for vy0 in vy0_m:
                for x0 in x0_m:
                    for y0 in y0_m:
                        Pa += chirp2(vz0,vx0,vy0, x0, y0)*f_vz(vz0, v0z)*f_vz(vx0, 0)*f_vz(vy0, 0)*f_r(x0, xc)*f_r(y0, yc)*dvz_s*dvx_s*dvy_s*dx_s*dy_s

    return Pa

def show_result():

    Pa = TS_Int()
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
ty = 0.25e-6 # pi/2 impulse duration
# a1 = 500000. # start chirp # 210 mks
# a2 = 50500000. # # end chirp # 210 mks
a1 = 25.1655e6 # start chirp # 210 mks
a2 = 25.165650e6 # # end chirp # 210 mks
na = 100# chirp points
a_range = np.linspace(a1, a2, na)
x0 = a_range[0]
nT = 100
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
W0 = np.pi/2/ty # Rabi freq 78539
T = 120e-3 # between impulse
v0z = 15128e-6*g # начальное смещение по вертикальной скорости
dw0 = keff*(v0z + v_s)*1 # start laser detuning
n = 1000 # количество рассчётных точек
Wg, We = 5e6*0, 3.5e6*0 # dynamic start AC shifts
D = 1e9
T_K = 5.5e-6
v_spread = np.sqrt(kb*T_K/mRb)
s_spread = 0.1e-3
xc = 0
yc = 0
ns = 100
rw = 7.5 # Интенсивность рамановского излучения падает в e квадрат

print(keff*h_/mRb*keff/W0/2)
#print(keff*g/2/np.pi, "a0")
print(keff*g/2/np.pi*1e-6)
print(np.sqrt(3*kb*T_K/mRb)*1e3)
print(dw0)

show_result()

