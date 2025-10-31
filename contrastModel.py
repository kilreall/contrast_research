import numpy as np 
import matplotlib.pyplot as plt


def maxwell_vz(vz, v0z):      # Функция распределения Максвелла для v_z
    prefactor = np.sqrt(mRb / (2 * np.pi * kb * T))
    exponent = - (mRb * (vz-v0z)**2) / (2 * kb * T)
    return prefactor * np.exp(exponent)



def dw(t, a, vz):
    # print(2*np.pi*a*t, "a")
    # print(keff*vz, "b")
    # print(dw0 + 2*np.pi*a*t - keff*vz, "c")
    # print(t*1e3, "T")
    return dw0 + 2*np.pi*a*t - keff*vz

def fR(r, dw, f, Dt): # laser phase
    return -keff*r+(w0+dw)*Dt+f

def RiM(t0, r, f, Dt, dw): # Rabi impulse Matrix (r(t0))

    We = np.sqrt(WR**2-(dw-dwAC)**2)

    M = np.zeros((2, 2), dtype=complex)

    M[0,0] = np.cos(WR*Dt/2)-1j*(dw-dwAC)/WR*np.sin(WR*Dt/2)
    M[0,1] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(-1j*(dw*t0+fR(r, dw, f, Dt)))
    M[1,0] = -1j*We/WR*np.sin(WR*Dt/2)*np.exp(1j*(dw*t0+fR(r, dw, f, Dt)))
    M[1,1] = np.cos(WR*Dt/2)+1j*(dw-dwAC)/WR*np.sin(WR*Dt/2)

    M = M * np.exp(-1j*(WgAC+WeAC-dw)*Dt/2)

    return M

def interference(a, vz0):
    
    c0 = np.array([1,0])
    M1 = RiM(0, 0, 0, ty, dw(0, a, vz0))
    c1 = M1@c0

    z2I = (vz0 + keff*h_/mRb)*T + g*T**2/2
    vz2I = vz0 + keff*h_/mRb + g*T

    z2II = vz0*T + g*T**2/2
    vz2II = vz0 + g*T

    M2I = RiM(T, z2I, 0, 2*ty, dw(T, a, vz2I))
    c2I = M2I@np.array([0,c1[1]])


    M2II = RiM(T, z2II, 0, 2*ty, dw(T, a, vz2II))
    c2II = M2II@np.array([c1[0],0])

    c2 = c2I + c2II

    z3I = z2I + (vz2I-keff*h_/mRb)*T+g*T
    vz3I = vz2I - keff*h_/mRb + g*T

    z3II = z2II + (vz2II+keff*h_/mRb)*T+g*T**2/2
    vz3II = vz2I + keff*h_/mRb + g*T**2/2

    M3I = RiM(2*T, z3I, 0, ty, dw(2*T, a, vz3I))
    c3I = M3I@np.array([c2[0],0])

    M3II = RiM(2*T, z3II, 0, ty, dw(2*T, a, vz3II))
    c3II = M3II@np.array([0,c2[1]])

    c3 = c3I + c3II


    return c3*c3.conjugate()

def tempdist(a):

    vz_d = np.linspace(-Dv+v0z, +Dv+v0z, 100) # скоростной диапазон
    Pvz = []
    for vz in vz_d:
        Pvz.append(interference(a, vz))

    return 1

def chirp(a1, a2, na):
    vz = 0
    P1 = []
    P2 = []
    a_range = np.linspace(a1, a2, na)
    for a in a_range:
        Pa = interference(a, vz)
        P1.append(Pa[0])
        P2.append(Pa[1])

    P1 = np.array(P1)
    P2 = np.array(P2)

    plt.plot(a_range, P2)
    plt.title("interference")
    plt.xlabel("chirp rate")
    plt.ylabel("Population")

    plt.show()

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
ty = 20e-6 # pi/2 impulse duration
a1 = 25.125e6 # start chirp
a2 = 25.130e6 # end chirp
na = 1 # chirp points
Dw = 1/ty # Raman pi/2 pulse width
Dv = Dw*c/(keff*c) # cutted speed width
WR = np.pi/2/ty # Rabi freq 78539
T = 5e-3 # between impulse
dw0 = 0. # start laser detuning
v0z = 0. # начальное смещение по вертикальной скорости
n = 1000 # количество рассчётных точек
dwAC, WgAC, WeAC = 0., 0., 0. # dynamic start AC shifts
g = 9.8

print(keff*h_/mRb*keff)
#chirp(a1, a2, na)