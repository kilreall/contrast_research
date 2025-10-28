import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sins(x, A, w, ph, s):
    return A*np.sin(w*x+ph) + s

def sinss(x, A, w, ph, s, al):
    return A*np.sin(w*x+ph) + s + al*(x-x0)

def expf(x, n, A, B):
    return A/x**n+B
    

c = 3*1e8
k =  (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9)/c + (384.2304844685*1e12 + 4.27167663181519*1e9 - 229.8518*1e6 - 1e9 - 6.83468261090429*1e9)/c
k = k*2*np.pi
#print(k)
# start_freq = 90582400/70*5282
# dt = 30e-3 # s для чирпирования
n = 101 # количество точек
T = 10200e-6 # s временной интервал между пи импульсами
M = 0
Tg = 0.00357# T1:0.4;T2:0.089;T4:0.0226;T6:0.0109;T8:0.0061;T10:0.00357;T12:0.0027; # пристрелка периода для fitа
gR = 9.68
Tf = 30528*1e-6 # полное време подготовки атомов
ty = 20e-6 # s длительность pi/2 импульса
Tpause = 500e-3
TF = Tf+2*T+Tpause+4*ty # point time
TAI = 2*T+4*ty
OR = np.pi/2/ty




r = 100000 # коэф единиц измерения
sk = 1 # коэф поправки для оценки погрешности


# чтение csv P(a)
file_path = r'2025-10-24_contrast_vs_T\norm_10200_us.csv' 
data = np.genfromtxt(file_path, delimiter=',', dtype=None, skip_header=1)
data = np.array(data.tolist())



chirp_rate = data[:,0]
intensity = data[:,1]
chirp0 = chirp_rate[:n]
x0 = chirp0[0]

initial_guess = [(np.max(intensity) - np.min(intensity))/2, 2*np.pi*T*T, 0, (np.max(intensity) - np.min(intensity))/2, 35e-10] 
par, cov = curve_fit(sinss, chirp_rate, intensity, p0=initial_guess)
A, w, ph, s, al = par
#print("A, s, A+s, s-A= ", A, s, s+A, s-A)
dw, dph, dA, ds = np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[0,0]), np.sqrt(cov[4,4])
dgE = 1/k/T**2/(A/dA*sk)
dgE = abs(dgE)
V = abs(A)/s
dV = dA*s-ds*abs(A)/s**2


print("T =", int(T*1e6))
print("sensitivity for experimental data =", dgE*np.sqrt(TF*n)*r, "mGal/.")
print("visibility =", V)
print("accuracy of visibility =", dV)
print("swing =", 2*abs(A))
print("accuracy of swing =", 2*dA)

plt.figure(1)
plt.title("data")
intensity = intensity #- al*(chirp_rate-x0)
plt.scatter(chirp_rate, intensity, color = "blue", s=20)
#plt.plot(chirp_rate, intensity, color = "blue")
plt.plot(chirp0, A*np.sin(w*chirp0+ph) + s + al*(chirp0-x0), color="orange")

plt.xlabel('chirp rate')
plt.ylabel('signal')

plt.figure(2)
plt.title("Contrast")
plt.xlabel('T')
plt.ylabel('C')
T = np.array([210, 600, 800, 1000, 1600, 2200, 2800, 3400, 4000, 4600, 5300, 6000, 6700, 7400, 8100, 8800, 9500, 10200])
C = np.array([0.116198897859155, 0.09632436964533655, 0.0874349394591908, 0.0861787578713447, 0.0820320812667242, 0.08035256996600361, 0.08014746720816358,
               0.07638466189981624, 0.07803898226931237, 0.07945535338142871, 0.0784225020609696, 0.07939911609779485, 0.07919190735204812, 0.07715253290165193,
                 0.07691569539537815, 0.07675128102950372, 0.07703996035647037, 0.07685091558694972])
dC = np.array([4.935730542415889e-05, 7.756051683107904e-05, 8.104271666836615e-05, 7.367936514455295e-05, 6.944350606374536e-05, 6.673833275679416e-05, 7.335072760126995e-05,
                7.245632300039677e-05, 8.998731744933044e-05, 5.9097150973543996e-05, 5.236484003803585e-05, 8.040674513672609e-05, 6.453106706139243e-05, 6.814110445812338e-05,
                 9.201771081999504e-05, 9.791574637389227e-05, 9.268788371941834e-05, 0.00010049771270828313])
plt.errorbar(T, C, yerr=dC, fmt='o-', color='blue', linewidth=1, markersize=2, capsize=7)


plt.figure(3)
plt.title("2A")
plt.xlabel('T')
plt.ylabel('Swing')
S = np.array([0.06757355855196533, 0.05735571583446123, 0.05282300953376252, 0.051989933157623645, 0.049606744234698116, 0.047943904939113303, 0.04799614544809886,
               0.04462333402065859, 0.04620165262121363, 0.04660298435686215, 0.04558890429045528, 0.04648673392637408, 0.046157108921699656, 0.04480220517488916,
                 0.044655113189632986, 0.045353895085609205, 0.045301123832414865, 0.045844798565183906])
dS = np.array([0.00033949757859891993, 0.0005210273470991036, 0.0005365819520300542, 0.000488525437012469, 0.0004593413770560613, 0.0004474079853747206, 0.0004899500904553138,
                0.0004961189317988506, 0.0006079987651859463, 0.00040303830294989936, 0.00036032631481353424, 0.0005493626545836457, 0.00044289150281916906, 0.00046940394636089586,
                  0.0006340246764255151, 0.0006628415682569212, 0.0006305456890805171, 0.0006739184453182254])
plt.errorbar(T, S, yerr=dS, fmt='o-', color='blue', linewidth=1, markersize=2, capsize=7)
initial_guess = [1, 13500, 0.448] 
par, cov = curve_fit(expf, T, S, p0=initial_guess, maxfev=10000)
n, A, B = par
plt.plot(T, expf(T, n, A, B), color="red")
print(n, A, B)


plt.show()