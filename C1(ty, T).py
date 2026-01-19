import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.size'] = 16 # Общий размер для всех элементов
plt.rcParams['figure.figsize'] = [8, 8] # Все новые графики будут 8x5 дюймов


# c against ty for certain T
ty = np.array([20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 1.25, 1, 0.5, 0.25])

ty_p = np.array([6, 9, 17, 30])
C_p = np.array([22, 17, 15, 13])/100

# T = 1 ms
C1= np.array([0.090, 0.097, 0.104, 0.116, 0.132, 0.159, 0.217, 0.373, 0.537, 0.573, 0.637, 0.657])

# T = 5 ms
C2 = np.array([0.091, 0.097, 0.105, 0.116, 0.132, 0.159, 0.216, 0.368, 0.524, 0.558, 0.615, 0.633])

# T = 10 ms
C3 = np.array([0.099, 0.105, 0.113, 0.124, 0.140, 0.169, 0.226, 0.381, 0.537, 0.571, 0.624, 0.639])

#T = 120
C4 = np.array([0.0366, 0.0385, 0.0444, 0.0478, 0.0491, 0.0559, 0.0779, 0.1317, 0.1838, 0.1956, 0.2159, 0.2220])

fig, ax = plt.subplots()
#plt.title("C againt ty for certain T")
plt.xlabel("τ, [мкс]")
plt.ylabel("C")


ax.plot(ty, C1, label="T = 1 мс", color="blue")
ax.plot(ty, C2, label="T = 5 мс", color="orange")
ax.plot(ty, C3, label="T = 10 мс", color ="green")
ax.plot(ty, C4, label="T = 120 мс", color="red")

ax.scatter(ty_p/2, C_p, color = "black")
ax.plot(ty_p/2, C_p, color = "black", label ="эксперимент")


ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
plt.grid()
plt.legend()
plt.show()
