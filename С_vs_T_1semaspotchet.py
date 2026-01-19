import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.size'] = 16 # Общий размер для всех элементов
plt.rcParams['figure.figsize'] = [8, 8] # Все новые графики будут 8x5 дюймов

T = np.array([1510, 2010, 2510, 3010, 3510, 4010, 4510, 5010])/1000
C = np.array([0.11053263996226463, 0.11639217828270054, 0.11880160712770997, 0.1181837979092125, 0.11893979492503828, 0.12168092677210288, 0.12117495027985267, 0.12167328679002996])

fix, ax = plt.subplots()
ax.plot(T, C, color = "black")
ax.scatter(T, C, color = "black")
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

plt.xlabel("T, [мс]")
plt.ylabel("C")
plt.ylim(0.08, 0.18)
plt.grid()

plt.show()