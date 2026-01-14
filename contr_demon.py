import numpy as np 
import matplotlib.pyplot as plt

a = np.linspace(-0.4, 0.4, 100)
P = -np.cos(20*a)*0.2 + 0.5
plt.xlabel("скорости чирпирования a-a0, [отн.ед]")
plt.ylabel("сигнал атомной интерференции, [отн. ед]")

plt.plot(a, P)
plt.legend()
plt.show()