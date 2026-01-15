import numpy as np 
import matplotlib.pyplot as plt

ty = np.array([6, 9, 17, 30])
C = np.array([22, 17, 15, 13])/100


#plt.title("C againt ty for certain T")
plt.xlabel("τ, [мкс]")
plt.ylabel("C")
plt.scatter(ty/2, C)
plt.plot(ty/2, C)



plt.show()
