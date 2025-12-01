import numpy as np 
import matplotlib.pyplot as plt

# c against ty for certain T
ty = np.array([20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 1.25, 1, 0.5, 0.25])*1e-6


# T = 1 ms
C1= np.array([0.47, 0.48, 0.49, 0.51, 0.54, 0.59, 0.66, 0.78, 0.91, 0.94, 0.98, 0.99])

# T = 5 ms
C2 = np.array([0.47, 0.48, 0.49, 0.51, 0.54, 0.59, 0.66, 0.79, 0.91, 0.94, 0.98, 0.99])

# T = 10 ms
C3 = np.array([0.47, 0.48, 0.49, 0.51, 0.54, 0.59, 0.66, 0.79, 0.91, 0.94, 0.98, 0.99])

#T = 120
C4 = np.array([0.47, 0.94, 0.99])

plt.figure(1)
plt.title("C againt ty for certain T")
plt.xlabel("ty, [uks]")
plt.ylabel("C")
plt.plot(ty*1e6, C2)




plt.show()
