import numpy as np

a = np.array([0.2, 0.25, 0.2, 0.15, 0.1,      # Camber limits
                        0.2, 0.2, 0.2, 0.2, 0.2,        # Thickness limits
                        45])

a[0:4] = np.array([0.0, 0.0, 0.0, 0.0])
print(a)