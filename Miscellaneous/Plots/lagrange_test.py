from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
F = np.sin(R)
G = (X**2 + Y**2 - 4)



# Plot the surface.

surf = ax.plot_surface(
    X, Y, F, cmap=cm.coolwarm, linewidth=0, antialiased=True)
"""

surf = ax.plot_surface(
    X, Y, G, cmap=cm.coolwarm, linewidth=0, antialiased=True)
"""
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
