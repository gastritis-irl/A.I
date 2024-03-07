import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load surface data
data = np.loadtxt('surface_100x100.txt')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
b = data[:, 3]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create obstacle blocks
obstacle_voxels = np.zeros((100, 100, 100), dtype=int)
obstacle_voxels[x[b == 1], y[b == 1], z[b == 1]] = 1
ax.voxels(x[b == 1], y[b == 1], z[b == 1], facecolors='red', edgecolor='k')

# Create non-obstacle points
ax.scatter(x[b == 0], y[b == 0], z[b == 0], c='b', marker='o')

# Set axis labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('3D Plot')

plt.show()
