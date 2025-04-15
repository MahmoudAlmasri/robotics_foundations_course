import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This script offers a visualization for the exercise 2 - Projection matrix of chapter 1.
It also solves it.
"""

# Given values
n = np.array([1, 2, -1])  # normal vector of the plane
x = np.array([2, 1, 0])  # vector to project

# Compute the projection matrix onto the plane (demonstration in video)
n_norm_sq = np.dot(n, n)
# np.outer(n, n) is equivalent ot n.T @ n if n was defined as np.matrix
P = np.eye(3) - np.outer(n, n) / n_norm_sq
print(f"The projection matrix is {P}")

# Project x onto the plane
x_proj = P @ x
print(f"The projection value is {x_proj}")


# Define the projection line from x to x_proj
proj_line = np.vstack((x, x_proj))

## Plane definition to plot
grid_size = 2
xx, yy = np.meshgrid(np.linspace(-grid_size, grid_size, 10),
                     np.linspace(-grid_size, grid_size, 10))
# Solve for z in the plane equation: n·[x, y, z] = 0 -> z = -(n1*x + n2*y)/n3
zz = -(n[0] * xx + n[1] * yy) / n[2]

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz, alpha=0.5, color='lightgray', edgecolor='none')
# Plot original vector x from origin
ax.quiver(0, 0, 0, x[0], x[1], x[2], color='blue',
          label=r'$\vec{x}$', linewidth=2)

# Plot projection of x onto the plane
ax.quiver(0, 0, 0, x_proj[0], x_proj[1], x_proj[2], color='green',
          label=r'$\mathrm{Proj}_\pi(\vec{x})$', linewidth=2)

# Connect x to its projection with a dashed red line
ax.plot(proj_line[:, 0], proj_line[:, 1],
        proj_line[:, 2], 'r--', label='Projection path')

# Formatting
ax.set_xlim([-2, 3])
ax.set_ylim([-2, 3])
ax.set_zlim([-2, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Projection of Vector onto a Plane in ℝ³')
ax.legend()
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.show()
