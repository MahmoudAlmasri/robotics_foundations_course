import matplotlib.pyplot as plt
import numpy as np

"""
This script makes a quick illustration for the first exercice of chapter 1, about vectors:
An infinitely small robot (point) is moving in the 2D plane. At a certain time, its velocity vector is v = (3, 4).
The robot is moving near a very long wall aligned along the direction of the vector d = (1, 1).
Tasks:
1- Compute the projection of v onto the direction d
2- Determine the orthogonal component of v w.r.t. d
3- Assuming that the robot is initially located at p0=(5,1). If the robot maintains its velocity, how long will it take before it hits the wall?
"""

# Given data as in the exercice
d = np.array([1, 1])  # Direction of the wall
v = np.array([-1, 2])  # Velocity vector
p0 = np.array([5, 1])  # Initial position of the robot

# Normalize direction
d_unit = d / np.linalg.norm(d)

# Create points along the wall. Alternatively, two points would suffice
# more info about what the outer product does: https://en.wikipedia.org/wiki/Outer_product
# Briefly, this generate a matrix 100 x 2 where the first column holds the x values of the points and the second column holds the y
# In other words, the line i of wall_points holds (x_pti, y_pti) = t_val[i] * d_unit; t_val[i] is a scalar.
t_vals = np.linspace(-10, 10, 100)
wall_points = np.outer(t_vals, d_unit)

# plot the wall line
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(wall_points[:, 0], wall_points[:, 1],
        'gray', label='Wall (line in direction d)')

# Plot the robot position: red "o"
ax.plot(p0[0], p0[1], 'ro', label='Robot position $p_0$')

# Plot the velocity vector as an arrow from p0
ax.arrow(p0[0], p0[1], v[0], v[1], head_width=0.3,
         head_length=0.3, fc='blue', ec='blue')

# Formatting
ax.set_aspect('equal')
ax.set_title('Robot Motion and Wall')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xticks(np.arange(-10, 11, 1))
ax.set_yticks(np.arange(-10, 11, 1))
ax.legend()
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
