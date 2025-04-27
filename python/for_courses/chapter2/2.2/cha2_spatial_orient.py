import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# This animation shows a rotating frame B relative to a static frame S.
# The code updates B's x, y, and z axes by applying a rotation matrix.
# Modify the rotation matrix R to explore different rotation behaviors.

# Set up the figure and 3D axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_title("Frame B rotating w.r.t. Frame S")
ax.view_init(elev=20, azim=30)

# Static frame S
ax.quiver(0, 0, 0, 1, 0, 0, color='gray', label='S-x')
ax.quiver(0, 0, 0, 0, 1, 0, color='lightgray', label='S-y')
ax.quiver(0, 0, 0, 0, 0, 1, color='darkgray', label='S-z')

# Rotating frame B placeholders
frame_b_x = ax.quiver(0, 0, 0, 1, 0, 0, color='red', label='B-x')
frame_b_y = ax.quiver(0, 0, 0, 0, 1, 0, color='green', label='B-y')
frame_b_z = ax.quiver(0, 0, 0, 0, 0, 1, color='blue', label='B-z')

# Legend
ax.legend()

# Update function for animation
def update(frame):
    theta = np.deg2rad(frame)
    # Elementatry Rotations
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta),  0.0, np.cos(theta)]

    ])
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    R_not_rot = np.array([
        [np.cos(theta), -np.sin(theta), np.tan(theta)],
        [np.sin(theta),  np.cos(theta), -np.tan(theta)],
        [np.cos(theta),  np.tan(theta), 0.5]
    ])

    R = Ry @ Rx @ Rz
    # R = Ry @ Rx
    # R = Rx @ Ry
    # R = R_not_rot

    # Apply rotation to basis vectors
    x_b = R @ np.array([1, 0, 0])
    y_b = R @ np.array([0, 1, 0])
    z_b = R @ np.array([0, 0, 1])

    # Update quiver directions
    frame_b_x.set_segments([[[0, 0, 0], x_b]])
    frame_b_y.set_segments([[[0, 0, 0], y_b]])
    frame_b_z.set_segments([[[0, 0, 0], z_b]])
    return frame_b_x, frame_b_y, frame_b_z


# Animate
ani = FuncAnimation(fig, update, frames=np.linspace(
    0, 360, 120), interval=100, blit=False)
plt.show()
