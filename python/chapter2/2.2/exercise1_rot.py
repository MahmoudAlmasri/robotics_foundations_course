# Recreate everything carefully
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rotation matrices about x, y, z


def rot_x(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


def rot_y(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rot_z(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


# Define the three rotations
R1 = rot_x(45)  # Rotation about x_s
R2 = rot_y(45)  # Rotation about y_b
R3 = rot_z(45)  # Rotation about z_s

# Intermediate frames
R_sa = R1                      # Frame a after first rotation
R_sb = R1 @ R2                  # Frame b after second rotation
R_sc = R3 @ R_sb                # Final frame c

# Draw frames


def draw_frame(ax, R, origin=[0, 0, 0], length=0.8, name=''):
    colors = ['r', 'g', 'b']  # x, y, z axes
    labels = ['x', 'y', 'z']
    for i in range(3):
        vec = R[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=colors[i], arrow_length_ratio=0.1)
        if name:
            ax.text(origin[0] + vec[0], origin[1] + vec[1], origin[2] + vec[2],
                    f'{labels[i]}_{name}', color=colors[i])


# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Frames {s}, {a}, {b}, {c}')
ax.grid(True)
ax.view_init(elev=30, azim=45)

# Draw frames
draw_frame(ax, np.eye(3), name='S')   # Fixed frame
# draw_frame(ax, R_sa, name='A')         # After first rotation
# draw_frame(ax, R_sb, name='B')         # After second rotation
draw_frame(ax, R_sc, name='C')         # Final frame

plt.show()
