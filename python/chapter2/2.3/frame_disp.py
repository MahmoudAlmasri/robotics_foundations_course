import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rot_z(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])


def rot_y(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def homogeneous_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Define transformation T (rotation + translation) It will be applied on S
R_T = rot_y(45)
t_T = np.array([1.0, 1.0, 1.0])
T = homogeneous_transform(R_T, t_T)

# Define space frame S
R_S = np.eye(3)
t_S = np.array([0.0, 0.0, 0.0])
S = homogeneous_transform(R_S, t_S)

# First frame A (pre-multiply -> T applied in space (fixed) frame)
A = T @ S

# Second frame B (post-multiply -> T applied in local (body) frame)
B = S @ T

# Function to draw frames
def draw_frame(ax, T, name='', length=1.0):
    colors = ['r', 'g', 'b']  # x, y, z axis colors
    axis_labels = ['x', 'y', 'z']
    origin = T[:3, 3]
    R = T[:3, :3]

    for i in range(3):
        vec = R[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=colors[i], arrow_length_ratio=0.1, linewidth=2)
        ax.text(origin[0] + vec[0]*1.05, origin[1] + vec[1]*1.05, origin[2] + vec[2]*1.05,
                f'{name}-{axis_labels[i]}', color=colors[i])


# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(
    'Homogeneous Transformation with non-identity {S}: Frames {S}, {A}, {B}')
ax.grid(True)

# Draw frames
draw_frame(ax, S, name='S')  # Space frame
draw_frame(ax, A, name='A')  # Frame A (pre-multiply)
draw_frame(ax, B, name='B')  # Frame B (post-multiply)

# Set viewing angle
ax.view_init(elev=30, azim=45)
plt.show()
