import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Given data
w = np.array([1, 2, 3])  # Rotation axis
theta_dot = 2            # Angular speed (rad/s)
t = 4                    # Time in seconds

# Step 2: Normalize w to get unit rotation axis
norm_w = np.linalg.norm(w)
w_hat = w / np.linalg.norm(w)

# Step 3: Compute total rotation angle
theta = norm_w * theta_dot * t  # Total rotation angle (radians)

# Print out the exponential coordinates:
exp_coord = w_hat * theta
print(f'Exponential coordinates w.theta\n: {exp_coord.reshape((3,1))}')

# Step 4: Skew-symmetric matrix of w_hat
def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


w_hat_skew = skew_symmetric(w_hat)

# Step 5: Rodrigues' rotation formula
I = np.eye(3)
R_ab = I + np.sin(theta) * w_hat_skew + \
    (1 - np.cos(theta)) * (w_hat_skew @ w_hat_skew)

# Step 6: Function to draw a frame with separate colors per axis


def draw_frame(ax, R, origin=[0, 0, 0], length=1.0, name=''):
    axis_colors = ['r', 'g', 'b']  # Colors: x-red, y-green, z-blue
    axis_labels = ['x', 'y', 'z']
    for i in range(3):
        vec = R[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=axis_colors[i], arrow_length_ratio=0.1, linewidth=1.5)
        if name:
            ax.text(origin[0] + vec[0]*1.1, origin[1] + vec[1]*1.1, origin[2] + vec[2]*1.1,
                    f'{axis_labels[i]}_{name}', color=axis_colors[i])


# Step 7: Plot the frames
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Frame {a} and rotated Frame {b} at t=4s (corrected)')
ax.grid(True)

# Draw frame a (initial frame)
draw_frame(ax, np.eye(3), name='A')

# Draw frame b (after rotation)
draw_frame(ax, R_ab, name='B')

# draw the rotaiton axis
ax.quiver(0, 0, 0,
          1, 2, 3,
          color='k', arrow_length_ratio=0.1, linewidth=1.5)
ax.text(1.1, 2.1, 3.1, f'w', color='k')
# Set viewing angle
ax.view_init(elev=30, azim=45)

plt.show()

# Display final rotation matrix
np.set_printoptions(precision=4, suppress=True)
print("Corrected Rotation matrix R_ab (Frame {a} to Frame {b} at t=4s):")
print(R_ab)
