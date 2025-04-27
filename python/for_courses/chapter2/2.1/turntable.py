import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setup the figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Two Frames rotating w.r.t. Frame S with Trajectories and Sticks')

# Draw static Frame S (space frame)
ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='gray', label='S-x')
ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='lightgray', label='S-y')

# Radii
r1 = 1.0 # Radius for object 1
r2 = 4.0  # Radius for object 2

# Draw trajectory circles
theta_circle = np.linspace(0, 2*np.pi, 300)
circle1_x = r1 * np.cos(theta_circle)
circle1_y = r1 * np.sin(theta_circle)
circle2_x = r2 * np.cos(theta_circle)
circle2_y = r2 * np.sin(theta_circle)

ax.plot(circle1_x, circle1_y, '--b', linewidth=0.8, label='Trajectory 1')
ax.plot(circle2_x, circle2_y, '--g', linewidth=0.8, label='Trajectory 2')

# Create quivers for Frame B1 (object 1, radius r1)
frame_b1_x = ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='blue')
frame_b1_y = ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='red')

# Create quivers for Frame B2 (object 2, radius r2)
frame_b2_x = ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='green')
frame_b2_y = ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='orange')

# Create sticks (lines) from center to the objects
stick, = ax.plot([0, 0], [0, 0], 'k--', linewidth=0.5)

# Animation function
def update(frame):
    theta = np.deg2rad(frame)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # Object 1
    center1 = r1 * np.array([np.cos(theta), np.sin(theta)])
    x_b1 = R @ np.array([1.0, 0])
    y_b1 = R @ np.array([0, 1.0])
    frame_b1_x.set_offsets(center1)
    frame_b1_x.set_UVC(x_b1[0], x_b1[1])
    frame_b1_y.set_offsets(center1)
    frame_b1_y.set_UVC(y_b1[0], y_b1[1])

    # Object 2
    center2 = r2 * np.array([np.cos(theta), np.sin(theta)])
    x_b2 = R @ np.array([1.0, 0])
    y_b2 = R @ np.array([0, 1.0])
    frame_b2_x.set_offsets(center2)
    frame_b2_x.set_UVC(x_b2[0], x_b2[1])
    frame_b2_y.set_offsets(center2)
    frame_b2_y.set_UVC(y_b2[0], y_b2[1])

    stick.set_data([0, center2[0]], [0, center2[1]])

    return frame_b1_x, frame_b1_y, frame_b2_x, frame_b2_y, stick

# Animate the planar rotation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 120), interval=50, blit=True)
plt.show()
