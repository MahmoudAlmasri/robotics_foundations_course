import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setup the figure
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Frame B rotating w.r.t. Frame S')

# Draw static Frame S (space frame)
ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='gray', label='S-x')
ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='lightgray', label='S-y')

# Create quivers for Frame B (body frame)
frame_b_x = ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='blue', label='B-x')
frame_b_y = ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='red', label='B-y')

# Legend
ax.legend(loc='upper left')

# Animation function
def update(frame):
    theta = np.deg2rad(frame)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    x_b = R @ np.array([1, 0])
    y_b = R @ np.array([0, 1])
    
    frame_b_x.set_UVC(x_b[0], x_b[1])
    frame_b_y.set_UVC(y_b[0], y_b[1])
    return frame_b_x, frame_b_y

# Animate the planar rotation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 120), interval=50, blit=True)
plt.show()
