import numpy as np

"""
This scipt solves the first exercice of chatper 1.
"""
# Given values
v = np.array([-1, 2])  # velocity vector
d = np.array([1, 1])  # direction of the wall
p0 = np.array([5, 1])  # initial position of the robot

# Step 1: Compute the projection of v onto d
d_norm_sq = np.dot(d, d)
proj_v_on_d = (np.dot(v, d) / d_norm_sq) * d

# Step 2: Compute the orthogonal component of v
v_perp = v - proj_v_on_d

# Step 3: Compute the distance from p0 to the wall
# In 2D, distance from point to line through origin in direction d is:
# distance = |d x p0| / ||d||
cross = np.cross(d, p0)  # cross = d[0]*p0[1] - d[1]*p0[0]
distance_to_wall = abs(cross) / np.linalg.norm(d)

# Step 4: Time to hit the wall = distance / speed in orthogonal direction
speed_toward_wall = np.linalg.norm(v_perp)
time_to_collision = distance_to_wall / speed_toward_wall

# Print the results using regular print statements
print("=== Solution to the Exercise ===")
print(f"1. Projection of v onto d: {proj_v_on_d}")
print(f"2. Orthogonal component of v: {v_perp}")
print(f"3. Distance from the robot to the wall: {distance_to_wall:.3f} m")
print(
    f"4. Speed toward the wall (orthogonal component of velocity): {speed_toward_wall:.3f} m/s")
print(
    f"5. Time until collision with the wall: {time_to_collision:.3f} seconds")
