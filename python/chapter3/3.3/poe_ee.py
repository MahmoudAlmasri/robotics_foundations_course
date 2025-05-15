import numpy as np

def skew(v):
    """Return the skew-symmetric matrix of a 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def vec_to_se3(S):
    """Convert a 6D spatial velocity vector into an se(3) matrix."""
    w = S[:3]
    v = S[3:]
    se3 = np.zeros((4, 4))
    se3[:3, :3] = skew(w)
    se3[:3, 3] = v
    return se3

def matrix_exp6(se3mat):
    """Compute the matrix exponential of an se(3) matrix using series expansion."""
    w_hat = se3mat[:3, :3]
    v = se3mat[:3, 3]
    theta = np.linalg.norm([w_hat[2,1], w_hat[0,2], w_hat[1,0]])
    
    if theta < 1e-10:
        return np.eye(4) + se3mat  # First-order approximation

    w = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])
    w_unit = w / theta
    w_hat_unit = skew(w_unit)

    R = (
        np.eye(3) +
        np.sin(theta) * w_hat_unit +
        (1 - np.cos(theta)) * w_hat_unit @ w_hat_unit
    )

    G = (
        np.eye(3) * theta +
        (1 - np.cos(theta)) * w_hat_unit +
        (theta - np.sin(theta)) * w_hat_unit @ w_hat_unit
    )
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = G @ (v / theta)
    return T


def fk_poe(B_list, M, theta_list):
    """
    Compute forward kinematics using the Product of Exponentials formula.
    
    Parameters:
    - B_list: list of 6D screw axes in the EE frame (shape: [n, 6])
    - M: home configuration of the end-effector (4x4 matrix)
    - theta_list: list of joint variables (length n)
    
    Returns:
    - T: 4x4 transformation matrix of the end-effector in the base frame
    """
    T = np.eye(4)
    for B, theta in zip(B_list, theta_list):
        exp_se3 = matrix_exp6(vec_to_se3(B) * theta)
        T = T @ exp_se3
    T =  M @ T
    return T


# Define screw axes (each 6D: [w; v]), shape = [n, 6]
B_list = [
    np.array([0, 0, 1, -1, 0, 0]),
    np.array([0, 0, 1, -0.5, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1]),
]

# Define home configuration of end-effector (M)
M = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0.5],
    [0, 0, 0, 1]
])

# Joint values (in radians)
theta_list = [np.pi/2, -np.pi/2, np.pi, 0.2]

# Compute forward kinematics
T = fk_poe(B_list, M, theta_list)
np.set_printoptions(precision=2, suppress=True)
print(T)
