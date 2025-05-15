import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SE3Math:
    """Helper class to perform SE3-related operations"""
    @staticmethod
    def skew(v):
        """Return the skew-symmetric matrix of a 3D vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def vec_to_se3(S):
        """Convert a 6D twist vector to an se(3) matrix."""
        w = S[:3]
        v = S[3:]
        se3 = np.zeros((4, 4))
        se3[:3, :3] = SE3Math.skew(w)
        se3[:3, 3] = v
        return se3

    @staticmethod
    def matrix_exp6(se3mat):
        """Compute the matrix exponential of an se(3) matrix using series expansion."""
        w_hat = se3mat[:3, :3]
        v = se3mat[:3, 3]
        theta = np.linalg.norm([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])

        if theta < 1e-10:
            return np.eye(4) + se3mat  # First-order approximation

        w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
        w_unit = w / theta
        w_hat_unit = SE3Math.skew(w_unit)

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


def generate_rrrp_trajectory(T, steps):
    """
    Generate a trajectory for a RRRP (Revolute-Revolute-Revolute-Prismatic) robotic arm.

    This function computes joint trajectories for a RRRP robotic arm over a specified
    time period and number of steps. The trajectory is defined as sinusoidal variations
    for the revolute joints and a sinusoidal variation with an offset for the prismatic joint.

    Args:
        T (float): Total duration of the trajectory in seconds.
        steps (int): Number of discrete steps in the trajectory.

    Returns:
        numpy.ndarray: A 2D array of shape (steps, 4) where each row represents the joint
        positions [q1, q2, q3, q4] at a specific time step. The first three columns correspond
        to the revolute joint angles (in radians), and the fourth column corresponds to the
        prismatic joint position (in meters).
    """
    t_vals = np.linspace(0, T, steps)
    q = np.zeros((steps, 4))
    q[:, 0] = np.pi / 4 * np.sin(2 * np.pi * t_vals / T)
    q[:, 1] = np.pi / 6 * np.sin(2 * np.pi * t_vals / T + np.pi / 3)
    q[:, 2] = np.pi / 8 * np.sin(2 * np.pi * t_vals / T + np.pi / 6)
    q[:, 3] = 0.2 + 0.1 * np.sin(2 * np.pi * t_vals / T)
    return q


class RRRP_Robot:
    """
    RRRP_Robot is a class that models a 4-DOF robot with three revolute joints and one prismatic joint.
    It provides methods to compute the forward kinematics using either the Product of Exponentials (PoE) 
    or Denavit-Hartenberg (DH) parameterization.

    Attributes:
        L1 (float): Length of the first link. Default is 1.0.
        L2 (float): Length of the second link. Default is 1.0.
        L3 (float): Length of the third link. Default is 0.5.
        L4 (float): Length of the fourth link. Default is 0.3.
        L5 (float): Length of the fifth link. Default is 0.15.
        M (np.ndarray): The home configuration matrix of the end-effector.
        method (str): The method used for forward kinematics ('poe' or 'dh').

    Methods:
        __init__(L1=1.0, L2=1.0, L3=0.5, L4=0.3, L5=0.15, method='poe'):
            Initializes the robot with given link lengths and method for forward kinematics.

        fk(q):
            Computes the forward kinematics for the given joint configuration `q`.
            Delegates to either `fk_poe` or `fk_dh` based on the selected method.

        fk_poe(q):
            Computes the forward kinematics using the Product of Exponentials (PoE) method.
            Args:
                q (list or np.ndarray): Joint configuration [theta1, theta2, theta3, theta4].
            Returns:
                np.ndarray: The transformation matrix of the end-effector.

        dh_transform(a, alpha, d, phi):
            Computes the Denavit-Hartenberg transformation matrix for given parameters.
            Args:
                a (float): Link length.
                alpha (float): Link twist.
                d (float): Link offset.
                phi (float): Joint angle.
            Returns:
                np.ndarray: The DH transformation matrix.

        fk_dh(q):
            Computes the forward kinematics using the Denavit-Hartenberg (DH) method.
            Args:
                q (list or np.ndarray): Joint configuration [theta1, theta2, theta3, theta4].
            Returns:
                np.ndarray: The transformation matrix of the end-effector.
    """

    def __init__(self, L1=1.0, L2=1.0, L3=0.5, L4=0.3, L5=0.15, method='poe'):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.L5 = L5
        # TODO: Fill the M matrix with the end-effector home configuration
        self.M = np.eye(4)

        assert method in ['poe', 'dh'], "Method must be either 'poe' or 'dh'"
        self.method = method

    def fk(self, q):
        return self.fk_poe(q) if self.method == 'poe' else self.fk_dh(q)

    def fk_poe(self, q):
        theta1, theta2, theta3, theta4 = q
        # TODO: define the screw matrices
        S1 = np.array([])
        S2 = np.array([])
        S3 = np.array([])
        S4 = np.array([])

        # TODO: compute the transform T of the end effector in the base frame knowing the screw axes and the joint variables q
        # T = 
        return np.eye(4)

    def dh_transform(self, a, alpha, d, phi):
        # NOTE: phi could be theta or theta+constant value.
        c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        # TODO compute the transform T of i in i-1 knowing the d-h parameters of the joint
        return np.eye(4)

    def fk_dh(self, q):
        theta1, theta2, theta3, theta4 = q

        # TODO: complete with the 4 D-H parameters for every joint and compute the FK
        # T1 = self.dh_transform()
        # T2 = self.dh_transform()
        # T3 = self.dh_transform()
        # T4 = self.dh_transform()
        # T5 = self.dh_transform()

        return np.eye(4)

def plot_trajectory(trajectory, label="RRRP Trajectory"):
    x = [T[0, 3] for T in trajectory]
    y = [T[1, 3] for T in trajectory]
    z = [T[2, 3] for T in trajectory]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label=label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-effector Trajectory")
    ax.legend()
    plt.grid(True)
    plt.show()


def main():
    # TODO: test wihth method='poe' once fk_poe is implemented
    # TODO: test wihth method='dh' once fk_dh and dh_transform are implemented

    robot = RRRP_Robot(method='poe')
    q_traj = generate_rrrp_trajectory(T=5, steps=100)
    trajectory = [robot.fk(q) for q in q_traj]
    plot_trajectory(trajectory)


main()
