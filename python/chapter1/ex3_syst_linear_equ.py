import numpy as np

"""
This is the solution for the 4th exercise of chapter 1.
"""

# A function to print the results


def print_values(solution, msg):
    print(msg)
    variable_names = ['x', 'y', 'z', 'w']
    for var, val in zip(variable_names, solution):
        print(f"{var} = {val:.3f}")


# Input - Coefficient matrix A
A = np.array([
    [1,  2, -1,  1],  # row1
    [2, -1,  1,  3],  # row2
    [3,  1,  2, -1],  # row3
    [-1, 4,  1,  2]  # row4
])

# Right-hand side vector b
b = np.array([4, 1, 7, 5])

# Solve the system A x = b with np.linalg.solve
solution_method1 = np.linalg.solve(A, b)
print_values(solution_method1, "with solve() method")

# Explicitely compute the inverse
A_inv = np.linalg.inv(A)
solution_method2 = A_inv @ b
print_values(solution_method2, "Explicit inverse computation")
