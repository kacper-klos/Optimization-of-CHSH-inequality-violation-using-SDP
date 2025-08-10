import numpy as np
import cvxpy as cp

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Standard measurement basis
A1 = Z
A2 = X
B1 = (Z + X) / np.sqrt(2)
B2 = (Z - X) / np.sqrt(2)

# CHSH operator
CHSH = np.kron(A1, B1 + B2) + np.kron(A2, B1 - B2)

# 2 qubit density matrix
rho = cp.Variable((4, 4), complex=True, hermitian=True)

# Optimization constraints
constrains = [
    # Positive semidefiniteness
    rho >> 0,
    # Probabilities sum to 1
    cp.trace(rho) == 1
]

# Objective function
objective = cp.Maximize(cp.real(cp.trace(CHSH @ rho)))

# Solver
problem = cp.Problem(objective, constrains)
problem.solve(solver = cp.SCS)

# Optimal CHSH value
print(f"Max CHSH value: {problem.value}")

# Optimal state
threshold = 1e-15
rho_pretty = rho.value.copy()
rho_pretty[np.abs(rho.value) < threshold] = 0
print(f"With qubit state:\n{rho_pretty}")
