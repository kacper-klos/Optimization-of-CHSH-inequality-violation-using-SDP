import numpy as np
import numpy.typing as npt
import cvxpy as cp

from typing import Tuple

def chsh_state_optimization(A1: npt.NDArray[np.complex128], 
                            A2: npt.NDArray[np.complex128], 
                            B1: npt.NDArray[np.complex128], 
                            B2: npt.NDArray[np.complex128]) -> Tuple[np.float64, npt.NDArray[np.complex128]]:
    # Assert proper input sizes
    assert(A1.shape == (2, 2))
    assert(A2.shape == (2, 2))
    assert(B1.shape == (2, 2))
    assert(B2.shape == (2, 2))

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

    return problem.value, rho.value

# Example

# Pauli matrices
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Solution for measurement which can obtain maximal violation
result_value, result_rho = chsh_state_optimization(Z, X, (Z + X) / np.sqrt(2), (Z - X) / np.sqrt(2))

# Optimal CHSH value
print(f"Max CHSH value: {result_value}")

# Optimal state
threshold = 1e-8
rho_pretty = result_rho.copy()
rho_pretty[np.abs(rho_pretty) < threshold] = 0
print(f"With qubit state:\n{rho_pretty}")
