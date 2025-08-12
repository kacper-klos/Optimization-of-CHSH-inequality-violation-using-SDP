import numpy as np
import cvxpy as cp
from typing import Tuple

def chsh_state_optimization(A1: np.ndarray, 
                            A2: np.ndarray, 
                            B1: np.ndarray, 
                            B2: np.ndarray,
                            mu: float = 1.0,
                            error_state: np.ndarray = np.zeros((4,4))) -> Tuple[np.float64, np.ndarray]:

    # Assert proper input sizes
    assert(A1.shape == (2, 2))
    assert(A2.shape == (2, 2))
    assert(B1.shape == (2, 2))
    assert(B2.shape == (2, 2))
    assert(error_state.shape == (4, 4))
    # Assert proper fraction
    assert(mu >= 0 and mu <= 1)

    # CHSH operator
    CHSH = np.kron(A1, B1 + B2) + np.kron(A2, B1 - B2)

    # 2 qubit density matrix
    rho = cp.Variable((4, 4), complex=True, hermitian=True)
    rho_effective = mu * rho + (1 - mu) * error_state

    # Optimization constraints
    constrains = [
        # Positive semidefiniteness
        rho >> 0,
        # Probabilities sum to 1
        cp.trace(rho) == 1
    ]

    # Objective function
    objective = cp.Maximize(cp.real(cp.trace(CHSH @ rho_effective)))

    # Solver
    problem = cp.Problem(objective, constrains)
    problem.solve(solver = cp.SCS)

    return problem.value, rho.value

