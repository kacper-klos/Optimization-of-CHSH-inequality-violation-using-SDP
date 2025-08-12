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

