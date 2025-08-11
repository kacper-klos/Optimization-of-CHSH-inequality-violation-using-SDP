import numpy as np
import numpy.typing as npt
import cvxpy as cp

def A_partial_trace(M: np.ndarray) -> np.ndarray:
    return (M[:2, :2] + M[2:, 2:]).copy()

def B_partial_trace(M: np.ndarray) -> np.ndarray:
    return np.array(
        [[np.trace(M[:2, :2]), np.trace(M[:2, 2:])],
         [np.trace(M[2:, :2]), np.trace(M[2:, 2:])]]
    )
