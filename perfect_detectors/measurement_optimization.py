import numpy as np
import numpy.typing as npt
import cvxpy as cp

def A_partial_trace(M: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(M.shape == (4, 4))

    return (M[:2, :2] + M[2:, 2:]).copy()

def B_partial_trace(M: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(M.shape == (4, 4))

    return np.array(
        [[np.trace(M[:2, :2]), np.trace(M[:2, 2:])],
         [np.trace(M[2:, :2]), np.trace(M[2:, 2:])]]
    )

def optimize_trace_product(K: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(K.shape == (2, 2))

    eigenvalues, U = np.linalg.eigh(K)
    A_diag = np.diag(np.sign(eigenvalues))
    return U.conj().T @ A_diag @ U

def first_subspace_K(Bs: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(Bs.shape == (2, 2))
    assert(rho.shape == (4, 4))

    M = np.kron(np.eye(2), Bs) @ rho
    return B_partial_trace(M)

def second_subspace_K(As: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(As.shape == (2, 2))
    assert(rho.shape == (4, 4))

    M = np.kron(As, np.eye(2)) @ rho
    return A_partial_trace(M)

def chsh_A1_optimize(B1: np.ndarray, B2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(B1.shape == (2, 2))
    assert(B2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = first_subspace_K(B1 + B2, rho)
    return optimize_trace_product(K)

def chsh_A2_optimize(B1: np.ndarray, B2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(B1.shape == (2, 2))
    assert(B2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = first_subspace_K(B1 - B2, rho)
    return optimize_trace_product(K)

def chsh_B1_optimize(A1: np.ndarray, A2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(A1.shape == (2, 2))
    assert(A2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = second_subspace_K(A1 + A2, rho)
    return optimize_trace_product(K)

def chsh_B2_optimize(A1: np.ndarray, A2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    # Assert proper input shape
    assert(A1.shape == (2, 2))
    assert(A2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = second_subspace_K(A1 - A2, rho)
    return optimize_trace_product(K)

