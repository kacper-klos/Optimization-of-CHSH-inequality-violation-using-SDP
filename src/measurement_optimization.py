import numpy as np

def A_partial_trace(M: np.ndarray) -> np.ndarray:
    """ Calculate partial trace over first matrix of 4x4 matrix
    resulting from kronecker product of two 2x2 matrix.

    Args:
        M: Matrix which partial trace will be calculated, must be 4x4

    Returns:
        2x2 array resulting from partial trace
    """

    # Assert proper input shape
    assert(M.shape == (4, 4))
    return (M[:2, :2] + M[2:, 2:]).copy()

def B_partial_trace(M: np.ndarray) -> np.ndarray:
    """ Calculate partial trace over second matrix of 4x4 matrix
    resulting from kronecker product of two 2x2 matrix.

    Args:
        M: Matrix which partial trace will be calculated, must be 4x4

    Returns:
        2x2 array resulting from partial trace
    """

    # Assert proper input shape
    assert(M.shape == (4, 4))
    return np.array(
        [[np.trace(M[:2, :2]), np.trace(M[:2, 2:])],
         [np.trace(M[2:, :2]), np.trace(M[2:, 2:])]]
    )

def optimize_trace_product(K: np.ndarray) -> np.ndarray:
    """ Finds optimla matrix A to maximize Tr(AK) where A have eigenvalues in range [-1; 1].

    Args:
        K: Matrix with which product is maximize.

    Returns:
        Optimal matrix for maximizing trace product with same size as K.
    """

    # Assert K to be square matrix
    assert(len(K.shape) == 2 and K.shape[0] == K.shape[1])

    eigenvalues, U = np.linalg.eigh(K)
    A_diag = np.diag(np.sign(eigenvalues))
    return U @ A_diag @ U.conj().T

def first_subspace_K(Bs: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Finds K_1 described in the README.

    Args:
        Bs: Expression involving measurement in second subspace (B1 + B2 or B1 - B2).
        rho: Density matrix of a quantum state.

    Returns:
        2x2 K_1 described in README.
    """

    # Assert proper input shape
    assert(Bs.shape == (2, 2))
    assert(rho.shape == (4, 4))

    M = np.kron(np.eye(2), Bs) @ rho
    return B_partial_trace(M)

def second_subspace_K(As: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Finds K_2 described in the README.

    Args:
        As: Expression involving measurement in second subspace (A1 + A2 or A1 - A2).
        rho: Density matrix of a quantum state.

    Returns:
        2x2 K_2 described in README.
    """
    # Assert proper input shape
    assert(As.shape == (2, 2))
    assert(rho.shape == (4, 4))

    M = np.kron(As, np.eye(2)) @ rho
    return A_partial_trace(M)

def chsh_A1_optimize(B1: np.ndarray, B2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Finds value of A1 measurement setting, which maximizes CHSH inequality.

    Args:
        B1: First Bob measurement setting.
        B2: Second Bob measurement setting.
        rho: Quantum state.

    Returns:
        Optimal Alice first measurement setting.
    """

    # Assert proper input shape
    assert(B1.shape == (2, 2))
    assert(B2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = first_subspace_K(B1 + B2, rho)
    return optimize_trace_product(K)

def chsh_A2_optimize(B1: np.ndarray, B2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Finds value of A2 measurement setting, which maximizes CHSH inequality.

    Args:
        B1: First Bob measurement setting.
        B2: Second Bob measurement setting.
        rho: Quantum state.

    Returns:
        Optimal Alice second measurement setting.
    """
    # Assert proper input shape
    assert(B1.shape == (2, 2))
    assert(B2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = first_subspace_K(B1 - B2, rho)
    return optimize_trace_product(K)

def chsh_B1_optimize(A1: np.ndarray, A2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Finds value of B1 measurement setting, which maximizes CHSH inequality.

    Args:
        A1: First Alice measurement setting.
        A2: Second Alice measurement setting.
        rho: Quantum state.

    Returns:
        Optimal Bob first measurement setting.
    """

    # Assert proper input shape
    assert(A1.shape == (2, 2))
    assert(A2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = second_subspace_K(A1 + A2, rho)
    return optimize_trace_product(K)

def chsh_B2_optimize(A1: np.ndarray, A2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Finds value of B2 measurement setting, which maximizes CHSH inequality.

    Args:
        A1: First Alice measurement setting.
        A2: Second Alice measurement setting.
        rho: Quantum state.

    Returns:
        Optimal Bob second measurement setting.
    """

    # Assert proper input shape
    assert(A1.shape == (2, 2))
    assert(A2.shape == (2, 2))
    assert(rho.shape == (4, 4))

    K = second_subspace_K(A1 - A2, rho)
    return optimize_trace_product(K)
