import numpy as np

# Pauli matrices
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Maximum possible CHSH value
CHSH_MAX = 2 * np.sqrt(2)
LOCAL_MAX = 2

# Threshold for printing values
THRESHOLD = 1e-8

# Detector error state
ZERO_KET = np.array([1, 0])
ERROR_STATE = np.identity(4) / 4

def random_measurement() -> np.ndarray:
    """ Returns random matrix which is valid measurement setting.

    Returns:
        Random measurement.
    """
    v = np.random.normal(size = 3)
    v /= np.linalg.norm(v)
    return X*v[0] + Y *v[1] + Z * v[2]

def array_pretty(A: np.ndarray, threshold = THRESHOLD) -> np.ndarray:
    """ Set zero to every value in array copy below threshold.

    Args:
        A: array which values will be set to zero.
        threshold: value under which values will be set to zero

    Returns:
        Copy with values set to zero.
    """

    A_copy = A.copy()
    # Zero real values
    small_real = np.abs(A_copy.real) < THRESHOLD
    A_copy.real[small_real] = 0
    # Zero imaginary values
    small_imag = np.abs(A_copy.imag) < THRESHOLD
    A_copy.imag[small_imag] = 0
    
    return A_copy
