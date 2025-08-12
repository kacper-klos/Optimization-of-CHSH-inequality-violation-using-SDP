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
ERROR_STATE = np.outer(np.kron(ZERO_KET, ZERO_KET), np.kron(ZERO_KET, ZERO_KET))

# Make random valid measurement matrix
def random_measurement():
    v = np.random.normal(size = 3)
    v /= np.linalg.norm(v)
    return X*v[0] + Y *v[1] + Z * v[2]

def array_pretty(A: np.ndarray) -> np.ndarray:
    A_copy = A.copy()
    
    small_real = np.abs(A_copy.real) < THRESHOLD
    A_copy.real[small_real] = 0
    
    small_imag = np.abs(A_copy.imag) < THRESHOLD
    A_copy.imag[small_imag] = 0
    
    return A_copy
