import numpy as np

# Pauli matrices
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Maximum possible CHSH value
CHSH_MAX = 2 * np.sqrt(2)

# Threshold for printing values
THRESHOLD = 1e-8
