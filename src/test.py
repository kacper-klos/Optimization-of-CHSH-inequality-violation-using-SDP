import numpy as np

import state_optimization as so
import measurement_optimization as mo
from const import X, Y, Z, CHSH_MAX, THRESHOLD

def random_measurement():
    v = np.random.normal(size = 3)
    v /= np.linalg.norm(v)
    return X*v[0] + Y *v[1] + Z * v[2]

def state_optimization_test():

    # Solution for measurement which can obtain maximal violation
    result_value, result_rho = so.chsh_state_optimization(Z, X, (Z + X) / np.sqrt(2), (Z - X) / np.sqrt(2))
    # Optimal CHSH value
    print(f"Max CHSH value: {result_value}, expected: {CHSH_MAX}, difference: {np.abs(result_value - CHSH_MAX)}")

    # Optimal state
    rho_pretty = result_rho.copy()
    rho_pretty[np.abs(rho_pretty) < THRESHOLD] = 0
    print(f"With qubit state:\n{rho_pretty}\n")

def measurment_optimization_test():
    A1 = Z
    A2 = X
    B1 = (Z + X) / np.sqrt(2)
    B2 = (Z - X) / np.sqrt(2)
    rho = 0.5 * np.array(
        [[1, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]
    )

    A1_opt = mo.chsh_A1_optimize(B1, B2, rho)
    A2_opt = mo.chsh_A2_optimize(B1, B2, rho)
    B1_opt = mo.chsh_B1_optimize(A1, A2, rho)
    B2_opt = mo.chsh_B2_optimize(A1, A2, rho)

    print(f"A1 optimized result\n{A1_opt},\nexpected:\n{A1}\n")
    print(f"A2 optimized result\n{A2_opt},\nexpected:\n{A2}\n")
    print(f"B1 optimized result\n{B1_opt},\nexpected:\n{B1}\n")
    print(f"B2 optimized result\n{B2_opt},\nexpected:\n{B2}\n")

def iterative_optimization_test():
    A1 = random_measurement()
    A2 = random_measurement()
    B1 = random_measurement()
    B2 = random_measurement()
    result = 0

    for i in range(100):
        result, rho = so.chsh_state_optimization(A1, A2, B1, B2)
        if i % 10 == 0:
            print(result)
        A1 = mo.chsh_A1_optimize(B1, B2, rho)
        A2 = mo.chsh_A2_optimize(B1, B2, rho)
        result, rho = so.chsh_state_optimization(A1, A2, B1, B2)
        B1 = mo.chsh_B1_optimize(A1, A2, rho)
        B2 = mo.chsh_B2_optimize(A1, A2, rho)

    print(f"Max CHSH value: {result}, expected: {CHSH_MAX}, difference: {np.abs(result - CHSH_MAX)}")


state_optimization_test()
measurment_optimization_test()
iterative_optimization_test()
