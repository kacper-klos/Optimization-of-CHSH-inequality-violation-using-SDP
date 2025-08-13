import numpy as np

import state_optimization as so
import measurement_optimization as mo
from const import X, Z, CHSH_MAX, array_pretty

def state_optimization_test():
    """ Prints found optimal state and CHSH violation and the true optimal value of violation """

    # Solution for measurement which can obtain maximal violation
    result_value, result_rho = so.chsh_state_optimization(Z, X, (Z + X) / np.sqrt(2), (Z - X) / np.sqrt(2))
    # Optimal CHSH value
    print(f"Max CHSH value: {result_value}, expected: {CHSH_MAX}, difference: {np.abs(result_value - CHSH_MAX)}")
    # Optimal state
    print(f"With qubit state:\n{array_pretty(result_rho)}\n")

def measurment_optimization_test():
    """ Prints found optimal measurement and the proper optimal measurement settings"""

    # Optimal measurements and state
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
    # Calculate optimal
    A1_opt = mo.chsh_A1_optimize(B1, B2, rho)
    A2_opt = mo.chsh_A2_optimize(B1, B2, rho)
    B1_opt = mo.chsh_B1_optimize(A1, A2, rho)
    B2_opt = mo.chsh_B2_optimize(A1, A2, rho)
    # Compare found state with actual optimal
    print(f"A1 optimized result\n{array_pretty(A1_opt)},\nexpected:\n{A1}\n")
    print(f"A2 optimized result\n{array_pretty(A2_opt)},\nexpected:\n{A2}\n")
    print(f"B1 optimized result\n{array_pretty(B1_opt)},\nexpected:\n{B1}\n")
    print(f"B2 optimized result\n{array_pretty(B2_opt)},\nexpected:\n{B2}\n")

state_optimization_test()
measurment_optimization_test()
