import numpy as np
from typing import Tuple

import state_optimization as so
import measurement_optimization as mo
from const import random_measurement, ERROR_STATE, LOCAL_MAX, array_pretty

def iterative_optimization_test(mu: float = 1,
                                error_state: np.ndarray = ERROR_STATE) -> Tuple[float, np.ndarray, np.ndarray]:
    A1 = random_measurement()
    A2 = random_measurement()
    B1 = random_measurement()
    B2 = random_measurement()
    result_old = -1
    result = 0
    rho = np.eye(4)

    while (result - result_old) / max(1, result_old) > 1e-4:
        result_old = result
        result, rho = so.chsh_state_optimization(A1, A2, B1, B2, mu, error_state)
        A1 = mo.chsh_A1_optimize(B1, B2, rho)
        A2 = mo.chsh_A2_optimize(B1, B2, rho)
        result, rho = so.chsh_state_optimization(A1, A2, B1, B2, mu, error_state)
        B1 = mo.chsh_B1_optimize(A1, A2, rho)
        B2 = mo.chsh_B2_optimize(A1, A2, rho)

    return result, array_pretty(rho), array_pretty(np.stack([A1, A2, B1, B2], axis = 0))

def testing_mu():

    measurement_setting_best = ()
    rho_best = np.eye(4)
    mu_best = 1.0
    violation_best = 0

    for i in range(43, 41, -1):
        mu = i/100
        max_violation = 0

        for _ in range(5000):
            max_violation, rho, measurement_setting = iterative_optimization_test(mu)
            if max_violation > LOCAL_MAX:
                violation_best = max_violation
                mu_best = mu
                measurement_setting_best = measurement_setting
                rho_best = rho

                break

    print(f"Low bound on detector accurancy: {mu_best}")
    print(f"Resulting in violation: {violation_best}")
    print(f"Obtrained with the state:\n{rho_best}")
    print(f"And measurement settings:\n{measurement_setting_best}")

testing_mu()
