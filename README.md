# Optimization of CHSH Inequality Violation Using SDP

This repository presents code for finding the most optimal quantum state and measurement device settings to achieve the maximum [CHSH inequality](https://en.wikipedia.org/wiki/CHSH_inequality)[^1] violation. The main objective is to determine the minimum detector accuracy that still allows detection of any CHSH violation. The code relies on [semidefinite programming (SDP)](https://en.wikipedia.org/wiki/Semidefinite_programming)[^2][^3] implemented in [CVXPY](https://www.cvxpy.org/)[^4].

## Perfect Measurement Devices

At the beginning, we will not consider imperfect detectors. First, we will search only for the optimal state while keeping the measurement settings fixed. Then, we will look for both the optimal state and the optimal measurement settings.

### Optimizing the State

This is the simplest case: the problem is a basic SDP because the CHSH inequality can be expressed as[^5]

$$
C = A_1 \otimes (B_1 + B_2) + A_2 \otimes (B_1 - B_2)
$$

where $A_1$ and $A_2$ are one party’s measurement settings, and $B_1$ and $B_2$ are the other’s. The task is to maximize

$$
\text{tr}(C \rho)
$$

where $\rho$ is the [density matrix](https://en.wikipedia.org/wiki/Density_matrix)[^6][^7] of the quantum state. The probabilities must sum to 1, giving the constraint

$$
\text{tr}(\rho) = 1
$$

The optimization code is in `perfect_detectors/state_optimization.py`, where the chosen measurement bases are well known for violating the CHSH inequality[^8].

---

# References
[^1]: https://en.wikipedia.org/wiki/CHSH_inequality  
[^2]: https://en.wikipedia.org/wiki/Semidefinite_programming  
[^3]: Robert M. Freund, *Introduction to Semidefinite Programming (SDP)*, 2004, https://ocw.mit.edu/courses/15-084j-nonlinear-programming-spring-2004/a632b565602fd2eb3be574c537eea095_lec23_semidef_opt.pdf  
[^4]: https://www.cvxpy.org/index.html  
[^5]: https://qubit.guide/6.3-chsh-inequality  
[^6]: https://en.wikipedia.org/wiki/Density_matrix  
[^7]: Caltech Ph219/CS219 Quantum Computation, https://www.preskill.caltech.edu/ph219/ph219_2021-22.html  
[^8]: https://en.wikipedia.org/wiki/Bell%27s_theorem
