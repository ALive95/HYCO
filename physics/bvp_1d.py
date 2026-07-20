"""
1D second-order BVP on [0, 1]:
  -λ u''(x) + k(x) u(x) = f(x),  u(0) = u(1) = 0.

k(x) is parametrised as a truncated Chebyshev series.
The FD system is assembled as a dense tridiagonal matrix and solved with
jnp.linalg.solve — fully JAX-differentiable w.r.t. k_vals.
"""
import jax.numpy as jnp


def make_grid(n):
    """
    Interior FD grid: x_i = i/(n+1), i = 1, ..., n.

    Args:
        n (int): Number of interior points.

    Returns:
        jnp.ndarray: (n,) grid coordinates in (0, 1).
    """
    return jnp.linspace(1.0 / (n + 1), n / (n + 1), n)


def chebyshev_basis_1d(x_01, p):
    """
    Chebyshev basis T_0, ..., T_{p-1} on [0, 1] via three-term recurrence.
    Maps x ∈ [0, 1] to xi = 2x - 1 ∈ [-1, 1] before evaluating.

    Args:
        x_01 (jnp.ndarray): (n,) points in [0, 1].
        p (int): Number of modes (static — determines unroll depth at JIT trace).

    Returns:
        jnp.ndarray: (n, p) basis matrix.
    """
    xi = 2.0 * x_01 - 1.0
    if p == 1:
        return jnp.ones((xi.shape[0], 1))
    T = [jnp.ones_like(xi), xi]
    for j in range(2, p):
        T.append(2.0 * xi * T[-1] - T[-2])
    return jnp.stack(T[:p], axis=1)


def eval_k_cheb(x_01, k_coeff):
    """
    Evaluate k(x) = Σ_j k_coeff[j] T_j(2x - 1).

    Args:
        x_01 (jnp.ndarray): (n,) points in [0, 1].
        k_coeff (jnp.ndarray): (p,) Chebyshev coefficients.

    Returns:
        jnp.ndarray: (n,) k values.
    """
    return chebyshev_basis_1d(x_01, k_coeff.shape[0]) @ k_coeff


def solve_bvp_1d(k_vals, f_vals, lam):
    """
    Solve -λu'' + k(x)u = f on the interior FD grid, u(0) = u(1) = 0.

    Assembles a dense tridiagonal system — correct for n ≤ ~512.
    Differentiable w.r.t. k_vals and f_vals via jnp.linalg.solve.

    Args:
        k_vals (jnp.ndarray): (n,) k at interior grid points.
        f_vals (jnp.ndarray): (n,) f at interior grid points.
        lam (float): Diffusion coefficient λ > 0.

    Returns:
        jnp.ndarray: (n,) solution u at interior grid points.
    """
    n = k_vals.shape[0]
    h = 1.0 / (n + 1)
    c = lam / h ** 2
    # main diagonal: 2λ/h² + k_i;  off-diagonals: -λ/h²
    diag = 2.0 * c + k_vals
    off  = -c * jnp.ones(n - 1)
    A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    return jnp.linalg.solve(A, f_vals)


def interp_fd(u_fd, n, x_query):
    """
    Linear interpolation of the FD solution (extended with boundary zeros)
    at arbitrary points x_query ∈ [0, 1].

    Args:
        u_fd (jnp.ndarray): (n,) interior solution values.
        n (int): Interior grid size (static).
        x_query (jnp.ndarray): (M,) query points in [0, 1].

    Returns:
        jnp.ndarray: (M,) interpolated values.
    """
    h = 1.0 / (n + 1)
    # extend with Dirichlet zeros at x=0 and x=1
    u_ext = jnp.concatenate([jnp.zeros(1), u_fd, jnp.zeros(1)])  # (n+2,)
    idx = x_query / h
    lo  = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, n)
    frac = idx - lo
    return (1.0 - frac) * u_ext[lo] + frac * u_ext[lo + 1]