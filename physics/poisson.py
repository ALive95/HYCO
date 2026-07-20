"""
Poisson BVP solver on a blob mesh with Chebyshev-basis forcing term.

Solves -Δu = f,  u = 0 on ∂Ω,
where f(x,y) = Σ c_{ij} T_i(x̃) T_j(ỹ),  (x̃, ỹ) ∈ [-1,1]² are bbox-normalised coords.

The N×N coefficient vector f_coeff (phy_params['f_coeff']) is the unknown to recover.
"""

import jax.numpy as jnp
import numpy as np
from collections import namedtuple

# Reuse mesh / FEM utilities from Helmholtz — no duplication.
from physics.helmholtz_fem import (
    FEMData,
    make_blob_mesh,
    precompute_fem,
    locate_points,
    interp_at_pts,
    sample_ghost_on_mesh,
)


PoissonData = namedtuple('PoissonData', [
    'fem',           # FEMData — mesh and precomputed FEM quantities
    'K_bc',          # (n_dof, n_dof) stiffness matrix with Dirichlet BCs applied
    'bc_mask',       # (n_dof,) 1.0 at boundary nodes, 0.0 at interior
    'cheb_at_quad',  # (E*Q, N*N) Chebyshev basis evaluated at all quadrature points
    'N',             # int — Chebyshev modes per direction; total coefficients = N*N
    'bbox',          # (2, 2) array — [[xmin, xmax], [ymin, ymax]] for normalisation
])


# ── Chebyshev helpers ─────────────────────────────────────────────────────────

def _chebyshev_polys(x, N):
    """
    Evaluate Chebyshev polynomials T_0, ..., T_{N-1} at points x via recurrence.

    Args:
        x (jnp.ndarray): (Q,) values in [-1, 1].
        N (int): Number of polynomials.

    Returns:
        jnp.ndarray: (Q, N) array of polynomial values.
    """
    if N == 1:
        return jnp.ones((x.shape[0], 1))
    T = [jnp.ones_like(x), x]
    for n in range(2, N):
        T.append(2.0 * x * T[-1] - T[-2])
    return jnp.stack(T[:N], axis=1)  # (Q, N)


def chebyshev_basis_2d(xy, N, bbox):
    """
    Evaluate the N² tensor-product Chebyshev basis {T_i(x̃) T_j(ỹ)} at Q points.

    Args:
        xy (jnp.ndarray): (Q, 2) physical coordinates.
        N (int): Modes per direction.
        bbox (jnp.ndarray): (2, 2) array [[xmin, xmax], [ymin, ymax]].

    Returns:
        jnp.ndarray: (Q, N*N) basis evaluations,
                     column order: (i, j) = (0,0), (0,1), ..., (N-1, N-1).
    """
    # map physical coords to [-1, 1]
    x_s = 2.0 * (xy[:, 0] - bbox[0, 0]) / (bbox[0, 1] - bbox[0, 0]) - 1.0
    y_s = 2.0 * (xy[:, 1] - bbox[1, 0]) / (bbox[1, 1] - bbox[1, 0]) - 1.0

    Tx = _chebyshev_polys(x_s, N)   # (Q, N)
    Ty = _chebyshev_polys(y_s, N)   # (Q, N)

    # outer product over mode indices
    basis = Tx[:, :, None] * Ty[:, None, :]  # (Q, N, N)
    return basis.reshape(xy.shape[0], N * N)  # (Q, N²)


def eval_f(xy, f_coeff, N, bbox):
    """
    Evaluate f(x,y) = Σ c_k φ_k(x,y) at Q points.

    Args:
        xy (jnp.ndarray): (Q, 2) evaluation points.
        f_coeff (jnp.ndarray): (N*N,) Chebyshev coefficients.
        N (int): Modes per direction.
        bbox (jnp.ndarray): (2, 2) bounding box.

    Returns:
        jnp.ndarray: (Q,) forcing values.
    """
    return chebyshev_basis_2d(xy, N, bbox) @ f_coeff


# ── FEM assembly ──────────────────────────────────────────────────────────────

def _assemble_stiffness(fem):
    """
    Assemble the global stiffness matrix K for -Δu (kappa = 1 everywhere).

    For P1 elements the gradient is constant per element, so:
      K_local[e,i,j] = (∇φ_i · ∇φ_j) * |detJ_e| * (area of reference triangle)

    The area of the reference triangle equals sum(quad_weights) = 1/2.

    Args:
        fem (FEMData): Precomputed FEM data.

    Returns:
        jnp.ndarray: (n_dof, n_dof) raw stiffness matrix (no BCs).
    """
    n_dof  = fem.nodes.shape[0]
    n_elem = fem.elements.shape[0]

    # grad_dot[e,i,j] = ∇φ_i · ∇φ_j  (constant per element for P1)
    grad_dot = jnp.einsum('eid,ejd->eij', fem.grad_phys, fem.grad_phys)  # (E,3,3)

    # element area in physical space = det_J * (1/2)  [ref triangle area = 1/2]
    area_weight = fem.det_J * jnp.sum(fem.quad_weights)  # (E,)

    K_local = grad_dot * area_weight[:, None, None]  # (E,3,3)

    K = jnp.zeros((n_dof, n_dof)).at[
        fem.scatter_rows.ravel(), fem.scatter_cols.ravel()
    ].add(K_local.reshape(n_elem, 9).ravel())

    return K


def _apply_dirichlet(fem, K, F):
    """
    Enforce homogeneous Dirichlet BCs by zeroing boundary rows/cols and
    setting the diagonal to 1, and zeroing the corresponding RHS entries.

    Args:
        fem (FEMData): Precomputed FEM data.
        K (jnp.ndarray): (n_dof, n_dof) raw stiffness matrix.
        F (jnp.ndarray): (n_dof,) raw load vector.

    Returns:
        tuple: K_bc (n_dof, n_dof), F_bc (n_dof,), bc_mask (n_dof,).
    """
    n_dof = fem.nodes.shape[0]
    bc = jnp.zeros(n_dof).at[fem.boundary_nodes].set(1.0)
    K_bc = K * (1.0 - bc[:, None]) * (1.0 - bc[None, :]) + jnp.diag(bc)
    F_bc = F * (1.0 - bc)
    return K_bc, F_bc, bc


def _assemble_load(fem, f_at_quad):
    """
    Assemble global load vector F[i] = Σ_e Σ_q w_q f(x_q) φ_i(x_q) |detJ_e|.

    Args:
        fem (FEMData): Precomputed FEM data.
        f_at_quad (jnp.ndarray): (E, Q) forcing values at quadrature points.

    Returns:
        jnp.ndarray: (n_dof,) load vector (no BCs).
    """
    n_dof = fem.nodes.shape[0]
    n_elem = fem.elements.shape[0]

    # F_local[e,i] = Σ_q w_q f_q φ_i(x_q) |detJ_e|
    F_local = jnp.einsum('eq,qi,q->ei', f_at_quad,
                         fem.basis_vals, fem.quad_weights) * fem.det_J[:, None]

    return jnp.zeros(n_dof).at[fem.elements.ravel()].add(F_local.ravel())


# ── Public interface ──────────────────────────────────────────────────────────

def precompute_poisson(fem, N):
    """
    Precompute all quantities that do not depend on f_coeff.

    Args:
        fem (FEMData): Precomputed FEM data.
        N (int): Chebyshev modes per direction (total unknowns = N²).

    Returns:
        PoissonData
    """
    # bounding box for Chebyshev normalisation
    nodes_np = np.array(fem.nodes)
    bbox = jnp.array([
        [nodes_np[:, 0].min(), nodes_np[:, 0].max()],
        [nodes_np[:, 1].min(), nodes_np[:, 1].max()],
    ])

    # precompute Chebyshev basis at all quadrature points — shape (E*Q, N²)
    qpts_flat = fem.phys_quad_pts.reshape(-1, 2)   # (E*Q, 2)
    cheb_at_quad = chebyshev_basis_2d(qpts_flat, N, bbox)

    # stiffness matrix with BCs (assembled once; independent of f)
    K_raw = _assemble_stiffness(fem)
    F_dummy = jnp.zeros(fem.nodes.shape[0])
    K_bc, _, bc_mask = _apply_dirichlet(fem, K_raw, F_dummy)

    return PoissonData(
        fem=fem,
        K_bc=K_bc,
        bc_mask=bc_mask,
        cheb_at_quad=cheb_at_quad,
        N=N,
        bbox=bbox,
    )


def solve_poisson_from_coeff(pdata, f_coeff):
    """
    Solve -Δu = f,  u|_∂Ω = 0, where f is given by Chebyshev coefficients.
    Differentiable w.r.t. f_coeff.

    Args:
        pdata (PoissonData): Precomputed Poisson data.
        f_coeff (jnp.ndarray): (N*N,) Chebyshev coefficients.

    Returns:
        jnp.ndarray: (n_dof,) nodal solution.
    """
    n_elem = pdata.fem.elements.shape[0]
    n_quad = pdata.fem.quad_weights.shape[0]

    # evaluate f at quadrature points via precomputed Chebyshev basis
    f_at_quad = (pdata.cheb_at_quad @ f_coeff).reshape(n_elem, n_quad)  # (E, Q)

    F_raw = _assemble_load(pdata.fem, f_at_quad)
    F_bc  = F_raw * (1.0 - pdata.bc_mask)

    return jnp.linalg.solve(pdata.K_bc, F_bc)


def solve_poisson_from_fn(fem, f_fn):
    """
    Solve -Δu = f_fn(x,y) with homogeneous Dirichlet BCs.
    Used to generate the reference solution; NOT differentiable w.r.t. f_fn.

    Args:
        fem (FEMData): Precomputed FEM data.
        f_fn (callable): (Q, 2) -> (Q,) forcing function.

    Returns:
        jnp.ndarray: (n_dof,) nodal reference solution.
    """
    qpts_flat = fem.phys_quad_pts.reshape(-1, 2)
    n_elem = fem.elements.shape[0]
    n_quad = fem.quad_weights.shape[0]
    f_at_quad = f_fn(qpts_flat).reshape(n_elem, n_quad)

    K_raw = _assemble_stiffness(fem)
    F_raw = _assemble_load(fem, f_at_quad)
    K_bc, F_bc, _ = _apply_dirichlet(fem, K_raw, F_raw)

    return jnp.linalg.solve(K_bc, F_bc)