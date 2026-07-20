import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import gmsh
from collections import namedtuple

FEMData = namedtuple('FEMData', [
    'nodes',           # (n_nodes, 2) node coordinates (jax)
    'elements',        # (n_elements, 3) element connectivity (jax)
    'boundary_nodes',  # (n_boundary,) boundary node indices (jax)
    'det_J',           # (n_elements,) |det J| per element
    'phys_quad_pts',   # (n_elements, n_quad, 2) quadrature points in physical coords
    'quad_weights',    # (n_quad,) reference quadrature weights
    'basis_vals',      # (n_quad, 3) P1 basis values at quadrature points
    'grad_phys',       # (n_elements, 3, 2) physical gradients of P1 basis functions
    'scatter_rows',    # (n_elements, 9) row indices for global assembly
    'scatter_cols',    # (n_elements, 9) col indices for global assembly
])


def make_blob_mesh(n_boundary=80, resolution=0.15):
    """
    Build a blob-shaped triangular mesh via Gmsh.

    Args:
        n_boundary (int): Number of spline control points on the boundary.
        resolution (float): Target mesh element size.

    Returns:
        tuple: nodes (n,2), elements (m,3), boundary_nodes (b,) as numpy arrays.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("blob")

    theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    r = 1.0 + 0.2 * np.cos(5 * theta) + 0.1 * np.sin(3 * theta)
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)

    tags = []
    for i in range(n_boundary):
        tags.append(gmsh.model.geo.addPoint(xs[i], ys[i], 0, resolution))

    gmsh.model.geo.addSpline(tags + [tags[0]], 1)
    gmsh.model.geo.addCurveLoop([1], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # extract all gmsh nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    all_coords = np.array(coords).reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # extract triangles (only 2D elements use the nodes we care about)
    _, _, tri_node_tags = gmsh.model.mesh.getElements(dim=2)
    raw_tri = np.array(tri_node_tags[0], dtype=int).reshape(-1, 3)

    # keep only nodes appearing in triangles and renumber compactly
    used_tags = np.unique(raw_tri.ravel())
    old_to_new = {int(t): i for i, t in enumerate(used_tags)}
    nodes = np.array([all_coords[tag_to_idx[int(t)]] for t in used_tags])
    elements = np.vectorize(lambda t: old_to_new[int(t)])(raw_tri)

    # boundary nodes: from 1D elements, intersected with used node set
    _, _, bnd_node_tags = gmsh.model.mesh.getElements(dim=1)
    raw_bnd = np.unique(np.array(bnd_node_tags[0], dtype=int))
    boundary_nodes = np.array([old_to_new[int(t)] for t in raw_bnd
                               if int(t) in old_to_new])

    gmsh.finalize()
    return nodes, elements, boundary_nodes


def precompute_fem(nodes, elements, boundary_nodes):
    """
    Precompute all element-wise FEM quantities for P1 triangular elements.

    Args:
        nodes (np.ndarray): (n,2) node coordinates.
        elements (np.ndarray): (m,3) element connectivity.
        boundary_nodes (np.ndarray): (b,) boundary node indices.

    Returns:
        FEMData
    """
    nodes_j = jnp.array(nodes, dtype=jnp.float64)
    elems_j = jnp.array(elements, dtype=jnp.int32)
    bnd_j = jnp.array(boundary_nodes, dtype=jnp.int32)

    # P1 reference triangle: vertices (0,0), (1,0), (0,1)
    # 3-point Gauss quadrature, exact for degree-2 polynomials
    quad_pts = jnp.array([[1.0/6, 1.0/6],
                          [2.0/3, 1.0/6],
                          [1.0/6, 2.0/3]])
    quad_w = jnp.array([1.0/6, 1.0/6, 1.0/6])

    # basis values at quadrature points: phi_0=1-xi-eta, phi_1=xi, phi_2=eta
    xi, eta = quad_pts[:, 0], quad_pts[:, 1]
    basis_vals = jnp.stack([1.0 - xi - eta, xi, eta], axis=1)  # (3_quad, 3_basis)

    # reference gradients (constant for P1)
    grad_ref = jnp.array([[-1.0, -1.0],
                          [ 1.0,  0.0],
                          [ 0.0,  1.0]])  # (3_basis, 2)

    # element geometry
    v0 = nodes_j[elems_j[:, 0]]  # (E, 2)
    v1 = nodes_j[elems_j[:, 1]]
    v2 = nodes_j[elems_j[:, 2]]

    # Jacobian J[e] = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
    J = jnp.stack([v1 - v0, v2 - v0], axis=2)  # (E, 2, 2)
    det_J = jnp.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])

    # J^{-T} for gradient mapping
    J_inv_T = jnp.stack([
        jnp.stack([ J[:, 1, 1], -J[:, 1, 0]], axis=1),
        jnp.stack([-J[:, 0, 1],  J[:, 0, 0]], axis=1),
    ], axis=1) / det_J[:, None, None]  # (E, 2, 2)

    # physical gradients of basis functions: grad_phys[e,i,:] = J_inv_T[e] @ grad_ref[i]
    grad_phys = jnp.einsum('edj,ij->eid', J_inv_T, grad_ref)  # (E, 3, 2)

    # physical quadrature points: x = v0 + J @ [xi, eta]
    phys_quad_pts = v0[:, None, :] + jnp.einsum('eij,qj->eqi', J, quad_pts)

    # scatter indices for global assembly: K_local[i,j] -> K[elems[e,i], elems[e,j]]
    ii = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    jj = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    scatter_rows = elems_j[:, ii]  # (E, 9)
    scatter_cols = elems_j[:, jj]

    return FEMData(
        nodes=nodes_j, elements=elems_j, boundary_nodes=bnd_j,
        det_J=det_J, phys_quad_pts=phys_quad_pts,
        quad_weights=quad_w, basis_vals=basis_vals,
        grad_phys=grad_phys,
        scatter_rows=scatter_rows, scatter_cols=scatter_cols,
    )


def kappa(xy, alpha1, c1):
    """
    Spatially varying diffusion coefficient.

    Args:
        xy (jnp.ndarray): (N,2) evaluation points.
        alpha1 (float): Amplitude of Gaussian bump.
        c1 (jnp.ndarray): (2,) center of Gaussian bump.

    Returns:
        jnp.ndarray: (N,) diffusion values.
    """
    return alpha1 * jnp.exp(-jnp.sum((xy - c1) ** 2, axis=1)) + 1.0


def eta(xy, alpha2, c2):
    """
    Spatially varying wave number coefficient.

    Args:
        xy (jnp.ndarray): (N,2) evaluation points.
        alpha2 (float): Amplitude of Gaussian bump.
        c2 (jnp.ndarray): (2,) center of Gaussian bump.

    Returns:
        jnp.ndarray: (N,) wave number values.
    """
    return alpha2 * jnp.exp(-jnp.sum((xy - c2) ** 2, axis=1)) + 1.0


def forcing(xy):
    """
    Fixed source term f(x,y) = 10 sin(x) cos(y).

    Args:
        xy (jnp.ndarray): (N,2) evaluation points.

    Returns:
        jnp.ndarray: (N,) forcing values.
    """
    return 10 * jnp.sin(xy[:, 0]) * jnp.cos(xy[:, 1])


def assemble_system(fem_data, params):
    """
    Assemble global stiffness matrix K and load vector F for:
    -div(kappa grad u) + eta^2 u = f  (weak form with homogeneous Dirichlet BC).

    Args:
        fem_data (FEMData): Precomputed FEM data.
        params (dict): {'alpha1', 'c1', 'alpha2', 'c2'}.

    Returns:
        tuple: (K, F) — (n_dof, n_dof) matrix and (n_dof,) vector.
    """
    n_dof = fem_data.nodes.shape[0]
    n_elem = fem_data.elements.shape[0]
    n_quad = fem_data.quad_weights.shape[0]

    # evaluate coefficients at all quadrature points
    qpts_flat = fem_data.phys_quad_pts.reshape(-1, 2)
    kap = kappa(qpts_flat, params['alpha1'], params['c1']).reshape(n_elem, n_quad)
    et = eta(qpts_flat, params['alpha2'], params['c2']).reshape(n_elem, n_quad)
    f = forcing(qpts_flat).reshape(n_elem, n_quad)

    # stiffness: gradients are constant per element (P1), so
    # K_stiff[e,i,j] = (sum_q w_q kappa_q) * |detJ| * grad_i . grad_j
    grad_dot = jnp.einsum('eid,ejd->eij',
                          fem_data.grad_phys, fem_data.grad_phys)     # (E,3,3)
    kap_avg = jnp.sum(kap * fem_data.quad_weights[None, :], axis=1)  # (E,)
    K_stiff = grad_dot * (kap_avg * fem_data.det_J)[:, None, None]

    # mass: M[e,i,j] = sum_q w_q eta^2_q phi_i_q phi_j_q * |detJ|
    et2 = et ** 2
    phi_ij = (fem_data.basis_vals[:, :, None]
              * fem_data.basis_vals[:, None, :])                       # (Q,3,3)
    M_local = jnp.einsum('eq,qij,q->eij', et2, phi_ij,
                         fem_data.quad_weights) * fem_data.det_J[:, None, None]

    # weak form: int kappa grad_u . grad_v + int eta^2 u v = int f v
    K_local = K_stiff + M_local

    # load: F[e,i] = sum_q w_q f_q phi_i_q * |detJ|
    F_local = jnp.einsum('eq,qi,q->ei', f, fem_data.basis_vals,
                         fem_data.quad_weights) * fem_data.det_J[:, None]

    # scatter to global matrix and vector
    K = jnp.zeros((n_dof, n_dof)).at[
        fem_data.scatter_rows.ravel(), fem_data.scatter_cols.ravel()
    ].add(K_local.reshape(n_elem, 9).ravel())

    F = jnp.zeros(n_dof).at[fem_data.elements.ravel()].add(F_local.ravel())

    return K, F


def solve_helmholtz(fem_data, params):
    """
    Solve the Helmholtz BVP with homogeneous Dirichlet boundary conditions.

    Args:
        fem_data (FEMData): Precomputed FEM data.
        params (dict): {'alpha1', 'c1', 'alpha2', 'c2'}.

    Returns:
        jnp.ndarray: (n_nodes,) nodal solution.
    """
    K, F = assemble_system(fem_data, params)

    n_dof = fem_data.nodes.shape[0]
    bc = jnp.zeros(n_dof).at[fem_data.boundary_nodes].set(1.0)

    # zero out BC rows/cols and set diagonal to 1
    K = K * (1.0 - bc[:, None]) * (1.0 - bc[None, :]) + jnp.diag(bc)
    F = F * (1.0 - bc)

    return jnp.linalg.solve(K, F)


def locate_points(fem_data, xy_query):
    """
    Find containing element and barycentric coordinates for each query point.
    Uses numpy (not differentiable); use interpolate_solution for diff eval.

    Args:
        fem_data (FEMData): Precomputed FEM data.
        xy_query (array-like): (Q,2) query coordinates.

    Returns:
        tuple: tri_indices (Q,) int32, bary_coords (Q,3) float64.
    """
    nodes = np.array(fem_data.nodes)
    elems = np.array(fem_data.elements)
    xy = np.asarray(xy_query)

    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]

    d00 = v1[:, 0] - v0[:, 0]
    d01 = v2[:, 0] - v0[:, 0]
    d10 = v1[:, 1] - v0[:, 1]
    d11 = v2[:, 1] - v0[:, 1]
    det = d00 * d11 - d01 * d10

    dx = xy[:, None, 0] - v0[None, :, 0]  # (Q, E)
    dy = xy[:, None, 1] - v0[None, :, 1]

    lam1 = ( d11[None, :] * dx - d01[None, :] * dy) / det[None, :]
    lam2 = (-d10[None, :] * dx + d00[None, :] * dy) / det[None, :]
    lam0 = 1.0 - lam1 - lam2

    inside = (lam0 >= -1e-8) & (lam1 >= -1e-8) & (lam2 >= -1e-8)
    tri_idx = np.argmax(inside, axis=1)

    arange = np.arange(xy.shape[0])
    bary = np.stack([lam0[arange, tri_idx],
                     lam1[arange, tri_idx],
                     lam2[arange, tri_idx]], axis=1)

    return jnp.array(tri_idx, dtype=jnp.int32), jnp.array(bary)


def interpolate_solution(fem_data, u_nodes, tri_indices, bary_coords):
    """
    Barycentric interpolation of a nodal field; differentiable w.r.t. u_nodes.

    Args:
        fem_data (FEMData): Precomputed FEM data.
        u_nodes (jnp.ndarray): (n_nodes,) nodal values.
        tri_indices (jnp.ndarray): (Q,) int32 element indices.
        bary_coords (jnp.ndarray): (Q,3) barycentric coordinates.

    Returns:
        jnp.ndarray: (Q,) interpolated values.
    """
    elem_nodes = fem_data.elements[tri_indices]   # (Q, 3)
    u_local = u_nodes[elem_nodes]                 # (Q, 3)
    return jnp.sum(bary_coords * u_local, axis=1)


def evaluate_solution(fem_data, u_nodes, xy_query):
    """
    Locate and interpolate in one call (not differentiable w.r.t. xy_query).

    Args:
        fem_data (FEMData): Precomputed FEM data.
        u_nodes (jnp.ndarray): (n_nodes,) nodal values.
        xy_query (array-like): (Q,2) query points.

    Returns:
        jnp.ndarray: (Q,) solution values.
    """
    tri_idx, bary = locate_points(fem_data, xy_query)
    return interpolate_solution(fem_data, u_nodes, tri_idx, bary)


# ── Functions from helmholtz_parametric_fem.py ────────────────────────────────

def sample_ghost_on_mesh(key, fem_data, n_ghost):
    """
    Sample points uniformly on the mesh (area-weighted by element).

    Args:
        key: JAX PRNG key.
        fem_data (FEMData): Precomputed FEM data.
        n_ghost (int): Number of points to sample.

    Returns:
        tuple: xy (H,2), tri_idx (H,), bary (H,3).
    """
    k1, k2 = jr.split(key)
    probs = fem_data.det_J / jnp.sum(fem_data.det_J)
    tri_idx = jr.choice(k1, fem_data.elements.shape[0],
                        shape=(n_ghost,), p=probs)
    # Convert uniform (u1,u2) on [0,1]^2 to barycentric via sorted-uniform trick
    u = jnp.sort(jr.uniform(k2, (n_ghost, 2)), axis=1)
    bary = jnp.stack([u[:, 0], u[:, 1] - u[:, 0], 1.0 - u[:, 1]], axis=1)
    v0 = fem_data.nodes[fem_data.elements[tri_idx, 0]]
    v1 = fem_data.nodes[fem_data.elements[tri_idx, 1]]
    v2 = fem_data.nodes[fem_data.elements[tri_idx, 2]]
    xy = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2
    return xy, tri_idx, bary


def interp_at_pts(fem_data, u_all, tri_idx, bary):
    """
    Barycentric interpolation supporting batched nodal fields.
    Differentiable w.r.t. u_all.

    Args:
        fem_data (FEMData): Precomputed FEM data.
        u_all (jnp.ndarray): (n_dof,) or (K, n_dof) nodal values.
        tri_idx (jnp.ndarray): (P,) element indices.
        bary (jnp.ndarray): (P,3) barycentric coordinates.

    Returns:
        jnp.ndarray: (P,) or (K, P) interpolated values.
    """
    elem_nodes = fem_data.elements[tri_idx]
    if u_all.ndim == 1:
        return jnp.sum(bary * u_all[elem_nodes], axis=1)
    return jnp.sum(bary[None, :, :] * u_all[:, elem_nodes], axis=2)
