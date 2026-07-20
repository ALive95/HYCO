import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

FVData = namedtuple('FVData', [
    'cell_centers',  # (n_cells,) midpoints of each cell
    'cell_edges',    # (n_cells+1,) interface positions
    'dx',            # scalar cell width
    'n_cells',       # int number of interior cells
    'ghost',         # int number of ghost cells per side
])


def make_mesh(x_left, x_right, n_cells):
    """
    Build a uniform 1D finite-volume mesh.

    Args:
        x_left (float): Left boundary.
        x_right (float): Right boundary.
        n_cells (int): Number of cells.

    Returns:
        tuple: cell_centers (n_cells,), cell_edges (n_cells+1,) as numpy arrays.
    """
    cell_edges = np.linspace(x_left, x_right, n_cells + 1)
    cell_centers = 0.5 * (cell_edges[:-1] + cell_edges[1:])
    return cell_centers, cell_edges


def precompute_fv(cell_centers, cell_edges, ghost=1):
    """
    Package mesh arrays into an FVData namedtuple.

    Args:
        cell_centers (np.ndarray): (n,) cell midpoints.
        cell_edges (np.ndarray): (n+1,) interface positions.
        ghost (int): Number of ghost cells per side.

    Returns:
        FVData
    """
    dx = cell_edges[1] - cell_edges[0]
    return FVData(
        cell_centers=jnp.array(cell_centers, dtype=jnp.float64),
        cell_edges=jnp.array(cell_edges, dtype=jnp.float64),
        dx=jnp.float64(dx),
        n_cells=cell_centers.shape[0],
        ghost=ghost,
    )


def flux(u, params):
    """
    Evaluate the physical flux f(u) by dispatching to params['flux_fn'].

    Args:
        u (jnp.ndarray): (N,) cell values.
        params (dict): Must contain 'flux_fn'.

    Returns:
        jnp.ndarray: (N,) flux values.
    """
    return params['flux_fn'](u, params)


def burgers_flux(u, params):
    """
    Burgers flux f(u) = u^2 / 2.

    Args:
        u (jnp.ndarray): (N,) cell values.
        params (dict): Unused.

    Returns:
        jnp.ndarray: (N,) flux values.
    """
    return 0.5 * u ** 2


def numerical_flux(u_left, u_right, params):
    """
    Lax-Friedrichs numerical flux at each interface.

    Args:
        u_left (jnp.ndarray): (N,) left states.
        u_right (jnp.ndarray): (N,) right states.
        params (dict): Must contain 'flux_fn' and 'max_wave_speed_fn'.

    Returns:
        jnp.ndarray: (N,) numerical flux values.
    """
    f_left = flux(u_left, params)
    f_right = flux(u_right, params)
    max_speed = params['max_wave_speed_fn'](u_left, u_right, params)
    return 0.5 * (f_left + f_right) - 0.5 * max_speed * (u_right - u_left)


def burgers_max_wave_speed(u_left, u_right, params):
    """
    Wave speed estimate for Burgers: max |f'(u)| = max |u|.

    Args:
        u_left (jnp.ndarray): (N,)
        u_right (jnp.ndarray): (N,)
        params (dict): Unused.

    Returns:
        jnp.ndarray: (N,) wave speeds.
    """
    return jnp.maximum(jnp.abs(u_left), jnp.abs(u_right))


def lwr_flux(u, params):
    """
    LWR Greenshields traffic flux: f(u) = v_max * u * (1 - u/rho_max).

    Args:
        u (jnp.ndarray): (N,) density.
        params (dict): Must contain 'v_max' and 'rho_max'.

    Returns:
        jnp.ndarray: (N,) flux values.
    """
    return params['v_max'] * u * (1.0 - u / params['rho_max'])


def lwr_max_wave_speed(u_left, u_right, params):
    """
    Wave speed estimate for LWR: max |f'(u)| = v_max * |1 - 2u/rho_max|.

    Args:
        u_left (jnp.ndarray): (N,)
        u_right (jnp.ndarray): (N,)
        params (dict): Must contain 'v_max' and 'rho_max'.

    Returns:
        jnp.ndarray: (N,) wave speeds.
    """
    s_left = jnp.abs(params['v_max'] * (1.0 - 2.0 * u_left / params['rho_max']))
    s_right = jnp.abs(params['v_max'] * (1.0 - 2.0 * u_right / params['rho_max']))
    return jnp.maximum(s_left, s_right)


def apply_bc(u, ghost, bc_left=0.0, bc_right=0.0):
    """
    Pad interior array with Dirichlet ghost cells.

    Args:
        u (jnp.ndarray): (n_cells,) interior values.
        ghost (int): Number of ghost cells per side.
        bc_left (float): Left Dirichlet value.
        bc_right (float): Right Dirichlet value.

    Returns:
        jnp.ndarray: (n_cells + 2*ghost,) padded array.
    """
    left_pad = jnp.full(ghost, bc_left)
    right_pad = jnp.full(ghost, bc_right)
    return jnp.concatenate([left_pad, u, right_pad])


def compute_rhs(fv_data, u, params, bc_left=0.0, bc_right=0.0):
    """
    Semi-discrete RHS: -1/dx * (F_{i+1/2} - F_{i-1/2}).

    Args:
        fv_data (FVData): Mesh data.
        u (jnp.ndarray): (n_cells,) cell values.
        params (dict): Flux parameters.
        bc_left (float): Left boundary value.
        bc_right (float): Right boundary value.

    Returns:
        jnp.ndarray: (n_cells,) RHS.
    """
    g = fv_data.ghost
    u_ext = apply_bc(u, g, bc_left, bc_right)

    u_left = u_ext[:-1]   # left state at each interface
    u_right = u_ext[1:]   # right state at each interface

    F = numerical_flux(u_left, u_right, params)  # (n_cells+1,) fluxes at interfaces
    return -(F[1:] - F[:-1]) / fv_data.dx


def time_step_ssprk2(fv_data, u, dt, params, bc_left=0.0, bc_right=0.0):
    """
    One SSP-RK2 (Heun) time step.

    Args:
        fv_data (FVData): Mesh data.
        u (jnp.ndarray): (n_cells,) current solution.
        dt (float): Time step size.
        params (dict): Flux parameters.
        bc_left (float): Left boundary value.
        bc_right (float): Right boundary value.

    Returns:
        jnp.ndarray: (n_cells,) updated solution.
    """
    k1 = compute_rhs(fv_data, u, params, bc_left, bc_right)
    u1 = u + dt * k1
    k2 = compute_rhs(fv_data, u1, params, bc_left, bc_right)
    return 0.5 * u + 0.5 * (u1 + dt * k2)


def solve_conservation_law(fv_data, u0, params, T, cfl=0.5, bc_left=0.0, bc_right=0.0):
    """
    Adaptive-dt SSP-RK2 solver until time T.

    Args:
        fv_data (FVData): Mesh data.
        u0 (jnp.ndarray): (n_cells,) initial condition.
        params (dict): Flux parameters.
        T (float): Final time.
        cfl (float): CFL number for adaptive step size.
        bc_left (float): Left boundary value.
        bc_right (float): Right boundary value.

    Returns:
        tuple: (t_hist, u_hist) — list of times and list of solution arrays.
    """
    u = u0
    t = 0.0
    t_hist = [t]
    u_hist = [u]

    while t < T:
        max_speed = jnp.max(params['max_wave_speed_fn'](u, u, params))
        max_speed = jnp.maximum(max_speed, 1e-10)
        dt = cfl * fv_data.dx / max_speed
        dt = jnp.minimum(dt, T - t)

        u = time_step_ssprk2(fv_data, u, dt, params, bc_left, bc_right)
        t = t + dt
        t_hist.append(float(t))
        u_hist.append(u)

    return t_hist, u_hist
