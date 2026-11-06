import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from physics.conservation_law import time_step_ssprk2
from physics.helmholtz_fem import solve_helmholtz, interp_at_pts, kappa, eta, forcing
from synthetic.nn import forward, forward_latent, forward_fourier_mlp
from physics.pod_rom import solve_rom, latent_rhs
from physics.poisson import solve_poisson_from_coeff
from physics.gray_scott import (
    rollout, sample_field, sample_points, merge_params, initial_condition,
    DOMAIN_L, ghost_grid, sample_snapshot_points, ic_smooth)


# ── Stopping criterion (eq. `stopping` in the paper, sec. Preliminaries) ──────
# ||Lambda^(k) - mean(Lambda^(k-Z:k-1))|| < eps: stop once the identified
# parameters have stabilized over the last Z epochs/iterations.

def _flatten_param_dict(d):
    """
    Flatten a physical-parameter dict into a flat array, with keys sorted for
    a deterministic ordering. Handles dict values that are Python floats,
    Python lists, or JAX/NumPy arrays/scalars. Uses numpy, not jax.numpy:
    this runs on already-host-side floats every epoch, so jnp would add
    needless host<->device transfers and force a sync via float().

    Args:
        d (dict): Physical parameter dict, e.g. {'alpha1': 2.0, 'c1': [0.1, 0.2]}.

    Returns:
        np.ndarray: Concatenated flat parameter vector.
    """
    return np.concatenate([np.atleast_1d(np.asarray(d[k])) for k in sorted(d)])


def stabilized(param_hist, window, tol):
    """
    Stabilization-based stopping criterion (eq. `stopping`): true once the
    current parameter estimate is within `tol` of the mean of the previous
    `window` estimates.

    Args:
        param_hist (list[dict]): Per-epoch/iteration physical-parameter dicts
            recorded so far, Lambda^(0), ..., Lambda^(k).
        window (int): Averaging window size Z.
        tol (float): Tolerance epsilon.

    Returns:
        bool: True if len(param_hist) > window and
            ||Lambda^(k) - mean(Lambda^(k-Z:k-1))|| < tol.
    """
    if len(param_hist) <= window:
        return False
    flat = [_flatten_param_dict(d) for d in param_hist[-(window + 1):]]
    lam_k, prev = flat[-1], np.stack(flat[:-1])
    return float(np.linalg.norm(lam_k - prev.mean(axis=0))) < tol



# ── FV helpers ────────────────────────────────────────────────────────────────

def solve_fv_fixed(fv_data, u0, params, n_steps, dt, bc_left=0.0, bc_right=0.0):
    """
    Fixed-dt SSP-RK2 FV solve compiled with lax.scan (differentiable).

    Args:
        fv_data (FVData): Mesh data.
        u0 (jnp.ndarray): (n_cells,) initial condition.
        params (dict): Flux parameters including 'flux_fn' and 'max_wave_speed_fn'.
        n_steps (int): Number of time steps.
        dt (float): Fixed time step size.
        bc_left (float): Left Dirichlet boundary value.
        bc_right (float): Right Dirichlet boundary value.

    Returns:
        jnp.ndarray: (n_steps+1, n_cells) solution snapshots.
    """
    def step(u, _):
        u_next = time_step_ssprk2(fv_data, u, dt, params, bc_left, bc_right)
        return u_next, u_next
    _, u_hist = jax.lax.scan(step, u0, None, length=n_steps)
    return jnp.concatenate([u0[None], u_hist], axis=0)


def interp_solution(u_hist, x_cells, T, tx, t_hist=None):
    """
    Bilinear interpolation of a 2D FV solution array at arbitrary (t, x) points.

    Args:
        u_hist (jnp.ndarray): (n_t, n_x) solution snapshots.
        x_cells (jnp.ndarray): (n_x,) cell center positions.
        T (float): Final time (first snapshot is at t=0, last at t=T).
        tx (jnp.ndarray): (Q, 2) query points with columns [t, x].
        t_hist (array | None): (n_t,) snapshot times. None means a uniform
            time grid on [0, T] (fixed-step solvers); pass the actual times
            for adaptive-step solutions.

    Returns:
        jnp.ndarray: (Q,) interpolated values.
    """
    n_t, n_x = u_hist.shape
    t_q, x_q = tx[:, 0], tx[:, 1]

    if t_hist is None:
        t_idx = t_q / T * (n_t - 1)
    else:
        t_hist = jnp.asarray(t_hist)
        t_k = jnp.clip(jnp.searchsorted(t_hist, t_q, side='right') - 1, 0, n_t - 2)
        t_idx = t_k + (t_q - t_hist[t_k]) / (t_hist[t_k + 1] - t_hist[t_k])
    t_lo = jnp.clip(jnp.floor(t_idx).astype(jnp.int32), 0, n_t - 2)
    t_f = t_idx - t_lo

    dx = x_cells[1] - x_cells[0]
    x_idx = (x_q - x_cells[0]) / dx
    x_lo = jnp.clip(jnp.floor(x_idx).astype(jnp.int32), 0, n_x - 2)
    x_f = x_idx - x_lo

    u00 = u_hist[t_lo, x_lo]
    u01 = u_hist[t_lo, x_lo + 1]
    u10 = u_hist[t_lo + 1, x_lo]
    u11 = u_hist[t_lo + 1, x_lo + 1]
    return (1 - t_f) * ((1 - x_f) * u00 + x_f * u01) + \
           t_f * ((1 - x_f) * u10 + x_f * u11)


def sample_ghost(key, n_ghost, t_range, x_range):
    """
    Sample collocation points uniformly in a [t_range] x [x_range] rectangle.

    Args:
        key: JAX PRNG key.
        n_ghost (int): Number of points.
        t_range (tuple): (t_min, t_max).
        x_range (tuple): (x_min, x_max).

    Returns:
        jnp.ndarray: (n_ghost, 2) points with columns [t, x].
    """
    k1, k2 = jr.split(key)
    t = jr.uniform(k1, (n_ghost,), minval=t_range[0], maxval=t_range[1])
    x = jr.uniform(k2, (n_ghost,), minval=x_range[0], maxval=x_range[1])
    return jnp.stack([t, x], axis=1)


def make_full_params(phy_params, flux_fn, wave_speed_fn):
    """
    Attach non-trainable callables to a trainable parameter dict.

    Args:
        phy_params (dict): Trainable physical parameters.
        flux_fn (callable): Physical flux function.
        wave_speed_fn (callable): Max wave speed function.

    Returns:
        dict: Full params dict expected by the FV solver.
    """
    return {**phy_params, 'flux_fn': flux_fn, 'max_wave_speed_fn': wave_speed_fn}


# ── HYCO conservation law ─────────────────────────────────────────────────────

def train_hyco(fv_data, u0, phy_params, syn_params,
               flux_fn, wave_speed_fn,
               tx_data, u_data,
               alpha, beta, schedule_syn, schedule_phy,
               n_ghost, n_epochs, n_steps, dt, T,
               t_range, x_range, key,
               bc_left=0.0, bc_right=0.0, lambda_reg=0, gamma_int=1.0,
               activation=jax.nn.relu, stab_window=1000, stab_tol=1e-4,
               n_finetune=200):
    """
    HYCO training for 1D conservation laws.

    Alternates between:
      syn_step: fit observations + match FV solution at ghost points.
      phy_step: update physical params so FV matches NN at ghost points.

    Args:
        fv_data (FVData): Coarse-mesh FV data.
        u0 (jnp.ndarray): (n_cells,) initial condition.
        phy_params (dict): Initial trainable physical parameters.
        syn_params (dict): Initial NN parameters.
        flux_fn (callable): Physical flux.
        wave_speed_fn (callable): Max wave speed.
        tx_data (jnp.ndarray): (M,2) observation coordinates [t,x].
        u_data (jnp.ndarray): (M,) observed values.
        alpha (float): Weight on observation loss in syn_step.
        beta (float): Weight on observation loss in phy_step.
        schedule_syn: Optax schedule or scalar for syn optimizer.
        schedule_phy: Optax schedule or scalar for phy optimizer.
        n_ghost (int): Ghost points per epoch.
        n_epochs (int): Training epochs (upper bound; may stop earlier, see stab_window).
        n_steps (int): FV time steps.
        dt (float): FV time step size.
        T (float): Final time.
        t_range (tuple): (t_min, t_max) for ghost sampling.
        x_range (tuple): (x_min, x_max) for ghost sampling.
        key: JAX PRNG key.
        bc_left (float): Left boundary value.
        bc_right (float): Right boundary value.
        lambda_reg (float): L2 regularisation weight on syn_params.
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`).
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
        n_finetune (int): Extra syn-only epochs after the main loop, with the
            physical model frozen (data + ghost-point consistency loss against
            the final FV solution). 0 disables.

    Returns:
        tuple: (phy_params, syn_params, history).
    """
    x_cells = fv_data.cell_centers

    opt_syn = optax.adam(schedule_syn)
    opt_phy = optax.adam(schedule_phy)
    state_syn = opt_syn.init(syn_params)
    state_phy = opt_phy.init(phy_params)

    history = {k: [] for k in ['L_syn', 'L_phy', 'L_int_syn', 'L_int_phy', 'L1', 'L2', 'params']}

    @jax.jit
    def syn_step(sp, state, u_hist, tx_g):
        def syn_objective(sp_):
            ls = jnp.mean((forward(sp_, tx_data, activation=activation) - u_data) ** 2)
            u_p = interp_solution(u_hist, x_cells, T, tx_g)
            li = jnp.mean((forward(sp_, tx_g, activation=activation) - u_p) ** 2)
            l_reg = sum(jnp.sum(w ** 2) for w in jax.tree_util.tree_leaves(sp_))
            return alpha * ls + gamma_int * li + lambda_reg * l_reg, (ls, li)
        (l2, (ls, li)), g = jax.value_and_grad(syn_objective, has_aux=True)(sp)
        updates, state = opt_syn.update(g, state, sp)
        sp = optax.apply_updates(sp, updates)
        return sp, state, l2, ls, li

    @jax.jit
    def phy_step(pp, state, sp, tx_g):
        def phy_objective(pp_):
            fp = make_full_params(pp_, flux_fn, wave_speed_fn)
            uh = solve_fv_fixed(fv_data, u0, fp, n_steps, dt, bc_left, bc_right)
            lp = jnp.mean((interp_solution(uh, x_cells, T, tx_data) - u_data) ** 2)
            u_s = forward(sp, tx_g, activation=activation)
            li = jnp.mean((u_s - interp_solution(uh, x_cells, T, tx_g)) ** 2)
            return beta * lp + gamma_int * li, (lp, li)
        (l1, (lp, li)), g = jax.value_and_grad(phy_objective, has_aux=True)(pp)
        updates, state = opt_phy.update(g, state, pp)
        pp = optax.apply_updates(pp, updates)
        return pp, state, l1, lp, li

    @jax.jit
    def solve_phy(pp):
        fp = make_full_params(pp, flux_fn, wave_speed_fn)
        return solve_fv_fixed(fv_data, u0, fp, n_steps, dt, bc_left, bc_right)

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        tx_g = sample_ghost(kg, n_ghost, t_range, x_range)

        u_hist = solve_phy(phy_params)

        syn_params, state_syn, l2, ls, li_s = syn_step(
            syn_params, state_syn, u_hist, tx_g)

        phy_params, state_phy, l1, lp, li_p = phy_step(
            phy_params, state_phy, syn_params, tx_g)

        history['params'].append({k: float(v) for k, v in phy_params.items()})
        history['L_syn'].append(float(ls))
        history['L_phy'].append(float(lp))
        history['L_int_syn'].append(float(li_s))
        history['L_int_phy'].append(float(li_p))
        history['L1'].append(float(l1))
        history['L2'].append(float(l2))

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={float(v):.6f}' if v.ndim == 0
                                else f'{k}={v}' for k, v in phy_params.items())
            print(f"[{epoch:4d}/{n_epochs}]  L_syn={ls:.4e}  L_int={li_p:.4e}  "
                  f"L1={l1:.4e}  L2={l2:.4e}  | {phy_str}")

        if stabilized(history['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    # ── Post-training syn-only fine-tuning (physical model frozen) ────────────
    # phy_params are held fixed; the FV solution is solved once and the NN is
    # trained for n_finetune more epochs against data + ghost-point consistency
    # with that frozen solution. phy-side history entries are repeated (frozen).
    if n_finetune > 0:
        u_hist = solve_phy(phy_params)
        for _ in range(n_finetune):
            key, kg = jr.split(key)
            tx_g = sample_ghost(kg, n_ghost, t_range, x_range)
            syn_params, state_syn, l2, ls, li_s = syn_step(
                syn_params, state_syn, u_hist, tx_g)
            history['params'].append({k: float(v) for k, v in phy_params.items()})
            history['L_syn'].append(float(ls))
            history['L_int_syn'].append(float(li_s))
            history['L2'].append(float(l2))
            history['L_phy'].append(history['L_phy'][-1] if history['L_phy'] else float('nan'))
            history['L_int_phy'].append(history['L_int_phy'][-1] if history['L_int_phy'] else float('nan'))
            history['L1'].append(history['L1'][-1] if history['L1'] else float('nan'))
        print(f"  Fine-tuned syn for {n_finetune} epochs (phy frozen).")

    return phy_params, syn_params, history


# ── PINN conservation law ─────────────────────────────────────────────────────
# Renamed: pde_residual → pde_residual_claw

def pde_residual_claw(syn_params, phy_params, flux_fn, tx_col,
                      activation=jax.nn.relu):
    """
    PDE residual u_t + (f(u))_x at collocation points for a 1D conservation law.

    Args:
        syn_params (dict): NN parameters.
        phy_params (dict): Trainable flux parameters.
        flux_fn (callable): Physical flux function.
        tx_col (jnp.ndarray): (N,2) collocation points [t, x].
        activation (callable): Element-wise nonlinearity used in the syn NN.

    Returns:
        jnp.ndarray: (N,) residuals.
    """
    def residual_single(tx):
        u_val, grad_tx = jax.value_and_grad(
            lambda tx_: forward(syn_params, tx_[None], activation=activation)[0]
        )(tx)
        u_t, u_x = grad_tx[0], grad_tx[1]
        f_prime = jax.grad(lambda u_: flux_fn(jnp.array([u_]), phy_params)[0])(u_val)
        return u_t + f_prime * u_x

    return jax.vmap(residual_single)(tx_col)


def train_pinn(syn_params, phy_params,
               flux_fn,
               tx_data, u_data,
               gamma, schedule,
               n_col, n_epochs,
               t_range, x_range, key,
               activation=jax.nn.tanh, stab_window=1000, stab_tol=1e-4,
               tx_ic=None, u_ic=None, gamma_ic=None,
               tx_bc_left=None, tx_bc_right=None, u_bc=None,
               num_flux_fn=None, gamma_bc=None):
    """
    PINN training for 1D conservation laws.
    Jointly optimises NN and physical parameters via data + PDE residual losses.

    Args:
        syn_params (dict): Initial NN parameters.
        phy_params (dict): Initial trainable physical parameters.
        flux_fn (callable): Physical flux function.
        tx_data (jnp.ndarray): (M,2) observation coordinates [t,x].
        u_data (jnp.ndarray): (M,) observed values.
        gamma (float): Weight on data loss relative to PDE residual loss.
        schedule: Optax schedule or scalar for the Adam optimizer.
        n_col (int): Number of PDE collocation points per epoch.
        n_epochs (int): Training epochs (upper bound; may stop earlier, see stab_window).
        t_range (tuple): (t_min, t_max) for collocation sampling.
        x_range (tuple): (x_min, x_max) for collocation sampling.
        key: JAX PRNG key.
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`).
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
        tx_ic (jnp.ndarray | None): (K,2) initial-datum coordinates [0, x]. If
            given, a soft IC loss ||u(0,x) - u_ic||^2 is added.
        u_ic (jnp.ndarray | None): (K,) initial-datum values.
        gamma_ic (float | None): Weight on the IC loss; defaults to gamma.
        tx_bc_left, tx_bc_right (jnp.ndarray | None): (K,2) boundary
            coordinates [t, x_left] and [t, x_right]. If given, the boundary
            penalty |f(u) - G(u_bc, u)|^2 (left) + |f(u) - G(u, u_bc)|^2
            (right) is added, G = num_flux_fn: incoming information is imposed
            and the outgoing trace is left free.
        u_bc (float | None): Boundary (reservoir) state.
        num_flux_fn (callable | None): Numerical flux G(u_left, u_right, params),
            e.g. physics.conservation_law.lwr_godunov_flux.
        gamma_bc (float | None): Weight on the boundary loss; defaults to gamma.

    Returns:
        tuple: (syn_params, phy_params, history).
    """
    w_ic = gamma if gamma_ic is None else gamma_ic
    w_bc = gamma if gamma_bc is None else gamma_bc
    all_params = {'syn': syn_params, 'phy': phy_params}
    opt = optax.adam(schedule)
    state = opt.init(all_params)

    @jax.jit
    def step(params, opt_state, tx_col):
        def objective(p):
            sp, pp = p['syn'], p['phy']
            l_data = jnp.mean((forward(sp, tx_data, activation=activation) - u_data) ** 2)
            res = pde_residual_claw(sp, pp, flux_fn, tx_col, activation=activation)
            l_pde = jnp.mean(res ** 2)
            l_total = gamma * l_data + l_pde
            if tx_ic is not None:
                l_ic = jnp.mean((forward(sp, tx_ic, activation=activation) - u_ic) ** 2)
                l_total = l_total + w_ic * l_ic
            if tx_bc_left is not None:
                u_l = forward(sp, tx_bc_left, activation=activation)
                u_r = forward(sp, tx_bc_right, activation=activation)
                r_l = flux_fn(u_l, pp) - num_flux_fn(jnp.full_like(u_l, u_bc), u_l, pp)
                r_r = flux_fn(u_r, pp) - num_flux_fn(u_r, jnp.full_like(u_r, u_bc), pp)
                l_bc = 0.5 * (jnp.mean(r_l ** 2) + jnp.mean(r_r ** 2))
                l_total = l_total + w_bc * l_bc
            return l_total, (l_data, l_pde)

        (l_total, (l_data, l_pde)), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, l_total, l_data, l_pde

    history = {k: [] for k in ['L_data', 'L_pde', 'L_total', 'params']}

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        tx_col = sample_ghost(kg, n_col, t_range, x_range)

        all_params, state, l_total, l_data, l_pde = step(all_params, state, tx_col)

        history['L_data'].append(float(l_data))
        history['L_pde'].append(float(l_pde))
        history['L_total'].append(float(l_total))
        history['params'].append({k: float(v) for k, v in all_params['phy'].items()})

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={float(v):.6f}' if v.ndim == 0
                                else f'{k}={v}' for k, v in all_params['phy'].items())
            print(f"  epoch {epoch:4d}  L_total={float(l_total):.3e}"
                  f"  L_data={float(l_data):.3e}  L_pde={float(l_pde):.3e}"
                  f"  {phy_str}")

        if stabilized(history['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    return all_params['syn'], all_params['phy'], history


# ── Ensemble Kalman Inversion — conservation law ──────────────────────────────

def train_eki_claw(
    fv_data, u0,
    flux_fn, wave_speed_fn,
    tx_data, u_data,
    init_params, T,
    n_steps, dt,
    bc_left=0.0, bc_right=0.0,
    n_ensemble=50, n_iter=30,
    init_spread=0.05, obs_noise=0.01,
    key=None, stab_window=5, stab_tol=1e-4,
):
    """
    Ensemble Kalman Inversion for LWR parameter identification.

    Maintains a J-member ensemble over (v_max, rho_max) and iterates the
    standard EKI update (Iglesias 2013) until convergence.  No gradients
    are used — only forward FV solves.

    Args:
        fv_data (FVData): Coarse finite-volume mesh data.
        u0 (jnp.ndarray): (n_cells,) initial condition.
        flux_fn (callable): LWR flux function.
        wave_speed_fn (callable): LWR max wave-speed function.
        tx_data (jnp.ndarray): (N, 2) observation coordinates [t, x].
        u_data (jnp.ndarray): (N,) observed density values.
        init_params (dict): {'v_max', 'rho_max'} used as ensemble centre.
        T (float): Final simulation time.
        n_steps (int): FV time steps.
        dt (float): FV time step size.
        bc_left (float): Left Dirichlet boundary value.
        bc_right (float): Right Dirichlet boundary value.
        n_ensemble (int): Ensemble size J.
        n_iter (int): EKI iterations (upper bound; may stop earlier, see stab_window).
        init_spread (float): Std of Gaussian initialisation noise around init_params.
        obs_noise (float): Observation noise std used to form Γ = obs_noise² I.
        key: JAX PRNG key; defaults to PRNGKey(0).
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`), applied to the ensemble mean.
        stab_tol (float): Tolerance epsilon for the stabilization criterion.

    Returns:
        tuple: (phy_params, history).
            phy_params (dict): Ensemble-mean {'v_max', 'rho_max'}.
            history (dict): Per-iteration ensemble mean/std and 'params'
                (ensemble-mean dict per iteration) for each parameter.
    """
    if key is None:
        key = jr.PRNGKey(0)

    n_obs = u_data.shape[0]
    theta0 = jnp.array([float(init_params['v_max']),
                         float(init_params['rho_max'])])

    # initialise ensemble around init_params; clip to physically valid range
    key, k_init = jr.split(key)
    ensemble = theta0[None] + init_spread * jr.normal(k_init, (n_ensemble, 2))
    ensemble = jnp.clip(ensemble, 0.05, 10.0)

    # Γ = obs_noise² I  (isotropic observation noise covariance)
    Gamma = (obs_noise ** 2) * jnp.eye(n_obs)

    hist = {'v_max_mean': [], 'rho_max_mean': [],
            'v_max_std':  [], 'rho_max_std':  [],
            'params': []}

    @jax.jit
    def fwd(v_max, rho_max):
        """FV forward map: parameters → predicted observations."""
        p = make_full_params(
            {'v_max': jnp.float64(v_max), 'rho_max': jnp.float64(rho_max)},
            flux_fn, wave_speed_fn,
        )
        u_hist = solve_fv_fixed(fv_data, u0, p, n_steps, dt, bc_left, bc_right)
        return interp_solution(u_hist, fv_data.cell_centers, T, tx_data)

    for it in range(n_iter):
        # forward evaluations for all ensemble members — Python loop (flux_fn
        # is a Python closure, not vmappable over params)
        G = jnp.stack([fwd(ensemble[j, 0], ensemble[j, 1])
                       for j in range(n_ensemble)])  # (J, N_obs)

        # ensemble statistics
        theta_mean = jnp.mean(ensemble, axis=0)  # (2,)
        G_mean     = jnp.mean(G, axis=0)         # (N_obs,)
        dtheta     = ensemble - theta_mean[None]  # (J, 2)
        dG         = G - G_mean[None]             # (J, N_obs)

        # cross-covariance (2, N_obs) and auto-covariance (N_obs, N_obs)
        C_tG = (dtheta.T @ dG) / (n_ensemble - 1)
        C_GG = (dG.T @ dG)     / (n_ensemble - 1)

        # perturbed observations: y_j = u_data + η_j,  η_j ~ N(0, Γ)
        key, k_noise = jr.split(key)
        noise = obs_noise * jr.normal(k_noise, (n_ensemble, n_obs))
        innov = (u_data[None] + noise) - G          # (J, N_obs)

        # EKI update:  θ_j ← θ_j + C_θG (C_GG + Γ)^{-1} (y_j - G_j)
        delta = (C_tG @ jnp.linalg.solve(C_GG + Gamma, innov.T)).T  # (J, 2)
        ensemble = jnp.clip(ensemble + delta, 0.05, 10.0)

        # record statistics
        m = jnp.mean(ensemble, axis=0)
        s = jnp.std(ensemble,  axis=0)
        hist['v_max_mean'].append(float(m[0]));   hist['rho_max_mean'].append(float(m[1]))
        hist['v_max_std'].append(float(s[0]));    hist['rho_max_std'].append(float(s[1]))
        hist['params'].append({'v_max': float(m[0]), 'rho_max': float(m[1])})

        if it % 5 == 0 or it == n_iter - 1:
            print(f"  [EKI {it:3d}]  "
                  f"v_max={float(m[0]):.4f}±{float(s[0]):.4f}  "
                  f"rho_max={float(m[1]):.4f}±{float(s[1]):.4f}")

        if stabilized(hist['params'], stab_window, stab_tol):
            print(f"  Stopped at EKI iteration {it}: ensemble mean stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    mean_theta = jnp.mean(ensemble, axis=0)
    phy_params = {
        'v_max':   jnp.float64(mean_theta[0]),
        'rho_max': jnp.float64(mean_theta[1]),
    }
    return phy_params, hist


# ── HYCO Helmholtz ────────────────────────────────────────────────────────────

def train_hyco_helmholtz(
    fem_coarse, syn_params, phy_params,
    schedule_syn, schedule_phy_amp, schedule_phy_ctr,
    sensor_xy, u_data, sensor_tri_c, sensor_bary_c,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    alpha, beta, n_ghost, n_ghost_pool, n_epochs, key,
    activation, gamma_int=1.0, stab_window=1000, stab_tol=1e-3,
):
    """
    HYCO training for Helmholtz BVP. `activation` is the syn-NN activation.

    Args:
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`); n_epochs is an upper bound and
            training may stop earlier once parameters stabilize.
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
    """
    opt_syn = optax.adam(schedule_syn)
    opt_phy = optax.multi_transform(
        {'amp': optax.adam(schedule_phy_amp),
         'ctr': optax.adam(schedule_phy_ctr)},
        {'alpha1': 'amp', 'alpha2': 'amp', 'c1': 'ctr', 'c2': 'ctr'},
    )
    st_syn = opt_syn.init(syn_params)
    st_phy = opt_phy.init(phy_params)

    hist = {k: [] for k in ['L_syn', 'L_phy', 'L_int', 'L1', 'L2', 'params']}

    @jax.jit
    def syn_step(sp, st, pp, g_idx):
        u_phy = solve_helmholtz(fem_coarse, pp)
        gxy, gtri, gbary = ghost_pool_xy[g_idx], ghost_pool_tri[g_idx], ghost_pool_bary[g_idx]
        def objective(sp_):
            ls = jnp.mean((forward(sp_, sensor_xy, activation=activation) - u_data) ** 2)
            u_syn_g = forward(sp_, gxy, activation=activation)
            u_phy_g = interp_at_pts(fem_coarse, u_phy, gtri, gbary)
            li = jnp.mean((u_syn_g - u_phy_g) ** 2)
            return alpha * ls + gamma_int * li, (ls, li)
        (l2, (ls, li)), g = jax.value_and_grad(objective, has_aux=True)(sp)
        updates, st = opt_syn.update(g, st, sp)
        return optax.apply_updates(sp, updates), st, l2, ls, li

    @jax.jit
    def phy_step(pp, st, sp, g_idx):
        gxy, gtri, gbary = ghost_pool_xy[g_idx], ghost_pool_tri[g_idx], ghost_pool_bary[g_idx]
        def objective(pp_):
            u_phy = solve_helmholtz(fem_coarse, pp_)
            u_phy_s = interp_at_pts(fem_coarse, u_phy, sensor_tri_c, sensor_bary_c)
            lp = jnp.mean((u_phy_s - u_data) ** 2)
            u_syn_g = forward(sp, gxy, activation=activation)
            u_phy_g = interp_at_pts(fem_coarse, u_phy, gtri, gbary)
            li = jnp.mean((u_syn_g - u_phy_g) ** 2)
            return beta * lp + gamma_int * li, (lp, li)
        (l1, (lp, li)), g = jax.value_and_grad(objective, has_aux=True)(pp)
        updates, st = opt_phy.update(g, st, pp)
        return optax.apply_updates(pp, updates), st, l1, lp, li

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        g_idx = jr.choice(kg, n_ghost_pool, shape=(n_ghost,), replace=False)
        syn_params, st_syn, l2, ls, li_s = syn_step(syn_params, st_syn, phy_params, g_idx)
        phy_params, st_phy, l1, lp, li_p = phy_step(phy_params, st_phy, syn_params, g_idx)

        hist['L_syn'].append(float(ls)); hist['L_phy'].append(float(lp))
        hist['L_int'].append(float(li_p)); hist['L1'].append(float(l1)); hist['L2'].append(float(l2))
        hist['params'].append({k: float(v) if v.ndim == 0 else v.tolist()
                               for k, v in phy_params.items()})

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            a1, a2 = float(phy_params['alpha1']), float(phy_params['alpha2'])
            c1, c2 = phy_params['c1'], phy_params['c2']
            print(f"  [{epoch:4d}]  Ls={ls:.2e} Lp={lp:.2e} Li={li_p:.2e}  "
                  f"a1={a1:.3f} c1=({float(c1[0]):.3f},{float(c1[1]):.3f})  "
                  f"a2={a2:.3f} c2=({float(c2[0]):.3f},{float(c2[1]):.3f})")

        if stabilized(hist['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    return phy_params, syn_params, hist


# ── PINN Helmholtz ────────────────────────────────────────────────────────────
# Renamed: pde_residual (in main_helmholtz_test.py) → pde_residual_helmholtz

def pde_residual_helmholtz(syn_params, phy_params, xy, activation):
    """PDE residual of -div(kappa grad u) + eta^2 u - f at collocation points."""
    def r_single(xy_):
        u_fn   = lambda x: forward(syn_params, x[None], activation=activation)[0]
        kap_fn = lambda x: kappa(x[None], phy_params['alpha1'], phy_params['c1'])[0]
        u_val    = u_fn(xy_)
        grad_u   = jax.grad(u_fn)(xy_)
        kap_val  = kap_fn(xy_)
        grad_kap = jax.grad(kap_fn)(xy_)
        et_val   = eta(xy_[None], phy_params['alpha2'], phy_params['c2'])[0]
        f_val    = forcing(xy_[None])[0]
        u_xx = jax.grad(lambda x: jax.grad(u_fn)(x)[0])(xy_)[0]
        u_yy = jax.grad(lambda x: jax.grad(u_fn)(x)[1])(xy_)[1]
        div_kap_grad_u = jnp.dot(grad_kap, grad_u) + kap_val * (u_xx + u_yy)
        return -div_kap_grad_u + et_val ** 2 * u_val - f_val
    return jax.vmap(r_single)(xy)


def train_pinn_helmholtz(
    pinn_syn, pinn_phy,
    schedule_syn, schedule_phy_amp, schedule_phy_ctr,
    sensor_xy, u_data,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    gamma, n_col, n_ghost_pool, n_epochs, key,
    activation, stab_window=1000, stab_tol=1e-3,
    gamma_init=1.0, gamma_warmup_frac=0.2,
    bnd_xy=None, gamma_bc=None,
):
    """
    PINN training for Helmholtz BVP. `activation` MUST be smooth (tanh) for grad-of-grad.

    Args:
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`); n_epochs is an upper bound and
            training may stop earlier once parameters stabilize.
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
        gamma_init (float): Starting value for the PDE-loss weight; linearly
            ramps up to `gamma` over the warmup, instead of `gamma` being
            fixed from epoch 0. Avoids weighting the residual heavily before
            the network's second derivatives are meaningful.
        gamma_warmup_frac (float): Fraction of n_epochs over which gamma
            ramps from gamma_init to gamma.
        bnd_xy (jnp.ndarray | None): (B, 2) boundary nodes. If given, the
            homogeneous Dirichlet penalty mean(u(bnd_xy)^2) is added.
        gamma_bc (float | None): Fixed weight on the boundary loss; defaults
            to gamma (its final value, not ramped).
    """
    w_bc = gamma if gamma_bc is None else gamma_bc
    all_pinn = {'syn': pinn_syn, 'phy': pinn_phy}
    labels = {
        'syn': jax.tree_util.tree_map(lambda _: 'syn', pinn_syn),
        'phy': {'alpha1': 'amp', 'alpha2': 'amp', 'c1': 'ctr', 'c2': 'ctr'},
    }
    opt_pinn = optax.multi_transform(
        {'syn': optax.adam(schedule_syn),
         'amp': optax.adam(schedule_phy_amp),
         'ctr': optax.adam(schedule_phy_ctr)},
        labels,
    )
    st_pinn = opt_pinn.init(all_pinn)

    hist = {k: [] for k in ['L_data', 'L_pde', 'params']}

    schedule_gamma = optax.linear_schedule(
        init_value=gamma_init, end_value=gamma,
        transition_steps=max(1, int(gamma_warmup_frac * n_epochs)))

    @jax.jit
    def pinn_step(params, st, xy_col, gamma_val):
        def objective(p):
            l_data = jnp.mean((forward(p['syn'], sensor_xy, activation=activation) - u_data) ** 2)
            res = pde_residual_helmholtz(p['syn'], p['phy'], xy_col, activation)
            l_pde = jnp.mean(res ** 2)
            l_total = l_data + gamma_val * l_pde
            if bnd_xy is not None:
                l_total = l_total + w_bc * jnp.mean(forward(p['syn'], bnd_xy, activation=activation) ** 2)
            return l_total, (l_data, l_pde)
        (lt, (ld, lp)), g = jax.value_and_grad(objective, has_aux=True)(params)
        updates, st = opt_pinn.update(g, st, params)
        return optax.apply_updates(params, updates), st, lt, ld, lp

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        g_idx = jr.choice(kg, n_ghost_pool, shape=(n_col,), replace=False)
        xy_col = ghost_pool_xy[g_idx]
        all_pinn, st_pinn, lt, ld, lp = pinn_step(all_pinn, st_pinn, xy_col, schedule_gamma(epoch))

        hist['L_data'].append(float(ld)); hist['L_pde'].append(float(lp))
        hist['params'].append({k: float(v) if v.ndim == 0 else v.tolist()
                               for k, v in all_pinn['phy'].items()})

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            pp = all_pinn['phy']
            a1, a2 = float(pp['alpha1']), float(pp['alpha2'])
            c1, c2 = pp['c1'], pp['c2']
            print(f"  [{epoch:4d}]  Ld={ld:.2e} Lp={lp:.2e}  "
                  f"a1={a1:.3f} c1=({float(c1[0]):.3f},{float(c1[1]):.3f})  "
                  f"a2={a2:.3f} c2=({float(c2[0]):.3f},{float(c2[1]):.3f})")

        if stabilized(hist['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    return all_pinn['syn'], all_pinn['phy'], hist

# ── SA-PINN Helmholtz ─────────────────────────────────────────────────────────

def train_sapinn_helmholtz(
    pinn_syn, pinn_phy,
    schedule_syn, schedule_phy_amp, schedule_phy_ctr,
    sensor_xy, u_data,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    gamma, n_col, n_ghost_pool, n_epochs, key,
    activation,
    lam_lr=1e-3,
    stab_window=1000,
    stab_tol=1e-3,
    gamma_init=1.0,
    gamma_warmup_frac=0.2,
    bnd_xy=None,
    gamma_bc=None,
):
    """
    Self-Adaptive PINN (SA-PINN) for Helmholtz parameter identification.

    Identical to plain PINN but adds a learnable weight vector λ ∈ R^{n_ghost_pool}
    over the collocation pool.  λ is updated by gradient ascent to up-weight
    collocation points where the PDE residual is large (hard regions).
    NN and physical parameters are updated by gradient descent as usual.

    Reference: McClenny & Braga-Neto (2022), arXiv:2009.04544.

    Args:
        pinn_syn (dict): Initial NN parameters.
        pinn_phy (dict): Initial physical parameters {'alpha1', 'c1', 'alpha2', 'c2'}.
        schedule_syn: Optax schedule for syn NN learning rate.
        schedule_phy_amp: Optax schedule for amplitude parameters.
        schedule_phy_ctr: Optax schedule for center parameters.
        sensor_xy (jnp.ndarray): (M, 2) sensor coordinates.
        u_data (jnp.ndarray): (M,) observed values.
        ghost_pool_xy (jnp.ndarray): (P, 2) pre-sampled collocation pool.
        ghost_pool_tri (jnp.ndarray): (P,) element indices (unused here, kept for API symmetry).
        ghost_pool_bary (jnp.ndarray): (P, 3) barycentric coords (unused here).
        gamma (float): Weight on PDE loss relative to data loss.
        n_col (int): Collocation points sampled per epoch from the pool.
        n_ghost_pool (int): Size of the collocation pool (= len(ghost_pool_xy)).
        n_epochs (int): Training epochs (upper bound; may stop earlier, see stab_window).
        key: JAX PRNG key.
        activation (callable): Element-wise nonlinearity; MUST be smooth (tanh).
        lam_lr (float): Learning rate for the self-adaptive weights λ.
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`).
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
        gamma_init (float): Starting value for the PDE-loss weight; linearly
            ramps up to `gamma` over the warmup, instead of `gamma` being
            fixed from epoch 0.
        gamma_warmup_frac (float): Fraction of n_epochs over which gamma
            ramps from gamma_init to gamma.
        bnd_xy (jnp.ndarray | None): (B, 2) boundary nodes. If given, the
            homogeneous Dirichlet penalty mean(u(bnd_xy)^2) is added with a
            fixed (non-adaptive) weight.
        gamma_bc (float | None): Weight on the boundary loss; defaults to gamma.

    Returns:
        tuple: (syn_params, phy_params, history).
    """
    w_bc = gamma if gamma_bc is None else gamma_bc
    all_pinn = {'syn': pinn_syn, 'phy': pinn_phy}
    lam = jnp.ones(n_ghost_pool)  # per-collocation-point weights, init to 1

    labels = {
        'syn': jax.tree_util.tree_map(lambda _: 'syn', pinn_syn),
        'phy': {'alpha1': 'amp', 'alpha2': 'amp', 'c1': 'ctr', 'c2': 'ctr'},
    }
    opt_params = optax.multi_transform(
        {'syn': optax.adam(schedule_syn),
         'amp': optax.adam(schedule_phy_amp),
         'ctr': optax.adam(schedule_phy_ctr)},
        labels,
    )
    opt_lam = optax.adam(lam_lr)

    st_params = opt_params.init(all_pinn)
    st_lam    = opt_lam.init(lam)

    hist = {k: [] for k in ['L_data', 'L_pde', 'params']}

    schedule_gamma = optax.linear_schedule(
        init_value=gamma_init, end_value=gamma,
        transition_steps=max(1, int(gamma_warmup_frac * n_epochs)))

    @jax.jit
    def sa_step(params, lam_, st_p, st_l, xy_col, col_idx, gamma_val):
        def objective(p, l):
            l_data = jnp.mean(
                (forward(p['syn'], sensor_xy, activation=activation) - u_data) ** 2)
            res    = pde_residual_helmholtz(p['syn'], p['phy'], xy_col, activation)
            # weighted residual: λ_i · r_i (λ indexed into pool by col_idx)
            l_pde  = jnp.mean((l[col_idx] * res) ** 2)
            l_total = l_data + gamma_val * l_pde
            if bnd_xy is not None:
                l_total = l_total + w_bc * jnp.mean(forward(p['syn'], bnd_xy, activation=activation) ** 2)
            return l_total, (l_data, l_pde)

        (_, (ld, lp)), (g_p, g_l) = jax.value_and_grad(
            objective, argnums=(0, 1), has_aux=True)(params, lam_)

        # θ: gradient descent
        u_p, st_p = opt_params.update(g_p, st_p, params)
        params = optax.apply_updates(params, u_p)

        # λ: gradient ascent (negate gradient) + clip to non-negative
        u_l, st_l = opt_lam.update(-g_l, st_l)
        lam_ = jnp.clip(optax.apply_updates(lam_, u_l), 0.0, None)

        return params, lam_, st_p, st_l, ld, lp

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        col_idx = jr.choice(kg, n_ghost_pool, shape=(n_col,), replace=False)
        xy_col  = ghost_pool_xy[col_idx]

        all_pinn, lam, st_params, st_lam, ld, lp = sa_step(
            all_pinn, lam, st_params, st_lam, xy_col, col_idx, schedule_gamma(epoch))

        hist['L_data'].append(float(ld))
        hist['L_pde'].append(float(lp))
        hist['params'].append({k: float(v) if v.ndim == 0 else v.tolist()
                               for k, v in all_pinn['phy'].items()})

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            pp = all_pinn['phy']
            a1, a2 = float(pp['alpha1']), float(pp['alpha2'])
            c1, c2 = pp['c1'], pp['c2']
            print(f"  [{epoch:4d}]  Ld={ld:.2e} Lp={lp:.2e}  "
                  f"a1={a1:.3f} c1=({float(c1[0]):.3f},{float(c1[1]):.3f})  "
                  f"a2={a2:.3f} c2=({float(c2[0]):.3f},{float(c2[1]):.3f})")

        if stabilized(hist['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    return all_pinn['syn'], all_pinn['phy'], hist


# ── Gauss-Newton with adjoint — Helmholtz ─────────────────────────────────────

def train_gauss_newton_helmholtz(
    fem_data,
    sensor_tri, sensor_bary,
    u_data,
    init_params,
    n_iter=50,
    damping=1e-4,
    stab_window=5,
    stab_tol=1e-4,
):
    """
    Gauss-Newton solver for Helmholtz parameter identification.

    Uses the FEM forward map (differentiable via JAX) to compute the Jacobian
    ∂G/∂θ ∈ R^{M×6} in forward mode (6 JVPs), then iterates:
        δθ = −(JᵀJ + λI)⁻¹ Jᵀ r,   r = G(θ) − u_data.

    Parameters are packed as θ = [α₁, c₁ₓ, c₁ᵧ, α₂, c₂ₓ, c₂ᵧ] ∈ R^6.
    Sensor predictions are obtained by barycentric interpolation on the
    supplied FEM mesh (same mesh as HYCO for a fair budget comparison).

    Args:
        fem_data (FEMData): FEM mesh (typically coarse, matching HYCO).
        sensor_tri (jnp.ndarray): (M,) element indices for sensor locations.
        sensor_bary (jnp.ndarray): (M, 3) barycentric coordinates.
        u_data (jnp.ndarray): (M,) observed values.
        init_params (dict): {'alpha1', 'c1', 'alpha2', 'c2'} starting point.
        n_iter (int): Gauss-Newton iterations (upper bound; may stop earlier,
            see stab_window).
        damping (float): Tikhonov damping λ added to JᵀJ.
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`).
        stab_tol (float): Tolerance epsilon for the stabilization criterion.

    Returns:
        tuple: (phy_params, history).
            phy_params (dict): Recovered {'alpha1', 'c1', 'alpha2', 'c2'}.
            history (dict): {'misfit': list of per-iteration MSE,
                             'params': list of per-iteration param dicts}.
    """
    def pack(p):
        """Pack parameter dict into a flat R^6 vector."""
        return jnp.concatenate([
            jnp.array([p['alpha1']]),
            p['c1'],
            jnp.array([p['alpha2']]),
            p['c2'],
        ])

    def unpack(v):
        """Unpack flat R^6 vector into parameter dict."""
        return {
            'alpha1': v[0],
            'c1':     v[1:3],
            'alpha2': v[3],
            'c2':     v[4:6],
        }

    def fwd(v):
        """FEM forward map: R^6 parameter vector → R^M sensor predictions."""
        u = solve_helmholtz(fem_data, unpack(v))
        return interp_at_pts(fem_data, u, sensor_tri, sensor_bary)

    v = pack(init_params)
    hist = {'misfit': [], 'params': []}

    for i in range(n_iter):
        r = fwd(v) - u_data                   # (M,) residual
        J = jax.jacfwd(fwd)(v)                # (M, 6) Jacobian via 6 JVPs
        H = J.T @ J + damping * jnp.eye(6)   # (6, 6) Gauss-Newton Hessian
        v = v - jnp.linalg.solve(H, J.T @ r) # full GN step

        misfit = float(jnp.mean(r ** 2))
        hist['misfit'].append(misfit)
        hist['params'].append(unpack(v))

        if i % 10 == 0 or i == n_iter - 1:
            p = unpack(v)
            print(f"  [GN {i:3d}]  misfit={misfit:.3e}  "
                  f"a1={float(p['alpha1']):.3f}  "
                  f"c1=({float(p['c1'][0]):.3f},{float(p['c1'][1]):.3f})  "
                  f"a2={float(p['alpha2']):.3f}  "
                  f"c2=({float(p['c2'][0]):.3f},{float(p['c2'][1]):.3f})")

        if stabilized(hist['params'], stab_window, stab_tol):
            print(f"  Stopped at GN iteration {i}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    return unpack(v), hist


# ── HYCO latent (NS / POD-ROM) ────────────────────────────────────────────────

def _interp_latent(z_hist, T, t_query):
    """
    Linear interpolation of a uniform latent trajectory at arbitrary times.

    Args:
        z_hist (jnp.ndarray): (S, r) trajectory at uniformly-spaced times in [0, T].
        T (float): Final time.
        t_query (jnp.ndarray): (M,) query times.

    Returns:
        jnp.ndarray: (M, r) interpolated latent states.
    """
    S    = z_hist.shape[0]
    idx  = t_query / T * (S - 1)
    lo   = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, S - 2)
    frac = idx - lo
    return (1 - frac[:, None]) * z_hist[lo] + frac[:, None] * z_hist[lo + 1]


def _observe_batch(z_batch, mean, basis, sensor_locs):
    """
    Decode latent states and extract physical sensor readings.

    Args:
        z_batch (jnp.ndarray): (M, r) latent states.
        mean (jnp.ndarray): (N^2,) POD mean field.
        basis (jnp.ndarray): (N^2, r) POD modes.
        sensor_locs (jnp.ndarray): (n_sensors,) flattened spatial indices.

    Returns:
        jnp.ndarray: (M, n_sensors) sensor readings.
    """
    return (mean[None] + z_batch @ basis.T)[:, sensor_locs]


def train_hyco_latent(y_data, t_data, sensor_locs, z0, nu_init,
                      alpha, beta, lr_syn, lr_phy,
                      n_ghost, n_epochs, print_every, n_warmup,
                      dt_rom, n_steps_rom, save_every_rom, T,
                      mean, basis, kx, ky, ksq, dealias, N, key,
                      activation=jax.nn.relu, gamma_int=1.0):
    """
    HYCO training in latent (POD) space for 2D Navier-Stokes viscosity identification.

    Compiled with jax.lax.scan — no Python overhead per epoch.
    syn_step: NN fits physical observations + matches ROM trajectory at ghost times.
    phy_step: nu is updated so ROM matches NN at ghost times.

    Args:
        y_data (jnp.ndarray): (M, n_sensors) physical observations.
        t_data (jnp.ndarray): (M,) observation times.
        sensor_locs (jnp.ndarray): (n_sensors,) flattened grid indices.
        z0 (jnp.ndarray): (r,) initial latent state.
        nu_init (float): Initial viscosity guess.
        alpha (float): Observation weight in syn_step.
        beta (float): Observation weight in phy_step.
        lr_syn (float): Adam learning rate for NN.
        lr_phy (float): Adam learning rate for viscosity.
        n_ghost (int): Ghost time points per epoch.
        n_epochs (int): Total training epochs (must be divisible by print_every).
        print_every (int): Print / log every this many epochs.
        n_warmup (int): NN-only epochs before updating nu.
        dt_rom (float): ROM integration time step.
        n_steps_rom (int): ROM integration steps.
        save_every_rom (int): ROM save frequency.
        T (float): Final time.
        mean (jnp.ndarray): (N^2,) POD mean.
        basis (jnp.ndarray): (N^2, r) POD modes.
        kx, ky, ksq, dealias: Spectral grid arrays.
        N (int): Physical grid size.
        key: JAX PRNG key.
        activation (callable): Element-wise nonlinearity for the latent NN.
            Default jax.nn.relu (HYCO usage).

    Returns:
        tuple: (nu_found, syn_params, history).
    """
    r          = z0.shape[0]
    key, k_init = jr.split(key)
    # init_latent_params imported via synthetic.nn (forward_latent used below)
    from synthetic.nn import init_latent_params
    syn_params = init_latent_params(k_init, r)
    nu         = nu_init

    # lr_syn / lr_phy may be floats (constant lr) or optax schedules.
    opt_syn   = optax.adam(lr_syn)
    opt_phy   = optax.adam(lr_phy)
    state_syn = opt_syn.init(syn_params)
    state_phy = opt_phy.init(nu)

    def syn_loss(sp, z_rom_hist, t_ghost):
        z_pred = forward_latent(sp, t_data, activation=activation)             # (M, r)
        y_pred = _observe_batch(z_pred, mean, basis, sensor_locs)
        L_obs  = jnp.mean((y_pred - y_data) ** 2)
        L_int  = jnp.mean((forward_latent(sp, t_ghost, activation=activation)
                           - _interp_latent(z_rom_hist, T, t_ghost)) ** 2)
        return alpha * L_obs + gamma_int * L_int, (L_obs, L_int)

    def phy_loss(nv, sp, t_ghost):
        z_rom = solve_rom(z0, nv, dt_rom, n_steps_rom, save_every_rom,
                          mean, basis, kx, ky, ksq, dealias, N)
        # skip _observe_batch entirely when beta=0 — beta is a compile-time constant
        if beta != 0.0:
            y_rom = _observe_batch(_interp_latent(z_rom, T, t_data), mean, basis, sensor_locs)
            L_obs = jnp.mean((y_rom - y_data) ** 2)
        else:
            L_obs = jnp.zeros(())
        L_int = jnp.mean((forward_latent(sp, t_ghost, activation=activation)
                          - _interp_latent(z_rom, T, t_ghost)) ** 2)
        # return z_rom so epoch_step can reuse it for syn_loss without a second forward solve
        return beta * L_obs + gamma_int * L_int, (L_obs, L_int, z_rom)

    def epoch_step(carry, key):
        sp, st_syn, nv, st_phy = carry
        t_ghost = jr.uniform(key, (n_ghost,), minval=0.0, maxval=T)

        # phy_step first: solve_rom runs once (forward) + once (backward through nu)
        # z_rom from the forward pass is returned in aux and reused by syn_step
        (_, (L_phy, L_int, z_rom_hist)), g_phy = jax.value_and_grad(
            phy_loss, has_aux=True)(nv, sp, t_ghost)
        upd_phy, st_phy = opt_phy.update(g_phy, st_phy, nv)
        nv = optax.apply_updates(nv, upd_phy)

        # syn_step: reuse z_rom from phy forward pass — no extra solve_rom call
        # NOTE: sp used here is from the previous epoch (phy used it too); order swapped vs original
        z_rom_sg = jax.lax.stop_gradient(z_rom_hist)
        (_, (L_syn, _)), g_syn = jax.value_and_grad(syn_loss, has_aux=True)(sp, z_rom_sg, t_ghost)
        upd_syn, st_syn = opt_syn.update(g_syn, st_syn, sp)
        sp = optax.apply_updates(sp, upd_syn)

        return (sp, st_syn, nv, st_phy), (L_syn, L_phy, L_int, nv)

    epoch_step_jit = jax.jit(epoch_step)

    carry = (syn_params, state_syn, nu, state_phy)
    history = {'L_syn': [], 'L_phy': [], 'L_int_phy': [], 'nu': []}

    if n_warmup > 0:
        print("Warmup starting...")
        def warmup_step(carry, key):
            sp, st_syn = carry
            t_ghost    = jr.uniform(key, (n_ghost,), minval=0.0, maxval=T)
            z_rom_hist = solve_rom(z0, nu, dt_rom, n_steps_rom, save_every_rom,
                                   mean, basis, kx, ky, ksq, dealias, N)
            (_, (L_syn, _)), g = jax.value_and_grad(syn_loss, has_aux=True)(sp, z_rom_hist, t_ghost)
            upd, st_syn = opt_syn.update(g, st_syn, sp)
            return (optax.apply_updates(sp, upd), st_syn), L_syn

        key, *wkeys = jr.split(key, n_warmup + 1)
        (syn_params, state_syn), L_syns_w = jax.lax.scan(
            jax.jit(warmup_step), (syn_params, state_syn), jnp.stack(wkeys))
        carry = (syn_params, state_syn, nu, state_phy)
        print(f"Warmup done ({n_warmup} epochs)  "
              f"L_syn: {float(L_syns_w[0]):.3e} -> {float(L_syns_w[-1]):.3e}")

    assert n_epochs % print_every == 0, "n_epochs must be divisible by print_every"
    n_chunks = n_epochs // print_every

    print(f"Training HYCO for {n_epochs} epochs  (r={r}, n_ghost={n_ghost}, "
          f"print_every={print_every})")

    for chunk in range(n_chunks):
        key, *chunk_keys = jr.split(key, print_every + 1)
        chunk_keys = jnp.stack(chunk_keys)

        carry, (L_syns, L_phys, L_ints, nus) = jax.lax.scan(
            epoch_step_jit, carry, chunk_keys)

        epoch_end = (chunk + 1) * print_every

        history['L_syn'].extend(L_syns.tolist())
        history['L_phy'].extend(L_phys.tolist())
        history['L_int_phy'].extend(L_ints.tolist())
        history['nu'].extend(nus.tolist())

        print(f"[{epoch_end:5d}/{n_epochs}]  "
              f"L_syn={float(L_syns[-1]):.3e}  "
              f"L_phy={float(L_phys[-1]):.3e}  "
              f"L_int={float(L_ints[-1]):.3e}  "
              f"nu={float(nus[-1]):.6f}")

    syn_params, state_syn, nu, state_phy = carry
    return nu, syn_params, history


# ── ROM-only baseline (NS) ────────────────────────────────────────────────────

def train_rom_only_ns(y_data, t_data, sensor_locs, z0, nu_init,
                      lr_phy, n_epochs, print_every,
                      dt_rom, n_steps_rom, save_every_rom, T,
                      mean, basis, kx, ky, ksq, dealias, N):
    """
    Direct gradient descent on ν using the ROM forward model. No NN surrogate.

    Minimises ||observed_sensors - ROM_prediction(ν)||² w.r.t. ν only.
    Isolates the NN contribution of HYCO: if this performs similarly to HYCO,
    the surrogate NN is not adding value.

    Args:
        y_data (jnp.ndarray): (M, n_sensors) physical observations.
        t_data (jnp.ndarray): (M,) observation times.
        sensor_locs (jnp.ndarray): (n_sensors,) flattened grid indices.
        z0 (jnp.ndarray): (r,) initial latent state.
        nu_init (float): Initial viscosity guess.
        lr_phy (float): Adam learning rate for ν.
        n_epochs (int): Training epochs.
        print_every (int): Print frequency.
        dt_rom, n_steps_rom, save_every_rom, T: ROM integration settings.
        mean, basis, kx, ky, ksq, dealias, N: POD and spectral grid data.

    Returns:
        tuple: (nu_found, history).
    """
    nu    = jnp.float64(nu_init)
    opt   = optax.adam(lr_phy)
    state = opt.init(nu)
    history = {'L': [], 'nu': []}

    @jax.jit
    def step(nv, st):
        def loss_fn(nv_):
            z_rom  = solve_rom(z0, nv_, dt_rom, n_steps_rom, save_every_rom,
                               mean, basis, kx, ky, ksq, dealias, N)
            y_pred = _observe_batch(
                _interp_latent(z_rom, T, t_data), mean, basis, sensor_locs)
            return jnp.mean((y_pred - y_data) ** 2)
        loss, g = jax.value_and_grad(loss_fn)(nv)
        updates, st = opt.update(g, st, nv)
        return optax.apply_updates(nv, updates), st, loss

    for epoch in range(n_epochs):
        nu, state, loss = step(nu, state)
        history['L'].append(float(loss))
        history['nu'].append(float(nu))
        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(f"  [{epoch:5d}]  loss={float(loss):.3e}  nu={float(nu):.6f}")
        # early stopping: patience=200, min relative improvement=1e-6
        if epoch >= 200:
            recent_best = min(history['L'][-200:])
            if (history['L'][-201] - recent_best) / (abs(history['L'][-201]) + 1e-12) < 1e-6:
                print(f"  Early stopping at epoch {epoch} (no improvement over 200 epochs).")
                break

    return nu, history


# ── Latent PINN baseline (NS) ─────────────────────────────────────────────────

def train_latent_pinn_ns(y_data, t_data, sensor_locs, z0, nu_init,
                         lr_syn, lr_phy, gamma_ode, n_col, n_epochs, print_every,
                         T, mean, basis, kx, ky, ksq, dealias, N, key,
                         activation=jax.nn.tanh):
    """
    Latent PINN for NS viscosity identification.

    Jointly trains the latent trajectory NN z_θ(t) and ν using:
      L = L_data(θ) + γ_ode · L_ode(θ, ν)
    where L_ode = ||dz_θ/dt(t_col) - f_ROM(z_θ(t_col), ν)||².

    Compared to HYCO, this uses joint (not alternating) optimisation and has
    no ghost-point coupling — it isolates the benefit of HYCO's alternation.

    activation MUST be smooth (tanh) because L_ode requires dz_θ/dt via AD.

    Args:
        y_data (jnp.ndarray): (M, n_sensors) physical observations.
        t_data (jnp.ndarray): (M,) observation times.
        sensor_locs (jnp.ndarray): (n_sensors,) flattened grid indices.
        z0 (jnp.ndarray): (r,) initial latent state.
        nu_init (float): Initial viscosity guess.
        lr_syn (float): Adam learning rate for NN parameters.
        lr_phy (float): Adam learning rate for ν.
        gamma_ode (float): Weight on ODE residual loss.
        n_col (int): ODE residual collocation points per epoch.
        n_epochs (int): Training epochs.
        print_every (int): Print frequency.
        T (float): Final time.
        mean, basis, kx, ky, ksq, dealias, N: POD and spectral grid data.
        key: JAX PRNG key.
        activation: Must be smooth; default tanh.

    Returns:
        tuple: (nu_found, syn_params, history).
    """
    from synthetic.nn import init_latent_params
    r = z0.shape[0]
    key, k_init = jr.split(key)
    syn_params  = init_latent_params(k_init, r)
    nu          = jnp.float64(nu_init)

    all_params = {'syn': syn_params, 'nu': nu}
    labels = {
        'syn': jax.tree_util.tree_map(lambda _: 'syn', syn_params),
        'nu': 'nu',
    }
    opt   = optax.multi_transform(
        {'syn': optax.adam(lr_syn), 'nu': optax.adam(lr_phy)}, labels)
    state = opt.init(all_params)
    history = {'L_data': [], 'L_ode': [], 'nu': []}

    @jax.jit
    def step(params, st, t_col):
        def loss_fn(p):
            sp, nv = p['syn'], p['nu']

            # data loss: sensor observations vs NN prediction
            z_pred = forward_latent(sp, t_data, activation=activation)
            y_pred = _observe_batch(z_pred, mean, basis, sensor_locs)
            L_data = jnp.mean((y_pred - y_data) ** 2)

            # ODE residual at collocation times
            z_col   = forward_latent(sp, t_col, activation=activation)  # (n_col, r)
            rhs_col = jax.vmap(
                lambda z: latent_rhs(z, mean, basis, kx, ky, ksq, nv, dealias, N)
            )(z_col)                                                     # (n_col, r)

            # dz_θ/dt via forward-mode AD over the time input
            def dz_at(t_s):
                return jax.jacfwd(
                    lambda t: forward_latent(sp, t.reshape(1), activation)[0]
                )(t_s)

            dz_col = jax.vmap(dz_at)(t_col)                            # (n_col, r)
            L_ode  = jnp.mean((dz_col - rhs_col) ** 2)

            return L_data + gamma_ode * L_ode, (L_data, L_ode)

        (_, (ld, lo)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, st = opt.update(g, st, params)
        return optax.apply_updates(params, updates), st, ld, lo

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        t_col = jr.uniform(kg, (n_col,), minval=0.0, maxval=T)
        all_params, state, ld, lo = step(all_params, state, t_col)
        history['L_data'].append(float(ld))
        history['L_ode'].append(float(lo))
        history['nu'].append(float(all_params['nu']))
        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(f"  [{epoch:5d}]  Ld={float(ld):.3e}  "
                  f"Lo={float(lo):.3e}  nu={float(all_params['nu']):.6f}")
        # early stopping on total loss: patience=200, min relative improvement=1e-6
        total = float(ld) + float(lo)
        history.setdefault('_L_total', []).append(total)
        if epoch >= 2000:
            recent_best = min(history['_L_total'][-2000:])
            if (history['_L_total'][-2001] - recent_best) / (abs(history['_L_total'][-2001]) + 1e-12) < 1e-6:
                print(f"  Early stopping at epoch {epoch} (no improvement over 200 epochs).")
                break

    history.pop('_L_total', None)  # internal bookkeeping, not saved
    return all_params['nu'], all_params['syn'], history


# ── HYCO Poisson ──────────────────────────────────────────────────────────────

def train_hyco_poisson(
    pdata,
    syn_params, phy_params,
    schedule_syn, schedule_phy,
    sensor_xy, u_data,
    sensor_tri, sensor_bary,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    alpha, beta, n_ghost, n_ghost_pool, n_epochs, key,
    activation=jax.nn.relu, gamma_int=1.0,
):
    """
    HYCO training for the Poisson BVP with unknown forcing term f.

    phy_params must contain a single key 'f_coeff' (N*N,) — the Chebyshev
    coefficients of f.  The solve is differentiable through f_coeff because
    Poisson is linear in f and jnp.linalg.solve supports AD.

    Alternates between:
      syn_step: fit sensor observations + match FEM solution at ghost points.
      phy_step: update f_coeff so FEM solution matches NN at ghost points.

    Args:
        pdata (PoissonData): Precomputed Poisson data (mesh + stiffness + Chebyshev).
        syn_params (dict): Initial NN parameters.
        phy_params (dict): {'f_coeff': (N*N,)} initial forcing coefficients.
        schedule_syn: Optax schedule or scalar for the syn Adam optimizer.
        schedule_phy: Optax schedule or scalar for the phy Adam optimizer.
        sensor_xy (jnp.ndarray): (M, 2) sensor coordinates.
        u_data (jnp.ndarray): (M,) observed solution values at sensors.
        sensor_tri (jnp.ndarray): (M,) element indices for sensor locations.
        sensor_bary (jnp.ndarray): (M, 3) barycentric coords for sensor locations.
        ghost_pool_xy (jnp.ndarray): (P, 2) pre-sampled ghost point coordinates.
        ghost_pool_tri (jnp.ndarray): (P,) element indices for ghost pool.
        ghost_pool_bary (jnp.ndarray): (P, 3) barycentric coords for ghost pool.
        alpha (float): Weight on sensor loss in syn_step.
        beta (float): Weight on sensor loss in phy_step.
        n_ghost (int): Ghost points drawn per epoch from the pool.
        n_ghost_pool (int): Size of the pre-sampled ghost pool.
        n_epochs (int): Training epochs.
        key: JAX PRNG key.
        activation (callable): Element-wise nonlinearity for the syn NN.
            Default jax.nn.relu (HYCO usage).

    Returns:
        tuple: (phy_params, syn_params, history).
    """
    opt_syn = optax.adam(schedule_syn)
    opt_phy = optax.adam(schedule_phy)
    st_syn  = opt_syn.init(syn_params)
    st_phy  = opt_phy.init(phy_params)

    hist = {k: [] for k in ['L_syn', 'L_phy', 'L_int', 'L1', 'L2']}

    @jax.jit
    def syn_step(sp, st, pp, g_idx):
        # FEM solution at current f_coeff (fixed — no grad through pp here)
        u_fem = solve_poisson_from_coeff(pdata, pp['f_coeff'])
        gxy   = ghost_pool_xy[g_idx]
        gtri  = ghost_pool_tri[g_idx]
        gbary = ghost_pool_bary[g_idx]

        def objective(sp_):
            ls      = jnp.mean((forward(sp_, sensor_xy, activation=activation) - u_data) ** 2)
            u_nn_g  = forward(sp_, gxy, activation=activation)
            u_fem_g = interp_at_pts(pdata.fem, u_fem, gtri, gbary)
            li      = jnp.mean((u_nn_g - u_fem_g) ** 2)
            return alpha * ls + gamma_int * li, (ls, li)

        (l2, (ls, li)), g = jax.value_and_grad(objective, has_aux=True)(sp)
        updates, st = opt_syn.update(g, st, sp)
        sp = optax.apply_updates(sp, updates)
        return sp, st, l2, ls, li

    @jax.jit
    def phy_step(pp, st, sp, g_idx):
        gxy   = ghost_pool_xy[g_idx]
        gtri  = ghost_pool_tri[g_idx]
        gbary = ghost_pool_bary[g_idx]

        def objective(pp_):
            u_fem    = solve_poisson_from_coeff(pdata, pp_['f_coeff'])
            # sensor loss
            u_fem_s  = interp_at_pts(pdata.fem, u_fem, sensor_tri, sensor_bary)
            lp       = jnp.mean((u_fem_s - u_data) ** 2)
            # coupling: NN should match FEM at ghost points
            u_nn_g   = forward(sp, gxy, activation=activation)
            u_fem_g  = interp_at_pts(pdata.fem, u_fem, gtri, gbary)
            li       = jnp.mean((u_nn_g - u_fem_g) ** 2)
            return beta * lp + gamma_int * li, (lp, li)

        (l1, (lp, li)), g = jax.value_and_grad(objective, has_aux=True)(pp)
        updates, st = opt_phy.update(g, st, pp)
        pp = optax.apply_updates(pp, updates)
        return pp, st, l1, lp, li

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        g_idx = jr.choice(kg, n_ghost_pool, shape=(n_ghost,), replace=False)

        syn_params, st_syn, l2, ls, li_s = syn_step(syn_params, st_syn, phy_params, g_idx)
        phy_params, st_phy, l1, lp, li_p = phy_step(phy_params, st_phy, syn_params, g_idx)

        hist['L_syn'].append(float(ls))
        hist['L_phy'].append(float(lp))
        hist['L_int'].append(float(li_p))
        hist['L1'].append(float(l1))
        hist['L2'].append(float(l2))

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            f_norm = float(jnp.linalg.norm(phy_params['f_coeff']))
            print(f"  [{epoch:4d}]  Ls={ls:.2e} Lp={lp:.2e} Li={li_p:.2e}  "
                  f"||f_coeff||={f_norm:.4f}")

    return phy_params, syn_params, hist



# ── HYCO Gray Scott ──────────────────────────────────────────────────────────────
    
def _to_log(d):
    """Map a positive-parameter dict to log space."""
    return {k: jnp.log(v) for k, v in d.items()}


def _from_log(d):
    """Map a log-space parameter dict back to positive values."""
    return {k: jnp.exp(v) for k, v in d.items()}


def _normalize(xyt, xyt_mean, xyt_std):
    """Z-score normalize (x,y,t) before it reaches the synthetic network."""
    return (xyt - xyt_mean) / xyt_std


def train_hyco_gs(phy_init, fixed_phy, syn_params,
                  xyt_data, u_data, xyt_mean, xyt_std,
                  alpha, beta, schedule_syn, schedule_phy,
                  n_ghost, n_epochs, phy_n, phy_steps, n_save_phy, T,
                  key, gamma_int=1.0, activation=jax.nn.relu,
                  n_pretrain=0, n_lbfgs=0, n_ghost_lbfgs='full',
                  stab_window=1000, stab_tol=1e-4, net_forward=forward):
    """
    HYCO training for the Gray-Scott system.

    Alternates between:
      syn_step: fit the (u,v) data with the NN and match u_phy at ghost points.
      phy_step: update the recovered diffusivities so the FD solution matches the
                NN at ghost points (data anchored only if beta > 0).
    Data are supplied to the synthetic model only (beta = 0 in the paper).
    Recovered parameters are optimised in log space (see module header).
    Network inputs (data, ghost points, dense L-BFGS ghost points) are z-score
    normalized using xyt_mean/xyt_std before every forward() call.

    Args:
        phy_init (dict): Initial values for the trainable (recovered) parameters,
            in physical units; assumed strictly positive.
        fixed_phy (dict): Parameters held constant (complement of phy_init).
        syn_params (dict): Initial NN parameters (R^3 -> R^2).
        xyt_data (jnp.ndarray): (M,3) observation coordinates [x,y,t].
        u_data (jnp.ndarray): (M,2) observed [u,v] values.
        xyt_mean (jnp.ndarray): (3,) per-coordinate means for normalization.
        xyt_std (jnp.ndarray): (3,) per-coordinate stds for normalization.
        alpha (float): Weight on the data loss in the syn step.
        beta (float): Weight on the data loss in the phy step (0 = none).
        schedule_syn, schedule_phy: Optax schedules or scalars.
        n_ghost (int | 'full'): Ghost points per epoch. 'full' uses the full
            phy_n x phy_n grid at every snapshot time (fixed set, exact FD
            values — matches the torch reference). An integer draws that many
            random spatial points per epoch, with time coordinates restricted
            to the snapshot times so sample_field never interpolates in time.
        n_epochs (int): Training epochs (upper bound; may stop earlier, see
            stab_window).
        phy_n (int): Physical-model grid resolution.
        phy_steps (int): Physical-model Euler steps (divisible by n_save_phy).
        n_save_phy (int): Saved snapshots of the physical model.
        T (float): Final time.
        key: JAX PRNG key.
        gamma_int (float): Interaction-loss weight (0 = uncoupled ablation).
        activation (callable): NN nonlinearity (ReLU for HYCO).
        n_pretrain (int): Data-only syn warmup epochs before the main loop.
        n_lbfgs (int): L-BFGS-B iterations post-training syn, phy frozen.
        n_ghost_lbfgs (int | 'full'): Ghost set used by L-BFGS. 'full' (default)
            uses the full grid x snapshot-time set; an integer draws that many
            random spatial points at snapshot times.
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`).
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
        net_forward (callable): Network apply function (params, x, activation)
            -> predictions. Default: shared residual `forward`; the Gray-Scott
            experiment passes the plain MLP `forward_mlp`.

    Returns:
        tuple: (phy_params, syn_params, history), phy_params in physical units.
    """
    opt_syn = optax.adam(schedule_syn)
    opt_phy = optax.adam(schedule_phy)
    state_syn = opt_syn.init(syn_params)

    theta = _to_log(phy_init)               # optimise log(D)
    state_phy = opt_phy.init(theta)

    dx_phy     = DOMAIN_L / phy_n
    dt_phy     = T / phy_steps
    save_every = phy_steps // n_save_phy
    u0p, v0p   = initial_condition(phy_n)

    # Snapshot times of the rollout (theta-independent). Ghost time coordinates
    # are restricted to these values so consistency targets are exact FD
    # snapshots rather than linear-in-time cross-fades.
    times_snap = jnp.linspace(0.0, T, n_save_phy + 1)
    xyt_g_full = ghost_grid(phy_n, times_snap) if (n_ghost == 'full'
                                                   or n_ghost_lbfgs == 'full') else None

    xyt_data_n = _normalize(xyt_data, xyt_mean, xyt_std)

    history = {k: [] for k in
               ['L_syn', 'L_phy', 'L_int_syn', 'L_int_phy', 'L1', 'L2', 'params']}

    def _solve(theta_):
        p = merge_params(_from_log(theta_), fixed_phy)
        return rollout(p, u0p, v0p, dx_phy, dt_phy, n_save_phy, save_every)

    phy_solve = jax.jit(_solve)             # constant solve reused in the syn step

    # ── Pretraining: syn-only fit on data, no ghost coupling ──────────────────
    if n_pretrain > 0:
        @jax.jit
        def pretrain_step(sp, state):
            def objective(sp_):
                return jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
            l, g = jax.value_and_grad(objective)(sp)
            updates, state = opt_syn.update(g, state, sp)
            return optax.apply_updates(sp, updates), state, l

        for epoch in range(n_pretrain):
            syn_params, state_syn, l = pretrain_step(syn_params, state_syn)
            if epoch % 200 == 0 or epoch == n_pretrain - 1:
                print(f"[pretrain {epoch:4d}/{n_pretrain}]  L_syn={float(l):.4e}")

    @jax.jit
    def syn_step(sp, state, U, V, times, xyt_g):
        # U, V are constant w.r.t. sp (current physical solution): the ghost target.
        phy_g = sample_field(U, V, times, dx_phy, phy_n, xyt_g)
        xyt_g_n = _normalize(xyt_g, xyt_mean, xyt_std)

        def objective(sp_):
            ls = jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
            li = jnp.mean((net_forward(sp_, xyt_g_n, activation=activation) - phy_g) ** 2)
            return alpha * ls + gamma_int * li, (ls, li)

        (l2, (ls, li)), g = jax.value_and_grad(objective, has_aux=True)(sp)
        updates, state = opt_syn.update(g, state, sp)
        return optax.apply_updates(sp, updates), state, l2, ls, li

    @jax.jit
    def phy_step(th, state, sp, xyt_g):
        xyt_g_n = _normalize(xyt_g, xyt_mean, xyt_std)
        syn_g = jax.lax.stop_gradient(net_forward(sp, xyt_g_n, activation=activation))

        def objective(th_):
            times, U, V = _solve(th_)        # differentiable rollout (remat chunks)
            li = jnp.mean((sample_field(U, V, times, dx_phy, phy_n, xyt_g) - syn_g) ** 2)
            lp = jnp.mean((sample_field(U, V, times, dx_phy, phy_n, xyt_data) - u_data) ** 2)
            return beta * lp + gamma_int * li, (lp, li)

        (l1, (lp, li)), g = jax.value_and_grad(objective, has_aux=True)(th)
        updates, state = opt_phy.update(g, state, th)
        return optax.apply_updates(th, updates), state, l1, lp, li

    for epoch in range(n_epochs):
        if n_ghost == 'full':
            xyt_g = xyt_g_full                  # fixed grid x snapshot set
        else:
            key, kg = jr.split(key)
            xyt_g = sample_snapshot_points(kg, n_ghost, times_snap)

        times, U, V = phy_solve(theta)
        syn_params, state_syn, l2, ls, li_s = syn_step(
            syn_params, state_syn, U, V, times, xyt_g)
        theta, state_phy, l1, lp, li_p = phy_step(
            theta, state_phy, syn_params, xyt_g)

        phy_now = {k: float(v) for k, v in _from_log(theta).items()}
        history['params'].append(phy_now)
        history['L_syn'].append(float(ls))
        history['L_phy'].append(float(lp))
        history['L_int_syn'].append(float(li_s))
        history['L_int_phy'].append(float(li_p))
        history['L1'].append(float(l1))
        history['L2'].append(float(l2))

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={v:.3e}' for k, v in phy_now.items())
            print(f"[{epoch:4d}/{n_epochs}]  L_syn={ls:.4e}  L_int={li_p:.4e}  "
                  f"L1={l1:.4e}  L2={l2:.4e}  | {phy_str}")

        if stabilized(history['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    # ── L-BFGS-B post-training of syn_params, phy frozen at final theta ──────
    if n_lbfgs > 0:
        from jax.flatten_util import ravel_pytree
        from scipy.optimize import minimize

        if n_ghost_lbfgs == 'full':
            xyt_g_dense = xyt_g_full            # full grid x snapshot set, exact values
        else:
            key, k_lbfgs = jr.split(key)
            xyt_g_dense = sample_snapshot_points(k_lbfgs, n_ghost_lbfgs, times_snap)
        times_f, U_f, V_f = phy_solve(theta)
        phy_g_dense = sample_field(U_f, V_f, times_f, dx_phy, phy_n, xyt_g_dense)
        xyt_g_dense_n = _normalize(xyt_g_dense, xyt_mean, xyt_std)

        flat0, unravel = ravel_pytree(syn_params)

        def lbfgs_objective(flat):
            sp = unravel(jnp.asarray(flat))
            def obj(sp_):
                ls = jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
                li = jnp.mean((net_forward(sp_, xyt_g_dense_n, activation=activation) - phy_g_dense) ** 2)
                return alpha * ls + gamma_int * li
            val, grad = jax.value_and_grad(obj)(sp)
            flat_grad, _ = ravel_pytree(grad)
            return float(val), np.asarray(flat_grad, dtype=np.float64)

        res = minimize(lbfgs_objective, np.asarray(flat0, dtype=np.float64),
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': n_lbfgs, 'maxcor': 100})
        syn_params = unravel(jnp.asarray(res.x))
        print(f"  L-BFGS-B: {res.nit} iters, final loss={res.fun:.4e}")

    return {k: jnp.exp(v) for k, v in theta.items()}, syn_params, history


def train_nn_gs(syn_params, xyt_data, u_data, xyt_mean, xyt_std, schedule, n_epochs, key,
                activation=jax.nn.relu, n_pretrain=0, n_lbfgs=0,
                patience=200, min_rel_improve=1e-6, net_forward=forward):
    """
    Pure data-driven NN baseline for Gray-Scott (no physics, interaction loss off).

    n_pretrain epochs run first as a distinct logged phase, using the same
    data-only objective as the main phase (NN has no other loss term to
    transition into; kept for structural parity with HYCO/PINN). Stopping is
    loss-based (patience/min_rel_improve), since there's no physical parameter
    dict for the stabilized() criterion to track.

    Args:
        syn_params (dict): Initial NN parameters (R^3 -> R^2).
        xyt_data (jnp.ndarray): (M,3) observation coordinates [x,y,t].
        u_data (jnp.ndarray): (M,2) observed [u,v] values.
        xyt_mean (jnp.ndarray): (3,) per-coordinate means for normalization.
        xyt_std (jnp.ndarray): (3,) per-coordinate stds for normalization.
        schedule: Optax schedule or scalar.
        n_epochs (int): Training epochs (upper bound; may stop earlier).
        key: JAX PRNG key (unused; kept for signature symmetry).
        activation (callable): NN nonlinearity (ReLU).
        n_pretrain (int): Data-only warmup epochs (logged separately).
        n_lbfgs (int): L-BFGS-B iterations after the Adam phases.
        patience (int): Epochs to look back for the loss-based stopping check.
        min_rel_improve (float): Minimum relative improvement over `patience`
            epochs required to keep going.
        net_forward (callable): Network apply function (see train_hyco_gs).

    Returns:
        tuple: (syn_params, history).
    """
    opt = optax.adam(schedule)
    state = opt.init(syn_params)
    history = {'L_syn': []}
    xyt_data_n = _normalize(xyt_data, xyt_mean, xyt_std)

    @jax.jit
    def step(sp, st):
        def objective(sp_):
            return jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
        l, g = jax.value_and_grad(objective)(sp)
        updates, st = opt.update(g, st, sp)
        return optax.apply_updates(sp, updates), st, l

    if n_pretrain > 0:
        for epoch in range(n_pretrain):
            syn_params, state, l = step(syn_params, state)
            history['L_syn'].append(float(l))
            if epoch % 200 == 0 or epoch == n_pretrain - 1:
                print(f"[NN pretrain {epoch:4d}/{n_pretrain}]  L_syn={float(l):.4e}")

    for epoch in range(n_epochs):
        syn_params, state, l = step(syn_params, state)
        history['L_syn'].append(float(l))
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"[NN {epoch:4d}/{n_epochs}]  L_syn={float(l):.4e}")
        if epoch >= patience:
            recent_best = min(history['L_syn'][-patience:])
            prev = history['L_syn'][-patience - 1]
            if (prev - recent_best) / (abs(prev) + 1e-12) < min_rel_improve:
                print(f"  Stopped at epoch {epoch}: no improvement over {patience} epochs.")
                break

    if n_lbfgs > 0:
        from jax.flatten_util import ravel_pytree
        from scipy.optimize import minimize

        flat0, unravel = ravel_pytree(syn_params)

        def lbfgs_objective(flat):
            sp = unravel(jnp.asarray(flat))
            def obj(sp_):
                return jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
            val, grad = jax.value_and_grad(obj)(sp)
            flat_grad, _ = ravel_pytree(grad)
            return float(val), np.asarray(flat_grad, dtype=np.float64)

        res = minimize(lbfgs_objective, np.asarray(flat0, dtype=np.float64),
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': n_lbfgs, 'maxcor': 100})
        syn_params = unravel(jnp.asarray(res.x))
        history['L_syn'].append(float(res.fun))
        print(f"[NN] L-BFGS-B: {res.nit} iters, final loss={res.fun:.4e}")

    return syn_params, history


def residual_gs(syn_params, full_params, xyt, xyt_mean, xyt_std, activation=jax.nn.tanh,
                net_forward=forward):
    """
    Gray-Scott PDE residual (sum of squared u- and v-equation residuals) per point.

    Residual matches the standard-form solver (see physics/gray_scott.py):
        R_u = u_t - D_u*Lap(u) + u*v^2 - F*(1-u)
        R_v = v_t - D_v*Lap(v) - u*v^2 + (F+k)*v
    Normalization is applied inside the differentiated closure so derivatives
    w.r.t. `pt` (physical space) are correct via autodiff chain rule. Requires
    a twice-differentiable activation (tanh).

    Args:
        syn_params (dict): NN parameters (R^3 -> R^2).
        full_params (dict): Full {D_u,D_v,F,k} (trainable merged over fixed).
        xyt (jnp.ndarray): (N,3) physical-space collocation points [x,y,t].
        xyt_mean (jnp.ndarray): (3,) per-coordinate means for normalization.
        xyt_std (jnp.ndarray): (3,) per-coordinate stds for normalization.
        activation (callable): NN nonlinearity (tanh).
        net_forward (callable): Network apply function (see train_hyco_gs).

    Returns:
        jnp.ndarray: (N,) squared residuals R_u^2 + R_v^2.
    """
    def single(pt):
        def net(q):
            q_n = (q - xyt_mean) / xyt_std
            return net_forward(syn_params, q_n[None], activation=activation)[0]
        u_fun = lambda q: net(q)[0]
        v_fun = lambda q: net(q)[1]
        u, v = u_fun(pt), v_fun(pt)
        gu, gv = jax.grad(u_fun)(pt), jax.grad(v_fun)(pt)
        Hu, Hv = jax.hessian(u_fun)(pt), jax.hessian(v_fun)(pt)
        Lu = Hu[0, 0] + Hu[1, 1]          # u_xx + u_yy
        Lv = Hv[0, 0] + Hv[1, 1]
        uvv = u * v * v
        Ru = gu[2] - full_params['D_u'] * Lu + uvv - full_params['F'] * (1.0 - u)
        Rv = gv[2] - full_params['D_v'] * Lv - uvv + (full_params['F'] + full_params['k']) * v
        return Ru ** 2 + Rv ** 2

    return jax.vmap(single)(xyt)


def train_pinn_gs(syn_params, phy_init, fixed_phy,
                  xyt_data, u_data, xyt_col, xyt_mean, xyt_std,
                  gamma, schedule, n_epochs, key,
                  activation=jax.nn.tanh, ic_n=64, gamma_ic=None,
                  n_pretrain=0, n_lbfgs=0,
                  stab_window=1000, stab_tol=1e-4,
                  schedule_phy=None, net_forward=forward,
                  per_n=0, gamma_per=None):
    """
    PINN baseline for Gray-Scott: joint NN + recovered-parameter optimisation.

    Loss = L_data + gamma_ic * L_ic + gamma * L_pde. gamma_ic defaults to gamma
    if not given. Collocation points are fixed (sampled once by the caller).
    Recovered parameters are optimised in log space (see module header), with
    their own (lower) learning rate schedule_phy — the PDE-residual gradient
    into phy_log is large and noisy early in training, and using the same LR
    as syn causes D_u/D_v to overshoot.

    n_pretrain epochs fit syn_params on data only (phy frozen at phy_init)
    before the joint data+pde+ic objective starts. n_lbfgs runs L-BFGS-B
    afterwards on syn_params only (phy frozen at its final value), reusing
    xyt_col directly since it's already a fixed, dense collocation set.

    Args:
        syn_params (dict): Initial NN parameters (R^3 -> R^2).
        phy_init (dict): Initial values for the trainable (recovered) parameters,
            in physical units; assumed strictly positive.
        fixed_phy (dict): Parameters held constant.
        xyt_data (jnp.ndarray): (M,3) observation coordinates [x,y,t].
        u_data (jnp.ndarray): (M,2) observed [u,v] values.
        xyt_col (jnp.ndarray): (N_col,3) fixed collocation points.
        xyt_mean (jnp.ndarray): (3,) per-coordinate means for normalization.
        xyt_std (jnp.ndarray): (3,) per-coordinate stds for normalization.
        gamma (float): Residual weight (paper: 100).
        schedule: Optax schedule or scalar for the syn NN.
        n_epochs (int): Training epochs (upper bound; may stop earlier, see
            stab_window).
        key: JAX PRNG key (unused; kept for signature symmetry).
        activation (callable): NN nonlinearity (tanh; required for 2nd derivatives).
        ic_n (int): Grid resolution for the initial-datum loss.
        gamma_ic (float | None): Weight on the IC loss; defaults to gamma.
        n_pretrain (int): Data-only syn warmup epochs before the joint loop.
        n_lbfgs (int): L-BFGS-B iterations post-training syn, phy frozen.
        stab_window (int): Averaging window Z for the stabilization stopping
            criterion (eq. `stopping`).
        stab_tol (float): Tolerance epsilon for the stabilization criterion.
        schedule_phy (optax schedule | None): LR schedule for phy_log; defaults
            to a constant 1e-4 (10x below a typical syn LR) if not given.
        net_forward (callable): Network apply function (see train_hyco_gs).
        per_n (int): Points per pair of opposite sides for the optional
            periodicity penalty (values and normal derivatives matched across
            x=0/x=L and y=0/y=L; derivatives scaled by L^2). 0 disables it.
            The points reuse the (x, y, t) of the first per_n collocation points.
        gamma_per (float | None): Weight on the periodicity loss; defaults to gamma.

    Returns:
        tuple: (syn_params, phy_params, history), phy_params in physical units.
    """
    w_ic = gamma if gamma_ic is None else gamma_ic
    w_per = gamma if gamma_per is None else gamma_per

    if per_n > 0:
        q = xyt_col[:per_n]
        zero, side = jnp.zeros(per_n), jnp.full(per_n, DOMAIN_L)
        per_pts = (jnp.stack([zero, q[:, 1], q[:, 2]], axis=1),   # x = 0
                   jnp.stack([side, q[:, 1], q[:, 2]], axis=1),   # x = L
                   jnp.stack([q[:, 0], zero, q[:, 2]], axis=1),   # y = 0
                   jnp.stack([q[:, 0], side, q[:, 2]], axis=1))   # y = L

    def periodic_loss(sp):
        if per_n == 0:
            return 0.0
        pred = lambda xyt: net_forward(sp, _normalize(xyt, xyt_mean, xyt_std), activation=activation)
        jac = jax.vmap(jax.jacfwd(lambda z: pred(z[None])[0]))      # (n, 2, 3)
        left, right, bottom, top = per_pts
        value = jnp.mean((pred(left) - pred(right)) ** 2) + jnp.mean((pred(bottom) - pred(top)) ** 2)
        deriv = (jnp.mean((jac(left)[:, :, 0] - jac(right)[:, :, 0]) ** 2)
                 + jnp.mean((jac(bottom)[:, :, 1] - jac(top)[:, :, 1]) ** 2))
        return value + DOMAIN_L ** 2 * deriv

    xs_ic = (jnp.arange(ic_n) + 0.5) * (DOMAIN_L / ic_n)
    Xic, Yic = jnp.meshgrid(xs_ic, xs_ic, indexing='ij')
    u0_ic, v0_ic = initial_condition(ic_n)
    xyt_ic = jnp.stack([Xic.ravel(), Yic.ravel(), jnp.zeros(ic_n * ic_n)], axis=1)
    uv_ic  = jnp.stack([u0_ic.ravel(), v0_ic.ravel()], axis=1)

    xyt_data_n = _normalize(xyt_data, xyt_mean, xyt_std)
    xyt_ic_n   = _normalize(xyt_ic, xyt_mean, xyt_std)

    all_params = {'syn': syn_params, 'phy_log': _to_log(phy_init)}

    # ── Pretraining: syn-only fit on data, phy frozen at phy_init ─────────────
    if n_pretrain > 0:
        opt_pre = optax.adam(schedule)
        state_pre = opt_pre.init(all_params['syn'])

        @jax.jit
        def pretrain_step(sp, st):
            def objective(sp_):
                return jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
            l, g = jax.value_and_grad(objective)(sp)
            updates, st = opt_pre.update(g, st, sp)
            return optax.apply_updates(sp, updates), st, l

        for epoch in range(n_pretrain):
            all_params['syn'], state_pre, l = pretrain_step(all_params['syn'], state_pre)
            if epoch % 200 == 0 or epoch == n_pretrain - 1:
                print(f"[PINN pretrain {epoch:4d}/{n_pretrain}]  L_syn={float(l):.4e}")

    if schedule_phy is None:
        schedule_phy = optax.constant_schedule(1e-4)

    labels = {
        'syn':     jax.tree_util.tree_map(lambda _: 'syn', all_params['syn']),
        'phy_log': jax.tree_util.tree_map(lambda _: 'phy', all_params['phy_log']),
    }
    opt = optax.multi_transform(
        {'syn': optax.adam(schedule), 'phy': optax.adam(schedule_phy)},
        labels,
    )
    state = opt.init(all_params)
    history = {k: [] for k in ['L_data', 'L_pde', 'L_ic', 'L_total', 'params']}

    @jax.jit
    def step(params, st):
        def objective(p):
            sp = p['syn']
            full = merge_params(_from_log(p['phy_log']), fixed_phy)
            l_data = jnp.mean((net_forward(sp, xyt_data_n, activation=activation) - u_data) ** 2)
            l_ic   = jnp.mean((net_forward(sp, xyt_ic_n, activation=activation) - uv_ic) ** 2)
            l_pde  = jnp.mean(residual_gs(sp, full, xyt_col, xyt_mean, xyt_std, activation=activation, net_forward=net_forward))
            return l_data + w_ic * l_ic + gamma * l_pde + w_per * periodic_loss(sp), (l_data, l_pde, l_ic)
        (l_tot, (l_data, l_pde, l_ic)), g = jax.value_and_grad(objective, has_aux=True)(params)
        updates, st = opt.update(g, st, params)
        return optax.apply_updates(params, updates), st, l_tot, l_data, l_pde, l_ic

    for epoch in range(n_epochs):
        all_params, state, l_tot, l_data, l_pde, l_ic = step(all_params, state)
        phy_now = {k: float(v) for k, v in _from_log(all_params['phy_log']).items()}
        history['L_data'].append(float(l_data))
        history['L_pde'].append(float(l_pde))
        history['L_ic'].append(float(l_ic))
        history['L_total'].append(float(l_tot))
        history['params'].append(phy_now)
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={v:.3e}' for k, v in phy_now.items())
            print(f"[PINN {epoch:4d}/{n_epochs}]  L_total={l_tot:.3e}  "
                  f"L_data={l_data:.3e}  L_pde={l_pde:.3e}  L_ic={l_ic:.3e}  | {phy_str}")

        if stabilized(history['params'], stab_window, stab_tol):
            print(f"  Stopped at epoch {epoch}: parameters stabilized "
                  f"(window={stab_window}, tol={stab_tol}).")
            break

    # ── L-BFGS-B post-training of syn_params, phy frozen at final value ──────
    if n_lbfgs > 0:
        from jax.flatten_util import ravel_pytree
        from scipy.optimize import minimize

        full_final = merge_params(_from_log(all_params['phy_log']), fixed_phy)
        flat0, unravel = ravel_pytree(all_params['syn'])

        def lbfgs_objective(flat):
            sp = unravel(jnp.asarray(flat))
            def obj(sp_):
                l_data = jnp.mean((net_forward(sp_, xyt_data_n, activation=activation) - u_data) ** 2)
                l_ic   = jnp.mean((net_forward(sp_, xyt_ic_n, activation=activation) - uv_ic) ** 2)
                l_pde  = jnp.mean(residual_gs(sp_, full_final, xyt_col, xyt_mean, xyt_std, activation=activation, net_forward=net_forward))
                return l_data + w_ic * l_ic + gamma * l_pde + w_per * periodic_loss(sp_)
            val, grad = jax.value_and_grad(obj)(sp)
            flat_grad, _ = ravel_pytree(grad)
            return float(val), np.asarray(flat_grad, dtype=np.float64)

        res = minimize(lbfgs_objective, np.asarray(flat0, dtype=np.float64),
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': n_lbfgs, 'maxcor': 100})
        all_params['syn'] = unravel(jnp.asarray(res.x))
        print(f"[PINN] L-BFGS-B: {res.nit} iters, final loss={res.fun:.4e}")

    phy_params = {k: jnp.exp(v) for k, v in all_params['phy_log'].items()}
    return all_params['syn'], phy_params, history

def make_opt_pinn_net(xyt_mean, xyt_std, tau_ic):
    """
    Build the opt-PINN network apply function: Fourier-feature MLP with the
    initial condition enforced as a hard constraint,

        (u, v)(x, y, t) = (u0, v0)(x, y) + (1 - exp(-t / tau_ic)) * N(x_n),

    where (u0, v0) is the smoothed analytic disc (ic_smooth) and N is the
    Fourier-feature MLP evaluated on the z-scored input x_n. At t = 0 the
    output equals the (smoothed) initial condition exactly, so no IC loss is
    needed. The returned callable has the standard net_forward signature
    (params, x_normalized, activation) and un-normalizes internally to get
    physical (x, y, t) for the constraint terms.

    Args:
        xyt_mean (jnp.ndarray): (3,) input means used for normalization.
        xyt_std (jnp.ndarray): (3,) input stds used for normalization.
        tau_ic (float): Ramp timescale of the hard IC factor.

    Returns:
        callable: net(params, x_n, activation) -> (N, 2) predictions.
    """
    def net(sp, x_n, activation=jax.nn.tanh):
        xyt = x_n * xyt_std + xyt_mean
        u0, v0 = ic_smooth(xyt[..., 0], xyt[..., 1])
        ramp = 1.0 - jnp.exp(-xyt[..., 2] / tau_ic)
        out = forward_fourier_mlp(sp, x_n, activation=activation)
        return jnp.stack([u0, v0], axis=-1) + ramp[..., None] * out
    return net


def train_opt_pinn_gs(syn_params, phy_init, fixed_phy,
                      xyt_data, u_data, xyt_mean, xyt_std,
                      gamma, schedule, n_epochs, key, n_col, T,
                      tau_ic=50.0, causal_eps=1.0, causal_bins=20,
                      activation=jax.nn.tanh,
                      n_pretrain=0, n_lbfgs=0, d_warmup=0, lr_phy=8e-3):
    """
    Best-practice PINN competitor ("opt-PINN") for Gray-Scott.

    Differences from the vanilla PINN baseline (train_pinn_gs), each standard
    in the PINN literature and disclosed in the paper:
      1. Fourier-feature embedding (fixed random B) against spectral bias.
      2. Hard initial condition (no L_ic; see make_opt_pinn_net).
      3. Causal residual weighting in time (Wang et al. 2022): collocation
         points are binned by t; bin k is weighted exp(-eps * sum_{j<k} L_j)
         so late times only contribute once early times are resolved.
      4. Collocation points resampled every epoch (n_col fresh points).
      5. Nondimensionalized residual: squared residuals are scaled by
         t_char^2 with t_char = 1/(F + k) (reaction timescale), so gamma
         weights an O(1) quantity. F, k are taken from the initial merged
         parameters; intended for RECOVER=('D_u','D_v') where they are fixed.
      6. Staged D learning: syn pretrained on data only (n_pretrain epochs);
         during joint training the phy learning rate is 0 for the first
         d_warmup epochs; global gradient clipping at norm 1.0 throughout.

    No stabilization stopping criterion: causal weighting resolves late times
    last, so parameter plateaus mid-training are expected and benign; the
    epoch budget is fixed.

    Args:
        syn_params (dict): Fourier-MLP parameters (init_params_fourier_mlp).
        phy_init (dict): Initial values of the recovered parameters.
        fixed_phy (dict): Parameters held at their true values.
        xyt_data (jnp.ndarray): (M,3) data coordinates (physical units).
        u_data (jnp.ndarray): (M,2) observed (u,v).
        xyt_mean, xyt_std (jnp.ndarray): (3,) normalization statistics.
        gamma (float): Weight on the nondimensionalized residual loss.
        schedule: Optax LR schedule for syn_params.
        n_epochs (int): Joint-training epochs (fixed; no early stop).
        key: JAX PRNG key (collocation resampling).
        n_col (int): Collocation points per epoch.
        T (float): Final time (collocation sampled over (0, T)).
        tau_ic (float): Hard-IC ramp timescale.
        causal_eps (float): Causality tolerance epsilon.
        causal_bins (int): Number of time bins for causal weighting.
        activation (callable): NN nonlinearity (tanh; residual needs C^2).
        n_pretrain (int): Data-only pretraining epochs (phy frozen).
        n_lbfgs (int): L-BFGS-B iterations on syn (phy frozen) afterwards.
        d_warmup (int): Joint epochs with phy LR held at 0.
        lr_phy (float): Adam LR for phy_log after warmup.

    Returns:
        tuple: (syn_params, phy_params, history), phy_params in physical units.
    """
    net = make_opt_pinn_net(xyt_mean, xyt_std, tau_ic)
    xyt_data_n = _normalize(xyt_data, xyt_mean, xyt_std)
    full0 = merge_params(phy_init, fixed_phy)
    t_char = 1.0 / float(full0['F'] + full0['k'])

    # ── Stage A: data-only pretraining (phy frozen, hard IC built in) ─────────
    if n_pretrain > 0:
        opt_pre = optax.adam(1e-3)
        st_pre = opt_pre.init(syn_params)

        @jax.jit
        def pre_step(sp, st):
            l, g = jax.value_and_grad(
                lambda p: jnp.mean((net(p, xyt_data_n, activation=activation)
                                    - u_data) ** 2))(sp)
            up, st = opt_pre.update(g, st, sp)
            return optax.apply_updates(sp, up), st, l

        for epoch in range(n_pretrain):
            syn_params, st_pre, l = pre_step(syn_params, st_pre)
            if epoch % 200 == 0 or epoch == n_pretrain - 1:
                print(f"[opt-PINN pretrain {epoch:4d}/{n_pretrain}]  L_syn={float(l):.4e}")

    # ── Stage B: joint training ────────────────────────────────────────────────
    all_params = {'syn': syn_params, 'phy_log': _to_log(phy_init)}
    labels = {
        'syn':     jax.tree_util.tree_map(lambda _: 'syn', all_params['syn']),
        'phy_log': jax.tree_util.tree_map(lambda _: 'phy', all_params['phy_log']),
    }
    # phy LR is 0 during warmup, lr_phy afterwards (staged D learning)
    phy_sched = lambda count: jnp.where(count < d_warmup, 0.0, lr_phy)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({'syn': optax.adam(schedule),
                               'phy': optax.adam(phy_sched)}, labels),
    )
    state = opt.init(all_params)
    history = {k: [] for k in ['L_data', 'L_pde', 'L_total', 'w_last', 'params']}

    @jax.jit
    def step(params, st, xyt_col):
        def objective(p):
            sp = p['syn']
            full = merge_params(_from_log(p['phy_log']), fixed_phy)
            l_data = jnp.mean((net(sp, xyt_data_n, activation=activation)
                               - u_data) ** 2)
            # Nondimensionalized squared residuals, per collocation point
            r2 = residual_gs(sp, full, xyt_col, xyt_mean, xyt_std,
                             activation=activation, net_forward=net) * t_char ** 2
            # Causal weighting over time bins
            bins = jnp.clip((xyt_col[:, 2] / T * causal_bins).astype(jnp.int32),
                            0, causal_bins - 1)
            cnt = jax.ops.segment_sum(jnp.ones_like(r2), bins, causal_bins)
            Lk = jax.ops.segment_sum(r2, bins, causal_bins) / (cnt + 1e-12)
            w = jax.lax.stop_gradient(
                jnp.exp(-causal_eps * (jnp.cumsum(Lk) - Lk)))
            l_pde = jnp.sum(w * Lk) / (jnp.sum(w) + 1e-12)
            return l_data + gamma * l_pde, (l_data, l_pde, w[-1])
        (l_tot, (l_data, l_pde, w_last)), g = jax.value_and_grad(
            objective, has_aux=True)(params)
        updates, st = opt.update(g, st, params)
        return optax.apply_updates(params, updates), st, l_tot, l_data, l_pde, w_last

    for epoch in range(n_epochs):
        key, kc = jr.split(key)
        xyt_col = sample_points(kc, n_col, (0.0, T),
                                (0.0, DOMAIN_L, 0.0, DOMAIN_L))
        all_params, state, l_tot, l_data, l_pde, w_last = step(all_params, state, xyt_col)
        phy_now = {k: float(v) for k, v in _from_log(all_params['phy_log']).items()}
        history['L_data'].append(float(l_data))
        history['L_pde'].append(float(l_pde))
        history['L_total'].append(float(l_tot))
        history['w_last'].append(float(w_last))
        history['params'].append(phy_now)
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={v:.3e}' for k, v in phy_now.items())
            print(f"[opt-PINN {epoch:4d}/{n_epochs}]  L_total={l_tot:.3e}  "
                  f"L_data={l_data:.3e}  L_pde={l_pde:.3e}  "
                  f"w_T={float(w_last):.2e}  | {phy_str}")

    # ── Stage C: L-BFGS-B on syn, phy frozen, unweighted residual ─────────────
    if n_lbfgs > 0:
        from jax.flatten_util import ravel_pytree
        from scipy.optimize import minimize

        key, kd = jr.split(key)
        xyt_dense = sample_points(kd, 2 * n_col, (0.0, T),
                                  (0.0, DOMAIN_L, 0.0, DOMAIN_L))
        full_final = merge_params(_from_log(all_params['phy_log']), fixed_phy)
        flat0, unravel = ravel_pytree(all_params['syn'])

        def lbfgs_objective(flat):
            sp = unravel(jnp.asarray(flat))
            def obj(sp_):
                l_data = jnp.mean((net(sp_, xyt_data_n, activation=activation)
                                   - u_data) ** 2)
                r2 = residual_gs(sp_, full_final, xyt_dense, xyt_mean, xyt_std,
                                 activation=activation, net_forward=net) * t_char ** 2
                return l_data + gamma * jnp.mean(r2)
            val, grad = jax.value_and_grad(obj)(sp)
            flat_grad, _ = ravel_pytree(grad)
            return float(val), np.asarray(flat_grad, dtype=np.float64)

        res = minimize(lbfgs_objective, np.asarray(flat0, dtype=np.float64),
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': n_lbfgs, 'maxcor': 100})
        all_params['syn'] = unravel(jnp.asarray(res.x))
        print(f"[opt-PINN] L-BFGS-B: {res.nit} iters, final loss={res.fun:.4e}")

    phy_params = {k: jnp.exp(v) for k, v in all_params['phy_log'].items()}
    return all_params['syn'], phy_params, history