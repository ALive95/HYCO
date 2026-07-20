import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from physics.conservation_law import time_step_ssprk2
from physics.helmholtz_fem import solve_helmholtz, interp_at_pts
from synthetic.nn import forward, forward_latent
from physics.pod_rom import solve_rom
from physics.helmholtz_fem import kappa, eta, forcing


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


def interp_solution(u_hist, x_cells, T, tx):
    """
    Bilinear interpolation of a 2D FV solution array at arbitrary (t, x) points.

    Args:
        u_hist (jnp.ndarray): (n_t, n_x) solution snapshots on uniform time grid.
        x_cells (jnp.ndarray): (n_x,) cell center positions.
        T (float): Final time (first snapshot is at t=0, last at t=T).
        tx (jnp.ndarray): (Q, 2) query points with columns [t, x].

    Returns:
        jnp.ndarray: (Q,) interpolated values.
    """
    n_t, n_x = u_hist.shape
    t_q, x_q = tx[:, 0], tx[:, 1]

    t_idx = t_q / T * (n_t - 1)
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
               bc_left=0.0, bc_right=0.0, lambda_reg=0):
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
        n_epochs (int): Training epochs.
        n_steps (int): FV time steps.
        dt (float): FV time step size.
        T (float): Final time.
        t_range (tuple): (t_min, t_max) for ghost sampling.
        x_range (tuple): (x_min, x_max) for ghost sampling.
        key: JAX PRNG key.
        bc_left (float): Left boundary value.
        bc_right (float): Right boundary value.
        lambda_reg (float): L2 regularisation weight on syn_params.

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
            ls = jnp.mean((forward(sp_, tx_data) - u_data) ** 2)
            u_p = interp_solution(u_hist, x_cells, T, tx_g)
            li = jnp.mean((forward(sp_, tx_g) - u_p) ** 2)
            l_reg = sum(jnp.sum(w ** 2) for w in jax.tree_util.tree_leaves(sp_))
            return alpha * ls + li + lambda_reg * l_reg, (ls, li)
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
            u_s = forward(sp, tx_g)
            li = jnp.mean((u_s - interp_solution(uh, x_cells, T, tx_g)) ** 2)
            return beta * lp + li, (lp, li)
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

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={float(v):.6f}' if v.ndim == 0
                                else f'{k}={v}' for k, v in phy_params.items())
            print(f"[{epoch:4d}/{n_epochs}]  L_syn={ls:.4e}  L_int={li_p:.4e}  "
                  f"L1={l1:.4e}  L2={l2:.4e}  | {phy_str}")

    return phy_params, syn_params, history


# ── PINN conservation law ─────────────────────────────────────────────────────
# Renamed: pde_residual → pde_residual_claw

def pde_residual_claw(syn_params, phy_params, flux_fn, tx_col):
    """
    PDE residual u_t + (f(u))_x at collocation points for a 1D conservation law.

    Args:
        syn_params (dict): NN parameters.
        phy_params (dict): Trainable flux parameters.
        flux_fn (callable): Physical flux function.
        tx_col (jnp.ndarray): (N,2) collocation points [t, x].

    Returns:
        jnp.ndarray: (N,) residuals.
    """
    def residual_single(tx):
        u_val, grad_tx = jax.value_and_grad(
            lambda tx_: forward(syn_params, tx_[None])[0]
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
               t_range, x_range, key):
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
        n_epochs (int): Training epochs.
        t_range (tuple): (t_min, t_max) for collocation sampling.
        x_range (tuple): (x_min, x_max) for collocation sampling.
        key: JAX PRNG key.

    Returns:
        tuple: (syn_params, phy_params, history).
    """
    all_params = {'syn': syn_params, 'phy': phy_params}
    opt = optax.adam(schedule)
    state = opt.init(all_params)

    @jax.jit
    def step(params, opt_state, tx_col):
        def objective(p):
            sp, pp = p['syn'], p['phy']
            l_data = jnp.mean((forward(sp, tx_data) - u_data) ** 2)
            res = pde_residual_claw(sp, pp, flux_fn, tx_col)
            l_pde = jnp.mean(res ** 2)
            return gamma * l_data + l_pde, (l_data, l_pde)

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

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            phy_str = ', '.join(f'{k}={float(v):.6f}' if v.ndim == 0
                                else f'{k}={v}' for k, v in all_params['phy'].items())
            print(f"  epoch {epoch:4d}  L_total={float(l_total):.3e}"
                  f"  L_data={float(l_data):.3e}  L_pde={float(l_pde):.3e}"
                  f"  {phy_str}")

    return all_params['syn'], all_params['phy'], history


# ── HYCO Helmholtz ────────────────────────────────────────────────────────────

def train_hyco_helmholtz(
    fem_coarse, syn_params, phy_params,
    schedule_syn, schedule_phy,
    sensor_xy, u_data, sensor_tri_c, sensor_bary_c,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    alpha, beta, n_ghost, n_ghost_pool, n_epochs, key,
):
    """
    HYCO training for the Helmholtz BVP.

    Alternates between:
      syn_step: fit sensor observations + match FEM solution at ghost points.
      phy_step: update PDE parameters so FEM matches NN at ghost points.

    Args:
        fem_coarse (FEMData): Coarse-mesh FEM data.
        syn_params (dict): Initial NN parameters.
        phy_params (dict): Initial PDE parameters {'alpha1','c1','alpha2','c2'}.
        schedule_syn: Optax schedule or scalar for syn optimizer.
        schedule_phy: Optax schedule or scalar for phy optimizer.
        sensor_xy (jnp.ndarray): (M,2) sensor locations.
        u_data (jnp.ndarray): (M,) sensor observations.
        sensor_tri_c (jnp.ndarray): (M,) coarse-mesh element indices for sensors.
        sensor_bary_c (jnp.ndarray): (M,3) barycentric coords for sensors.
        ghost_pool_xy (jnp.ndarray): (P,2) pre-sampled ghost point coords.
        ghost_pool_tri (jnp.ndarray): (P,) element indices for ghost pool.
        ghost_pool_bary (jnp.ndarray): (P,3) barycentric coords for ghost pool.
        alpha (float): Weight on sensor loss in syn_step.
        beta (float): Weight on sensor loss in phy_step.
        n_ghost (int): Ghost points drawn per epoch from the pool.
        n_ghost_pool (int): Size of the pre-sampled ghost pool.
        n_epochs (int): Training epochs.
        key: JAX PRNG key.

    Returns:
        tuple: (phy_params, syn_params, history).
    """
    opt_syn = optax.adam(schedule_syn)
    opt_phy = optax.adam(schedule_phy)
    st_syn = opt_syn.init(syn_params)
    st_phy = opt_phy.init(phy_params)

    hist = {k: [] for k in ['L_syn', 'L_phy', 'L_int', 'L1', 'L2', 'params']}

    @jax.jit
    def syn_step(sp, st, pp, g_idx):
        u_phy = solve_helmholtz(fem_coarse, pp)
        gxy   = ghost_pool_xy[g_idx]
        gtri  = ghost_pool_tri[g_idx]
        gbary = ghost_pool_bary[g_idx]

        def objective(sp_):
            ls = jnp.mean((forward(sp_, sensor_xy) - u_data) ** 2)
            u_syn_g = forward(sp_, gxy)
            u_phy_g = interp_at_pts(fem_coarse, u_phy, gtri, gbary)
            li = jnp.mean((u_syn_g - u_phy_g) ** 2)
            return alpha * ls + li, (ls, li)

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
            u_phy = solve_helmholtz(fem_coarse, pp_)
            u_phy_s = interp_at_pts(fem_coarse, u_phy, sensor_tri_c, sensor_bary_c)
            lp = jnp.mean((u_phy_s - u_data) ** 2)
            u_syn_g = forward(sp, gxy)
            u_phy_g = interp_at_pts(fem_coarse, u_phy, gtri, gbary)
            li = jnp.mean((u_syn_g - u_phy_g) ** 2)
            return beta * lp + li, (lp, li)

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
        hist['params'].append({k: float(v) if v.ndim == 0 else v.tolist()
                               for k, v in phy_params.items()})

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            a1, a2 = float(phy_params['alpha1']), float(phy_params['alpha2'])
            c1, c2 = phy_params['c1'], phy_params['c2']
            print(f"  [{epoch:4d}]  Ls={ls:.2e} Lp={lp:.2e} Li={li_p:.2e}  "
                  f"a1={a1:.3f} c1=({float(c1[0]):.3f},{float(c1[1]):.3f})  "
                  f"a2={a2:.3f} c2=({float(c2[0]):.3f},{float(c2[1]):.3f})")

    return phy_params, syn_params, hist


# ── PINN Helmholtz ────────────────────────────────────────────────────────────
# Renamed: pde_residual (in main_helmholtz_test.py) → pde_residual_helmholtz

def pde_residual_helmholtz(syn_params, phy_params, xy):
    """
    PDE residual of div(kappa grad u) + eta^2 u - f at collocation points.

    Args:
        syn_params (dict): NN parameters.
        phy_params (dict): PDE parameters {'alpha1','c1','alpha2','c2'}.
        xy (jnp.ndarray): (N,2) collocation points.

    Returns:
        jnp.ndarray: (N,) residuals.
    """
    def r_single(xy_):
        u_fn = lambda x: forward(syn_params, x[None])[0]
        u_val = u_fn(xy_)
        grad_u = jax.grad(u_fn)(xy_)
        kap_fn = lambda x: kappa(x[None], phy_params['alpha1'], phy_params['c1'])[0]
        kap_val = kap_fn(xy_)
        grad_kap = jax.grad(kap_fn)(xy_)
        et_val = eta(xy_[None], phy_params['alpha2'], phy_params['c2'])[0]
        f_val = forcing(xy_[None])[0]
        # second-order directional derivatives for the Laplacian
        u_xx = jax.grad(lambda x: jax.grad(u_fn)(x)[0])(xy_)[0]
        u_yy = jax.grad(lambda x: jax.grad(u_fn)(x)[1])(xy_)[1]
        div_kap_grad_u = jnp.dot(grad_kap, grad_u) + kap_val * (u_xx + u_yy)
        return div_kap_grad_u + et_val ** 2 * u_val - f_val

    return jax.vmap(r_single)(xy)


def train_pinn_helmholtz(
    pinn_syn, pinn_phy,
    schedule_pinn,
    sensor_xy, u_data,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    gamma, n_col, n_ghost_pool, n_epochs, key,
):
    """
    PINN training for the Helmholtz BVP.
    Jointly optimises NN and PDE parameters via data + PDE residual losses.

    Args:
        pinn_syn (dict): Initial NN parameters.
        pinn_phy (dict): Initial PDE parameters {'alpha1','c1','alpha2','c2'}.
        schedule_pinn: Optax schedule or scalar for the Adam optimizer.
        sensor_xy (jnp.ndarray): (M,2) sensor locations.
        u_data (jnp.ndarray): (M,) sensor observations.
        ghost_pool_xy (jnp.ndarray): (P,2) pre-sampled collocation point coords.
        ghost_pool_tri (jnp.ndarray): (P,) element indices (used only for sampling).
        ghost_pool_bary (jnp.ndarray): (P,3) barycentric coords (used only for sampling).
        gamma (float): Weight on PDE residual loss.
        n_col (int): Collocation points drawn per epoch from the pool.
        n_ghost_pool (int): Size of the ghost pool.
        n_epochs (int): Training epochs.
        key: JAX PRNG key.

    Returns:
        tuple: (syn_params, phy_params, history).
    """
    all_pinn = {'syn': pinn_syn, 'phy': pinn_phy}
    opt_pinn = optax.adam(schedule_pinn)
    st_pinn = opt_pinn.init(all_pinn)

    hist = {k: [] for k in ['L_data', 'L_pde', 'params']}

    @jax.jit
    def pinn_step(params, st, xy_col):
        def objective(p):
            l_data = jnp.mean((forward(p['syn'], sensor_xy) - u_data) ** 2)
            res = pde_residual_helmholtz(p['syn'], p['phy'], xy_col)
            l_pde = jnp.mean(res ** 2)
            return l_data + gamma * l_pde, (l_data, l_pde)

        (lt, (ld, lp)), g = jax.value_and_grad(objective, has_aux=True)(params)
        updates, st = opt_pinn.update(g, st, params)
        params = optax.apply_updates(params, updates)
        return params, st, lt, ld, lp

    for epoch in range(n_epochs):
        key, kg = jr.split(key)
        g_idx = jr.choice(kg, n_ghost_pool, shape=(n_col,), replace=False)
        xy_col = ghost_pool_xy[g_idx]

        all_pinn, st_pinn, lt, ld, lp = pinn_step(all_pinn, st_pinn, xy_col)

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

    return all_pinn['syn'], all_pinn['phy'], hist


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
                      mean, basis, kx, ky, ksq, dealias, N, key):
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

    Returns:
        tuple: (nu_found, syn_params, history).
    """
    r          = z0.shape[0]
    key, k_init = jr.split(key)
    # init_latent_params imported via synthetic.nn (forward_latent used below)
    from synthetic.nn import init_latent_params
    syn_params = init_latent_params(k_init, r)
    nu         = nu_init

    opt_syn   = optax.adam(lr_syn)
    opt_phy   = optax.adam(lr_phy)
    state_syn = opt_syn.init(syn_params)
    state_phy = opt_phy.init(nu)

    def syn_loss(sp, z_rom_hist, t_ghost):
        z_pred = forward_latent(sp, t_data)                                   # (M, r)
        y_pred = _observe_batch(z_pred, mean, basis, sensor_locs)
        L_obs  = jnp.mean((y_pred - y_data) ** 2)
        L_int  = jnp.mean((forward_latent(sp, t_ghost)
                           - _interp_latent(z_rom_hist, T, t_ghost)) ** 2)
        return alpha * L_obs + L_int, (L_obs, L_int)

    def phy_loss(nv, sp, t_ghost):
        z_rom  = solve_rom(z0, nv, dt_rom, n_steps_rom, save_every_rom,
                           mean, basis, kx, ky, ksq, dealias, N)
        y_rom  = _observe_batch(_interp_latent(z_rom, T, t_data), mean, basis, sensor_locs)
        L_obs  = jnp.mean((y_rom - y_data) ** 2)
        L_int  = jnp.mean((forward_latent(sp, t_ghost)
                           - _interp_latent(z_rom, T, t_ghost)) ** 2)
        return beta * L_obs + L_int, (L_obs, L_int)

    def epoch_step(carry, key):
        sp, st_syn, nv, st_phy = carry
        t_ghost = jr.uniform(key, (n_ghost,), minval=0.0, maxval=T)

        # ROM forward pass (fixed nu — no grad here)
        z_rom_hist = solve_rom(z0, nv, dt_rom, n_steps_rom, save_every_rom,
                               mean, basis, kx, ky, ksq, dealias, N)

        (_, (L_syn, _)), g_syn = jax.value_and_grad(syn_loss, has_aux=True)(sp, z_rom_hist, t_ghost)
        upd_syn, st_syn = opt_syn.update(g_syn, st_syn, sp)
        sp = optax.apply_updates(sp, upd_syn)

        # phy_step: ROM re-runs with grad tracked through nu
        (_, (L_phy, L_int)), g_phy = jax.value_and_grad(phy_loss, has_aux=True)(nv, sp, t_ghost)
        upd_phy, st_phy = opt_phy.update(g_phy, st_phy, nv)
        nv = optax.apply_updates(nv, upd_phy)

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




# ── HYCO Poisson ──────────────────────────────────────────────────────────────

def train_hyco_poisson(
    pdata,
    syn_params, phy_params,
    schedule_syn, schedule_phy,
    sensor_xy, u_data,
    sensor_tri, sensor_bary,
    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
    alpha, beta, n_ghost, n_ghost_pool, n_epochs,
    lambda_reg, lambda_smooth, key,
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
        lambda_reg (float): Tikhonov Regularization weight for the forcing learning.
        lambda_smooth (float): Regularization weight for the forcing learning (modewise).
        key: JAX PRNG key.

    Returns:
        tuple: (phy_params, syn_params, history).
    """
    from physics.poisson import solve_poisson_from_coeff
    from physics.helmholtz_fem import interp_at_pts

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
            ls      = jnp.mean((forward(sp_, sensor_xy) - u_data) ** 2)
            u_nn_g  = forward(sp_, gxy)
            u_fem_g = interp_at_pts(pdata.fem, u_fem, gtri, gbary)
            li      = jnp.mean((u_nn_g - u_fem_g) ** 2)
            return alpha * ls + li, (ls, li)

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
            u_nn_g   = forward(sp, gxy)
            u_fem_g  = interp_at_pts(pdata.fem, u_fem, gtri, gbary)
            li       = jnp.mean((u_nn_g - u_fem_g) ** 2)
            fc = pp_['f_coeff']
            l_reg = lambda_reg * jnp.sum(fc ** 2) \
                    + lambda_smooth * jnp.sum(mode_weights * fc ** 2)
            return beta * lp + li + l_reg, (lp, li)

        (l1, (lp, li)), g = jax.value_and_grad(objective, has_aux=True)(pp)
        updates, st = opt_phy.update(g, st, pp)
        pp = optax.apply_updates(pp, updates)
        return pp, st, l1, lp, li

    # precompute once before the loop — shape (N*N,)
    N = pdata.N
    idx_i = jnp.repeat(jnp.arange(N), N)  # [0,0,...,1,1,...,N-1,N-1]
    idx_j = jnp.tile(jnp.arange(N), N)  # [0,1,...,N-1, 0,1,...,N-1]
    mode_weights = (idx_i ** 2 + idx_j ** 2).astype(jnp.float64)  # (N²,)

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
