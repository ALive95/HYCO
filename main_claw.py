"""
HYCO and PINN for LWR traffic flow (conservation law) parameter identification.

Run modes (set MODE in main):
  "basic"       — train HYCO only (default alpha, beta, gamma_int=1.0).
  "comparison"  — train PINN, EKI, and the HYCO ablation (gamma_int=0.0, no
                  ghost-point coupling) alongside HYCO, for the comparison figure.
  "sweep"       — sweep ALPHA_GRID x BETA_GRID for HYCO only.
  "full"        — basic + comparison in sequence.

SEEDS controls multi-seed aggregation (orthogonal to MODE).
LOAD=True skips training if a checkpoint already exists for that method/tag.
"""
import jax
jax.config.update("jax_enable_x64", True)

import os
import pickle

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from physics.conservation_law import (
    make_mesh, precompute_fv, solve_conservation_law, lwr_flux, lwr_max_wave_speed,
    lwr_godunov_flux,
)
from synthetic.nn import init_params, forward
from utils.training import (
    train_hyco, train_pinn, train_eki_claw,
    solve_fv_fixed, interp_solution,
    sample_ghost, make_full_params,
)
from utils.model_io import save_claw, load_claw
from utils.plotting import plot_claw_basic, plot_claw_comparison
from utils.experiment import (
    DEFAULT_SEEDS, time_block, aggregate_seeds, print_aggregated,
    make_run_tag, sweep_alpha_beta_grid,
)


# ── Activation choices ────────────────────────────────────────────────────────
# HYCO does not need higher derivatives of the syn NN — ReLU is fine.
# PINN's residual takes one space and one time derivative — tanh is the safe
# default (still works with ReLU here, but matches the Helmholtz convention).
HYCO_ACTIVATION = jax.nn.relu
PINN_ACTIVATION = jax.nn.tanh


TRUE_PARAMS = {'v_max': jnp.float64(1.0), 'rho_max': jnp.float64(1.0)}
INIT_PARAMS = {'v_max': jnp.float64(0.5), 'rho_max': jnp.float64(0.6)}


def ic(cell_centers):
    """Traffic-jam initial condition: density 0.9 in [0.5, 1.5], else 0.1."""
    return jnp.where(
        (cell_centers >= 0.5) & (cell_centers <= 1.5),
        jnp.float64(0.9), jnp.float64(0.1),
    )


def pinn_bc_kwargs(x_cells, u0, T, x_left, x_right, bc_val, n_bc):
    """
    Optional PINN initial/boundary penalties (config PINN_BC=True).

    Initial profile at the reference cell centers; at both ends, flux
    compatibility with the reservoir state through the Godunov flux, which
    imposes incoming information and leaves the outgoing trace free.
    Both weights default to the PINN weight GAMMA.
    """
    t_bc = jnp.linspace(0.0, T, n_bc)
    return dict(
        tx_ic=jnp.stack([jnp.zeros_like(x_cells), x_cells], axis=1), u_ic=u0,
        tx_bc_left=jnp.stack([t_bc, jnp.full_like(t_bc, x_left)], axis=1),
        tx_bc_right=jnp.stack([t_bc, jnp.full_like(t_bc, x_right)], axis=1),
        u_bc=bc_val, num_flux_fn=lwr_godunov_flux,
    )


def run_experiment(seed, alpha, beta, config, run_tag="", mode="basic", load=False):
    """
    Run one or more methods for the LWR experiment under the given mode.

    Data setup always runs (deterministic from seed), so the ablation (trained
    as part of "comparison") sees exactly the same reference solution and
    sensor locations as HYCO in "basic".

    Args:
        seed (int): PRNG seed.
        alpha (float): HYCO syn-step weight on data loss.
        beta (float): HYCO phy-step weight on data loss (0.0 = no phy data loss).
        config (dict): Run-invariant settings.
        run_tag (str): Filename suffix; "" means default filenames.
        mode (str): One of "basic", "comparison", "sweep", "full".
        load (bool): If True, load checkpoint when available instead of retraining.

    Returns:
        dict: Scalar metrics for each method that ran.
    """
    X_LEFT, X_RIGHT = config['X_LEFT'], config['X_RIGHT']
    T        = config['T']
    T_DATA   = config['T_DATA']
    X_DATA_R = config['X_DATA_R']
    CFL      = config['CFL']
    N_FINE   = config['N_FINE']
    N_COARSE = config['N_COARSE']
    N_DATA   = config['N_DATA']
    N_GHOST  = config['N_GHOST']
    N_COL    = config['N_COL']
    N_EPOCHS = config['N_EPOCHS']
    GAMMA    = config['GAMMA']
    PINN_BC  = config.get('PINN_BC', False)
    pinn_tag = '_bc' if PINN_BC else ''

    schedule_syn_hyco = optax.cosine_decay_schedule(5e-4, N_EPOCHS, alpha=0.01)
    schedule_phy_hyco = optax.cosine_decay_schedule(1e-3, N_EPOCHS, alpha=0.01)
    schedule_pinn     = optax.cosine_decay_schedule(1e-4, N_EPOCHS, alpha=0.01)

    # ── Upfront key splits (order fixed regardless of mode) ───────────────────
    # Each method always gets the same keys for a given seed, even when other
    # methods are skipped — this ensures mode-independence.
    key = jr.PRNGKey(seed)
    key, k_data                    = jr.split(key)
    key, k_hyco_init, k_hyco_train = jr.split(key, 3)
    key, k_abl_init,  k_abl_train  = jr.split(key, 3)
    key, k_pinn_init, k_pinn_train = jr.split(key, 3)
    key, k_eki                     = jr.split(key)
    _, k_eval                      = jr.split(key)

    # ── 2. Data setup (always runs) ────────────────────────────────────────────
    centers_fine, edges_fine = make_mesh(X_LEFT, X_RIGHT, N_FINE)
    fv_fine = precompute_fv(centers_fine, edges_fine)
    u0_fine = ic(fv_fine.cell_centers)
    BC_VAL  = float(TRUE_PARAMS['rho_max']) * 0.1

    true_params_full = {**TRUE_PARAMS, 'flux_fn': lwr_flux,
                        'max_wave_speed_fn': lwr_max_wave_speed}
    t_hist_ref, u_hist_ref = solve_conservation_law(
        fv_fine, u0_fine, true_params_full, T, cfl=CFL,
        bc_left=BC_VAL, bc_right=BC_VAL)
    u_ref_array = jnp.stack(u_hist_ref)
    print(f"Reference: {N_FINE} cells, {len(u_hist_ref)} time steps")

    tx_data = sample_ghost(k_data, N_DATA,
                           t_range=(0.0, T_DATA), x_range=(X_LEFT, X_DATA_R))
    # Observations keep the archived convention (uniform snapshot times), so that
    # saved checkpoints are replayed on their original training data; the
    # difference from the correctly timed values is below 1e-4.
    u_data  = interp_solution(u_ref_array, fv_fine.cell_centers, T, tx_data)
    print(f"Data: {N_DATA} points in t in [0, {T_DATA}], x in [{X_LEFT}, {X_DATA_R}]")

    centers_c, edges_c = make_mesh(X_LEFT, X_RIGHT, N_COARSE)
    fv_coarse = precompute_fv(centers_c, edges_c)
    u0_coarse = ic(fv_coarse.cell_centers)
    dt_coarse = CFL * float(fv_coarse.dx) / float(TRUE_PARAMS['v_max'])
    N_STEPS   = int(np.ceil(T / dt_coarse))
    dt_coarse = jnp.float64(T / N_STEPS)

    t_range = (jnp.float64(0.0), jnp.float64(T))
    x_range = (jnp.float64(X_LEFT), jnp.float64(X_RIGHT))

    # ── Results containers (None = method not run this call) ──────────────────
    syn_hyco = phy_hyco = hist_hyco = t_hyco_elapsed = None
    syn_abl  = phy_abl  = hist_abl  = t_abl_elapsed  = None
    syn_pinn = phy_pinn = hist_pinn = t_pinn_elapsed  = None
    phy_eki  = hist_eki = t_eki_elapsed = None

    # ── HYCO (basic / full) ───────────────────────────────────────────────────
    if mode in ("basic", "full"):
        hyco_path = f'saved_models/claw_hyco{run_tag}.pkl'
        if load and os.path.exists(hyco_path):
            print(f"Loading HYCO from {hyco_path}")
            syn_hyco, phy_hyco, hist_hyco, t_hyco_elapsed = load_claw(hyco_path)
        else:
            print(f"\n--- HYCO  seed={seed}  alpha={alpha}  beta={beta} ---")
            with time_block("HYCO") as _t:
                phy_hyco, syn_hyco, hist_hyco = train_hyco(
                    fv_coarse, u0_coarse,
                    dict(INIT_PARAMS), init_params(k_hyco_init, hidden=128),
                    lwr_flux, lwr_max_wave_speed,
                    tx_data, u_data,
                    alpha=alpha, beta=beta,
                    schedule_syn=schedule_syn_hyco, schedule_phy=schedule_phy_hyco,
                    n_ghost=N_GHOST, n_epochs=N_EPOCHS,
                    n_steps=N_STEPS, dt=dt_coarse, T=T,
                    t_range=t_range, x_range=x_range,
                    key=k_hyco_train, bc_left=BC_VAL, bc_right=BC_VAL,
                    activation=HYCO_ACTIVATION,
                )
            t_hyco_elapsed = _t.elapsed
            print(f"  HYCO: v_max={float(phy_hyco['v_max']):.4f}  "
                  f"rho_max={float(phy_hyco['rho_max']):.4f} (true 1.0)")
            save_claw(hyco_path, syn_hyco, phy_hyco, hist_hyco, elapsed=t_hyco_elapsed)

    # ── HYCO ablation (gamma_int=0) — trained as part of comparison ───────────
    if mode in ("comparison", "full"):
        abl_path = f'saved_models/claw_hyco_abl{run_tag}.pkl'
        if load and os.path.exists(abl_path):
            print(f"Loading HYCO ablation from {abl_path}")
            syn_abl, phy_abl, hist_abl, t_abl_elapsed = load_claw(abl_path)
        else:
            print(f"\n--- HYCO (ablation, gamma_int=0)  seed={seed}  "
                  f"alpha={alpha}  beta={beta} ---")
            with time_block("HYCO_abl") as _t:
                phy_abl, syn_abl, hist_abl = train_hyco(
                    fv_coarse, u0_coarse,
                    dict(INIT_PARAMS), init_params(k_abl_init, hidden=128),
                    lwr_flux, lwr_max_wave_speed,
                    tx_data, u_data,
                    alpha=alpha, beta=beta,
                    schedule_syn=schedule_syn_hyco, schedule_phy=schedule_phy_hyco,
                    n_ghost=N_GHOST, n_epochs=N_EPOCHS,
                    n_steps=N_STEPS, dt=dt_coarse, T=T,
                    t_range=t_range, x_range=x_range,
                    key=k_abl_train, bc_left=BC_VAL, bc_right=BC_VAL,
                    activation=HYCO_ACTIVATION, gamma_int=0.0,
                    stab_window=100000, stab_tol=1e-14,n_finetune=0
                )
            t_abl_elapsed = _t.elapsed
            print(f"  HYCO-abl: v_max={float(phy_abl['v_max']):.4f}  "
                  f"rho_max={float(phy_abl['rho_max']):.4f} (true 1.0)")
            save_claw(abl_path, syn_abl, phy_abl, hist_abl, elapsed=t_abl_elapsed)

    # ── PINN (comparison / full) ───────────────────────────────────────────────
    if mode in ("comparison", "full"):
        pinn_path = f'saved_models/claw_pinn{run_tag}{pinn_tag}.pkl'
        if load and os.path.exists(pinn_path):
            print(f"Loading PINN from {pinn_path}")
            syn_pinn, phy_pinn, hist_pinn, t_pinn_elapsed = load_claw(pinn_path)
        else:
            print(f"\n--- PINN  seed={seed} ---")
            with time_block("PINN") as _t:
                syn_pinn, phy_pinn, hist_pinn = train_pinn(
                    init_params(k_pinn_init, hidden=128), dict(INIT_PARAMS),
                    lwr_flux, tx_data, u_data,
                    gamma=GAMMA, schedule=schedule_pinn,
                    n_col=N_COL, n_epochs=N_EPOCHS,
                    t_range=t_range, x_range=x_range, key=k_pinn_train,
                    activation=PINN_ACTIVATION,
                    **(pinn_bc_kwargs(fv_fine.cell_centers, u0_fine, T, X_LEFT, X_RIGHT,
                                      BC_VAL, N_COL) if PINN_BC else {}),
                )
            t_pinn_elapsed = _t.elapsed
            print(f"  PINN: v_max={float(phy_pinn['v_max']):.4f}  "
                  f"rho_max={float(phy_pinn['rho_max']):.4f} (true 1.0)")
            save_claw(pinn_path, syn_pinn, phy_pinn, hist_pinn, elapsed=t_pinn_elapsed)

    # ── EKI (comparison / full) ────────────────────────────────────────────────
    if mode in ("comparison", "full"):
        eki_path = f'saved_models/claw_eki{run_tag}.pkl'
        if load and os.path.exists(eki_path):
            print(f"Loading EKI from {eki_path}")
            with open(eki_path, 'rb') as fh:
                d = pickle.load(fh)
            phy_eki, hist_eki = d['phy_params'], d['history']
            t_eki_elapsed = d.get('elapsed')
        else:
            print(f"\n--- EKI  seed={seed} ---")
            with time_block("EKI") as _t:
                phy_eki, hist_eki = train_eki_claw(
                    fv_coarse, u0_coarse,
                    lwr_flux, lwr_max_wave_speed,
                    tx_data, u_data,
                    init_params=INIT_PARAMS, T=T,
                    n_steps=N_STEPS, dt=dt_coarse,
                    bc_left=BC_VAL, bc_right=BC_VAL,
                    key=k_eki,
                )
            t_eki_elapsed = _t.elapsed
            print(f"  EKI: v_max={float(phy_eki['v_max']):.4f}  "
                  f"rho_max={float(phy_eki['rho_max']):.4f} (true 1.0)")
            with open(eki_path, 'wb') as fh:
                pickle.dump({'phy_params': phy_eki, 'history': hist_eki,
                            'elapsed': t_eki_elapsed}, fh)

    # ── Load missing checkpoints for comparison ────────────────────────────────
    # Comparison plots need both HYCO and PINN.  Load whichever is missing
    # (not trained this run) if a checkpoint exists.
    if syn_hyco is None:
        _p = f'saved_models/claw_hyco{run_tag}.pkl'
        if os.path.exists(_p):
            print(f"Loading HYCO from {_p} for comparison.")
            syn_hyco, phy_hyco, hist_hyco, t_hyco_elapsed = load_claw(_p)
    if syn_pinn is None:
        _p = f'saved_models/claw_pinn{run_tag}{pinn_tag}.pkl'
        if os.path.exists(_p):
            print(f"Loading PINN from {_p} for comparison.")
            syn_pinn, phy_pinn, hist_pinn, t_pinn_elapsed = load_claw(_p)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    tx_eval     = sample_ghost(k_eval, 2000, (0.0, T), (X_LEFT, X_RIGHT))
    u_true_eval = interp_solution(u_ref_array, fv_fine.cell_centers, T, tx_eval,
                                  t_hist=t_hist_ref)
    norm        = jnp.sqrt(jnp.mean(u_true_eval ** 2))

    def param_error(p):
        """Relative RMS error in (v_max, rho_max) against true values of 1.0."""
        err = (float(p['v_max']) - 1.0) ** 2 + (float(p['rho_max']) - 1.0) ** 2
        return float(jnp.sqrt(jnp.array(err / 2.0)))

    def sol_error(syn, activation):
        """Relative L2 solution error (eq:normL2) on the full-domain eval set."""
        u_eval = forward(syn, tx_eval, activation=activation)
        return float(jnp.sqrt(jnp.mean((u_eval - u_true_eval) ** 2)) / norm)

    def data_error(syn, activation):
        """Data discrepancy (eq:dataerr): MSE against observations at tx_data."""
        u_at_data = forward(syn, tx_data, activation=activation)
        return float(jnp.mean((u_at_data - u_data) ** 2))

    metrics = {}
    rows    = []  # (name, v_max, rho_max, e_p, e_s, e_d)

    if syn_hyco is not None:
        e_s = sol_error(syn_hyco, HYCO_ACTIVATION)
        e_p = param_error(phy_hyco)
        e_d = data_error(syn_hyco, HYCO_ACTIVATION)
        vmax, rmax = float(phy_hyco['v_max']), float(phy_hyco['rho_max'])
        metrics.update({'e_s_hyco': e_s, 'e_p_hyco': e_p, 'e_d_hyco': e_d,
                        'v_max_hyco': vmax, 'rho_max_hyco': rmax,
                        't_hyco_s': t_hyco_elapsed})
        rows.append(('HYCO', vmax, rmax, e_p, e_s, e_d))

    if syn_abl is not None:
        e_s = sol_error(syn_abl, HYCO_ACTIVATION)
        e_p = param_error(phy_abl)
        e_d = data_error(syn_abl, HYCO_ACTIVATION)
        vmax, rmax = float(phy_abl['v_max']), float(phy_abl['rho_max'])
        metrics.update({'e_s_abl': e_s, 'e_p_abl': e_p, 'e_d_abl': e_d,
                        'v_max_abl': vmax, 'rho_max_abl': rmax,
                        't_abl_s': t_abl_elapsed})
        rows.append(('HYCO (abl)', vmax, rmax, e_p, e_s, e_d))

    if syn_pinn is not None:
        e_s = sol_error(syn_pinn, PINN_ACTIVATION)
        e_p = param_error(phy_pinn)
        e_d = data_error(syn_pinn, PINN_ACTIVATION)
        vmax, rmax = float(phy_pinn['v_max']), float(phy_pinn['rho_max'])
        metrics.update({'e_s_pinn': e_s, 'e_p_pinn': e_p, 'e_d_pinn': e_d,
                        'v_max_pinn': vmax, 'rho_max_pinn': rmax,
                        't_pinn_s': t_pinn_elapsed})
        rows.append(('PINN', vmax, rmax, e_p, e_s, e_d))

    if phy_eki is not None:
        e_p = param_error(phy_eki)
        # Run the FV forward with the recovered params, then evaluate e_s and e_d
        # exactly like the other methods: e_s on the full-domain eval set against
        # the reference (eq:normL2), e_d at the observation points (eq:dataerr).
        full_p_eki = make_full_params(phy_eki, lwr_flux, lwr_max_wave_speed)
        u_hist_eki = solve_fv_fixed(fv_coarse, u0_coarse, full_p_eki,
                                    N_STEPS, dt_coarse, bc_left=BC_VAL, bc_right=BC_VAL)
        u_eki_eval    = interp_solution(u_hist_eki, fv_coarse.cell_centers, T, tx_eval)
        e_s_eki       = float(jnp.sqrt(jnp.mean((u_eki_eval - u_true_eval) ** 2)) / norm)
        u_eki_at_data = interp_solution(u_hist_eki, fv_coarse.cell_centers, T, tx_data)
        e_d_eki       = float(jnp.mean((u_eki_at_data - u_data) ** 2))
        vmax, rmax = float(phy_eki['v_max']), float(phy_eki['rho_max'])
        metrics.update({'e_s_eki': e_s_eki, 'e_p_eki': e_p, 'e_d_eki': e_d_eki,
                        'v_max_eki': vmax, 'rho_max_eki': rmax,
                        't_eki_s': t_eki_elapsed})
        rows.append(('EKI', vmax, rmax, e_p, e_s_eki, e_d_eki))

    if rows:
        print(f"\n{'Method':<12s}  {'v_max':>8s}  {'rho_max':>8s}  "
              f"{'e_p':>10s}  {'e_s':>10s}  {'e_d':>12s}")
        print("-" * 68)
        for name, vmax, rmax, ep, es, ed in rows:
            print(f"{name:<12s}  {vmax:8.4f}  {rmax:8.4f}  "
                  f"{ep:10.4f}  {es:10.4f}  {ed:12.4e}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs('figures/claw', exist_ok=True)

    # Shared spacetime evaluation grids
    N_T_PLOT, N_X_PLOT = 200, 300
    t_grid_plot = np.linspace(0.0, T, N_T_PLOT)
    x_grid_plot = np.linspace(X_LEFT, X_RIGHT, N_X_PLOT)
    TT, XX = np.meshgrid(t_grid_plot, x_grid_plot, indexing='ij')
    tx_grid = jnp.stack([jnp.array(TT.ravel()), jnp.array(XX.ravel())], axis=1)
    u_ref_grid = np.array(interp_solution(
        u_ref_array, fv_fine.cell_centers, T, tx_grid,
        t_hist=t_hist_ref)).reshape(N_T_PLOT, N_X_PLOT)

    def _hyco_phy_grid(phy):
        full_p = make_full_params(phy, lwr_flux, lwr_max_wave_speed)
        u_hist = solve_fv_fixed(fv_coarse, u0_coarse, full_p,
                                N_STEPS, dt_coarse, bc_left=BC_VAL, bc_right=BC_VAL)
        return np.array(interp_solution(
            u_hist, fv_coarse.cell_centers, T, tx_grid)).reshape(N_T_PLOT, N_X_PLOT)

    DATA_REGION = (0.0, T_DATA, X_LEFT, X_DATA_R)

    if mode in ("basic", "full") and syn_hyco is not None:
        u_hyco_syn_grid = np.array(forward(
            syn_hyco, tx_grid, activation=HYCO_ACTIVATION)).reshape(N_T_PLOT, N_X_PLOT)
        u_hyco_phy_grid = _hyco_phy_grid(phy_hyco)
        plot_claw_basic(
            x_grid_plot, t_grid_plot, u_ref_grid,
            u_hyco_phy_grid, u_hyco_syn_grid,
            hist_hyco, TRUE_PARAMS, INIT_PARAMS,
            tx_data=np.array(tx_data), data_region=DATA_REGION,
            save_path=f'figures/claw/basic{run_tag}.png',
        )

    if mode in ("comparison", "full"):
        # row1_fields: (title, grid) per space-time panel, in display order.
        # param_curves: (label, v_hist, rho_hist, color) for methods with an
        #   epoch history (HYCO, PINN, HYCO-abl).
        # param_hlines: (label, v_val, rho_val, color) for methods with only a
        #   converged value (EKI).
        COLORS = {'HYCO': 'tab:blue', 'PINN': 'tab:orange',
                  'EKI': 'tab:green', 'HYCO-abl': 'tab:red'}
        row1_fields  = [('Reference', u_ref_grid)]
        param_curves = []
        param_hlines = []

        if syn_hyco is not None:
            row1_fields.append((r'HYCO $u_{phy}$', _hyco_phy_grid(phy_hyco)))
            row1_fields.append((r'HYCO $u_{syn}$', np.array(forward(
                syn_hyco, tx_grid, activation=HYCO_ACTIVATION)).reshape(N_T_PLOT, N_X_PLOT)))
            param_curves.append(('HYCO',
                np.array([p['v_max']   for p in hist_hyco['params']]),
                np.array([p['rho_max'] for p in hist_hyco['params']]),
                COLORS['HYCO']))

        if syn_pinn is not None:
            row1_fields.append(('PINN', np.array(forward(
                syn_pinn, tx_grid, activation=PINN_ACTIVATION)).reshape(N_T_PLOT, N_X_PLOT)))
            param_curves.append(('PINN',
                np.array([p['v_max']   for p in hist_pinn['params']]),
                np.array([p['rho_max'] for p in hist_pinn['params']]),
                COLORS['PINN']))

        if phy_eki is not None:
            row1_fields.append(('EKI', _hyco_phy_grid(phy_eki)))
            param_hlines.append(('EKI', float(phy_eki['v_max']), float(phy_eki['rho_max']),
                                 COLORS['EKI']))

        if syn_abl is not None:
            row1_fields.append((r'HYCO-abl $u_{syn}$', np.array(forward(
                syn_abl, tx_grid, activation=HYCO_ACTIVATION)).reshape(N_T_PLOT, N_X_PLOT)))
            param_curves.append(('HYCO-abl',
                np.array([p['v_max']   for p in hist_abl['params']]),
                np.array([p['rho_max'] for p in hist_abl['params']]),
                COLORS['HYCO-abl']))

        if len(row1_fields) > 1:
            plot_claw_comparison(
                x_grid_plot, t_grid_plot, row1_fields, param_curves,
                TRUE_PARAMS, INIT_PARAMS, param_hlines=param_hlines,
                tx_data=np.array(tx_data), data_region=DATA_REGION,
                save_path=f'figures/claw/comparison{run_tag}.png',
            )

    return metrics


def main():
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('figures',      exist_ok=True)

    # ── Run mode ──────────────────────────────────────────────────────────────
    MODE  = "comparison"   # "basic" | "comparison" | "sweep" | "full"
    SEEDS = (42, 33, 17, 23, 2)     # len > 1 → aggregate over seeds (works with any MODE)
    LOAD  = True     # load checkpoint if available, skip retraining

    # ── Default (alpha, beta) ─────────────────────────────────────────────────
    ALPHA = 1.0
    BETA  = 0.0

    # ── Sweep grid (only used when MODE="sweep") ──────────────────────────────
    ALPHA_GRID = (0.1, 1.0, 10.0)
    BETA_GRID  = (0.0, 0.1, 1.0)

    # ── Other config ──────────────────────────────────────────────────────────
    config = dict(
        X_LEFT=0.0, X_RIGHT=3.0, T=2.0, T_DATA=0.5, X_DATA_R=1.5, CFL=0.5,
        N_FINE=2000, N_COARSE=50,
        N_DATA=100, N_GHOST=2000, N_COL=2000,
        N_EPOCHS=20000, GAMMA=5.0,
        # PINN initial/boundary penalties. False reproduces the paper's PINN
        # (data + residual only); True adds them (checkpoints get a "_bc" suffix).
        PINN_BC=False,
    )

    multi_seed = len(SEEDS) > 1
    ab_pairs   = sweep_alpha_beta_grid(ALPHA_GRID, BETA_GRID) if MODE == "sweep" \
                 else [(ALPHA, BETA)]

    all_results = []
    for seed in SEEDS:
        for (a, b) in ab_pairs:
            tag = make_run_tag(seed=seed, alpha=a, beta=b,
                               multi_seed=multi_seed, sweep_ab=(MODE == "sweep"))
            label = tag or "(default)"
            print(f"\n{'=' * 72}\n>>> RUN {label}  seed={seed}  "
                  f"alpha={a}  beta={b}  mode={MODE}\n{'=' * 72}")
            with time_block(f"run {label}") as t_total:
                metrics = run_experiment(seed, a, b, config,
                                         run_tag=tag, mode=MODE, load=LOAD)
            metrics.update({'seed': seed, 'alpha': a, 'beta': b,
                            't_total_s': t_total.elapsed})
            all_results.append(metrics)

    # ── Persist per-run metrics ────────────────────────────────────────────────
    results_path = 'saved_models/claw_results.pkl'
    with open(results_path, 'wb') as fout:
        pickle.dump(all_results, fout)
    print(f'Saved per-run metrics to {results_path}')

    # ── Aggregation ───────────────────────────────────────────────────────────
    if multi_seed and MODE != "sweep":
        print_aggregated(aggregate_seeds(all_results),
                         title=f"summary across {len(SEEDS)} seeds")
    elif MODE == "sweep":
        print(f"\n=== sweep summary "
              f"({'mean over ' + str(len(SEEDS)) + ' seeds' if multi_seed else 'single seed'}) ===")
        print(f"{'alpha':>8} {'beta':>8} {'e_s_hyco':>12} {'e_p_hyco':>10} "
              f"{'t_hyco_s':>10}")
        for (a, b) in ab_pairs:
            rows = [r for r in all_results if r['alpha'] == a and r['beta'] == b]
            mean = lambda k: float(np.mean([r[k] for r in rows if k in r]))
            print(f"{a:8g} {b:8g} {mean('e_s_hyco'):12.4e} "
                  f"{mean('e_p_hyco'):10.4f} {mean('t_hyco_s'):10.2f}")


if __name__ == "__main__":
    main()