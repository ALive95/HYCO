"""
HYCO and PINN for Helmholtz BVP parameter identification on a blob mesh.

Run modes (set MODE in main):
  "basic"       — train HYCO only (default alpha, beta, gamma_int=1.0).
  "ablation"    — train HYCO with gamma_int=0.0 (no ghost-point coupling).
                  Uses the same seed/data as "basic" for a fair comparison.
  "comparison"  — train PINN only (no HYCO).
  "sweep"       — sweep ALPHA_GRID x BETA_GRID for HYCO only.
  "full"        — basic + ablation + comparison in sequence.

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

from physics.helmholtz_fem import (
    make_blob_mesh, precompute_fem, solve_helmholtz, locate_points,
    sample_ghost_on_mesh,
)
from synthetic.nn import init_params, forward
from utils.training import (
    train_hyco_helmholtz, train_pinn_helmholtz,
    train_sapinn_helmholtz, train_gauss_newton_helmholtz,
)
from utils.model_io import save_claw, load_claw
from utils.plotting import (
    plot_helmholtz_basic, plot_helmholtz_ablation, plot_helmholtz_comparison,
)
from utils.experiment import (
    DEFAULT_SEEDS, time_block, aggregate_seeds, print_aggregated,
    make_run_tag, sweep_alpha_beta_grid,
)


# ── Activation choices ────────────────────────────────────────────────────────
# HYCO does not need second derivatives of the syn NN — ReLU is fine.
# PINN's residual takes Laplacian via grad-of-grad — MUST be smooth (tanh).
HYCO_ACTIVATION = jax.nn.relu
PINN_ACTIVATION = jax.nn.tanh


# ── Problem-level constants (fixed across runs) ───────────────────────────────
TRUE_PARAMS = {
    'alpha1': jnp.float64(4.0),  'c1': jnp.array([-0.5, -0.5]),
    'alpha2': jnp.float64(10.0), 'c2': jnp.array([0.5,  0.3]),
}
INIT_PARAMS = {
    'alpha1': jnp.float64(2.0),  'c1': jnp.array([-0.2, -0.1]),
    'alpha2': jnp.float64(5.5),  'c2': jnp.array([ 0.2,  0.1]),
}


def run_experiment(seed, alpha, beta, config, run_tag="", mode="basic", load=False,
                   noise_std=0.0):
    """
    Run one or more methods for the Helmholtz experiment under the given mode.

    Data setup always runs (deterministic from seed), so ablation and basic
    see exactly the same reference solution and sensor locations.

    Args:
        seed (int): PRNG seed.
        alpha (float): HYCO syn-step weight on sensor data.
        beta (float): HYCO phy-step weight on sensor data (0.0 = no phy data loss).
        config (dict): Run-invariant settings.
        run_tag (str): Filename suffix; "" means default filenames.
        mode (str): One of "basic", "ablation", "comparison", "sweep", "full".
        load (bool): If True, load checkpoint when available instead of retraining.
        noise_std (float): Std of additive Gaussian noise on sensor observations
            (0.0 = clean data).

    Returns:
        dict: Scalar metrics for each method that ran.
    """
    M_SENSORS    = config['M_SENSORS']
    N_GHOST      = config['N_GHOST']
    N_GHOST_POOL = config['N_GHOST_POOL']
    N_EPOCHS     = config['N_EPOCHS']
    RES_FINE     = config['RES_FINE']
    RES_COARSE   = config['RES_COARSE']

    # HYCO schedules (amp/ctr split for phy parameters).
    schedule_syn          = optax.cosine_decay_schedule(1e-4, N_EPOCHS, alpha=0.01)
    schedule_phy_amp      = optax.cosine_decay_schedule(1e-3, N_EPOCHS, alpha=0.01)
    schedule_phy_ctr      = optax.cosine_decay_schedule(1e-2, N_EPOCHS, alpha=0.01)

    # PINN schedules/hyperparameters — tunable independently of HYCO/SA-PINN.
    PINN_N_EPOCHS         = config['PINN_N_EPOCHS']
    PINN_GAMMA            = config['PINN_GAMMA']
    PINN_N_COL            = config['PINN_N_COL']
    PINN_GAMMA_INIT         = config['PINN_GAMMA_INIT']
    PINN_GAMMA_WARMUP_FRAC  = config['PINN_GAMMA_WARMUP_FRAC']
    PINN_BC                 = config.get('PINN_BC', False)
    pinn_tag                = '_bc' if PINN_BC else ''
    schedule_syn_pinn     = optax.cosine_decay_schedule(
        config['PINN_LR_SYN'], PINN_N_EPOCHS, alpha=0.01)
    schedule_phy_amp_pinn = optax.cosine_decay_schedule(
        config['PINN_LR_PHY_AMP'], PINN_N_EPOCHS, alpha=0.01)
    schedule_phy_ctr_pinn = optax.cosine_decay_schedule(
        config['PINN_LR_PHY_CTR'], PINN_N_EPOCHS, alpha=0.01)

    # SA-PINN schedules/hyperparameters — tunable independently of HYCO/PINN.
    SAPINN_N_EPOCHS         = config['SAPINN_N_EPOCHS']
    SAPINN_GAMMA            = config['SAPINN_GAMMA']
    SAPINN_N_COL            = config['SAPINN_N_COL']
    SAPINN_LAM_LR           = config['SAPINN_LAM_LR']
    SAPINN_GAMMA_INIT         = config['SAPINN_GAMMA_INIT']
    SAPINN_GAMMA_WARMUP_FRAC  = config['SAPINN_GAMMA_WARMUP_FRAC']
    schedule_syn_sapinn     = optax.cosine_decay_schedule(
        config['SAPINN_LR_SYN'], SAPINN_N_EPOCHS, alpha=0.01)
    schedule_phy_amp_sapinn = optax.cosine_decay_schedule(
        config['SAPINN_LR_PHY_AMP'], SAPINN_N_EPOCHS, alpha=0.01)
    schedule_phy_ctr_sapinn = optax.cosine_decay_schedule(
        config['SAPINN_LR_PHY_CTR'], SAPINN_N_EPOCHS, alpha=0.01)

    # ── Upfront key splits (order fixed regardless of mode) ───────────────────
    key = jr.PRNGKey(seed)
    key, k_sensors                    = jr.split(key)
    key, k_noise                      = jr.split(key)
    key, k_ghost                      = jr.split(key)
    key, k_hyco_init, k_hyco_train    = jr.split(key, 3)
    key, k_abl_init,  k_abl_train     = jr.split(key, 3)
    key, k_pinn_init, k_pinn_train    = jr.split(key, 3)
    _, k_sapinn_init, k_sapinn_train  = jr.split(key, 3)
    # GN is deterministic — no key needed.

    # ── 2. Data setup (always runs) ────────────────────────────────────────────
    print("Building meshes...")
    nodes_fine, elems_fine, bnd_fine = make_blob_mesh(resolution=RES_FINE)
    fem_fine   = precompute_fem(nodes_fine, elems_fine, bnd_fine)
    nodes_c, elems_c, bnd_c = make_blob_mesh(resolution=RES_COARSE)
    fem_coarse = precompute_fem(nodes_c, elems_c, bnd_c)
    print(f"  Fine: {fem_fine.nodes.shape[0]} nodes, "
          f"Coarse: {fem_coarse.nodes.shape[0]} nodes")

    u_ref = solve_helmholtz(fem_fine, TRUE_PARAMS)

    interior_mask = np.ones(fem_fine.nodes.shape[0], dtype=bool)
    interior_mask[np.array(bnd_fine)] = False
    interior_idx = np.where(interior_mask)[0]
    sensor_idx   = jr.choice(k_sensors, jnp.array(interior_idx),
                             shape=(M_SENSORS,), replace=False)
    sensor_xy    = fem_fine.nodes[sensor_idx]
    u_clean      = u_ref[sensor_idx]
    u_data       = u_clean + noise_std * jr.normal(k_noise, u_clean.shape)
    sensor_tri_c, sensor_bary_c = locate_points(fem_coarse, sensor_xy)

    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary = \
        sample_ghost_on_mesh(k_ghost, fem_coarse, N_GHOST_POOL)

    # Optional homogeneous Dirichlet penalty for PINN / SA-PINN (fine boundary nodes).
    bnd_xy = fem_fine.nodes[np.array(bnd_fine)] if PINN_BC else None

    # ── Results containers (None = method not run this call) ──────────────────
    syn_hyco   = phy_hyco   = hist_hyco   = t_hyco_elapsed   = None
    syn_abl    = phy_abl    = hist_abl    = t_abl_elapsed    = None
    syn_pinn   = phy_pinn   = hist_pinn   = t_pinn_elapsed   = None
    syn_sapinn = phy_sapinn = hist_sapinn = t_sapinn_elapsed  = None
    phy_gn     = hist_gn    = t_gn_elapsed = None

    # ── HYCO (basic / full) ───────────────────────────────────────────────────
    if mode in ("basic", "full"):
        hyco_path = f'saved_models/helmholtz_hyco{run_tag}.pkl'
        if load and os.path.exists(hyco_path):
            print(f"Loading HYCO from {hyco_path}")
            with open(hyco_path, 'rb') as _fh:
                _d = pickle.load(_fh)
            if 'syn_params' in _d:
                syn_hyco, phy_hyco, hist_hyco = _d['syn_params'], _d['phy_params'], _d['history']
            else:
                syn_hyco, phy_hyco, hist_hyco = _d['syn_hyco'], _d['phy_hyco'], _d['hist_hyco']
            t_hyco_elapsed = _d.get('elapsed')
        else:
            print(f"\n--- HYCO  seed={seed}  alpha={alpha}  beta={beta} ---")
            with time_block("HYCO") as _t:
                phy_hyco, syn_hyco, hist_hyco = train_hyco_helmholtz(
                    fem_coarse, init_params(k_hyco_init, hidden=256), dict(INIT_PARAMS),
                    schedule_syn, schedule_phy_amp, schedule_phy_ctr,
                    sensor_xy, u_data, sensor_tri_c, sensor_bary_c,
                    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
                    alpha=alpha, beta=beta,
                    n_ghost=N_GHOST, n_ghost_pool=N_GHOST_POOL, n_epochs=N_EPOCHS,
                    key=k_hyco_train, activation=HYCO_ACTIVATION,
                )
            t_hyco_elapsed = _t.elapsed
            save_claw(hyco_path, syn_hyco, phy_hyco, hist_hyco, elapsed=t_hyco_elapsed)

    # ── HYCO ablation (gamma_int=0) — same seed/data as basic ─────────────────
    if mode in ("ablation", "full"):
        abl_path = f'saved_models/helmholtz_hyco_abl{run_tag}.pkl'
        if load and os.path.exists(abl_path):
            print(f"Loading HYCO ablation from {abl_path}")
            with open(abl_path, 'rb') as _fh:
                _d = pickle.load(_fh)
            if 'syn_params' in _d:
                syn_abl, phy_abl, hist_abl = _d['syn_params'], _d['phy_params'], _d['history']
            else:
                syn_abl, phy_abl, hist_abl = _d['syn_hyco'], _d['phy_hyco'], _d['hist_hyco']
            t_abl_elapsed = _d.get('elapsed')
        else:
            print(f"\n--- HYCO (ablation, gamma_int=0)  seed={seed}  "
                  f"alpha={alpha}  beta={beta} ---")
            with time_block("HYCO_abl") as _t:
                phy_abl, syn_abl, hist_abl = train_hyco_helmholtz(
                    fem_coarse, init_params(k_abl_init, hidden=256), dict(INIT_PARAMS),
                    schedule_syn, schedule_phy_amp, schedule_phy_ctr,
                    sensor_xy, u_data, sensor_tri_c, sensor_bary_c,
                    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
                    alpha=alpha, beta=beta,
                    n_ghost=N_GHOST, n_ghost_pool=N_GHOST_POOL, n_epochs=N_EPOCHS,
                    key=k_abl_train, activation=HYCO_ACTIVATION, gamma_int=0.0,
                )
            t_abl_elapsed = _t.elapsed
            save_claw(abl_path, syn_abl, phy_abl, hist_abl, elapsed=t_abl_elapsed)

    # ── PINN (comparison / full) ───────────────────────────────────────────────
    if mode in ("comparison", "full"):
        pinn_path = f'saved_models/helmholtz_pinn{run_tag}{pinn_tag}.pkl'
        if load and os.path.exists(pinn_path):
            print(f"Loading PINN from {pinn_path}")
            with open(pinn_path, 'rb') as _fh:
                _d = pickle.load(_fh)
            if 'syn_params' in _d:
                syn_pinn, phy_pinn, hist_pinn = _d['syn_params'], _d['phy_params'], _d['history']
            else:
                syn_pinn, phy_pinn, hist_pinn = _d['syn_hyco'], _d['phy_hyco'], _d['hist_hyco']
            t_pinn_elapsed = _d.get('elapsed')
        else:
            print(f"\n--- PINN  seed={seed} ---")
            with time_block("PINN") as _t:
                syn_pinn, phy_pinn, hist_pinn = train_pinn_helmholtz(
                    init_params(k_pinn_init, hidden=256), dict(INIT_PARAMS),
                    schedule_syn_pinn, schedule_phy_amp_pinn, schedule_phy_ctr_pinn,
                    sensor_xy, u_data,
                    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
                    gamma=PINN_GAMMA, n_col=PINN_N_COL, n_ghost_pool=N_GHOST_POOL,
                    n_epochs=PINN_N_EPOCHS,
                    key=k_pinn_train, activation=PINN_ACTIVATION,
                    gamma_init=PINN_GAMMA_INIT, gamma_warmup_frac=PINN_GAMMA_WARMUP_FRAC,
                    bnd_xy=bnd_xy,
                )
            t_pinn_elapsed = _t.elapsed
            save_claw(pinn_path, syn_pinn, phy_pinn, hist_pinn, elapsed=t_pinn_elapsed)

    # ── SA-PINN (comparison / full) ───────────────────────────────────────────
    if mode in ("comparison", "full"):
        sapinn_path = f'saved_models/helmholtz_sapinn{run_tag}{pinn_tag}.pkl'
        if load and os.path.exists(sapinn_path):
            print(f"Loading SA-PINN from {sapinn_path}")
            with open(sapinn_path, 'rb') as _fh:
                _d = pickle.load(_fh)
            if 'syn_params' in _d:
                syn_sapinn, phy_sapinn, hist_sapinn = _d['syn_params'], _d['phy_params'], _d['history']
            else:
                syn_sapinn, phy_sapinn, hist_sapinn = _d['syn_hyco'], _d['phy_hyco'], _d['hist_hyco']
            t_sapinn_elapsed = _d.get('elapsed')
        else:
            print(f"\n--- SA-PINN  seed={seed} ---")
            with time_block("SA-PINN") as _t:
                syn_sapinn, phy_sapinn, hist_sapinn = train_sapinn_helmholtz(
                    init_params(k_sapinn_init, hidden=256), dict(INIT_PARAMS),
                    schedule_syn_sapinn, schedule_phy_amp_sapinn, schedule_phy_ctr_sapinn,
                    sensor_xy, u_data,
                    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary,
                    gamma=SAPINN_GAMMA, n_col=SAPINN_N_COL, n_ghost_pool=N_GHOST_POOL,
                    n_epochs=SAPINN_N_EPOCHS, key=k_sapinn_train,
                    activation=PINN_ACTIVATION, lam_lr=SAPINN_LAM_LR,
                    gamma_init=SAPINN_GAMMA_INIT, gamma_warmup_frac=SAPINN_GAMMA_WARMUP_FRAC,
                    bnd_xy=bnd_xy,
                )
            t_sapinn_elapsed = _t.elapsed
            save_claw(sapinn_path, syn_sapinn, phy_sapinn, hist_sapinn, elapsed=t_sapinn_elapsed)

    # ── Gauss-Newton (comparison / full) ──────────────────────────────────────
    if mode in ("comparison", "full"):
        gn_path = f'saved_models/helmholtz_gn{run_tag}.pkl'
        if load and os.path.exists(gn_path):
            print(f"Loading GN from {gn_path}")
            with open(gn_path, 'rb') as fh:
                d = pickle.load(fh)
            phy_gn, hist_gn = d['phy_params'], d['history']
            t_gn_elapsed = d.get('elapsed')
        else:
            print(f"\n--- Gauss-Newton  seed={seed} ---")
            with time_block("GN") as _t:
                phy_gn, hist_gn = train_gauss_newton_helmholtz(
                    fem_coarse,
                    sensor_tri_c, sensor_bary_c,
                    u_data, dict(INIT_PARAMS),
                )
            t_gn_elapsed = _t.elapsed
            with open(gn_path, 'wb') as fh:
                pickle.dump({'phy_params': phy_gn, 'history': hist_gn,
                            'elapsed': t_gn_elapsed}, fh)

    # ── Load HYCO for comparison if not trained this run ──────────────────────
    # In "comparison" mode HYCO is not trained, but we want its results in the
    # evaluation table and plots.  Load silently if a checkpoint exists.
    if syn_hyco is None:
        _hyco_path = f'saved_models/helmholtz_hyco{run_tag}.pkl'
        if os.path.exists(_hyco_path):
            print(f"Loading HYCO from {_hyco_path} for comparison.")
            with open(_hyco_path, 'rb') as _fh:
                _d = pickle.load(_fh)
            # support both new format (syn_params/phy_params/history)
            # and old monolithic format (syn_hyco/phy_hyco/hist_hyco)
            if 'syn_params' in _d:
                syn_hyco, phy_hyco, hist_hyco = \
                    _d['syn_params'], _d['phy_params'], _d['history']
            else:
                syn_hyco, phy_hyco, hist_hyco = \
                    _d['syn_hyco'], _d['phy_hyco'], _d['hist_hyco']
            t_hyco_elapsed = _d.get('elapsed')

    # ── Evaluation ─────────────────────────────────────────────────────────────
    norm   = jnp.sqrt(jnp.mean(u_ref ** 2))
    e_s    = lambda u_: float(jnp.sqrt(jnp.mean((u_ - u_ref) ** 2)) / norm)
    e_d    = lambda u_: float(jnp.mean((u_[sensor_idx] - u_data) ** 2))

    true_vec = jnp.concatenate([jnp.array([TRUE_PARAMS['alpha1']]), TRUE_PARAMS['c1'],
                                jnp.array([TRUE_PARAMS['alpha2']]), TRUE_PARAMS['c2']])
    def e_p(pp):
        rec_vec = jnp.concatenate([jnp.array([pp['alpha1']]), pp['c1'],
                                   jnp.array([pp['alpha2']]), pp['c2']])
        return float(jnp.linalg.norm(rec_vec - true_vec) / jnp.linalg.norm(true_vec))

    metrics = {}
    rows    = []

    if syn_hyco is not None:
        u_hyco_phy = solve_helmholtz(fem_fine, phy_hyco)
        u_hyco_syn = forward(syn_hyco, fem_fine.nodes, activation=HYCO_ACTIVATION)
        es_phy, es_syn, ep = e_s(u_hyco_phy), e_s(u_hyco_syn), e_p(phy_hyco)
        ed_phy, ed_syn = e_d(u_hyco_phy), e_d(u_hyco_syn)
        metrics.update({'e_s_hyco_phy': es_phy, 'e_s_hyco_syn': es_syn,
                        'e_d_hyco_phy': ed_phy, 'e_d_hyco_syn': ed_syn,
                        'e_p_hyco': ep, 't_hyco_s': t_hyco_elapsed})
        rows.append(('HYCO Phy', es_phy, ed_phy, ep))
        rows.append(('HYCO Syn', es_syn, ed_syn, None))

    if syn_abl is not None:
        u_abl_phy = solve_helmholtz(fem_fine, phy_abl)
        u_abl_syn = forward(syn_abl, fem_fine.nodes, activation=HYCO_ACTIVATION)
        es_phy, es_syn, ep = e_s(u_abl_phy), e_s(u_abl_syn), e_p(phy_abl)
        ed_phy, ed_syn = e_d(u_abl_phy), e_d(u_abl_syn)
        metrics.update({'e_s_abl_phy': es_phy, 'e_s_abl_syn': es_syn,
                        'e_d_abl_phy': ed_phy, 'e_d_abl_syn': ed_syn,
                        'e_p_abl': ep, 't_abl_s': t_abl_elapsed})
        rows.append(('HYCO-abl Phy', es_phy, ed_phy, ep))
        rows.append(('HYCO-abl Syn', es_syn, ed_syn, None))

    if syn_pinn is not None:
        u_pinn_syn = forward(syn_pinn, fem_fine.nodes, activation=PINN_ACTIVATION)
        es_syn, ep = e_s(u_pinn_syn), e_p(phy_pinn)
        ed_syn = e_d(u_pinn_syn)
        metrics.update({'e_s_pinn': es_syn, 'e_d_pinn': ed_syn,
                        'e_p_pinn': ep, 't_pinn_s': t_pinn_elapsed})
        rows.append(('PINN', es_syn, ed_syn, ep))

    if syn_sapinn is not None:
        u_sapinn_syn = forward(syn_sapinn, fem_fine.nodes, activation=PINN_ACTIVATION)
        es_syn, ep = e_s(u_sapinn_syn), e_p(phy_sapinn)
        ed_syn = e_d(u_sapinn_syn)
        metrics.update({'e_s_sapinn': es_syn, 'e_d_sapinn': ed_syn,
                        'e_p_sapinn': ep, 't_sapinn_s': t_sapinn_elapsed})
        rows.append(('SA-PINN', es_syn, ed_syn, ep))

    if phy_gn is not None:
        u_gn = solve_helmholtz(fem_fine, phy_gn)
        es, ep = e_s(u_gn), e_p(phy_gn)
        ed = e_d(u_gn)
        metrics.update({'e_s_gn': es, 'e_d_gn': ed, 'e_p_gn': ep, 't_gn_s': t_gn_elapsed})
        rows.append(('Gauss-Newton', es, ed, ep))

    if rows:
        print(f"\n{'Method':<18s}  {'e_s':>10s}  {'e_d':>10s}  {'e_p':>10s}")
        print("-" * 54)
        for name, es, ed, ep in rows:
            ep_str = f"{ep:10.4f}" if ep is not None else f"{'--':>10s}"
            print(f"{name:<18s}  {es:10.4f}  {ed:10.4e}  {ep_str}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs('figures/helmholtz', exist_ok=True)

    if mode in ("basic", "full") and syn_hyco is not None:
        u_hyco_phy = solve_helmholtz(fem_fine, phy_hyco)
        u_hyco_syn = forward(syn_hyco, fem_fine.nodes, activation=HYCO_ACTIVATION)
        plot_helmholtz_basic(
            fem_fine,
            np.array(u_ref), np.array(u_hyco_phy), np.array(u_hyco_syn),
            np.array(sensor_xy), phy_hyco, TRUE_PARAMS, hist_hyco,
            save_path=f'figures/helmholtz/basic{run_tag}.png',
        )

    if mode in ("ablation", "full") and syn_abl is not None and syn_hyco is not None:
        plot_helmholtz_ablation(
            hist_hyco, hist_abl,
            save_path=f'figures/helmholtz/ablation{run_tag}.png',
        )

    if mode in ("comparison", "full"):
        # Build methods list: (label, u_pred, phy_params, e_s, e_p, t_s)
        # Only include methods that actually ran / were loaded.
        comp_methods = []
        def _add(label, u_pred_fn, phy, es_key, ep_key, t_key):
            if phy is None:
                return
            u_p = np.array(u_pred_fn())
            comp_methods.append((
                label, u_p, phy,
                metrics.get(es_key, float('nan')),
                metrics.get(ep_key, float('nan')),
                metrics.get(t_key),
            ))
        if syn_hyco is not None:
            _add('HYCO',
                 lambda: solve_helmholtz(fem_fine, phy_hyco), phy_hyco,
                 'e_s_hyco_phy', 'e_p_hyco', 't_hyco_s')
        if syn_abl is not None:
            _add('HYCO-abl',
                 lambda: solve_helmholtz(fem_fine, phy_abl), phy_abl,
                 'e_s_abl_phy', 'e_p_abl', 't_abl_s')
        if syn_pinn is not None:
            _add('PINN',
                 lambda: forward(syn_pinn, fem_fine.nodes, activation=PINN_ACTIVATION), phy_pinn,
                 'e_s_pinn', 'e_p_pinn', 't_pinn_s')
        if syn_sapinn is not None:
            _add('SA-PINN',
                 lambda: forward(syn_sapinn, fem_fine.nodes, activation=PINN_ACTIVATION), phy_sapinn,
                 'e_s_sapinn', 'e_p_sapinn', 't_sapinn_s')
        if phy_gn is not None:
            _add('GN',
                 lambda: solve_helmholtz(fem_fine, phy_gn), phy_gn,
                 'e_s_gn', 'e_p_gn', 't_gn_s')
        if comp_methods:
            plot_helmholtz_comparison(
                fem_fine, np.array(u_ref), TRUE_PARAMS, np.array(sensor_xy),
                comp_methods,
                save_path=f'figures/helmholtz/comparison{run_tag}.png',
            )

    return metrics


def main():
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('figures',      exist_ok=True)

    # ── Run mode ──────────────────────────────────────────────────────────────
    MODE  = "basic"   # "basic" | "ablation" | "comparison" | "sweep" | "full"
    SEEDS = (42, 33, 17, 23, 2)    # len > 1 → aggregate over seeds (works with any MODE)
    LOAD  = True     # load checkpoint if available, skip retraining

    # ── Default (alpha, beta) ─────────────────────────────────────────────────
    ALPHA = 10.0
    BETA  = 1.0

    # ── Sweep grid (only used when MODE="sweep") ──────────────────────────────
    ALPHA_GRID = (1.0, 10.0, 100.0)
    BETA_GRID  = (0.0, 1.0, 100.0)

    # ── Noise sweep (std of additive Gaussian noise on sensor data) ──────────
    NOISE_GRID = (0.0, )

    # ── Other config ──────────────────────────────────────────────────────────
    config = dict(
        M_SENSORS=25, N_GHOST=1000, N_GHOST_POOL=30000,
        N_EPOCHS=30000, RES_FINE=0.05, RES_COARSE=0.2,
        # PINN hyperparameters — independent of HYCO and SA-PINN.
        PINN_LR_SYN=1e-4, PINN_LR_PHY_AMP=1e-3, PINN_LR_PHY_CTR=1e-2,
        PINN_GAMMA=100.0, PINN_N_COL=2000, PINN_N_EPOCHS=30000,
        PINN_GAMMA_INIT=1.0, PINN_GAMMA_WARMUP_FRAC=0.2,
        # SA-PINN hyperparameters — independent of HYCO and PINN.
        SAPINN_LR_SYN=1e-4, SAPINN_LR_PHY_AMP=1e-3, SAPINN_LR_PHY_CTR=1e-2,
        SAPINN_GAMMA=100.0, SAPINN_N_COL=2000, SAPINN_N_EPOCHS=30000,
        SAPINN_LAM_LR=1e-3,
        SAPINN_GAMMA_INIT=1.0, SAPINN_GAMMA_WARMUP_FRAC=0.2,
        # Dirichlet boundary penalty for PINN and SA-PINN. False reproduces the
        # paper's baselines; True adds it (checkpoints get a "_bc" suffix).
        PINN_BC=False,
    )

    multi_seed  = len(SEEDS) > 1
    multi_noise = 0
    #len(NOISE_GRID) > 1
    ab_pairs    = sweep_alpha_beta_grid(ALPHA_GRID, BETA_GRID) if MODE == "sweep" \
                  else [(ALPHA, BETA)]

    all_results = []
    for noise in NOISE_GRID:
        for seed in SEEDS:
            for (a, b) in ab_pairs:
                tag = make_run_tag(seed=seed, alpha=a, beta=b,
                                   multi_seed=multi_seed, sweep_ab=(MODE == "sweep"),
                                   noise=noise, sweep_noise=multi_noise)
                label = tag or "(default)"
                print(f"\n{'=' * 72}\n>>> RUN {label}  seed={seed}  "
                      f"alpha={a}  beta={b}  noise={noise}  mode={MODE}\n{'=' * 72}")
                with time_block(f"run {label}") as t_total:
                    metrics = run_experiment(seed, a, b, config,
                                             run_tag=tag, mode=MODE, load=LOAD,
                                             noise_std=noise)
                metrics.update({'seed': seed, 'alpha': a, 'beta': b, 'noise': noise,
                                't_total_s': t_total.elapsed})
                all_results.append(metrics)

    # ── Persist per-run metrics ────────────────────────────────────────────────
    results_path = 'saved_models/helmholtz_results.pkl'
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
        print(f"{'alpha':>8} {'beta':>8} {'e_s_hyco_phy':>14} {'e_p_hyco':>10} "
              f"{'t_hyco_s':>10}")
        for (a, b) in ab_pairs:
            rows = [r for r in all_results if r['alpha'] == a and r['beta'] == b]
            mean = lambda k: float(np.mean([r[k] for r in rows if k in r]))
            print(f"{a:8g} {b:8g} {mean('e_s_hyco_phy'):14.4e} "
                  f"{mean('e_p_hyco'):10.4f} {mean('t_hyco_s'):10.2f}")


if __name__ == "__main__":
    main()