"""
HYCO / NN / PINN for the Gray-Scott reaction-diffusion system (forward + (D_u,D_v)
identification), folded into the shared MODE/LOAD/SEEDS harness.

Run modes (set MODE in main):
  "basic"       — train HYCO only (gamma_int=1.0, beta=0.0).
  "ablation"    — train HYCO with gamma_int=0.0 (no ghost-point coupling).
  "comparison"  — train NN and PINN baselines.
  "full"        — basic + ablation + comparison in sequence.

SEEDS controls multi-seed aggregation (orthogonal to MODE).
LOAD=True skips training if a checkpoint already exists for that method/tag.

Configurable beyond the paper defaults:
  * PHY_N / PHY_STEPS / PHY_SAVE  — physical-model grid + step count. Defaults
    match the reference; coarsening PHY_N is allowed but degrades results.
  * RECOVER                       — which of {D_u,D_v,F,k} are identified.
  * DATA_T_RANGE / DATA_REGION    — localise the data in time and/or to a spatial
    quadrant (or an explicit bounding box).
  * N_PRETRAIN / N_LBFGS          — HYCO/NN data-only warmup epochs and post-training
    L-BFGS-B refinement iterations (see utils/training.py).
  * GAMMA_IC                      — PINN initial-datum loss weight (separate from
    GAMMA, the PDE-residual weight).

NOTE: epoch counts are fixed (paper). A per-method stopping criterion is a
separate revision item and is intentionally not implemented here.
"""
import jax
jax.config.update("jax_enable_x64", True)

import os
import pickle

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from physics.gray_scott import (
    TRUE_PARAMS_GS, DOMAIN_L, generate_reference, forward_phy, sample_field,
    sample_points, quadrant_box, merge_params,
)
from synthetic.nn import init_params_mlp, forward_mlp, init_params_fourier_mlp
from utils.training import (train_hyco_gs, train_nn_gs, train_pinn_gs,
                            train_opt_pinn_gs, make_opt_pinn_net)
from utils.model_io import save_gs, load_gs
from utils.plotting import plot_gs_comparison
from utils.experiment import (
    DEFAULT_SEEDS, time_block, aggregate_seeds, print_aggregated, make_run_tag,
)


# Gray-Scott uses the plain (non-residual) MLP: the shared residual net's
# nonnegative-increment ReLU blocks plateau on the oscillatory GS fields.
HYCO_ACTIVATION = jax.nn.relu
PINN_ACTIVATION = jax.nn.tanh    # PINN residual needs twice-differentiable activation

ALL_PARAM_NAMES = ('D_u', 'D_v', 'F', 'k')

# Initial guesses for identification. D_u, D_v match the paper; F, k default to
# their true values (override here if they are placed in RECOVER).
INIT_GS = {'D_u': 0.10, 'D_v': 0.05, 'F': 0.018, 'k': 0.051}


def load_gs_compat(path, method):
    """
    Load a GS checkpoint, rejecting pre-MLP (residual-net) parameter formats.

    Args:
        path (str): Checkpoint path.
        method (str): Method name for the warning message.

    Returns:
        tuple | None: (syn, phy, hist) or None if the format is incompatible.
    """
    r = load_gs(path)
    syn = r[0]
    if syn is not None and 'W0' not in syn:
        print(f"WARNING: {method} checkpoint {path} has the old residual-net "
              f"parameter format; ignoring it. Delete or move the file to "
              f"retrain cleanly.")
        return None
    return r


def split_params(recover):
    """
    Split the parameter set into trainable (recovered) and fixed (known) dicts.

    Args:
        recover (sequence[str]): Names in {D_u,D_v,F,k} to identify.

    Returns:
        tuple: (phy_init, fixed_phy) dicts.
    """
    recover = tuple(recover)
    phy_init  = {p: jnp.float64(INIT_GS[p]) for p in recover}
    fixed_phy = {p: jnp.float64(TRUE_PARAMS_GS[p])
                 for p in ALL_PARAM_NAMES if p not in recover}
    return phy_init, fixed_phy


def grid_query(n, times):
    """
    Build (x,y,t) query coordinates for every grid point at every given time.

    Args:
        n (int): Grid resolution.
        times (sequence[float]): Time values.

    Returns:
        jnp.ndarray: (len(times), n, n, 3) coordinates [x,y,t].
    """
    xs = (jnp.arange(n) + 0.5) * (DOMAIN_L / n)
    X, Y = jnp.meshgrid(xs, xs, indexing='ij')
    out = []
    for t in times:
        T = jnp.full_like(X, float(t))
        out.append(jnp.stack([X, Y, T], axis=-1))           # (n,n,3)
    return jnp.stack(out, axis=0)                            # (K,n,n,3)


def net_field(syn, n, times, activation, xyt_mean, xyt_std,
              net_forward=forward_mlp):
    """Evaluate a (u,v) network on the full grid at the given times -> (K,n,n,2).

    Args:
        syn: NN parameters.
        n (int): Grid resolution.
        times (sequence[float]): Visualization time points.
        activation (callable): NN nonlinearity.
        xyt_mean (jnp.ndarray): (3,) per-coordinate means used during training.
        xyt_std (jnp.ndarray): (3,) per-coordinate stds used during training.
    """
    K = len(times)
    xyt = grid_query(n, times).reshape(-1, 3)
    # Apply the same z-score normalization used during training.
    xyt_n = (xyt - xyt_mean) / xyt_std
    pred = net_forward(syn, xyt_n, activation=activation)      # (K*n*n, 2)
    return pred.reshape(K, n, n, 2)


def phy_field(full_params, ref, phy_n, phy_steps, phy_save, times):
    """Solve the physical model and sample (u,v) on the reference grid at `times`."""
    T = ref['T']
    times_p, U_p, V_p = forward_phy(full_params, phy_n, T, phy_steps, phy_save)
    dx_p = DOMAIN_L / phy_n
    K, n = len(times), ref['n']
    xyt = grid_query(n, times).reshape(-1, 3)
    vals = sample_field(U_p, V_p, times_p, dx_p, phy_n, xyt)  # (K*n*n, 2)
    return vals.reshape(K, n, n, 2)


def run_experiment(seed, config, run_tag="", mode="basic", load=False):
    """
    Run the requested Gray-Scott methods under one mode / seed.

    Args:
        seed (int): PRNG seed.
        config (dict): Run-invariant settings.
        run_tag (str): Filename suffix ("" -> default filenames).
        mode (str): "basic" | "ablation" | "comparison" | "full".
        load (bool): Load checkpoints instead of retraining when available.

    Returns:
        dict: Scalar metrics for each method that ran.
    """
    REF_N, REF_T   = config['REF_N'], config['REF_T']
    REF_STEPS      = config['REF_STEPS']
    REF_SAVE       = config['REF_SAVE']
    PHY_N          = config['PHY_N']
    PHY_STEPS      = config['PHY_STEPS']
    PHY_SAVE       = config['PHY_SAVE']
    RECOVER        = tuple(config['RECOVER'])
    M, H, N_COL    = config['M'], config['H'], config['N_COL']
    DATA_T_RANGE   = config['DATA_T_RANGE']
    DATA_REGION    = config['DATA_REGION']
    HIDDEN         = config['HIDDEN']
    LR             = config['LR']
    LR_PHY         = config['LR_PHY']
    ALPHA, BETA    = config['ALPHA'], config['BETA']
    ABLATION_BETA  = config['ABLATION_BETA']
    GAMMA          = config['GAMMA']
    GAMMA_IC       = config['GAMMA_IC']
    LR_PHY_PINN    = config['LR_PHY_PINN']
    E_HYCO, E_NN, E_PINN = config['E_HYCO'], config['E_NN'], config['E_PINN']
    N_PRETRAIN            = config.get('N_PRETRAIN', 0)
    N_LBFGS               = config.get('N_LBFGS', 0)
    # opt-PINN (best-practice PINN competitor)
    N_FREQ      = config.get('N_FREQ', 64)
    SIGMA_X     = config.get('SIGMA_X', 3.0)
    SIGMA_T     = config.get('SIGMA_T', 1.0)
    TAU_IC      = config.get('TAU_IC', 50.0)
    CAUSAL_EPS  = config.get('CAUSAL_EPS', 1.0)
    CAUSAL_BINS = config.get('CAUSAL_BINS', 20)
    D_WARMUP    = config.get('D_WARMUP', 1000)
    GAMMA_OPT   = config.get('GAMMA_OPT', 1.0)

    phy_init, fixed_phy = split_params(RECOVER)

    # ── Reference (solved once, cached; independent of seed) ──────────────────
    ref_path = f"saved_models/gray_scott_ref_n{REF_N}_s{REF_STEPS}.npz"
    ref = generate_reference(n=REF_N, T=REF_T, n_steps=REF_STEPS, n_save=REF_SAVE,
                             cache_path=ref_path)
    dx_ref = ref['dx']

    # ── Key splits (fixed order, mode-independent) ────────────────────────────
    key = jr.PRNGKey(seed)
    key, k_data                    = jr.split(key)
    key, k_hyco_init, k_hyco_train = jr.split(key, 3)
    key, k_abl_init,  k_abl_train  = jr.split(key, 3)
    key, k_nn_init,   k_nn_train   = jr.split(key, 3)
    key, k_pinn_init, k_pinn_train = jr.split(key, 3)
    key, k_col                     = jr.split(key)
    _,   k_opt_init,  k_opt_train  = jr.split(key, 3)

    # ── Data (optionally localised in time / space) ───────────────────────────
    box      = quadrant_box(DATA_REGION)
    xyt_data = sample_points(k_data, M, DATA_T_RANGE, box)
    u_data   = sample_field(ref['U'], ref['V'], ref['times'], dx_ref, REF_N, xyt_data)
    # Normalization stats: z-score applied to network inputs everywhere (data,
    # ghost/collocation points, IC points, eval grid) — see utils/training.py.
    xyt_mean = xyt_data.mean(0)
    xyt_std  = xyt_data.std(0) + 1e-8
    print(f"Data: M={M} points, t in {DATA_T_RANGE}, region={DATA_REGION} box={box}")
    print(f"Recovering: {RECOVER}")

    schedule_syn = optax.constant_schedule(LR)        # paper: Adam, lr 1e-3
    schedule_phy = optax.constant_schedule(LR_PHY)    # log-space params tolerate a larger lr

    syn_hyco = phy_hyco = hist_hyco = t_hyco = None
    syn_abl  = phy_abl  = hist_abl  = t_abl  = None
    syn_nn   =            hist_nn   = t_nn   = None
    syn_pinn = phy_pinn = hist_pinn = t_pinn = None
    syn_opt  = phy_opt  = hist_opt  = t_opt  = None

    # ── HYCO ───────────────────────────────────────────────────────────────────
    if mode in ("basic", "full"):
        path = f"saved_models/gray_scott_hyco{run_tag}.pkl"
        r = (load_gs_compat(path, 'HYCO')
             if load and os.path.exists(path) else None)
        if r is not None:
            print(f"Loading HYCO from {path}")
            syn_hyco, phy_hyco, hist_hyco = r; t_hyco = 0.0
        else:
            print(f"\n--- HYCO  seed={seed} ---")
            with time_block("HYCO") as _t:
                phy_hyco, syn_hyco, hist_hyco = train_hyco_gs(
                    phy_init, fixed_phy,
                    init_params_mlp(k_hyco_init, d_in=3, hidden=HIDDEN, d_out=2),
                    xyt_data, u_data, xyt_mean, xyt_std,
                    alpha=ALPHA, beta=BETA,
                    schedule_syn=schedule_syn, schedule_phy=schedule_phy,
                    n_ghost=H, n_epochs=E_HYCO,
                    phy_n=PHY_N, phy_steps=PHY_STEPS, n_save_phy=PHY_SAVE, T=REF_T,
                    key=k_hyco_train, gamma_int=1.0, activation=HYCO_ACTIVATION,
                    n_pretrain=N_PRETRAIN, n_lbfgs=N_LBFGS, net_forward=forward_mlp)
            t_hyco = _t.elapsed
            save_gs(path, syn_hyco, phy_hyco, hist_hyco)

    # ── HYCO ablation (gamma_int=0) ────────────────────────────────────────────
    if mode in ("ablation", "full"):
        path = f"saved_models/gray_scott_hyco_abl{run_tag}.pkl"
        r = (load_gs_compat(path, 'HYCO ablation')
             if load and os.path.exists(path) else None)
        if r is not None:
            print(f"Loading HYCO ablation from {path}")
            syn_abl, phy_abl, hist_abl = r; t_abl = 0.0
        else:
            print(f"\n--- HYCO (ablation, gamma_int=0)  seed={seed} ---")
            with time_block("HYCO_abl") as _t:
                phy_abl, syn_abl, hist_abl = train_hyco_gs(
                    phy_init, fixed_phy,
                    init_params_mlp(k_abl_init, d_in=3, hidden=HIDDEN, d_out=2),
                    xyt_data, u_data, xyt_mean, xyt_std,
                    alpha=ALPHA, beta=ABLATION_BETA,
                    schedule_syn=schedule_syn, schedule_phy=schedule_phy,
                    n_ghost=H, n_epochs=E_HYCO,
                    phy_n=PHY_N, phy_steps=PHY_STEPS, n_save_phy=PHY_SAVE, T=REF_T,
                    key=k_abl_train, gamma_int=0.0, activation=HYCO_ACTIVATION,
                    n_pretrain=N_PRETRAIN, n_lbfgs=N_LBFGS, net_forward=forward_mlp)
            t_abl = _t.elapsed
            save_gs(path, syn_abl, phy_abl, hist_abl)

    # ── NN baseline ────────────────────────────────────────────────────────────
    if mode in ("comparison", "full"):
        path = f"saved_models/gray_scott_nn{run_tag}.pkl"
        r = (load_gs_compat(path, 'NN')
             if load and os.path.exists(path) else None)
        if r is not None:
            print(f"Loading NN from {path}")
            syn_nn, _, hist_nn = r; t_nn = 0.0
        else:
            print(f"\n--- NN  seed={seed} ---")
            with time_block("NN") as _t:
                syn_nn, hist_nn = train_nn_gs(
                    init_params_mlp(k_nn_init, d_in=3, hidden=HIDDEN, d_out=2),
                    xyt_data, u_data, xyt_mean, xyt_std, schedule=schedule_syn, n_epochs=E_NN,
                    key=k_nn_train, activation=HYCO_ACTIVATION,
                    n_pretrain=N_PRETRAIN, n_lbfgs=N_LBFGS, net_forward=forward_mlp)
            t_nn = _t.elapsed
            save_gs(path, syn_nn, None, hist_nn)

    # ── PINN baseline ──────────────────────────────────────────────────────────
    if mode in ("comparison", "full"):
        path = f"saved_models/gray_scott_pinn{run_tag}.pkl"
        r = (load_gs_compat(path, 'PINN')
             if load and os.path.exists(path) else None)
        if r is not None:
            print(f"Loading PINN from {path}")
            syn_pinn, phy_pinn, hist_pinn = r; t_pinn = 0.0
        else:
            print(f"\n--- PINN  seed={seed} ---")
            xyt_col = sample_points(k_col, N_COL, (0.0, REF_T), (0.0, DOMAIN_L, 0.0, DOMAIN_L))
            with time_block("PINN") as _t:
                syn_pinn, phy_pinn, hist_pinn = train_pinn_gs(
                    init_params_mlp(k_pinn_init, d_in=3, hidden=HIDDEN, d_out=2),
                    phy_init, fixed_phy,
                    xyt_data, u_data, xyt_col, xyt_mean, xyt_std,
                    gamma=GAMMA, schedule=schedule_syn, n_epochs=E_PINN,
                    key=k_pinn_train, activation=PINN_ACTIVATION, gamma_ic=GAMMA_IC,
                    n_pretrain=N_PRETRAIN, n_lbfgs=N_LBFGS, net_forward=forward_mlp,
                    schedule_phy=optax.constant_schedule(LR_PHY_PINN))
            t_pinn = _t.elapsed
            save_gs(path, syn_pinn, phy_pinn, hist_pinn)

    # ── opt-PINN competitor (best-practice PINN: Fourier features, hard IC,
    #    causal weighting, resampled collocation, nondimensional residual) ─────
    if mode in ("comparison", "full") and config.get('RUN_OPT_PINN', False):
        path = f"saved_models/gray_scott_opt_pinn{run_tag}.pkl"
        r = (load_gs_compat(path, 'opt-PINN')
             if load and os.path.exists(path) else None)
        if r is not None:
            print(f"Loading opt-PINN from {path}")
            syn_opt, phy_opt, hist_opt = r; t_opt = 0.0
        else:
            print(f"\n--- opt-PINN  seed={seed} ---")
            with time_block("opt-PINN") as _t:
                syn_opt, phy_opt, hist_opt = train_opt_pinn_gs(
                    init_params_fourier_mlp(k_opt_init, n_freq=N_FREQ, hidden=HIDDEN,
                                            d_out=2, sigma_space=SIGMA_X,
                                            sigma_time=SIGMA_T),
                    phy_init, fixed_phy,
                    xyt_data, u_data, xyt_mean, xyt_std,
                    gamma=GAMMA_OPT, schedule=schedule_syn, n_epochs=E_PINN,
                    key=k_opt_train, n_col=N_COL, T=REF_T,
                    tau_ic=TAU_IC, causal_eps=CAUSAL_EPS, causal_bins=CAUSAL_BINS,
                    activation=PINN_ACTIVATION,
                    n_pretrain=N_PRETRAIN, n_lbfgs=N_LBFGS,
                    d_warmup=D_WARMUP, lr_phy=LR_PHY_PINN)
            t_opt = _t.elapsed
            save_gs(path, syn_opt, phy_opt, hist_opt)

    # ── Load missing checkpoints for plotting/eval ────────────────────────────
    def _maybe_load(method, cur_syn):
        if cur_syn is not None:
            return None
        p = f"saved_models/gray_scott_{method}{run_tag}.pkl"
        if os.path.exists(p):
            r = load_gs_compat(p, method)
            if r is not None:
                print(f"Loading {method} from {p} for comparison.")
            return r
        return None

    r = _maybe_load('hyco', syn_hyco)
    if r: syn_hyco, phy_hyco, hist_hyco = r
    r = _maybe_load('nn', syn_nn)
    if r: syn_nn, _, hist_nn = r
    r = _maybe_load('pinn', syn_pinn)
    if r: syn_pinn, phy_pinn, hist_pinn = r
    if config.get('RUN_OPT_PINN', False):
        r = _maybe_load('opt_pinn', syn_opt)
        if r: syn_opt, phy_opt, hist_opt = r

    # ── Evaluation: normalised L2 over the reference grid x snapshot times ─────
    ref_stack = jnp.stack([ref['U'], ref['V']], axis=-1)     # (K,n,n,2)
    ref_norm  = jnp.sqrt(jnp.sum(ref_stack ** 2))
    eval_times = ref['times']

    def e_s_net(syn, activation, net_forward=forward_mlp):
        pred = net_field(syn, REF_N, eval_times, activation, xyt_mean, xyt_std,
                         net_forward=net_forward)
        return float(jnp.sqrt(jnp.sum((pred - ref_stack) ** 2)) / ref_norm)

    def e_s_phy(phy):
        full = merge_params(phy, fixed_phy)
        pred = phy_field(full, ref, PHY_N, PHY_STEPS, PHY_SAVE, eval_times)
        return float(jnp.sqrt(jnp.sum((pred - ref_stack) ** 2)) / ref_norm)

    def e_p(phy):
        if not RECOVER:
            return float('nan')
        errs = [abs(float(phy[p]) - TRUE_PARAMS_GS[p]) / abs(TRUE_PARAMS_GS[p])
                for p in RECOVER]
        return float(np.mean(errs))

    metrics, rows = {}, []
    if syn_hyco is not None:
        es_s, es_p, ep = e_s_net(syn_hyco, HYCO_ACTIVATION), e_s_phy(phy_hyco), e_p(phy_hyco)
        metrics.update({'e_s_hyco_syn': es_s, 'e_s_hyco_phy': es_p,
                        'e_p_hyco': ep, 't_hyco_s': t_hyco,
                        **{f'{p}_hyco': float(phy_hyco[p]) for p in RECOVER}})
        rows += [('HYCO Physical', es_p, ep), ('HYCO Synthetic', es_s, float('nan'))]
    if syn_abl is not None:
        es_s, es_p, ep = e_s_net(syn_abl, HYCO_ACTIVATION), e_s_phy(phy_abl), e_p(phy_abl)
        metrics.update({'e_s_abl_syn': es_s, 'e_s_abl_phy': es_p,
                        'e_p_abl': ep, 't_abl_s': t_abl})
        rows += [('HYCO-abl Physical', es_p, ep), ('HYCO-abl Synthetic', es_s, float('nan'))]
    if syn_nn is not None:
        es = e_s_net(syn_nn, HYCO_ACTIVATION)
        metrics.update({'e_s_nn': es, 't_nn_s': t_nn})
        rows.append(('NN', es, float('nan')))
    if syn_pinn is not None:
        es, ep = e_s_net(syn_pinn, PINN_ACTIVATION), e_p(phy_pinn)
        metrics.update({'e_s_pinn': es, 'e_p_pinn': ep, 't_pinn_s': t_pinn,
                        **{f'{p}_pinn': float(phy_pinn[p]) for p in RECOVER}})
        rows.append(('PINN', es, ep))
    if syn_opt is not None:
        opt_net = make_opt_pinn_net(xyt_mean, xyt_std, TAU_IC)
        es, ep = e_s_net(syn_opt, PINN_ACTIVATION, net_forward=opt_net), e_p(phy_opt)
        metrics.update({'e_s_opt_pinn': es, 'e_p_opt_pinn': ep, 't_opt_pinn_s': t_opt,
                        **{f'{p}_opt_pinn': float(phy_opt[p]) for p in RECOVER}})
        rows.append(('opt-PINN', es, ep))

    if rows:
        print(f"\n{'Method':<20s}  {'e_s':>10s}  {'e_p':>10s}")
        print("-" * 44)
        for name, es, ep in rows:
            print(f"{name:<20s}  {es:10.4f}  {ep:10.4f}")

    # ── Plot: 5-row u-snapshot comparison ─────────────────────────────────────
    os.makedirs('figures/gray_scott', exist_ok=True)
    snap_idx   = np.linspace(0, len(eval_times) - 1, 5).astype(int)
    snap_times = [float(eval_times[i]) for i in snap_idx]

    def _u_snaps_net(syn, activation, net_forward=forward_mlp):
        return np.array(net_field(syn, REF_N, snap_times, activation, xyt_mean,
                                  xyt_std, net_forward=net_forward)[..., 0])

    def _u_snaps_phy(phy):
        full = merge_params(phy, fixed_phy)
        return np.array(phy_field(full, ref, PHY_N, PHY_STEPS, PHY_SAVE, snap_times)[..., 0])

    truth_u = np.array(ref['U'])[snap_idx]
    row_fields = [
        ('HYCO Physical',  _u_snaps_phy(phy_hyco) if syn_hyco is not None else None),
        ('HYCO Synthetic', _u_snaps_net(syn_hyco, HYCO_ACTIVATION) if syn_hyco is not None else None),
        ('NN',             _u_snaps_net(syn_nn, HYCO_ACTIVATION) if syn_nn is not None else None),
        ('PINN',           _u_snaps_net(syn_pinn, PINN_ACTIVATION) if syn_pinn is not None else None),
        ('Ground truth',   truth_u),
    ]
    if any(f is not None for _, f in row_fields[:-1]):
        plot_gs_comparison(snap_times, row_fields,
                           save_path=f"figures/gray_scott/{mode}{run_tag}.png",
                           cmap=config.get('CMAP', 'viridis'))

    return metrics


def main():
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('figures',      exist_ok=True)

    # ── Run mode ──────────────────────────────────────────────────────────────
    MODE  = "comparison"      # "basic" | "ablation" | "comparison" | "full"
    SEEDS = (42,)       # len > 1 -> aggregate over seeds
    LOAD  = True

    config = dict(
        # Reference (solved once, cached)
        REF_N=64, REF_T=2000.0, REF_STEPS=5000, REF_SAVE=50,
        # Physical model used in HYCO (coarsening PHY_N is allowed but degrades results)
        PHY_N=64, PHY_STEPS=5000, PHY_SAVE=50,
        # Identification target: any subset of ('D_u','D_v','F','k')
        RECOVER=('D_u', 'D_v'),
        # Data: counts + localisation. DATA_REGION in full/BL/BR/TL/TR or (x0,x1,y0,y1)
        # H (ghost points): 'full' = full PHY_N x PHY_N grid at every snapshot
        # time (exact FD targets, matches the torch reference); or an integer
        # for random spatial points at snapshot times.
        M=5000, H=100000, N_COL=10000,
        DATA_T_RANGE=(0.0, 2000.0), DATA_REGION='full',
        # Network / optimisation (paper)
        HIDDEN=128, LR=1e-3, LR_PHY=8e-3,
        ALPHA=10.0, BETA=0.0, ABLATION_BETA=0.0,
        # PDE-residual weight (PINN). Kept separate from GAMMA_IC below.
        GAMMA=100.0,
        # PINN initial-datum loss weight — set high to enforce the IC strongly
        # (matches the working torch reference's lambda_ic=100). Flagged: this
        # is a new config key, not previously present; confirm if a different
        # value is intended.
        GAMMA_IC=100.0,
        # PINN phy_log (D_u,D_v) learning rate — kept below LR to stop the
        # diffusivities overshooting during joint PDE-residual optimization.
        # New key, untuned beyond "clearly lower than LR"; adjust if needed.
        LR_PHY_PINN=1e-4,
        # opt-PINN (best-practice PINN competitor; see train_opt_pinn_gs)
        N_FREQ=64, SIGMA_X=3.0, SIGMA_T=1.0,   # Fourier features (normalized units)
        TAU_IC=50.0,                            # hard-IC ramp timescale
        CAUSAL_EPS=1.0, CAUSAL_BINS=20,         # causal residual weighting
        D_WARMUP=1000,                          # joint epochs with D frozen
        GAMMA_OPT=1.0,                          # weight on NONDIMENSIONAL residual
        RUN_OPT_PINN=False,                     # opt-PINN off (kept for later)
        CMAP='twilight',                         # comparison-figure colormap
        # Fixed epoch counts (paper); no stopping criterion here
        E_HYCO=10000, E_NN=60000, E_PINN=20000,
        # Pre-training epochs for HYCO NN (data-only fit before ghost coupling)
        N_PRETRAIN=2000,
        # Max L-BFGS-B iterations for post-training the synthetic model
        N_LBFGS=2000,
    )

    multi_seed = len(SEEDS) > 1
    all_results = []
    for seed in SEEDS:
        tag = make_run_tag(seed=seed, multi_seed=multi_seed)
        label = tag or "(default)"
        print(f"\n{'=' * 72}\n>>> RUN {label}  seed={seed}  mode={MODE}\n{'=' * 72}")
        with time_block(f"run {label}") as t_total:
            metrics = run_experiment(seed, config, run_tag=tag, mode=MODE, load=LOAD)
        metrics.update({'seed': seed, 't_total_s': t_total.elapsed})
        all_results.append(metrics)

    results_path = 'saved_models/gray_scott_results.pkl'
    with open(results_path, 'wb') as fout:
        pickle.dump(all_results, fout)
    print(f'Saved per-run metrics to {results_path}')

    if multi_seed:
        print_aggregated(aggregate_seeds(all_results),
                         title=f"summary across {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()