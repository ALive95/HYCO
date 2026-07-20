"""
Standalone sensitivity study for HYCO on the Helmholtz inverse problem.

Produces the three studies for the paper's Helmholtz sensitivity subsection:

  1. (alpha, beta) matrices  -- 5 heatmaps: e_s, e_p, e_d for the physical model
     (top row) and e_s, e_d for the synthetic model (bottom row).
  2. ghost-point sweep       -- final e_s (phy/syn), model-model discrepancy, and
     wall-clock time as the number of ghost points H grows; plus an optional
     L_int-history plot for the appendix.
  3. noise sweep             -- recovered coefficients (e_p) and solution error
     (e_s) as observational noise increases.

Only HYCO is trained (the three studies characterise HYCO's own machinery;
alpha/beta/H do not exist for the baselines). A Gauss-Newton overlay for the
noise study is available behind a flag but OFF by default.

This script drives HYCO through the same building blocks as main_helmholtz.py and
mirrors its data-setup (identical PRNG-split order), so a given seed reproduces
the main experiment's HYCO run. Nothing in main_helmholtz.py is modified.

Run:  python sensitivity_helmholtz.py
Cluster note: heavy. Set SMOKE=True for a fast structural check first.
"""
import jax
jax.config.update("jax_enable_x64", True)

import os
import pickle
import csv

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from physics.helmholtz_fem import (
    make_blob_mesh, precompute_fem, solve_helmholtz, locate_points,
    sample_ghost_on_mesh,
)
from synthetic.nn import init_params, forward
from utils.training import train_hyco_helmholtz, train_gauss_newton_helmholtz
from utils.experiment import time_block

# Reuse the problem definition and activation from the main script verbatim.
from main_helmholtz import TRUE_PARAMS, INIT_PARAMS, HYCO_ACTIVATION


# ── Output locations ──────────────────────────────────────────────────────────
FIG_DIR      = "figures/helmholtz/sensitivity"
RESULTS_PATH = "saved_models/sensitivity_results.pkl"   # scalar metrics, one dict
HIST_PATH    = "saved_models/sensitivity_ghost_hist.pkl" # L_int histories (ghost)


# ── Configuration (mirrors main_helmholtz.py defaults) ────────────────────────
def build_config(smoke=False):
    """Return the run-invariant config; `smoke` shrinks epochs for a quick check."""
    return dict(
        M_SENSORS=25, N_GHOST_POOL=30000,
        N_EPOCHS=500 if smoke else 30000,
        RES_FINE=0.05, RES_COARSE=0.2,
        # Defaults for the knobs NOT being swept in a given study.
        DEFAULT_ALPHA=10.0, DEFAULT_BETA=1.0, DEFAULT_NGHOST=1000,
    )


# Study grids (edit here). A 4x4 alpha/beta grid is enough to show a plateau.
ALPHA_GRID  = (0.1, 1.0, 10.0, 100.0)
BETA_GRID   = (0.0, 0.1, 1.0, 10.0)
GHOST_GRID  = (50, 100, 200, 500, 1000, 2000)
NOISE_GRID  = (0.0, 0.01, 0.02, 0.04)   # fractions of ||u*|| (<=5% by request)

SEEDS = (42,)                            # >1 -> mean/std; heavier
NOISE_SEEDS = (42, 43, 44, 45, 46)       # noise study runs over these (mean/std)
NOISE_INCLUDE_GN = False                 # overlay Gauss-Newton in the noise study


# ── Data setup (mirrors main_helmholtz.run_experiment) ────────────────────────
class Data:
    """Container for one (seed, noise) data realisation and its eval closures."""
    pass


def setup_data(seed, noise_frac, config):
    """
    Build meshes, reference solution, sensors, ghost pool, and eval closures.

    The PRNG-split order matches main_helmholtz.run_experiment exactly, so the
    HYCO init/train keys coincide with the main experiment for the same seed.

    Args:
        seed (int): PRNG seed.
        noise_frac (float): Observational noise std as a fraction of the RMS of
            the reference solution (0.0 = clean).
        config (dict): Output of build_config.

    Returns:
        Data: populated container.
    """
    key = jr.PRNGKey(seed)
    key, k_sensors             = jr.split(key)
    key, k_noise               = jr.split(key)
    key, k_ghost               = jr.split(key)
    key, k_hyco_init, k_hyco_train = jr.split(key, 3)

    nodes_f, elems_f, bnd_f = make_blob_mesh(resolution=config['RES_FINE'])
    fem_fine   = precompute_fem(nodes_f, elems_f, bnd_f)
    nodes_c, elems_c, bnd_c = make_blob_mesh(resolution=config['RES_COARSE'])
    fem_coarse = precompute_fem(nodes_c, elems_c, bnd_c)

    u_ref = solve_helmholtz(fem_fine, TRUE_PARAMS)
    rms   = float(jnp.sqrt(jnp.mean(u_ref ** 2)))
    noise_std = noise_frac * rms                       # relative -> absolute std

    interior_mask = np.ones(fem_fine.nodes.shape[0], dtype=bool)
    interior_mask[np.array(bnd_f)] = False
    interior_idx = np.where(interior_mask)[0]
    sensor_idx = jr.choice(k_sensors, jnp.array(interior_idx),
                           shape=(config['M_SENSORS'],), replace=False)
    sensor_xy = fem_fine.nodes[sensor_idx]
    u_clean   = u_ref[sensor_idx]
    u_data    = u_clean + noise_std * jr.normal(k_noise, u_clean.shape)
    sensor_tri_c, sensor_bary_c = locate_points(fem_coarse, sensor_xy)

    ghost_pool_xy, ghost_pool_tri, ghost_pool_bary = \
        sample_ghost_on_mesh(k_ghost, fem_coarse, config['N_GHOST_POOL'])

    # Ground-truth parameter vector (order matches e_p below).
    true_vec = jnp.concatenate([jnp.array([TRUE_PARAMS['alpha1']]), TRUE_PARAMS['c1'],
                                jnp.array([TRUE_PARAMS['alpha2']]), TRUE_PARAMS['c2']])
    norm = jnp.sqrt(jnp.mean(u_ref ** 2))

    d = Data()
    d.fem_fine, d.fem_coarse = fem_fine, fem_coarse
    d.u_ref = u_ref
    d.sensor_idx, d.sensor_xy, d.u_data = sensor_idx, sensor_xy, u_data
    d.sensor_tri_c, d.sensor_bary_c = sensor_tri_c, sensor_bary_c
    d.ghost_pool_xy = ghost_pool_xy
    d.ghost_pool_tri, d.ghost_pool_bary = ghost_pool_tri, ghost_pool_bary
    d.k_hyco_init, d.k_hyco_train = k_hyco_init, k_hyco_train
    d.noise_std = noise_std

    d.e_s = lambda u_: float(jnp.sqrt(jnp.mean((u_ - u_ref) ** 2)) / norm)
    d.e_d = lambda u_: float(jnp.mean((u_[sensor_idx] - u_data) ** 2))

    def e_p(pp):
        rec = jnp.concatenate([jnp.array([pp['alpha1']]), pp['c1'],
                               jnp.array([pp['alpha2']]), pp['c2']])
        return float(jnp.linalg.norm(rec - true_vec) / jnp.linalg.norm(true_vec))
    d.e_p = e_p
    return d


# ── One HYCO run ──────────────────────────────────────────────────────────────
def train_hyco_once(d, alpha, beta, n_ghost, config):
    """
    Train HYCO once on a prepared Data object and evaluate on the fine mesh.

    Args:
        d (Data): Output of setup_data.
        alpha, beta (float): HYCO loss weights.
        n_ghost (int): Ghost-point minibatch size H (drawn from the pool/epoch).
        config (dict): Output of build_config.

    Returns:
        dict: scalar metrics, plus 'L_int_hist' (list) for the ghost study.
    """
    schedule_syn     = optax.cosine_decay_schedule(1e-4, config['N_EPOCHS'], alpha=0.01)
    schedule_phy_amp = optax.cosine_decay_schedule(1e-3, config['N_EPOCHS'], alpha=0.01)
    schedule_phy_ctr = optax.cosine_decay_schedule(1e-2, config['N_EPOCHS'], alpha=0.01)

    with time_block() as t:
        phy, syn, hist = train_hyco_helmholtz(
            d.fem_coarse, init_params(d.k_hyco_init, hidden=256), dict(INIT_PARAMS),
            schedule_syn, schedule_phy_amp, schedule_phy_ctr,
            d.sensor_xy, d.u_data, d.sensor_tri_c, d.sensor_bary_c,
            d.ghost_pool_xy, d.ghost_pool_tri, d.ghost_pool_bary,
            alpha=alpha, beta=beta,
            n_ghost=n_ghost, n_ghost_pool=config['N_GHOST_POOL'],
            n_epochs=config['N_EPOCHS'],
            key=d.k_hyco_train, activation=HYCO_ACTIVATION,
        )

    u_phy = solve_helmholtz(d.fem_fine, phy)
    u_syn = forward(syn, d.fem_fine.nodes, activation=HYCO_ACTIVATION)
    # Domain-wide discrepancy between the two final predictions (MSE, L_int units).
    disc = float(jnp.mean((u_syn - u_phy) ** 2))

    return {
        'e_s_phy': d.e_s(u_phy), 'e_s_syn': d.e_s(u_syn),
        'e_d_phy': d.e_d(u_phy), 'e_d_syn': d.e_d(u_syn),
        'e_p':     d.e_p(phy),
        'disc':    disc,
        'time_s':  t.elapsed,
        'epochs':  len(hist['L_int']),
        'L_int_hist': [float(x) for x in hist['L_int']],
    }


# ── Caching ───────────────────────────────────────────────────────────────────
def load_cache(path):
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    return {}


def save_cache(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)


def get_or_run(cache, key, run_fn, load):
    """Return cache[key], computing and storing it via run_fn() if absent."""
    if load and key in cache:
        print(f"  [cache] {key}")
        return cache[key]
    val = run_fn()
    cache[key] = val
    save_cache(cache, RESULTS_PATH)   # persist after every run (crash-safe)
    return val


def mean_over_seeds(cache, key_fn, metric, seeds=SEEDS):
    """Mean of `metric` over `seeds` for the run identified by key_fn(seed)."""
    vals = [cache[key_fn(s)][metric] for s in seeds if key_fn(s) in cache]
    return float(np.mean(vals)) if vals else float('nan')


def std_over_seeds(cache, key_fn, metric, seeds=SEEDS):
    """Std of `metric` over `seeds` for the run identified by key_fn(seed)."""
    vals = [cache[key_fn(s)][metric] for s in seeds if key_fn(s) in cache]
    return float(np.std(vals)) if vals else float('nan')


# ── Study 1: alpha/beta matrices ──────────────────────────────────────────────
def study_alpha_beta(cache, config, load):
    """Sweep the (alpha, beta) grid and render the 5 error matrices."""
    print("\n=== Study 1: alpha/beta matrices ===")
    H = config['DEFAULT_NGHOST']
    for seed in SEEDS:
        for a in ALPHA_GRID:
            for b in BETA_GRID:
                key = ('ab', seed, a, b, H)
                print(f">>> ab  seed={seed} alpha={a} beta={b}")
                data = setup_data(seed, 0.0, config)
                get_or_run(cache, key,
                           lambda data=data, a=a, b=b: train_hyco_once(data, a, b, H, config),
                           load)

    # Assemble metric -> (len(BETA) x len(ALPHA)) matrix, averaged over seeds.
    def mat(metric):
        M = np.full((len(BETA_GRID), len(ALPHA_GRID)), np.nan)
        for i, b in enumerate(BETA_GRID):
            for j, a in enumerate(ALPHA_GRID):
                M[i, j] = mean_over_seeds(cache, lambda s, a=a, b=b: ('ab', s, a, b, H), metric)
        return M

    mats = {
        'e_s_phy': mat('e_s_phy'), 'e_p': mat('e_p'), 'e_d_phy': mat('e_d_phy'),
        'e_s_syn': mat('e_s_syn'), 'e_d_syn': mat('e_d_syn'),
    }
    plot_ab_matrices(mats)
    dump_ab_csv(mats)


def plot_ab_matrices(mats):
    """Render 5 heatmaps: physical (e_s,e_p,e_d) on top, synthetic (e_s,e_d) below."""
    titles = {
        'e_s_phy': r'$\mathsf{e}_s$ (phy)', 'e_p': r'$\mathsf{e}_p$',
        'e_d_phy': r'$\mathsf{e}_d$ (phy)',
        'e_s_syn': r'$\mathsf{e}_s$ (syn)', 'e_d_syn': r'$\mathsf{e}_d$ (syn)',
    }
    layout = [['e_s_phy', 'e_p', 'e_d_phy'], ['e_s_syn', None, 'e_d_syn']]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for i, row in enumerate(layout):
        for j, name in enumerate(row):
            ax = axes[i, j]
            if name is None:
                ax.axis('off')
                continue
            M = mats[name]
            Mpos = np.where(M > 0, M, np.nan)
            norm = LogNorm(vmin=np.nanmin(Mpos), vmax=np.nanmax(Mpos)) \
                if np.isfinite(np.nanmin(Mpos)) else None
            im = ax.imshow(M, cmap='viridis_r', norm=norm, aspect='auto', origin='lower')
            ax.set_title(titles[name])
            ax.set_xticks(range(len(ALPHA_GRID)));  ax.set_xticklabels([f'{a:g}' for a in ALPHA_GRID])
            ax.set_yticks(range(len(BETA_GRID)));   ax.set_yticklabels([f'{b:g}' for b in BETA_GRID])
            ax.set_xlabel(r'$\alpha$');  ax.set_ylabel(r'$\beta$')
            for p in range(M.shape[0]):
                for q in range(M.shape[1]):
                    ax.text(q, p, f'{M[p, q]:.2g}', ha='center', va='center',
                            color='w', fontsize=7)
            # Outline the robust plateau (alpha in [1,10], beta in [0.1,1]) on the
            # parameter-error panel, where the paper's robustness claim is made.
            if name == 'e_p':
                a_idx = [j for j, a in enumerate(ALPHA_GRID) if 1.0 <= a <= 10.0]
                b_idx = [i for i, b in enumerate(BETA_GRID) if 0.1 <= b <= 1.0]
                if a_idx and b_idx:
                    ax.add_patch(Rectangle(
                        (min(a_idx) - 0.5, min(b_idx) - 0.5),
                        len(a_idx), len(b_idx),
                        fill=False, edgecolor='#d1495b', lw=3, zorder=6))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = f'{FIG_DIR}/ab_matrices.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}")


def dump_ab_csv(mats):
    """Write one CSV per matrix (rows=beta, cols=alpha) for dropping into LaTeX."""
    for name, M in mats.items():
        out = f'{FIG_DIR}/ab_{name}.csv'
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['beta\\alpha'] + [f'{a:g}' for a in ALPHA_GRID])
            for i, b in enumerate(BETA_GRID):
                w.writerow([f'{b:g}'] + [f'{M[i, j]:.6e}' for j in range(len(ALPHA_GRID))])
    print(f"  saved ab_*.csv")


# ── Study 2: ghost-point sweep ────────────────────────────────────────────────
def study_ghost(cache, hist_cache, config, load):
    """Sweep H and report time, final precision, and model-model discrepancy."""
    print("\n=== Study 2: ghost-point sweep ===")
    a, b = config['DEFAULT_ALPHA'], config['DEFAULT_BETA']
    for seed in SEEDS:
        for H in GHOST_GRID:
            key = ('ghost', seed, a, b, H)
            print(f">>> ghost  seed={seed} H={H}")
            data = setup_data(seed, 0.0, config)
            res = get_or_run(cache, key,
                             lambda data=data, H=H: train_hyco_once(data, a, b, H, config),
                             load)
            hist_cache[('ghost', seed, H)] = res['L_int_hist']
    save_cache(hist_cache, HIST_PATH)

    m = lambda H, k: mean_over_seeds(cache, lambda s, H=H: ('ghost', s, a, b, H), k)
    table = [(H, m(H, 'e_s_phy'), m(H, 'e_s_syn'), m(H, 'disc'), m(H, 'e_p'), m(H, 'time_s'))
             for H in GHOST_GRID]
    plot_ghost(table)
    plot_loss_histories(hist_cache)
    dump_ghost_csv(table)


def plot_ghost(table):
    """Errors + discrepancy vs H on a log axis, wall-clock time on a twin axis."""
    H   = [r[0] for r in table]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(H, [r[1] for r in table], 'o-', label=r'$\mathsf{e}_s$ (phy)')
    ax.plot(H, [r[2] for r in table], 's-', label=r'$\mathsf{e}_s$ (syn)')
    ax.plot(H, [r[3] for r in table], '^-', label='discrepancy')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('number of ghost points $H$'); ax.set_ylabel('error / discrepancy')
    ax.grid(True, which='both', alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(H, [r[5] for r in table], 'x--', color='gray', label='time (s)')
    ax2.set_ylabel('wall-clock time (s)')
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8, loc='best')
    fig.tight_layout()
    out = f'{FIG_DIR}/ghost_sweep.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}")


def plot_loss_histories(hist_cache):
    """Appendix candidate: L_int vs epoch for each H (first seed in SEEDS)."""
    seed = SEEDS[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for H in GHOST_GRID:
        h = hist_cache.get(('ghost', seed, H))
        if h:
            ax.plot(np.arange(len(h)), h, label=f'H={H}', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel(r'$\mathcal{L}_{int}$ (minibatch)')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    out = f'{FIG_DIR}/ghost_Lint_history.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}")


def dump_ghost_csv(table):
    out = f'{FIG_DIR}/ghost_sweep.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['H', 'e_s_phy', 'e_s_syn', 'discrepancy', 'e_p', 'time_s'])
        for r in table:
            w.writerow([r[0]] + [f'{x:.6e}' for x in r[1:]])
    print(f"  saved {out}")


# ── Study 3: noise sweep ──────────────────────────────────────────────────────
def study_noise(cache, config, load):
    """Sweep observational noise over NOISE_SEEDS and report mean/std of e_p, e_s."""
    print("\n=== Study 3: noise sweep ===")
    a, b, H = config['DEFAULT_ALPHA'], config['DEFAULT_BETA'], config['DEFAULT_NGHOST']
    for seed in NOISE_SEEDS:
        for nf in NOISE_GRID:
            key = ('noise', seed, a, b, H, nf)
            print(f">>> noise  seed={seed} frac={nf}")
            data = setup_data(seed, nf, config)
            get_or_run(cache, key,
                       lambda data=data, key=key:
                           train_hyco_once(data, a, b, H, config, model_path(key)),
                       load)
            if NOISE_INCLUDE_GN:
                gnkey = ('noise_gn', seed, nf)
                get_or_run(cache, gnkey,
                           lambda data=data: run_gn(data),
                           load)

    def stat(nf, metric, fn):
        return fn(cache, lambda s, nf=nf: ('noise', s, a, b, H, nf), metric, NOISE_SEEDS)
    # rows: (nf, e_p_mean, e_p_std, e_s_mean, e_s_std)
    rows = [(nf,
             stat(nf, 'e_p', mean_over_seeds), stat(nf, 'e_p', std_over_seeds),
             stat(nf, 'e_s_phy', mean_over_seeds), stat(nf, 'e_s_phy', std_over_seeds))
            for nf in NOISE_GRID]

    gn_rows = None
    if NOISE_INCLUDE_GN:
        def gstat(nf, metric, fn):
            return fn(cache, lambda s, nf=nf: ('noise_gn', s, nf), metric, NOISE_SEEDS)
        gn_rows = [(nf,
                    gstat(nf, 'e_p', mean_over_seeds), gstat(nf, 'e_p', std_over_seeds),
                    gstat(nf, 'e_s', mean_over_seeds), gstat(nf, 'e_s', std_over_seeds))
                   for nf in NOISE_GRID]
    plot_noise(rows, gn_rows)
    dump_noise_csv(rows, gn_rows)


def run_gn(d):
    """Gauss-Newton inversion on the same (noisy) data, for the noise overlay."""
    with time_block() as t:
        phy_gn, _ = train_gauss_newton_helmholtz(
            d.fem_coarse, d.sensor_tri_c, d.sensor_bary_c, d.u_data, dict(INIT_PARAMS))
    u_gn = solve_helmholtz(d.fem_fine, phy_gn)
    return {'e_p': d.e_p(phy_gn), 'e_s': d.e_s(u_gn), 'time_s': t.elapsed}


def plot_noise(rows, gn_rows):
    """Mean +/- std of recovered-coefficient and solution error vs noise level."""
    x = [r[0] * 100 for r in rows]                     # percent
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_ylim(1e-2, 5e-1)   # e_p
    ax[1].set_ylim(1e-2, 5e-1)   # e_s
    ax[0].errorbar(x, [r[1] for r in rows], yerr=[r[2] for r in rows],
                   fmt='o-', capsize=3, label='HYCO')
    ax[1].errorbar(x, [r[3] for r in rows], yerr=[r[4] for r in rows],
                   fmt='o-', capsize=3, label='HYCO')
    if gn_rows is not None:
        ax[0].errorbar(x, [r[1] for r in gn_rows], yerr=[r[2] for r in gn_rows],
                       fmt='s--', capsize=3, label='Gauss-Newton')
        ax[1].errorbar(x, [r[3] for r in gn_rows], yerr=[r[4] for r in gn_rows],
                       fmt='s--', capsize=3, label='Gauss-Newton')
    ax[0].set_ylabel(r'$\mathsf{e}_p$'); ax[1].set_ylabel(r'$\mathsf{e}_s$ (phy)')
    for a_ in ax:
        a_.set_xlabel(r'noise level (% of $\|u^\star\|$)')
        a_.set_yscale('log'); a_.grid(True, which='both', alpha=0.3); a_.legend(fontsize=8)
    fig.tight_layout()
    out = f'{FIG_DIR}/noise_sweep.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}")


def dump_noise_csv(rows, gn_rows):
    out = f'{FIG_DIR}/noise_sweep.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        header = ['noise_frac', 'hyco_e_p_mean', 'hyco_e_p_std',
                  'hyco_e_s_mean', 'hyco_e_s_std']
        if gn_rows is not None:
            header += ['gn_e_p_mean', 'gn_e_p_std', 'gn_e_s_mean', 'gn_e_s_std']
        w.writerow(header)
        for i, r in enumerate(rows):
            line = [r[0]] + [f'{v:.6e}' for v in r[1:]]
            if gn_rows is not None:
                line += [f'{v:.6e}' for v in gn_rows[i][1:]]
            w.writerow(line)
    print(f"  saved {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    SMOKE = False           # True -> few epochs, for a quick structural check
    LOAD  = True           # reuse cached per-run results if present
    RUN   = ('ab', 'ghost', 'noise')   # which studies to run

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    config     = build_config(smoke=SMOKE)
    cache      = load_cache(RESULTS_PATH)
    hist_cache = load_cache(HIST_PATH)

    if 'ab' in RUN:
        study_alpha_beta(cache, config, LOAD)
    if 'ghost' in RUN:
        study_ghost(cache, hist_cache, config, LOAD)
    if 'noise' in RUN:
        study_noise(cache, config, LOAD)

    print("\nDone. Figures in", FIG_DIR)


if __name__ == "__main__":
    main()