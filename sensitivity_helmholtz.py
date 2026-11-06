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

Run: python sensitivity_helmholtz.py
Referee losses: python sensitivity_helmholtz.py --study losses --report-only
  (add --checkpoint-dir saved_models/helmholtz_losses/full/357f6addc52e16f0 to reuse the archived sweep)
Small structural run: python sensitivity_helmholtz.py --study losses --smoke
"""
import jax
jax.config.update("jax_enable_x64", True)

import os
import pickle
import csv
import io
import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from importlib import metadata as import_metadata

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from physics.helmholtz_fem import (
    FEMData, make_blob_mesh, precompute_fem, solve_helmholtz, locate_points,
    sample_ghost_on_mesh, interp_at_pts,
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


def mean_over_seeds(cache, key_fn, metric, seeds=None):
    """Mean of `metric` over `seeds` for the run identified by key_fn(seed)."""
    seeds = SEEDS if seeds is None else seeds
    vals = [cache[key_fn(s)][metric] for s in seeds if key_fn(s) in cache]
    return float(np.mean(vals)) if vals else float('nan')


def std_over_seeds(cache, key_fn, metric, seeds=None):
    """Std of `metric` over `seeds` for the run identified by key_fn(seed)."""
    seeds = SEEDS if seeds is None else seeds
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


def _short_label(value, position=None):
    """2 significant digits, exponent without leading zeros (e.g. 3.1e-5)."""
    label = f'{value:.2g}'
    if 'e' in label:
        mantissa, exponent = label.split('e')
        label = f'{mantissa}e{int(exponent)}'
    return label


@plt.rc_context({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix',
                 'text.usetex': False, 'axes.formatter.use_mathtext': False,
                 'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})
def plot_ab_losses(mats, fig_dir):
    """Paper figure: same 5-panel layout as plot_ab_matrices, with the coarse
    training losses L_phy, L_syn in place of the data discrepancies."""
    titles = {
        'e_s_phy': r'$\mathsf{e}_s^{\mathrm{phy}}$',
        'e_s_syn': r'$\mathsf{e}_s^{\mathrm{syn}}$',
        'e_p': r'$\mathsf{e}_p$',
        'L_phy': r'$\mathcal{L}_{\mathrm{phy}}\equiv\mathsf{e}_d^{\mathrm{phy}}$',
        'L_syn': r'$\mathcal{L}_{\mathrm{syn}}\equiv\mathsf{e}_d^{\mathrm{syn}}$',
    }
    layout = [['e_s_phy', 'e_p', 'L_phy'], ['e_s_syn', None, 'L_syn']]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.3))
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.085, top=0.94,
                        wspace=0.36, hspace=0.30)
    for i, row in enumerate(layout):
        for j, name in enumerate(row):
            ax = axes[i, j]
            if name is None:
                ax.axis('off')
                continue
            M = mats[name]
            lo, hi = M.min(), M.max()
            im = ax.imshow(M, cmap='viridis_r', origin='lower', aspect='equal',
                           norm=LogNorm(vmin=lo / 10 if lo == hi else lo, vmax=hi))
            ax.set_title(titles[name], fontsize=16, pad=10)
            ax.set_xticks(range(len(ALPHA_GRID)), [f'{a:g}' for a in ALPHA_GRID])
            ax.set_yticks(range(len(BETA_GRID)), [f'{b:g}' for b in BETA_GRID])
            ax.set_xlabel(r'$\alpha$', fontsize=14, labelpad=3)
            ax.set_ylabel(r'$\beta$', fontsize=14, labelpad=3)
            ax.tick_params(length=3, labelsize=10)
            for p in range(M.shape[0]):
                for q in range(M.shape[1]):
                    rgba = im.cmap(im.norm(M[p, q]))
                    luminance = np.dot((0.2126, 0.7152, 0.0722), rgba[:3])
                    ax.text(q, p, _short_label(M[p, q]), ha='center', va='center',
                            color='black' if luminance > 0.45 else 'white', fontsize=9)
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.10)
            cbar = fig.colorbar(im, cax=cax)
            cbar.locator = FixedLocator(np.geomspace(im.norm.vmin, hi, 3))
            cbar.formatter = FuncFormatter(_short_label)
            cbar.update_ticks()
            cbar.ax.yaxis.set_minor_locator(NullLocator())
            cbar.ax.tick_params(length=3, labelsize=9, pad=3)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = Path(fig_dir) / f'ab_matrices_training_losses.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  saved {out}")
    plt.close(fig)


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
                       lambda data=data: train_hyco_once(data, a, b, H, config),
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


# ── Referee loss reporting (separate from the original caches) ────────────────
LOSS_SCHEMA = 1
LOSS_DATA_FIELDS = (
    'u_ref', 'sensor_idx', 'sensor_xy', 'u_data', 'sensor_tri_c', 'sensor_bary_c',
    'ghost_pool_xy', 'ghost_pool_tri', 'ghost_pool_bary',
    'k_hyco_init', 'k_hyco_train', 'noise_std',
)


def loss_config(smoke=False, epochs=None):
    """Keep production settings unchanged; isolate the smaller structural check."""
    config = build_config(False)
    config.update(HIDDEN=256, LR_SYN=1e-4, LR_PHY_AMP=1e-3, LR_PHY_CTR=1e-2,
                  LR_FINAL_FRACTION=0.01, STAB_WINDOW=1000, STAB_TOL=1e-3)
    if smoke:
        config.update(N_EPOCHS=5, M_SENSORS=4, N_GHOST_POOL=32,
                      DEFAULT_NGHOST=8, HIDDEN=8, RES_FINE=0.3, RES_COARSE=0.5)
    if epochs is not None:
        if epochs < 1:
            raise ValueError('epochs must be positive')
        config['N_EPOCHS'] = epochs
    return config


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return np.asarray(value).tolist()


def _json(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def loss_source_identity():
    """Fingerprint code and numerical packages, independently of output paths."""
    root = Path(__file__).resolve().parent
    names = ('sensitivity_helmholtz.py', 'main_helmholtz.py', 'utils/training.py',
             'synthetic/nn.py', 'physics/helmholtz_fem.py')
    versions = {}
    for name in ('jax', 'jaxlib', 'optax', 'gmsh', 'numpy'):
        try:
            versions[name] = import_metadata.version(name)
        except import_metadata.PackageNotFoundError:
            versions[name] = 'unknown'
    return {'files': {name: hashlib.sha256((root / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
                      for name in names}, 'packages': versions}


def loss_run_spec(config, source=None):
    return {'schema': LOSS_SCHEMA, 'config': config,
            'source': loss_source_identity() if source is None else source,
            'true_params': _jsonable(TRUE_PARAMS), 'initial_params': _jsonable(INIT_PARAMS),
            'precision': 'float64', 'activation': 'relu', 'noise_fraction': 0.0,
            'update_order': 'synthetic_then_physical',
            'terminal_losses': 'unweighted_sensor_mse_at_returned_checkpoint'}


def _spec_key(spec):
    """Settings that identify a run; source hashes are recorded but not enforced."""
    return {k: v for k, v in spec.items() if k != 'source'}


def loss_output_paths(config, smoke, spec, output_dir=None):
    """Every changed budget/configuration has its own namespace."""
    mode = 'smoke' if smoke else ('full' if config == loss_config() else 'custom')
    fingerprint = hashlib.sha256(_json(_spec_key(spec)).encode()).hexdigest()
    suffix = Path(mode) / fingerprint[:16]
    if output_dir is None:
        return (Path('saved_models/helmholtz_losses') / suffix,
                Path('figures/helmholtz/losses') / suffix, fingerprint)
    out = Path(output_dir) / suffix
    return out, out / 'figures', fingerprint


def _atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        kwargs = {} if binary else {'encoding': 'utf-8', 'newline': ''}
        with os.fdopen(fd, 'wb' if binary else 'w', **kwargs) as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def unweighted_data_losses(physical_at_sensors, synthetic_at_sensors, observations):
    """No weights or interaction; require matching vectors to prevent broadcasting."""
    phy, syn, obs = (np.asarray(x, dtype=float) for x in
                     (physical_at_sensors, synthetic_at_sensors, observations))
    if obs.ndim != 1 or phy.shape != obs.shape or syn.shape != obs.shape or not obs.size:
        raise ValueError('sensor predictions and observations must be matching nonempty vectors')
    if not all(np.all(np.isfinite(x)) for x in (phy, syn, obs)):
        raise ValueError('nonfinite sensor values')
    return {'L_phy': float(np.mean((phy - obs) ** 2)),
            'L_syn': float(np.mean((syn - obs) ** 2))}


def terminal_data_losses(data, phy, syn):
    """Evaluate losses and Figure 6.3 errors at the same returned checkpoint."""
    coarse = solve_helmholtz(data.fem_coarse, phy)
    phy_sensor = interp_at_pts(data.fem_coarse, coarse,
                               data.sensor_tri_c, data.sensor_bary_c)
    fine = solve_helmholtz(data.fem_fine, phy)
    syn_fine = forward(syn, data.fem_fine.nodes, activation=HYCO_ACTIVATION)
    syn_sensor = syn_fine[data.sensor_idx]
    losses = unweighted_data_losses(phy_sensor, syn_sensor, data.u_data)
    losses['L_phy_eval'] = unweighted_data_losses(
        fine[data.sensor_idx], syn_sensor, data.u_data)['L_phy']
    norm = np.sqrt(np.mean(np.asarray(data.u_ref) ** 2))
    losses['e_s_phy'] = float(np.sqrt(np.mean(np.asarray(fine - data.u_ref) ** 2)) / norm)
    losses['e_s_syn'] = float(np.sqrt(np.mean(np.asarray(syn_fine - data.u_ref) ** 2)) / norm)
    def parameter_vector(params):
        return np.concatenate([np.atleast_1d(np.asarray(params[key]))
                               for key in ('alpha1', 'c1', 'alpha2', 'c2')])
    true = parameter_vector(TRUE_PARAMS)
    losses['e_p'] = float(np.linalg.norm(parameter_vector(phy) - true) / np.linalg.norm(true))
    return losses


def _array_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(_json([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_loss_checkpoint(path, data, phy, syn, history, metadata):
    """Save numeric arrays and JSON only; all mesh and sampling provenance travels with it."""
    arrays = {f'{group}__{key}': np.asarray(value)
              for group, params in (('phy', phy), ('syn', syn))
              for key, value in params.items()}
    for group, fem in (('coarse', data.fem_coarse), ('fine', data.fem_fine)):
        arrays.update({f'{group}__{key}': np.asarray(getattr(fem, key))
                       for key in FEMData._fields})
    arrays.update({f'data__{name}': np.asarray(getattr(data, name))
                   for name in LOSS_DATA_FIELDS})
    history_json = _json(history)
    metadata = dict(metadata, array_sha256=_array_digest(arrays),
                    history_sha256=hashlib.sha256(history_json.encode()).hexdigest())
    arrays.update(metadata_json=np.asarray(_json(metadata)),
                  history_json=np.asarray(history_json))
    _atomic_write(path, lambda fh: np.savez_compressed(fh, **arrays), binary=True)


def load_loss_checkpoint(path, spec, seed, alpha, beta):
    """Reject mismatched settings, and never execute objects from a checkpoint."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json']))
        expected = {'spec': _spec_key(spec), 'seed': seed, 'alpha': alpha, 'beta': beta}
        stored = dict(metadata, spec=_spec_key(metadata.get('spec') or {}))
        if any(stored.get(k) != _jsonable(v) for k, v in expected.items()):
            raise ValueError('incompatible checkpoint configuration')
        arrays = {name: saved[name] for name in saved.files
                  if name not in ('metadata_json', 'history_json')}
        if metadata.get('array_sha256') != _array_digest(arrays):
            raise ValueError('checkpoint array checksum mismatch')
        history_json = str(saved['history_json'])
        if metadata.get('history_sha256') != hashlib.sha256(history_json.encode()).hexdigest():
            raise ValueError('checkpoint history checksum mismatch')
    data = Data()
    for group in ('coarse', 'fine'):
        fem = FEMData(*(jnp.asarray(arrays[f'{group}__{name}']) for name in FEMData._fields))
        setattr(data, f'fem_{group}', fem)
    for name in LOSS_DATA_FIELDS:
        setattr(data, name, jnp.asarray(arrays[f'data__{name}']))
    phy = {name[5:]: jnp.asarray(value) for name, value in arrays.items() if name.startswith('phy__')}
    syn = {name[5:]: jnp.asarray(value) for name, value in arrays.items() if name.startswith('syn__')}
    return data, phy, syn, json.loads(history_json), metadata


def train_loss_case(data, alpha, beta, config):
    def schedule(rate):
        return optax.cosine_decay_schedule(rate, config['N_EPOCHS'],
                                           alpha=config['LR_FINAL_FRACTION'])
    with time_block() as timing:
        phy, syn, history = train_hyco_helmholtz(
            data.fem_coarse, init_params(data.k_hyco_init, hidden=config['HIDDEN']),
            dict(INIT_PARAMS), schedule(config['LR_SYN']),
            schedule(config['LR_PHY_AMP']), schedule(config['LR_PHY_CTR']),
            data.sensor_xy, data.u_data, data.sensor_tri_c, data.sensor_bary_c,
            data.ghost_pool_xy, data.ghost_pool_tri, data.ghost_pool_bary,
            alpha=alpha, beta=beta, n_ghost=config['DEFAULT_NGHOST'],
            n_ghost_pool=config['N_GHOST_POOL'], n_epochs=config['N_EPOCHS'],
            key=data.k_hyco_train, activation=HYCO_ACTIVATION,
            stab_window=config['STAB_WINDOW'], stab_tol=config['STAB_TOL'])
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), (phy, syn))
    return phy, syn, history, timing.elapsed


class _PlainCache(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('legacy summary must contain plain data only')


def legacy_loss_summary(path, config):
    """The old file has no configuration provenance; use only its original settings."""
    if config != loss_config() or not Path(path).is_file():
        return {}
    with open(path, 'rb') as fh:
        return _PlainCache(fh).load()


def _loss_checkpoint_name(seed, alpha, beta):
    return f'loss_seed{seed}_a{alpha:g}_b{beta:g}.npz'


def _loss_csv(path, rows, fields):
    content = io.StringIO(newline='')
    writer = csv.DictWriter(content, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows({key: _json(value) if isinstance(value, (dict, list)) else value
                      for key, value in row.items()} for row in rows)
    _atomic_write(path, lambda fh: fh.write(content.getvalue()))


def write_loss_report(rows, out, manifest):
    fields = ('seed', 'alpha', 'beta', 'H', 'status', 'L_phy', 'L_syn', 'L_phy_eval',
              'e_s_phy', 'e_s_syn', 'e_p',
              'loss_source', 'L_syn_legacy', 'L_phy_eval_legacy', 'epochs',
              'training_elapsed_s', 'checkpoint', 'origin_hardware', 'origin_packages', 'reason')
    _loss_csv(out / 'losses.csv', rows, fields)
    manifest = dict(manifest, rows=rows,
                    complete=sum(row['L_phy'] is not None for row in rows),
                    missing=sum(row['L_phy'] is None for row in rows))
    _atomic_write(out / 'run.json', lambda fh: fh.write(_json(manifest) + '\n'))


def _loss_mean_matrix(rows, seeds, field):
    matrix = np.full((len(BETA_GRID), len(ALPHA_GRID)), np.nan)
    for i, beta in enumerate(BETA_GRID):
        for j, alpha in enumerate(ALPHA_GRID):
            values = [r[field] for r in rows if r['alpha'] == alpha and r['beta'] == beta
                      and r[field] is not None]
            if len(values) == len(seeds):
                matrix[i, j] = np.mean(values)
    return matrix


def study_losses(args):
    """Recover exact final losses, or compute only missing alpha/beta cases."""
    config = loss_config(args.smoke, args.epochs)
    seeds = tuple(SEEDS if args.seeds is None else args.seeds)
    spec = loss_run_spec(config)
    out, fig_dir, fingerprint = loss_output_paths(config, args.smoke, spec, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        legacy = legacy_loss_summary(args.legacy_cache, config)
    except (OSError, ValueError, pickle.UnpicklingError, EOFError) as error:
        print(f'Legacy summary unavailable: {error}')
        legacy = {}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=Path(__file__).resolve().parent,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {'schema': LOSS_SCHEMA, 'study': 'losses', 'spec': spec,
                'fingerprint': fingerprint, 'seeds': seeds,
                'alpha_grid': ALPHA_GRID, 'beta_grid': BETA_GRID,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_commit': commit, 'report_only': args.report_only,
                'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                             'python': platform.python_version(),
                             'devices': [str(device) for device in jax.devices()]},
                'definitions': {'L_phy': 'Unweighted sensor MSE using the coarse training FEM.',
                                'L_syn': 'Unweighted sensor MSE using the synthetic network.',
                                'L_phy_eval': 'Fine-FEM sensor MSE, not the physical training loss.',
                                'e_s_phy': 'Relative RMS physical-field error using the fine FEM.',
                                'e_s_syn': 'Relative RMS synthetic-field error on the fine mesh.',
                                'e_p': 'Relative Euclidean error in all six physical parameters.',
                                'figure': 'ab_matrices_training_losses: layout of ab_matrices with the coarse training loss L_phy in place of e_d (phy); all panels use the same checkpoints.',
                                'legacy': 'Legacy summary lacks verified configuration/checkpoint provenance.',
                                'history': 'Training losses precede block updates; terminal losses are reevaluated.',
                                'training_elapsed_s': 'Includes training JIT compilation; excludes terminal diagnostics.',
                                'checkpoint': 'Final returned parameters, full history, exact meshes and sample arrays.'}}
    rows, prepared = [], {}
    for seed in seeds:
        for alpha in ALPHA_GRID:
            for beta in BETA_GRID:
                old = legacy.get(('ab', seed, alpha, beta, config['DEFAULT_NGHOST']), {})
                row = dict(seed=seed, alpha=alpha, beta=beta, H=config['DEFAULT_NGHOST'],
                           status='missing_checkpoint', L_phy=None, L_syn=old.get('e_d_syn'),
                           L_phy_eval=old.get('e_d_phy'), loss_source='legacy_summary' if old else 'missing',
                           e_s_phy=None, e_s_syn=None, e_p=None,
                           L_syn_legacy=old.get('e_d_syn'), L_phy_eval_legacy=old.get('e_d_phy'),
                           epochs=old.get('epochs'), training_elapsed_s=None, checkpoint=None,
                           origin_hardware=None, origin_packages=None,
                           reason='No compatible final checkpoint; coarse training loss is unavailable.')
                checkpoint = out / 'checkpoints' / _loss_checkpoint_name(seed, alpha, beta)
                candidates = [checkpoint]
                if args.checkpoint_dir:
                    candidates += sorted(Path(args.checkpoint_dir).rglob(checkpoint.name))
                loaded = None
                if args.load:
                    for candidate in dict.fromkeys(candidates):
                        if not candidate.is_file():
                            continue
                        try:
                            loaded = load_loss_checkpoint(candidate, spec, seed, alpha, beta)
                            checkpoint = candidate
                            break
                        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                            print(f'Ignoring {candidate}: {error}')
                            row['reason'] = str(error)
                if loaded is None and not args.report_only:
                    if seed not in prepared:
                        prepared[seed] = setup_data(seed, 0.0, config)
                    data = prepared[seed]
                    print(f'>>> losses seed={seed} alpha={alpha:g} beta={beta:g}')
                    phy, syn, history, elapsed = train_loss_case(data, alpha, beta, config)
                    metadata = {'spec': spec, 'seed': seed, 'alpha': alpha, 'beta': beta,
                                'origin': 'new_computation', 'source_commit': commit,
                                'hardware': manifest['hardware'],
                                'packages': spec['source']['packages'],
                                'training_elapsed_s': elapsed,
                                'epochs': len(history['L_int']),
                                'created_utc': datetime.now(timezone.utc).isoformat()}
                    save_loss_checkpoint(checkpoint, data, phy, syn, history, metadata)
                    loaded = (data, phy, syn, history, metadata)
                    row['status'] = 'new_computation'
                if loaded is not None:
                    data, phy, syn, history, metadata = loaded
                    row.update(terminal_data_losses(data, phy, syn),
                               status='recovered_checkpoint' if row['status'] != 'new_computation' else row['status'],
                               loss_source='returned_checkpoint', epochs=len(history['L_int']),
                               training_elapsed_s=metadata.get('training_elapsed_s'),
                               origin_hardware=metadata.get('hardware'),
                               origin_packages=metadata.get('packages'),
                               checkpoint=str(checkpoint.resolve()), reason='')
                rows.append(row)
                write_loss_report(rows, out, manifest)
    mats = {name: _loss_mean_matrix(rows, seeds, name)
            for name in ('e_s_phy', 'e_s_syn', 'e_p', 'L_phy', 'L_syn')}
    ready = all(np.all(np.isfinite(M) & (M > 0)) for M in mats.values())
    if ready:
        plot_ab_losses(mats, fig_dir)
    else:
        print('Loss figure withheld: some checkpoint errors/losses are missing.')
    manifest['full_sensitivity_figure_ready'] = ready
    write_loss_report(rows, out, manifest)
    missing = sum(row['L_phy'] is None for row in rows)
    print(f'Loss report: {out / "losses.csv"} ({len(rows) - missing} complete, {missing} missing)')
    print(f'Figures: {fig_dir}')
    return rows


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('all', 'ab', 'ghost', 'noise', 'losses'), default='all',
                        help='losses reports the 16 alpha/beta cases; default retains the original three studies')
    parser.add_argument('--smoke', action='store_true', help='small, isolated structural run; not paper results')
    parser.add_argument('--epochs', type=int, help='override the epoch ceiling in an isolated configuration')
    parser.add_argument('--seeds', type=int, nargs='+', help='space-separated seeds (losses default: 42)')
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument('--load', dest='load', action='store_true', help='reuse compatible checkpoints (default)')
    reuse.add_argument('--refresh', dest='load', action='store_false', help='recompute the selected cases')
    parser.set_defaults(load=True)
    parser.add_argument('--output-dir', type=Path, help='override output root; fingerprints keep configurations separate')
    parser.add_argument('--report-only', '--cache-only', dest='report_only', action='store_true',
                        help='losses only: recover available values and report missing cases without training')
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='losses only: also search for compatible self-contained loss_*.npz checkpoints here')
    parser.add_argument('--legacy-cache', type=Path, default=Path(RESULTS_PATH),
                        help='read-only original sensitivity summary; never supplies coarse L_phy')
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.seeds is not None and len(args.seeds) != len(set(args.seeds)):
        parser.error('--seeds must be distinct')
    if args.seeds is not None and any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in args.seeds):
        parser.error('--seeds must be unsigned 32-bit integers (0 through 4294967295)')
    if (args.report_only or args.checkpoint_dir) and args.study != 'losses':
        parser.error('--report-only and --checkpoint-dir require --study losses')
    if args.report_only and not args.load:
        parser.error('--report-only cannot be combined with --refresh')
    return args


def main(argv=None):
    global FIG_DIR, RESULTS_PATH, HIST_PATH, SEEDS, NOISE_SEEDS
    args = parse_args(argv)
    if args.study == 'losses':
        return study_losses(args)
    if args.seeds is not None:
        SEEDS = NOISE_SEEDS = tuple(args.seeds)
    config = build_config(smoke=args.smoke)
    if args.epochs is not None:
        config['N_EPOCHS'] = args.epochs
    if args.output_dir is not None or args.smoke or args.epochs is not None or args.seeds is not None:
        identity = {'config': config, 'seeds': SEEDS, 'noise_seeds': NOISE_SEEDS,
                    'smoke': args.smoke, 'source': loss_source_identity()}
        suffix = hashlib.sha256(_json(identity).encode()).hexdigest()[:16]
        root = (args.output_dir or Path('saved_models/sensitivity_custom')) / suffix
        FIG_DIR = str(root / 'figures')
        RESULTS_PATH = str(root / 'results.pkl')
        HIST_PATH = str(root / 'ghost_hist.pkl')
    run = ('ab', 'ghost', 'noise') if args.study == 'all' else (args.study,)

    os.makedirs(FIG_DIR, exist_ok=True)
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    cache      = load_cache(RESULTS_PATH)
    hist_cache = load_cache(HIST_PATH)

    if 'ab' in run:
        study_alpha_beta(cache, config, args.load)
    if 'ghost' in run:
        study_ghost(cache, hist_cache, config, args.load)
    if 'noise' in run:
        study_noise(cache, config, args.load)

    print("\nDone. Figures in", FIG_DIR)


if __name__ == "__main__":
    main()
