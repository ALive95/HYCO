"""
HYCO on the LWR traffic model: alternating versus simultaneous updates
(paper, Section 7.4).

For every seed, both update rules start from the same network/physical
parameters, see the same observations and ghost points, and use the same
per-block Adam settings; only the order of the two block updates differs:
  - alternating:  synthetic update, then physical update with the new synthetic model;
  - simultaneous: both gradients from the old parameter pair, then both updates.
The fine reference uses a fixed time step, so its snapshots are uniformly
spaced in physical time.

Run:   python alternation_claw.py           # train (cached runs are reused)
Plot:  python alternation_claw.py --plot    # figures/claw/alternation_5000.{png,pdf}
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
import csv
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from physics.conservation_law import make_mesh, precompute_fv
from synthetic.nn import init_params
from utils.training import sample_ghost, interp_solution
from utils.alternation import init_state, make_update_step, make_evaluator, solve_model
from utils.plotting import plot_claw_alternation, CLAW_ALTERNATION_METRICS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'saved_models' / 'claw_alternation'
FIG = ROOT / 'figures' / 'claw'
SEEDS = (42, 33, 17, 23, 2)
METHODS = ('alternating', 'simultaneous')


# ── Configuration (paper, Section 7.4) ────────────────────────────────────────
def build_config(n_epochs):
    """Same problem as main_claw.py; synthetic learning rate 2e-3, cosine decay over n_epochs."""
    return dict(n_epochs=n_epochs, n_data=100, n_ghost=2000, hidden=128,
                n_coarse=50, n_fine=2000,
                T=2.0, x_left=0.0, x_right=3.0, cfl=0.5,
                t_data=0.5, x_data_left=0.0, x_data_right=1.5,
                alpha=1.0, beta=0.0, gamma_int=1.0,
                lr_syn=2e-3, lr_phy=1e-3, schedule='cosine', decay_alpha=0.01,
                n_eval=2000, n_fixed=2000, eval_every=100)


# ── Data setup ────────────────────────────────────────────────────────────────
def setup_data(seed, config):
    key = jr.PRNGKey(seed)
    key, kd = jr.split(key)
    key, ki, kt = jr.split(key, 3)
    _, ke, kf = jr.split(key, 3)
    d = SimpleNamespace(T=config['T'], x_range=(config['x_left'], config['x_right']), bc=0.1)
    for name, cells in [('coarse', config['n_coarse']), ('fine', config['n_fine'])]:
        fv = precompute_fv(*make_mesh(*d.x_range, cells))
        u0 = jnp.where((fv.cell_centers >= 0.5) & (fv.cell_centers <= 1.5), 0.9, 0.1)
        n_steps = int(np.ceil(d.T / (config['cfl'] * float(fv.dx))))
        setattr(d, f'fv_{name}', fv)
        setattr(d, f'u0_{name}', u0)
        setattr(d, 'n_steps' if name == 'coarse' else 'n_steps_fine', n_steps)
        setattr(d, 'dt' if name == 'coarse' else 'dt_fine', d.T / n_steps)
    d.tx_data = sample_ghost(kd, config['n_data'], (0.0, config['t_data']),
                             (config['x_data_left'], config['x_data_right']))
    d.tx_eval = sample_ghost(ke, config['n_eval'], (0.0, d.T), d.x_range)
    d.tx_fixed = sample_ghost(kf, config['n_fixed'], (0.0, d.T), d.x_range)
    true = dict(v_max=jnp.float64(1.0), rho_max=jnp.float64(1.0))
    # Fixed dt makes all reference snapshots uniformly spaced in physical time.
    reference = jax.jit(lambda: solve_model(d, true, fine=True))()
    d.u_data = interp_solution(reference, d.fv_fine.cell_centers, d.T, d.tx_data)
    d.u_eval_ref = interp_solution(reference, d.fv_fine.cell_centers, d.T, d.tx_eval)
    sp = init_params(ki, hidden=config['hidden'])
    pp = dict(v_max=jnp.float64(0.5), rho_max=jnp.float64(0.6))
    return d, init_state(sp, pp, config, kt)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def host(value):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        jax.device_get(value))


def json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f'Cannot encode {type(value).__name__}.')


def save_pickle(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(host(value), stream)
    temporary.replace(path)


def provenance():
    sources = ['alternation_claw.py', 'utils/alternation.py', 'utils/training.py',
               'physics/conservation_law.py', 'synthetic/nn.py']
    return dict(python=sys.version, platform=platform.platform(), jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                               for p in sources})


# ── One method on one seed (resumable) ────────────────────────────────────────
def run_method(folder, data, initial, config, method):
    path = folder / f'{method}.pkl'
    if path.exists():
        with path.open('rb') as stream:
            checkpoint = pickle.load(stream)
        state = jax.tree_util.tree_map(jnp.asarray, checkpoint['state'])
        raw, evaluations = checkpoint['raw_history'], checkpoint['evaluations']
        if checkpoint['status'] == 'complete':
            print(f'  {method}: cached', flush=True)
            return checkpoint
    else:
        state, raw, evaluations = initial, {}, []
    step = make_update_step(data, config, method)
    evaluate = make_evaluator(data, config)
    blocks = {}
    started = time.perf_counter()
    if not evaluations:
        evaluations.append(dict(cycle=0, **{k: float(v) for k, v in evaluate(state).items()}))
    while int(state['cycle']) < config['n_epochs']:
        n = min(config['eval_every'], config['n_epochs'] - int(state['cycle']))
        if n not in blocks:
            blocks[n] = jax.jit(lambda st: jax.lax.scan(lambda s, _: step(s), st, None, length=n))
        state, history = blocks[n](state)
        history = host(history)
        for key, values in history.items():
            raw.setdefault(key, []).extend(np.asarray(values).tolist())
        metrics = {k: float(v) for k, v in evaluate(state).items()}
        evaluations.append(dict(cycle=int(state['cycle']), **metrics))
        finite = all(np.all(np.isfinite(values)) for values in history.values())
        finite = finite and all(np.isfinite(list(metrics.values())))
        positive = min(raw['v_max']) > 0.0 and min(raw['rho_max']) > 0.0
        cfl_max = max(raw['cfl_bound'])
        density_min = min(r['density_min'] for r in evaluations)
        density_max = max(r['density_max'] for r in evaluations)
        valid_cfl = cfl_max <= 1.0 and metrics['max_courant'] <= 1.0
        valid_density = density_min >= 0.1 - 1e-8 and density_max <= 0.9 + 1e-8
        status = 'running'
        if not finite or not positive:
            status = 'invalid_parameters'
        elif not valid_cfl:
            status = 'invalid_cfl'
        elif not valid_density:
            status = 'invalid_density'
        if int(state['cycle']) == config['n_epochs'] and status == 'running':
            status = 'complete'
        checkpoint = dict(state=state, raw_history=raw, evaluations=evaluations,
                          status=status, config=config, method=method,
                          numerical_validity=dict(finite=finite, positive=positive,
                              valid_cfl=valid_cfl, valid_density=valid_density,
                              max_cfl_bound=cfl_max, density_min=density_min, density_max=density_max),
                          elapsed_this_session=time.perf_counter() - started)
        save_pickle(path, checkpoint)
        print(f"  {method} {int(state['cycle']):4d}: ep={metrics['e_p']:.5f} "
              f"es_phy={metrics['e_s_phy']:.5f} es_syn={metrics['e_s_syn']:.5f} "
              f"v={metrics['v_max']:.4f} rho={metrics['rho_max']:.4f}", flush=True)
        if status.startswith('invalid'):
            break
    return host(checkpoint)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_seed(name, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for method, result in runs.items():
        records = result['evaluations']
        for ax, metric in zip(axes, ['e_p', 'e_s_phy', 'e_s_syn']):
            ax.plot([r['cycle'] for r in records], [r[metric] for r in records], label=method)
            ax.set(xlabel='Full update cycles', ylabel=metric)
            ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(name + ' (alternating vs simultaneous)')
    fig.tight_layout()
    fig.savefig(FIG / f'alternation_{name}.png', dpi=180)
    plt.close(fig)


def load_runs(seeds, epochs):
    """Load the completed alternating/simultaneous pair of every seed."""
    runs = {}
    for seed in seeds:
        folder = OUT / f'seed{seed}_epochs{epochs}'
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        if any(summary[m]['status'] != 'complete' or summary[m]['cycle'] != epochs for m in METHODS):
            raise ValueError(f'Seed {seed} is missing a completed method at {epochs} epochs.')
        with (folder / 'evaluations.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        pair = {}
        for method in METHODS:
            history = [row for row in rows if row['method'] == method]
            pair[method] = dict(cycle=np.asarray([int(row['cycle']) for row in history]),
                                **{k: np.asarray([float(row[k]) for row in history])
                                   for k in CLAW_ALTERNATION_METRICS})
        if not np.array_equal(pair['alternating']['cycle'], pair['simultaneous']['cycle']):
            raise ValueError(f'Paired evaluation cycles differ for seed {seed}.')
        runs[seed] = pair
    return runs


def plot_all(seeds, epochs):
    """Paper figure and the mean histories it shows."""
    runs = load_runs(seeds, epochs)
    mean = plot_claw_alternation(runs, seeds, FIG / f'alternation_{epochs}')
    cycles = runs[seeds[0]]['alternating']['cycle']
    with (FIG / f'alternation_{epochs}_mean.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'cycle', *CLAW_ALTERNATION_METRICS])
        for method in METHODS:
            for i, cycle in enumerate(cycles):
                writer.writerow([method, int(cycle), *[mean[method][k][i] for k in CLAW_ALTERNATION_METRICS]])
    for k in CLAW_ALTERNATION_METRICS:
        a, b = mean['alternating'][k][-1], mean['simultaneous'][k][-1]
        print(f'{k}: alternating={a:.6g}, simultaneous={b:.6g}, reduction={100 * (1 - a / b):+.2f}%')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--plot', action='store_true', help='only plot completed runs')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Epoch count must be positive.')
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if args.plot:
        return plot_all(args.seeds, args.epochs)

    config = build_config(args.epochs)
    for seed in args.seeds:
        name = f'seed{seed}_epochs{args.epochs}'
        folder = OUT / name
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists():
            # Existing run: resume it only if it used the same settings.
            stored = json.loads(manifest_path.read_text(encoding='utf-8'))['config']
            if any(stored.get(k) != v for k, v in config.items()):
                raise ValueError(f'{folder} was produced with different settings.')
        else:
            folder.mkdir(parents=True)
            manifest = dict(seed=seed, config=config, provenance=provenance())
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'\n{name}', flush=True)
        data, initial = setup_data(seed, config)
        if not (folder / 'paired_data_and_initial.pkl').exists():
            save_pickle(folder / 'paired_data_and_initial.pkl', dict(data=vars(data), initial=initial))
        runs = {method: run_method(folder, data, initial, config, method) for method in METHODS}
        summary = {method: dict(status=result['status'],
                                numerical_validity=result.get('numerical_validity'),
                                **result['evaluations'][-1])
                   for method, result in runs.items()}
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2, default=json_default), encoding='utf-8')
        with (folder / 'evaluations.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['method', *runs['alternating']['evaluations'][0]])
            writer.writeheader()
            for method, result in runs.items():
                writer.writerows(dict(method=method, **r) for r in result['evaluations'])
        plot_seed(name, runs)
    if set(args.seeds) == set(SEEDS):
        plot_all(args.seeds, args.epochs)


if __name__ == '__main__':
    main()
