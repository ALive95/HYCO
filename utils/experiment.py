"""
Experiment harness shared by all main_*.py scripts.

Provides:
  - DEFAULT_SEEDS         : project-wide convention (23, 42, 888)
  - time_block            : context manager that records wall-clock time
  - aggregate_seeds       : mean / std across runs for scalar metrics
  - print_aggregated      : pretty-print of aggregated results
  - make_run_tag          : path suffix builder ("", "_seed42", "_a10_b1", ...)
  - sweep_alpha_beta_grid : helper to flatten an (alpha, beta) sweep to a list

The conventions are:
  * Default behaviour is one seed, one (alpha, beta), no path suffix —
    so existing saved_models / figures filenames are preserved.
  * Multi-seed and alpha/beta sweep are opt-in by setting SEEDS / SWEEP_AB
    in the calling script.
"""
import time
from contextlib import contextmanager

import numpy as np


DEFAULT_SEEDS = (23, 42, 888)


@contextmanager
def time_block(name=None):
    """
    Context manager that records elapsed wall-clock time.

    Usage:
        with time_block("training") as t:
            ...
        t.elapsed   # seconds (also printed if name is not None)

    Args:
        name (str | None): Label printed on exit; if None, nothing is printed.

    Yields:
        object: Has an `.elapsed` float attribute set on exit.
    """
    class _T:
        elapsed = 0.0
    t = _T()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - t0
        if name is not None:
            print(f"[time] {name}: {t.elapsed:.2f}s")


def aggregate_seeds(per_seed_results):
    """
    Compute mean and std across seeds for every scalar metric present in all runs.

    Non-scalar entries (arrays, dicts, lists) are silently skipped.

    Args:
        per_seed_results (list[dict]): One dict per seed, mapping metric name to value.

    Returns:
        dict: {metric_name: (mean, std)} only for keys that are scalar in every run.
    """
    if not per_seed_results:
        return {}
    common = set(per_seed_results[0].keys())
    for r in per_seed_results[1:]:
        common &= set(r.keys())

    out = {}
    for k in common:
        vals = [r[k] for r in per_seed_results]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            arr = np.asarray(vals, dtype=float)
            out[k] = (float(arr.mean()), float(arr.std()))
    return out


def print_aggregated(agg, title=None):
    """
    Pretty-print {metric: (mean, std)} as a small table.

    Args:
        agg (dict): Output of aggregate_seeds.
        title (str | None): Optional header.
    """
    if title is not None:
        print(f"\n=== {title} ===")
    if not agg:
        print("  (no scalar metrics aggregated)")
        return
    width = max(len(k) for k in agg)
    for k in sorted(agg):
        m, s = agg[k]
        print(f"  {k:<{width}}  {m:.4e} +/- {s:.2e}")


def make_run_tag(seed=None, alpha=None, beta=None,
                 multi_seed=False, sweep_ab=False,
                 noise=None, sweep_noise=False):
    """
    Build a filename suffix that disambiguates runs.

    Conventions:
      single seed, no sweep -> ""        (preserves default filenames)
      multi-seed            -> "_seed{N}"
      sweep                 -> "_a{ALPHA}_b{BETA}"
      noise sweep           -> "_n{NOISE}"
      combinations concatenate in the order above.

    Floats are formatted with %g to keep paths short.

    Args:
        seed (int | None): Seed value (used iff multi_seed).
        alpha (float | None): Alpha value (used iff sweep_ab).
        beta (float | None): Beta value (used iff sweep_ab).
        multi_seed (bool): Whether this is a multi-seed run.
        sweep_ab (bool): Whether this is part of an alpha/beta sweep.
        noise (float | None): Noise std (used iff sweep_noise).
        sweep_noise (bool): Whether this is part of a noise-level sweep.

    Returns:
        str: Suffix beginning with "_" (or empty string).
    """
    parts = []
    if multi_seed and seed is not None:
        parts.append(f"seed{int(seed)}")
    if sweep_ab and alpha is not None and beta is not None:
        parts.append(f"a{alpha:g}_b{beta:g}")
    if sweep_noise and noise is not None:
        parts.append(f"n{noise:g}")
    return ("_" + "_".join(parts)) if parts else ""


def sweep_alpha_beta_grid(alpha_grid, beta_grid):
    """
    Flatten an (alpha, beta) sweep into a list of (alpha, beta) pairs.

    Args:
        alpha_grid (sequence[float]): Alpha values to sweep.
        beta_grid (sequence[float]): Beta values to sweep.

    Returns:
        list[tuple[float, float]]: All (alpha, beta) combinations, alpha-major.
    """
    return [(float(a), float(b)) for a in alpha_grid for b in beta_grid]