"""
Gray-Scott reaction-diffusion physical model (physics/gray_scott.py).

Standard Gray-Scott system on the periodic unit square Omega = [0,1]^2:

    u_t = D_u * Lap(u) - u*v^2 + F*(1 - u)
    v_t = D_v * Lap(v) + u*v^2 - (F + k)*v

NOTE ON SIGNS: the paper's eq:gray_scott block is written with flipped signs on
the reaction / feed terms, which diverges and is internally inconsistent with the
PINN residual block. This module implements the standard (convergent) form that
produced the figures. Reconcile the tex when paper editing resumes.

Provides:
  - TRUE_PARAMS_GS, DEFAULT_INIT_GS : paper parameter values.
  - initial_condition               : characteristic-function ICs from the paper.
  - rollout                         : explicit-Euler FD time stepping, written as a
                                      chunked, rematerialised lax.scan so the
                                      backward pass through ~thousands of steps is
                                      memory-feasible (the efficiency rewrite).
  - forward_phy                     : convenience wrapper (build IC + rollout).
  - sample_field                    : differentiable bilinear(space)+linear(time)
                                      interpolation of (u,v) at arbitrary (x,y,t).
  - sample_points                   : random (x,y,t) sampler with optional spatial
                                      box / time-window localisation.
  - quadrant_box                    : map a quadrant label to a spatial bounding box.
  - merge_params                    : merge trainable + fixed parameter dicts.
  - generate_reference              : solve once at (n, n_steps) and cache to .npz.
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


# Ground-truth and paper initial guess for the diffusivities.
# Domain side length (square domain [0, DOMAIN_L]^2).
DOMAIN_L = 100.0

# Ground-truth and initial-guess parameters (matching the original GS run on the
# DOMAIN_L = 100 domain).
TRUE_PARAMS_GS    = {'D_u': 0.20, 'D_v': 0.08, 'F': 0.018, 'k': 0.051}
DEFAULT_INIT_GS   = {'D_u': 0.10, 'D_v': 0.05, 'F': 0.018, 'k': 0.051}


def initial_condition(n):
    """
    Initial condition on an n x n periodic grid of [0, DOMAIN_L]^2.

    Background state (u=1, v=0) is the trivial steady state. A central ball of
    radius 0.1*DOMAIN_L is perturbed to (u=0.5, v=0.5) to seed pattern formation.

    Args:
        n (int): Grid resolution per axis.

    Returns:
        tuple: (u0, v0), each jnp.ndarray of shape (n, n).
    """
    xs = (jnp.arange(n) + 0.5) * (DOMAIN_L / n)   # cell-centred periodic grid
    X, Y = jnp.meshgrid(xs, xs, indexing='ij')
    c = 0.5 * DOMAIN_L
    r = 0.1 * DOMAIN_L
    in_ball = (X - c) ** 2 + (Y - c) ** 2 <= r ** 2
    # Background (u=1, v=0) is the trivial steady state; the central perturbation
    # (u=0.5, v=0.5) seeds Gray-Scott pattern formation.
    u0 = jnp.where(in_ball, 0.5, 1.0).astype(jnp.float64)
    v0 = jnp.where(in_ball, 0.5, 0.0).astype(jnp.float64)
    return u0, v0


def _laplacian(f, dx):
    """Periodic 5-point Laplacian of a 2D field (via jnp.roll)."""
    return (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1)
            - 4.0 * f) / dx ** 2


def _euler_step(state, params, dx, dt):
    """One explicit-Euler step of the standard Gray-Scott system."""
    u, v = state
    uvv = u * v * v
    du = params['D_u'] * _laplacian(u, dx) - uvv + params['F'] * (1.0 - u)
    dv = params['D_v'] * _laplacian(v, dx) + uvv - (params['F'] + params['k']) * v
    return (u + dt * du, v + dt * dv)


def rollout(params, u0, v0, dx, dt, n_save, save_every):
    """
    Explicit-Euler FD rollout, returning snapshots including t=0.

    Implemented as an outer lax.scan over n_save chunks, each chunk an inner
    lax.scan of `save_every` steps wrapped in jax.checkpoint. This keeps the
    backward pass (e.g. d/d(D_u,D_v) in the HYCO physical step) memory-bounded by
    the number of saved snapshots rather than the total step count.

    Total steps performed = n_save * save_every.

    Args:
        params (dict): Must contain D_u, D_v, F, k.
        u0, v0 (jnp.ndarray): (n, n) initial fields.
        dx (float): Grid spacing (= 1/n).
        dt (float): Time step.
        n_save (int): Number of saved chunks (snapshots, excluding t=0).
        save_every (int): Euler steps per chunk.

    Returns:
        tuple: (times, U, V) with shapes (n_save+1,), (n_save+1, n, n), (n_save+1, n, n).
    """
    @jax.checkpoint
    def chunk(state, _):
        state, _ = lax.scan(
            lambda s, _: (_euler_step(s, params, dx, dt), None),
            state, None, length=save_every)
        return state, state            # emit (u, v) after each chunk

    (_, _), (Us, Vs) = lax.scan(chunk, (u0, v0), None, length=n_save)
    U = jnp.concatenate([u0[None], Us], axis=0)
    V = jnp.concatenate([v0[None], Vs], axis=0)
    T = n_save * save_every * dt
    times = jnp.linspace(0.0, T, n_save + 1)
    return times, U, V


def forward_phy(params, n, T, n_steps, n_save):
    """
    Build the paper IC and roll the physical model out on an n x n grid.

    Args:
        params (dict): D_u, D_v, F, k.
        n (int): Grid resolution.
        T (float): Final time.
        n_steps (int): Total Euler steps (must be divisible by n_save).
        n_save (int): Number of saved snapshots (excluding t=0).

    Returns:
        tuple: (times, U, V).
    """
    if n_steps % n_save != 0:
        raise ValueError(f"n_steps ({n_steps}) must be divisible by n_save ({n_save}).")
    u0, v0 = initial_condition(n)
    dx = DOMAIN_L / n
    dt = T / n_steps
    return rollout(params, u0, v0, dx, dt, n_save, n_steps // n_save)


def sample_field(U, V, times, dx, n, xyt):
    """
    Interpolate (u, v) at query points: bilinear in (periodic) space, linear in time.

    Differentiable w.r.t. U, V (query coordinates are treated as constants), so it
    can be used inside the HYCO physical step to read u_phy at ghost points.

    Args:
        U, V (jnp.ndarray): (K, n, n) snapshots.
        times (jnp.ndarray): (K,) uniformly spaced snapshot times.
        dx (float): Grid spacing (= 1/n).
        n (int): Grid resolution.
        xyt (jnp.ndarray): (M, 3) query coordinates [x, y, t].

    Returns:
        jnp.ndarray: (M, 2) interpolated [u, v] values.
    """
    x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]

    # Spatial bilinear weights on the cell-centred periodic grid (centres at (i+0.5)*dx).
    gx = x / dx - 0.5
    gy = y / dx - 0.5
    i0 = jnp.floor(gx).astype(jnp.int32)
    j0 = jnp.floor(gy).astype(jnp.int32)
    fx = gx - i0
    fy = gy - j0
    i0m, i1m = i0 % n, (i0 + 1) % n
    j0m, j1m = j0 % n, (j0 + 1) % n

    # Temporal linear weights (uniform times).
    K = times.shape[0]
    dt_save = times[1] - times[0]
    kf = (t - times[0]) / dt_save
    k0 = jnp.clip(jnp.floor(kf).astype(jnp.int32), 0, K - 1)
    k1 = jnp.clip(k0 + 1, 0, K - 1)
    ft = jnp.clip(kf - k0, 0.0, 1.0)

    def _bilinear(W, k):
        return ((1 - fx) * (1 - fy) * W[k, i0m, j0m]
                + fx * (1 - fy) * W[k, i1m, j0m]
                + (1 - fx) * fy * W[k, i0m, j1m]
                + fx * fy * W[k, i1m, j1m])

    def _gather(W):
        return _bilinear(W, k0) * (1 - ft) + _bilinear(W, k1) * ft

    return jnp.stack([_gather(U), _gather(V)], axis=1)


def quadrant_box(region):
    """
    Map a spatial-region label to a bounding box (x0, x1, y0, y1) on [0, DOMAIN_L]^2.

    Args:
        region (str | tuple | None): One of 'full', 'BL', 'BR', 'TL', 'TR', or an
            explicit (x0, x1, y0, y1) tuple, or None (== 'full').

    Returns:
        tuple: (x0, x1, y0, y1).
    """
    if region is None or region == 'full':
        return (0.0, DOMAIN_L, 0.0, DOMAIN_L)
    if isinstance(region, (tuple, list)) and len(region) == 4:
        return tuple(float(r) for r in region)
    h = 0.5 * DOMAIN_L
    boxes = {
        'BL': (0.0, h, 0.0, h),
        'BR': (h, DOMAIN_L, 0.0, h),
        'TL': (0.0, h, h, DOMAIN_L),
        'TR': (h, DOMAIN_L, h, DOMAIN_L),
    }
    if region not in boxes:
        raise ValueError(f"Unknown region {region!r}; expected full/BL/BR/TL/TR or a 4-tuple.")
    return boxes[region]


def sample_points(key, m, t_range, box=(0.0, DOMAIN_L, 0.0, DOMAIN_L)):
    """
    Draw m random (x, y, t) points in a spatial box and time window.

    Args:
        key: JAX PRNG key.
        m (int): Number of points.
        t_range (tuple): (t_min, t_max).
        box (tuple): (x0, x1, y0, y1) spatial bounds.

    Returns:
        jnp.ndarray: (m, 3) points [x, y, t].
    """
    kx, ky, kt = jax.random.split(key, 3)
    x0, x1, y0, y1 = box
    x = jax.random.uniform(kx, (m,), minval=x0, maxval=x1)
    y = jax.random.uniform(ky, (m,), minval=y0, maxval=y1)
    t = jax.random.uniform(kt, (m,), minval=t_range[0], maxval=t_range[1])
    return jnp.stack([x, y, t], axis=1)


def ic_smooth(x, y, width=DOMAIN_L / 64.0):
    """
    Smoothed analytic initial condition (disc), twice differentiable in (x,y).

    Approximates initial_condition()'s sharp disc with a logistic edge of the
    given width (default: one grid cell), for use as a hard constraint in
    PINN-style models where the output must be differentiated.

    Args:
        x (jnp.ndarray): x coordinates (any shape).
        y (jnp.ndarray): y coordinates (same shape as x).
        width (float): Edge smoothing length scale.

    Returns:
        tuple: (u0, v0) arrays with the shape of x.
    """
    c, r0 = 0.5 * DOMAIN_L, 0.1 * DOMAIN_L
    r = jnp.sqrt((x - c) ** 2 + (y - c) ** 2)
    s = jax.nn.sigmoid((r0 - r) / width)     # ~1 inside the disc, ~0 outside
    return 1.0 - 0.5 * s, 0.5 * s


def ghost_grid(n, times):
    """
    Full cell-centred n x n grid at every snapshot time.

    Time coordinates coincide with the physical-model snapshot times, so
    sample_field never interpolates in time (targets are exact FD snapshots,
    matching the reference torch implementation).

    Args:
        n (int): Grid resolution per axis.
        times (jnp.ndarray): (K,) snapshot times.

    Returns:
        jnp.ndarray: (K*n*n, 3) points [x, y, t].
    """
    xs = (jnp.arange(n) + 0.5) * (DOMAIN_L / n)
    X, Y = jnp.meshgrid(xs, xs, indexing='ij')
    x_f, y_f = X.flatten(), Y.flatten()
    out = [jnp.stack([x_f, y_f, jnp.full_like(x_f, t)], axis=1) for t in times]
    return jnp.concatenate(out, axis=0)


def sample_snapshot_points(key, m, times, box=(0.0, DOMAIN_L, 0.0, DOMAIN_L)):
    """
    Draw m ghost points with random spatial locations but time coordinates
    restricted to the given snapshot times (uniform over the snapshots).

    Spatial bilinear interpolation in sample_field is harmless; temporal
    interpolation is not (linear cross-fade of a translating pattern), hence
    the restriction of t to exact snapshot times.

    Args:
        key: JAX PRNG key.
        m (int): Number of points.
        times (jnp.ndarray): (K,) snapshot times.
        box (tuple): (x0, x1, y0, y1) spatial bounds.

    Returns:
        jnp.ndarray: (m, 3) points [x, y, t].
    """
    kx, ky, kt = jax.random.split(key, 3)
    x0, x1, y0, y1 = box
    x = jax.random.uniform(kx, (m,), minval=x0, maxval=x1)
    y = jax.random.uniform(ky, (m,), minval=y0, maxval=y1)
    t = times[jax.random.randint(kt, (m,), 0, times.shape[0])]
    return jnp.stack([x, y, t], axis=1)


def merge_params(trainable, fixed):
    """
    Merge a trainable-parameter dict over a fixed-parameter dict.

    Args:
        trainable (dict): Parameters being optimised (override fixed).
        fixed (dict): Parameters held constant.

    Returns:
        dict: Full parameter dict {D_u, D_v, F, k}.
    """
    return {**fixed, **trainable}


def generate_reference(n=64, T=2000.0, n_steps=5000, n_save=500,
                       params=None, cache_path=None, recompute=False):
    """
    Generate (or load from cache) the reference Gray-Scott solution.

    Solved once at the given resolution / step count and cached to .npz, mirroring
    the per-method checkpointing used by the other mains.

    Args:
        n (int): Reference grid resolution.
        T (float): Final time.
        n_steps (int): Total Euler steps (must be divisible by n_save).
        n_save (int): Number of saved snapshots (excluding t=0).
        params (dict | None): True parameters (defaults to TRUE_PARAMS_GS).
        cache_path (str | None): .npz path; if it exists and recompute is False, load it.
        recompute (bool): Force re-solving even if the cache exists.

    Returns:
        dict: {'times' (K,), 'U' (K,n,n), 'V' (K,n,n), 'n', 'dx', 'T'}.
    """
    params = dict(params) if params is not None else dict(TRUE_PARAMS_GS)

    if cache_path and os.path.exists(cache_path) and not recompute:
        d = np.load(cache_path)
        print(f"Loaded Gray-Scott reference from {cache_path}")
        return {'times': jnp.array(d['times']), 'U': jnp.array(d['U']),
                'V': jnp.array(d['V']), 'n': int(d['n']),
                'dx': float(d['dx']), 'T': float(d['T'])}

    print(f"Solving Gray-Scott reference: {n}x{n} grid, {n_steps} steps, T={T}")
    times, U, V = forward_phy(params, n, T, n_steps, n_save)
    times.block_until_ready()
    out = {'times': times, 'U': U, 'V': V, 'n': n, 'dx': DOMAIN_L / n, 'T': T}

    if cache_path:
        np.savez(cache_path,
                 times=np.array(times), U=np.array(U), V=np.array(V),
                 n=n, dx=DOMAIN_L / n, T=T)
        print(f"Cached reference to {cache_path}")
    return out