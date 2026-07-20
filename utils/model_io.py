import pickle
import numpy as np
import jax.numpy as jnp


# ── NS (Navier-Stokes / latent HYCO) ─────────────────────────────────────────
# Renamed: save_model → save_ns,  load_model → load_ns

def save_ns(path, mean, basis, sigma, syn_params, nu_found, nu_true,
            history, z0, N, T, r, snaps_vis, vis_times, sensor_locs):
    """
    Save all NS-HYCO model state to a .npz file.

    Args:
        path (str): Output path (without .npz extension).
        mean (jnp.ndarray): (N^2,) POD mean.
        basis (jnp.ndarray): (N^2, r) POD modes.
        sigma (jnp.ndarray): (r,) singular values.
        syn_params (dict): Latent NN parameters.
        nu_found (float): Recovered viscosity.
        nu_true (float): Ground-truth viscosity.
        history (dict): Training loss history.
        z0 (jnp.ndarray): (r,) initial latent state.
        N (int): Physical grid size.
        T (float): Final time.
        r (int): POD rank.
        snaps_vis (jnp.ndarray): (n_vis, N, N) true snapshots at vis_times.
        vis_times (list): Visualisation time values.
        sensor_locs (jnp.ndarray): (n_sensors,) flattened spatial sensor indices.
    """
    data = {
        'mean':        np.array(mean),
        'basis':       np.array(basis),
        'sigma':       np.array(sigma),
        'nu_found':    float(nu_found),
        'nu_true':     float(nu_true),
        'z0':          np.array(z0),
        'N':           N,
        'T':           T,
        'r':           r,
        'snaps_vis':   np.array(snaps_vis),
        'vis_times':   np.array(vis_times),
        'sensor_locs': np.array(sensor_locs),
    }
    for k, v in syn_params.items():
        data[f'syn_{k}'] = np.array(v)
    for k, v in history.items():
        data[f'hist_{k}'] = np.array(v)

    np.savez(path, **data)
    print(f"Model saved to {path}.npz")


def load_ns(path):
    """
    Load NS-HYCO model state from a .npz file.

    Args:
        path (str): Path to .npz file (extension optional).

    Returns:
        dict: Loaded model state with keys matching save_ns arguments.
    """
    if not path.endswith('.npz'):
        path = path + '.npz'
    d = np.load(path)

    syn_params = {k[4:]: jnp.array(d[k]) for k in d.files if k.startswith('syn_')}
    history    = {k[5:]: d[k].tolist()   for k in d.files if k.startswith('hist_')}

    return {
        'mean':        jnp.array(d['mean']),
        'basis':       jnp.array(d['basis']),
        'sigma':       jnp.array(d['sigma']),
        'nu_found':    float(d['nu_found']),
        'nu_true':     float(d['nu_true']),
        'z0':          jnp.array(d['z0']),
        'N':           int(d['N']),
        'T':           float(d['T']),
        'r':           int(d['r']),
        'snaps_vis':   jnp.array(d['snaps_vis']),
        'vis_times':   d['vis_times'].tolist(),
        'sensor_locs': jnp.array(d['sensor_locs']),
        'syn_params':  syn_params,
        'history':     history,
    }


# ── Conservation law ──────────────────────────────────────────────────────────

def save_claw(path, syn_params, phy_params, history, elapsed=None):
    """
    Save conservation-law model (syn_params, phy_params, history) to a .pkl file.

    Args:
        path (str): Output path (should end in .pkl).
        syn_params (dict): NN parameters.
        phy_params (dict): Recovered physical parameters.
        history (dict): Training loss history.
        elapsed (float | None): Training wall-clock time in seconds, so tables
            can be regenerated from a checkpoint without retraining.
    """
    with open(path, 'wb') as f:
        pickle.dump({'syn_params': syn_params, 'phy_params': phy_params,
                     'history': history, 'elapsed': elapsed}, f)
    print(f"Model saved to {path}")


def load_claw(path):
    """
    Load conservation-law model from a .pkl file.

    Args:
        path (str): Path to .pkl file.

    Returns:
        tuple: (syn_params, phy_params, history, elapsed). elapsed is None
            for checkpoints saved before wall-clock tracking was added.
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['syn_params'], d['phy_params'], d['history'], d.get('elapsed')


# ── Helmholtz ─────────────────────────────────────────────────────────────────

def save_helmholtz(path, syn_hyco, phy_hyco, hist_hyco,
                   syn_pinn, phy_pinn, hist_pinn,
                   syn_unc=None, phy_unc=None, hist_unc=None):
    """
    Save HYCO, PINN, and (optionally) uncoupled ablation Helmholtz models to a .pkl file.

    Args:
        path (str): Output path (should end in .pkl).
        syn_hyco (dict): HYCO NN parameters.
        phy_hyco (dict): HYCO recovered PDE parameters.
        hist_hyco (dict): HYCO training history.
        syn_pinn (dict): PINN NN parameters.
        phy_pinn (dict): PINN recovered PDE parameters.
        hist_pinn (dict): PINN training history.
        syn_unc (dict or None): Uncoupled NN parameters.
        phy_unc (dict or None): Uncoupled recovered PDE parameters.
        hist_unc (dict or None): Uncoupled training history.
    """
    with open(path, 'wb') as f:
        pickle.dump({
            'syn_hyco': syn_hyco, 'phy_hyco': phy_hyco, 'hist_hyco': hist_hyco,
            'syn_pinn': syn_pinn, 'phy_pinn': phy_pinn, 'hist_pinn': hist_pinn,
            'syn_unc':  syn_unc,  'phy_unc':  phy_unc,  'hist_unc':  hist_unc,
        }, f)
    print(f"Model saved to {path}")


def load_helmholtz(path):
    """
    Load Helmholtz models from a .pkl file.

    Args:
        path (str): Path to .pkl file.

    Returns:
        tuple: (syn_hyco, phy_hyco, hist_hyco, syn_pinn, phy_pinn, hist_pinn,
                syn_unc, phy_unc, hist_unc).
                Uncoupled entries are None if the file predates the ablation run.
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return (d['syn_hyco'], d['phy_hyco'], d['hist_hyco'],
            d['syn_pinn'],  d['phy_pinn'],  d['hist_pinn'],
            d.get('syn_unc'), d.get('phy_unc'), d.get('hist_unc'))




# ── Poisson ───────────────────────────────────────────────────────────────────

def save_poisson(path, syn_params, phy_params, history):
    """
    Save Poisson-HYCO model state to a .pkl file.

    Args:
        path (str): Output path (should end in .pkl).
        syn_params (dict): NN parameters.
        phy_params (dict): {'f_coeff': (N*N,)} recovered forcing coefficients.
        history (dict): Training loss history.
    """
    with open(path, 'wb') as f:
        pickle.dump({'syn_params': syn_params, 'phy_params': phy_params,
                     'history': history}, f)
    print(f"Model saved to {path}")


def load_poisson(path):
    """
    Load Poisson-HYCO model state from a .pkl file.

    Args:
        path (str): Path to .pkl file.

    Returns:
        tuple: (syn_params, phy_params, history).
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['syn_params'], d['phy_params'], d['history']


# ── Gray-Scott ────────────────────────────────────────────────────────────────

def save_gs(path, syn_params, phy_params, history):
    """
    Save a Gray-Scott method's state to a .pkl file.

    Args:
        path (str): Output path (should end in .pkl).
        syn_params (dict | None): NN parameters (None for parameter-only methods).
        phy_params (dict | None): Recovered physical parameters (None for NN).
        history (dict): Training loss / parameter history.
    """
    with open(path, 'wb') as f:
        pickle.dump({'syn_params': syn_params, 'phy_params': phy_params,
                     'history': history}, f)
    print(f"Model saved to {path}")


def load_gs(path):
    """
    Load a Gray-Scott method's state from a .pkl file.

    Args:
        path (str): Path to .pkl file.

    Returns:
        tuple: (syn_params, phy_params, history).
    """
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['syn_params'], d['phy_params'], d['history']