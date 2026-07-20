import jax
import jax.numpy as jnp
import jax.random as jr


# ── Spatial / space-time network (d_in -> 1) ──────────────────────────────────
# Covers NN.py (d_in=2) and synth_net_3d.py (d_in=3, hardcoded).
# Renamed: synth_net_3d.init_params / forward  →  same names, d_in parameter added.

def init_params(key, d_in=2, hidden=256, d_out=1):
    """
    Initialise a 3-layer residual MLP: R^d_in -> R^d_out with He initialisation.

    Args:
        key: JAX PRNG key.
        d_in (int): Input dimension (2 for (t,x) or (x,y); 3 for (x,y,eta)).
        hidden (int): Width of each hidden layer.
        d_out (int): Output dimension (1 for scalar fields; 2 for (u,v)).

    Returns:
        dict: Network parameters.
    """
    keys = jr.split(key, 6)
    s_in = jnp.sqrt(2.0 / d_in)
    s_h  = jnp.sqrt(2.0 / hidden)
    return {
        'W_in':  jr.normal(keys[0], (hidden, d_in))    * s_in,
        'b_in':  jnp.zeros(hidden),
        'W1':    jr.normal(keys[1], (hidden, hidden))   * s_h,
        'b1':    jnp.zeros(hidden),
        'W2':    jr.normal(keys[2], (hidden, hidden))   * s_h,
        'b2':    jnp.zeros(hidden),
        'W_out': jr.normal(keys[3], (d_out, hidden))    * s_h,
        'b_out': jnp.zeros(d_out),
    }


def forward(params, x, activation=jax.nn.relu):
    """
    Evaluate the spatial / space-time MLP.

    Args:
        params (dict): Network parameters from init_params.
        x (jnp.ndarray): (N, d_in) input array.
        activation (callable): Element-wise nonlinearity. Default jax.nn.relu
            (HYCO usage). For PINN consumers that need second derivatives
            (e.g. Laplacian autodiff) pass a smooth activation such as
            jax.nn.tanh.

    Returns:
        jnp.ndarray: (N,) for d_out == 1, else (N, d_out) predicted values.
    """
    h = activation(x @ params['W_in'].T + params['b_in'])
    h = h + activation(h @ params['W1'].T + params['b1'])   # residual
    h = h + activation(h @ params['W2'].T + params['b2'])   # residual
    out = h @ params['W_out'].T + params['b_out']           # (N, d_out)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


# ── Plain (non-residual) MLP ──────────────────────────────────────────────────
# Torch-equivalent of SimpleNeuralModel / GrayScottPINN (Gray-Scott experiment).
# Added because the residual net above (nonnegative-increment ReLU blocks)
# plateaus when fitting the oscillatory Gray-Scott fields; a plain feedforward
# of the same width fits ~3x better under an identical budget.

def init_params_mlp(key, d_in=3, hidden=128, d_out=2, depth=4):
    """
    Initialise a plain feedforward MLP: R^d_in -> R^d_out, `depth` hidden
    layers of width `hidden`, He initialisation.

    Args:
        key: JAX PRNG key.
        d_in (int): Input dimension.
        hidden (int): Width of each hidden layer.
        d_out (int): Output dimension.
        depth (int): Number of hidden layers (torch reference: 4).

    Returns:
        dict: Network parameters with keys W0..W{depth-1}, b0.., W_out, b_out.
    """
    keys = jr.split(key, depth + 1)
    p = {'W0': jr.normal(keys[0], (hidden, d_in)) * jnp.sqrt(2.0 / d_in),
         'b0': jnp.zeros(hidden)}
    s_h = jnp.sqrt(2.0 / hidden)
    for l in range(1, depth):
        p[f'W{l}'] = jr.normal(keys[l], (hidden, hidden)) * s_h
        p[f'b{l}'] = jnp.zeros(hidden)
    p['W_out'] = jr.normal(keys[depth], (d_out, hidden)) * s_h
    p['b_out'] = jnp.zeros(d_out)
    return p


def forward_mlp(params, x, activation=jax.nn.relu):
    """
    Evaluate the plain feedforward MLP (no residual connections).

    Args:
        params (dict): Network parameters from init_params_mlp.
        x (jnp.ndarray): (N, d_in) input array.
        activation (callable): Element-wise nonlinearity (ReLU for HYCO/NN,
            tanh for PINN consumers needing second derivatives).

    Returns:
        jnp.ndarray: (N,) for d_out == 1, else (N, d_out) predicted values.
    """
    h = x
    l = 0
    while f'W{l}' in params:
        h = activation(h @ params[f'W{l}'].T + params[f'b{l}'])
        l += 1
    out = h @ params['W_out'].T + params['b_out']
    return out.squeeze(-1) if out.shape[-1] == 1 else out


# ── Fourier-feature MLP (opt-PINN) ────────────────────────────────────────────
# Random Fourier features (Tancik et al. 2020) counter the spectral bias that
# makes a plain tanh net collapse to smooth fields on pattern-forming PDEs.

def init_params_fourier_mlp(key, n_freq=64, hidden=128, d_out=2, depth=4,
                            sigma_space=3.0, sigma_time=1.0):
    """
    Initialise a Fourier-feature MLP: x -> [sin(2*pi*Bx), cos(2*pi*Bx)] -> MLP.

    The embedding matrix B is drawn once and kept FIXED (stop-gradient in the
    forward pass). Inputs are assumed z-score normalized; sigmas are frequency
    scales in normalized units, with separate values for space and time.

    Args:
        key: JAX PRNG key.
        n_freq (int): Number of random frequencies (embedding dim = 2*n_freq).
        hidden (int): Width of each hidden layer.
        d_out (int): Output dimension.
        depth (int): Number of hidden layers.
        sigma_space (float): Frequency std for the (x, y) columns of B.
        sigma_time (float): Frequency std for the t column of B.

    Returns:
        dict: MLP parameters (see init_params_mlp) plus key 'B' of shape
            (n_freq, 3).
    """
    kx, kt, kmlp = jr.split(key, 3)
    B = jnp.concatenate([jr.normal(kx, (n_freq, 2)) * sigma_space,
                         jr.normal(kt, (n_freq, 1)) * sigma_time], axis=1)
    p = init_params_mlp(kmlp, d_in=2 * n_freq, hidden=hidden,
                        d_out=d_out, depth=depth)
    p['B'] = B
    return p


def forward_fourier_mlp(params, x, activation=jax.nn.tanh):
    """
    Evaluate the Fourier-feature MLP.

    Args:
        params (dict): Parameters from init_params_fourier_mlp.
        x (jnp.ndarray): (N, 3) normalized inputs.
        activation (callable): Hidden-layer nonlinearity (tanh for PINN use).

    Returns:
        jnp.ndarray: (N,) for d_out == 1, else (N, d_out) predictions.
    """
    B = jax.lax.stop_gradient(params['B'])   # embedding is fixed, not trained
    proj = 2.0 * jnp.pi * (x @ B.T)
    emb = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
    return forward_mlp(params, emb, activation=activation)


# ── Latent / time-only network (1 -> r) ───────────────────────────────────────
# From HYCO_latent_1_.py.
# Renamed: init_syn_params  →  init_latent_params
#          forward_syn       →  forward_latent

def init_latent_params(key, r, hidden=128):
    """
    Initialise a 3-layer residual MLP: R^1 -> R^r with He initialisation.
    Used for the latent trajectory model t -> z(t).

    Args:
        key: JAX PRNG key.
        r (int): Latent dimension (POD rank).
        hidden (int): Width of each hidden layer.

    Returns:
        dict: Network parameters.
    """
    keys = jr.split(key, 6)
    s_in = jnp.sqrt(2.0 / 1)
    s_h  = jnp.sqrt(2.0 / hidden)
    return {
        'W_in':  jr.normal(keys[0], (hidden, 1))       * s_in,
        'b_in':  jnp.zeros(hidden),
        'W1':    jr.normal(keys[1], (hidden, hidden))   * s_h,
        'b1':    jnp.zeros(hidden),
        'W2':    jr.normal(keys[2], (hidden, hidden))   * s_h,
        'b2':    jnp.zeros(hidden),
        'W_out': jr.normal(keys[3], (r, hidden))        * s_h,
        'b_out': jnp.zeros(r),
    }


def forward_latent(params, t, activation=jax.nn.relu):
    """
    Evaluate the latent trajectory model.

    Args:
        params (dict): Network parameters from init_latent_params.
        t (jnp.ndarray): (M,) time values.
        activation (callable): Element-wise nonlinearity. Default jax.nn.relu;
            switch to a smooth activation if a downstream consumer needs
            higher-order derivatives.

    Returns:
        jnp.ndarray: (M, r) latent state predictions.
    """
    x = t[:, None]
    h = activation(x @ params['W_in'].T + params['b_in'])
    h = h + activation(h @ params['W1'].T + params['b1'])   # residual
    h = h + activation(h @ params['W2'].T + params['b2'])   # residual
    return h @ params['W_out'].T + params['b_out']