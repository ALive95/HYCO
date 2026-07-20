import jax
import jax.numpy as jnp
from physics.ns_spectral import spectral_rhs


def build_pod(snapshots, r):
    """SVD-based POD. Returns (mean, basis, sigma) with basis: (N^2, r)."""
    S, N, _ = snapshots.shape
    data     = snapshots.reshape(S, -1)
    mean     = jnp.mean(data, axis=0)
    centered = data - mean[None, :]
    _, sigma, Vt = jnp.linalg.svd(centered, full_matrices=False)
    basis = Vt[:r].T                  # (N^2, r) — columns are POD modes
    return mean, basis, sigma[:r]


def encode(omega, mean, basis):
    """Project single field (N,N) to latent coords (r,)."""
    return basis.T @ (omega.ravel() - mean)


def decode(z, mean, basis, N):
    """Reconstruct field (N,N) from latent coords (r,)."""
    return (mean + basis @ z).reshape(N, N)


def encode_batch(snapshots, mean, basis):
    """Project snapshot array (S,N,N) to latent trajectories (S,r)."""
    S    = snapshots.shape[0]
    data = snapshots.reshape(S, -1) - mean[None, :]
    return data @ basis


def latent_rhs(z, mean, basis, kx, ky, ksq, nu, dealias, N):
    """Project NS RHS onto POD basis: dz/dt = basis^T * RHS(decode(z))."""
    omega     = decode(z, mean, basis, N)
    omega_hat = jnp.fft.rfft2(omega)
    rhs_hat   = spectral_rhs(omega_hat, kx, ky, ksq, nu, dealias)
    rhs_phys  = jnp.fft.irfft2(rhs_hat).ravel()
    return basis.T @ rhs_phys


def solve_rom(z0, nu, dt, n_steps, save_every, mean, basis, kx, ky, ksq, dealias, N):
    """Integrate latent ODE with RK4. Returns (n_saves+1, r) trajectory."""
    n_saves = n_steps // save_every

    def rhs(z):
        return latent_rhs(z, mean, basis, kx, ky, ksq, nu, dealias, N)

    def rk4(z, _):
        k1 = rhs(z)
        k2 = rhs(z + 0.5 * dt * k1)
        k3 = rhs(z + 0.5 * dt * k2)
        k4 = rhs(z + dt * k3)
        return z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), None

    def chunk(z, _):
        z, _ = jax.lax.scan(rk4, z, None, length=save_every)
        return z, z

    _, z_snaps = jax.lax.scan(chunk, z0, None, length=n_saves)
    return jnp.concatenate([z0[None], z_snaps], axis=0)
