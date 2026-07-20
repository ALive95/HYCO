import jax
import jax.numpy as jnp
import jax.random as jr


def setup_grid(N):
    """Build physical and spectral grids on [0, 2pi]^2."""
    dx   = 2 * jnp.pi / N
    x1d  = jnp.arange(N) * dx
    x, y = jnp.meshgrid(x1d, x1d, indexing='ij')

    kx1d  = jnp.fft.fftfreq(N) * N
    ky1d  = jnp.fft.rfftfreq(N) * N
    kx, ky = jnp.meshgrid(kx1d, ky1d, indexing='ij')
    ksq   = kx**2 + ky**2

    # 2/3 dealiasing mask
    dealias = ((jnp.abs(kx) < N / 3) & (jnp.abs(ky) < N / 3)).astype(jnp.float64)
    return x, y, kx, ky, ksq, dealias


def spectral_rhs(omega_hat, kx, ky, ksq, nu, dealias):
    """Vorticity RHS: -(u·∇)ω + ν∇²ω in Fourier space."""
    ksq_safe = jnp.where(ksq > 0, ksq, 1.0)
    psi_hat  = jnp.where(ksq > 0, omega_hat / ksq_safe, 0.0 + 0j)   # streamfunction

    # Velocities and vorticity gradients in physical space (dealiased)
    u    = jnp.fft.irfft2(1j * ky * psi_hat   * dealias)
    v    = jnp.fft.irfft2(-1j * kx * psi_hat  * dealias)
    dwdx = jnp.fft.irfft2(1j * kx * omega_hat * dealias)
    dwdy = jnp.fft.irfft2(1j * ky * omega_hat * dealias)

    nl_hat = jnp.fft.rfft2(u * dwdx + v * dwdy)          # nonlinear term
    return -dealias * nl_hat - nu * ksq * omega_hat       # advection + diffusion


def solve(omega0, nu, dt, n_steps, save_every, kx, ky, ksq, dealias):
    """Integrate vorticity equation with RK4. Returns (n_saves+1, N, N) snapshots."""
    omega_hat = jnp.fft.rfft2(omega0)
    n_saves   = n_steps // save_every

    def rhs(wh):
        return spectral_rhs(wh, kx, ky, ksq, nu, dealias)

    def rk4(wh, _):
        k1 = rhs(wh)
        k2 = rhs(wh + 0.5 * dt * k1)
        k3 = rhs(wh + 0.5 * dt * k2)
        k4 = rhs(wh + dt * k3)
        return wh + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), None

    def chunk(wh, _):
        # Advance save_every steps, return physical snapshot
        wh, _ = jax.lax.scan(rk4, wh, None, length=save_every)
        return wh, jnp.fft.irfft2(wh)

    _, snaps = jax.lax.scan(chunk, omega_hat, None, length=n_saves)
    return jnp.concatenate([omega0[None], snaps], axis=0)


def random_vorticity(key, N, k_peak=4, bandwidth=2.0):
    """Random unit-RMS vorticity field peaked at wavenumber k_peak."""
    kx1d  = jnp.fft.fftfreq(N) * N
    ky1d  = jnp.fft.rfftfreq(N) * N
    kx, ky = jnp.meshgrid(kx1d, ky1d, indexing='ij')
    k_mag  = jnp.sqrt(kx**2 + ky**2)

    envelope  = jnp.exp(-0.5 * ((k_mag - k_peak) / bandwidth) ** 2)
    k1, k2    = jr.split(key)
    omega_hat = envelope * (jr.normal(k1, (N, N//2+1)) + 1j * jr.normal(k2, (N, N//2+1)))
    omega     = jnp.fft.irfft2(omega_hat)
    return omega / jnp.std(omega)
