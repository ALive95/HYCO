"""Alternating and simultaneous HYCO updates for the LWR problem (paper, Section 7.4)."""

import jax
import jax.numpy as jnp
import optax

from physics.conservation_law import lwr_flux, lwr_max_wave_speed
from synthetic.nn import forward
from utils.training import solve_fv_fixed, interp_solution, sample_ghost


def make_optimizers(config):
    def rate(name):
        if config.get('schedule', 'cosine') == 'constant':
            return config[name]
        if config.get('schedule', 'cosine') != 'cosine':
            raise ValueError('Unknown learning-rate schedule.')
        return optax.cosine_decay_schedule(
            config[name], config['n_epochs'], alpha=config.get('decay_alpha', 0.01))
    return optax.adam(rate('lr_syn')), optax.adam(rate('lr_phy'))


def init_state(sp, pp, config, key):
    opt_syn, opt_phy = make_optimizers(config)
    return dict(syn_params=sp, phy_params=pp, st_syn=opt_syn.init(sp),
                st_phy=opt_phy.init(pp), key=key, cycle=jnp.int32(0))


def solve_model(data, pp, fine=False):
    mesh = data.fv_fine if fine else data.fv_coarse
    u0 = data.u0_fine if fine else data.u0_coarse
    steps = data.n_steps_fine if fine else data.n_steps
    dt = data.dt_fine if fine else data.dt
    full = {**pp, 'flux_fn': lwr_flux, 'max_wave_speed_fn': lwr_max_wave_speed}
    return solve_fv_fixed(mesh, u0, full, steps, dt,
                          bc_left=data.bc, bc_right=data.bc)


def make_losses(data, config):
    def losses(sp, pp, tx_g):
        uh = solve_model(data, pp)
        us = forward(sp, data.tx_data)
        up = interp_solution(uh, data.fv_coarse.cell_centers, data.T, data.tx_data)
        ug = interp_solution(uh, data.fv_coarse.cell_centers, data.T, tx_g)
        ls = jnp.mean((us - data.u_data) ** 2)
        lp = jnp.mean((up - data.u_data) ** 2)
        li = jnp.mean((forward(sp, tx_g) - ug) ** 2)
        return ls, lp, li

    def potential(sp, pp, tx_g):
        ls, lp, li = losses(sp, pp, tx_g)
        value = config['alpha'] * ls + config['beta'] * lp + config['gamma_int'] * li
        return value, (ls, lp, li)
    return losses, potential


def make_update_step(data, config, method):
    if method not in ('alternating', 'simultaneous'):
        raise ValueError('Unknown update method.')
    opt_syn, opt_phy = make_optimizers(config)
    _, potential = make_losses(data, config)

    @jax.jit
    def step(state):
        key, kg = jax.random.split(state['key'])
        tx_g = sample_ghost(kg, config['n_ghost'], (0.0, data.T), data.x_range)
        sp, pp = state['syn_params'], state['phy_params']
        if method == 'simultaneous':
            (value, (ls, lp, li)), (gs, gp) = jax.value_and_grad(
                potential, argnums=(0, 1), has_aux=True)(sp, pp, tx_g)
        else:
            (value, (ls, lp, li)), gs = jax.value_and_grad(
                potential, argnums=0, has_aux=True)(sp, pp, tx_g)
        us, st_syn = opt_syn.update(gs, state['st_syn'], sp)
        new_sp = optax.apply_updates(sp, us)
        if method == 'alternating':
            gp = jax.grad(lambda p: potential(new_sp, p, tx_g)[0])(pp)
        up, st_phy = opt_phy.update(gp, state['st_phy'], pp)
        new_pp = optax.apply_updates(pp, up)
        state = dict(syn_params=new_sp, phy_params=new_pp, st_syn=st_syn,
                     st_phy=st_phy, key=key, cycle=state['cycle'] + 1)
        # Stochastic losses refer to the old pair for both methods.
        history = dict(sample_potential_before=value, sample_L_syn_before=ls,
                       sample_L_phy_before=lp, sample_L_int_before=li,
                       v_max=new_pp['v_max'], rho_max=new_pp['rho_max'],
                       cycle=state['cycle'])
        # A per-cycle bound uses the initial density interval.
        endpoint_speed = lwr_max_wave_speed(jnp.min(data.u0_coarse),
                                            jnp.max(data.u0_coarse), new_pp)
        history['cfl_bound'] = endpoint_speed * data.dt / data.fv_coarse.dx
        return state, history
    return step


def make_evaluator(data, config):
    losses, _ = make_losses(data, config)
    norm = jnp.linalg.norm(data.u_eval_ref)

    @jax.jit
    def evaluate(state):
        sp, pp = state['syn_params'], state['phy_params']
        ls, lp, li = losses(sp, pp, data.tx_fixed)
        fine = interp_solution(solve_model(data, pp, fine=True),
                               data.fv_fine.cell_centers, data.T, data.tx_eval)
        coarse_history = solve_model(data, pp)
        coarse = interp_solution(coarse_history, data.fv_coarse.cell_centers,
                                 data.T, data.tx_eval)
        syn = forward(sp, data.tx_eval)
        return dict(
            L_syn=ls, L_phy=lp, L_int=li,
            objective=config['alpha'] * ls + config['beta'] * lp + config['gamma_int'] * li,
            e_p=jnp.sqrt(((pp['v_max'] - 1.0) ** 2 + (pp['rho_max'] - 1.0) ** 2) / 2),
            e_s_phy=jnp.linalg.norm(fine - data.u_eval_ref) / norm,
            e_s_phy_coarse=jnp.linalg.norm(coarse - data.u_eval_ref) / norm,
            e_s_syn=jnp.linalg.norm(syn - data.u_eval_ref) / norm,
            max_courant=jnp.max(lwr_max_wave_speed(coarse_history, coarse_history, pp))
                        * data.dt / data.fv_coarse.dx,
            density_min=jnp.min(coarse_history), density_max=jnp.max(coarse_history),
            v_max=pp['v_max'], rho_max=pp['rho_max'])
    return evaluate
