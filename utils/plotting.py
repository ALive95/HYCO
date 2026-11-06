"""
All plotting and visualisation functions.
Every function receives data as explicit arguments and saves figures under figures/.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.animation import FuncAnimation, PillowWriter


# Accent colour used to frame the HYCO panels in comparison figures, so the
# proposed method is immediately distinguishable from the competitors.
HYCO_ACCENT = '#d1495b'


def _frame_axes(ax, color=HYCO_ACCENT, lw=3.5):
    """
    Draw a thick coloured border around an axes to visually highlight it.

    Args:
        ax (plt.Axes): Axes to frame.
        color (str): Border colour.
        lw (float): Border line width.

    Returns:
        None
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_zorder(10)


# ── NS (Navier-Stokes / latent HYCO) ─────────────────────────────────────────

def plot_ns_comparison(fields_true, fields_rom, fields_syn,
                       errors_rom, errors_syn,
                       vis_times, nu_found, nu_true, r,
                       save_path='figures/fig_comparison.png'):
    """
    5-row panel: True | HYCO phys | HYCO syn | phys error | syn error.

    Rows 0-2 share one colorbar (vorticity range of the true field).
    Rows 3-4 share one colorbar (absolute error, 0 to emax).

    Args:
        fields_true (list[np.ndarray]): True vorticity fields, each (N, N).
        fields_rom (list[np.ndarray]): ROM-decoded fields, each (N, N).
        fields_syn (list[np.ndarray]): NN-decoded fields, each (N, N).
        errors_rom (list[np.ndarray]): True - ROM, each (N, N).
        errors_syn (list[np.ndarray]): True - NN, each (N, N).
        vis_times (list[float]): Visualisation times.
        nu_found (float): Recovered viscosity.
        nu_true (float): Ground-truth viscosity.
        r (int): POD rank.
        save_path (str): Output file path.
    """
    vmin = min(f.min() for f in fields_true)
    vmax = max(f.max() for f in fields_true)
    emax = max(max(np.abs(e).max() for e in errors_rom),
               max(np.abs(e).max() for e in errors_syn))

    # absolute errors for the error rows
    abs_err_rom = [np.abs(e) for e in errors_rom]
    abs_err_syn = [np.abs(e) for e in errors_syn]

    n_cols = len(vis_times)
    fig = plt.figure(figsize=(16, 16))
    gs  = GridSpec(5, n_cols + 1, width_ratios=[1] * n_cols + [0.05])

    row_labels  = ['True $u$', f'HYCO phys\n$\\nu$={nu_found:.5f}', 'HYCO syn',
                   'Phys |error|', 'Syn |error|']
    fields_rows = [fields_true, fields_rom, fields_syn, abs_err_rom, abs_err_syn]
    vmins       = [vmin, vmin, vmin,    0,    0]
    vmaxs       = [vmax, vmax, vmax, emax, emax]
    cmaps       = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'Reds', 'Reds']

    im_sol = im_err = None
    for row, (label, fields, vlo, vhi, cmap) in enumerate(
            zip(row_labels, fields_rows, vmins, vmaxs, cmaps)):
        for col, (field, tv) in enumerate(zip(fields, vis_times)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(field.T, origin='lower', cmap=cmap,
                           vmin=vlo, vmax=vhi, extent=[0, 2 * np.pi, 0, 2 * np.pi])
            if row == 0:
                ax.set_title(f't = {tv:.1f}', fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row in (3, 4) and col == 0:
                ax.set_xlabel(f'RMS={float(np.std(field)):.3f}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

            if row == 2 and col == n_cols - 1:
                im_sol = im   # anchor for the solution colorbar
            if row == 4 and col == n_cols - 1:
                im_err = im   # anchor for the error colorbar

    # single colorbar spanning solution rows 0-2
    cax_sol = fig.add_subplot(gs[0:3, -1])
    fig.colorbar(im_sol, cax=cax_sol, label='vorticity')

    # single colorbar spanning error rows 3-4
    cax_err = fig.add_subplot(gs[3:5, -1])
    fig.colorbar(im_err, cax=cax_err, label='|error|')

    fig.suptitle(f'HYCO — 2D NS  nu_true={nu_true}  nu_found={nu_found:.5f}  r={r}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ns_history(history, nu_true, r, sigma,
                    save_path='figures/fig_history.png'):
    """
    3-panel: training losses | viscosity convergence | POD spectrum.

    Args:
        history (dict): Training history with keys 'L_syn', 'L_phy', 'L_int_phy', 'nu'.
        nu_true (float): Ground-truth viscosity.
        r (int): POD rank used.
        sigma (np.ndarray): All singular values from build_pod.
        save_path (str): Output file path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].semilogy(history['L_syn'], label='L_syn')
    axes[0].semilogy(history['L_phy'], label='L_phy')
    axes[0].semilogy(history['L_int_phy'], label='L_int')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].set_title('Training losses'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['nu'])
    axes[1].axhline(nu_true, color='r', linestyle='--', label=f'nu_true={nu_true}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('nu')
    axes[1].legend(); axes[1].set_title('Viscosity convergence'); axes[1].grid(True, alpha=0.3)

    r_plot = min(50, len(sigma))
    axes[2].semilogy(np.array(sigma[:r_plot]), 'ko-', markersize=3)
    axes[2].axvline(r - 1, color='r', linestyle='--', alpha=0.6, label=f'r={r}')
    axes[2].set_xlabel('Mode'); axes[2].set_ylabel('Singular value')
    axes[2].legend(); axes[2].set_title('POD spectrum'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ns_enstrophy(vis_times, ens_true, t_rom, ens_rom, nu_found,
                      save_path='figures/fig_enstrophy.png'):
    """
    Enstrophy decay: true snapshots vs ROM trajectory.

    Args:
        vis_times (list[float]): Times of the true snapshots.
        ens_true (list[float]): Enstrophy Z = 0.5 * mean(omega^2) at vis_times.
        t_rom (np.ndarray): Time axis for ROM trajectory.
        ens_rom (list[float]): Enstrophy along ROM trajectory.
        nu_found (float): Recovered viscosity.
        save_path (str): Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.array(vis_times), ens_true, 'ko', markersize=6, label='True (vis times)')
    ax.plot(t_rom, ens_rom, 'b--', label=f'ROM nu={nu_found:.4f}')
    ax.set_xlabel('Time t'); ax.set_ylabel('Enstrophy Z(t)')
    ax.set_title('Enstrophy decay'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ns_basic(fields_true, fields_rom, fields_syn,
                  errors_rom, errors_syn,
                  vis_times, nu_found, nu_true, r, sigma, history,
                  save_path='figures/ns/basic.png'):
    """
    5-row field panel: True | HYCO phys | HYCO syn | phys error | syn error.

    Rows 0-2 share one vorticity colorbar; rows 3-4 share one error colorbar.
    Saves both a PNG (original) and a bilinear-interpolated PDF (HD).

    Args:
        fields_true (list[np.ndarray]): True vorticity fields, each (N, N).
        fields_rom (list[np.ndarray]): ROM-decoded fields, each (N, N).
        fields_syn (list[np.ndarray]): NN-decoded fields, each (N, N).
        errors_rom (list[np.ndarray]): True - ROM, each (N, N).
        errors_syn (list[np.ndarray]): True - NN, each (N, N).
        vis_times (list[float]): Visualisation times.
        nu_found (float): Recovered viscosity.
        nu_true (float): Ground-truth viscosity.
        r (int): POD rank.
        sigma (np.ndarray): All singular values from build_pod.
        history (dict): Training history (unused; kept for call-site compatibility).
        save_path (str): Output file path (PNG); PDF uses same base name.
    """
    vmin = min(f.min() for f in fields_true)
    vmax = max(f.max() for f in fields_true)
    emax = max(max(np.abs(e).max() for e in errors_rom),
               max(np.abs(e).max() for e in errors_syn))
    abs_err_rom = [np.abs(e) for e in errors_rom]
    abs_err_syn = [np.abs(e) for e in errors_syn]

    n_cols = len(vis_times)
    fig = plt.figure(figsize=(4 * n_cols + 1, 4 * 5))
    gs  = GridSpec(5, n_cols + 1, width_ratios=[1] * n_cols + [0.05],
                   hspace=0.04, wspace=0.04)

    row_labels  = ['True $u$', 'HYCO phys', 'HYCO syn',
                   'Phys |error|', 'Syn |error|']
    fields_rows = [fields_true, fields_rom, fields_syn, abs_err_rom, abs_err_syn]
    vmins       = [vmin, vmin, vmin,    0,    0]
    vmaxs       = [vmax, vmax, vmax, emax, emax]
    cmaps       = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'Reds', 'Reds']

    im_sol = im_err = None
    field_ims = []
    for row, (label, fields, vlo, vhi, cmap) in enumerate(
            zip(row_labels, fields_rows, vmins, vmaxs, cmaps)):
        for col, (field, tv) in enumerate(zip(fields, vis_times)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(field.T, origin='lower', cmap=cmap,
                           vmin=vlo, vmax=vhi, extent=[0, 2 * np.pi, 0, 2 * np.pi],
                           interpolation='none')
            field_ims.append(im)
            if row == 0:
                ax.set_title(f't = {tv:.1f}', fontsize=36)
            if col == 0:
                ax.set_ylabel(label, fontsize=36)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 2 and col == n_cols - 1:
                im_sol = im
            if row == 4 and col == n_cols - 1:
                im_err = im

    cax_sol = fig.add_subplot(gs[0:3, -1])
    fig.colorbar(im_sol, cax=cax_sol)
    cax_err = fig.add_subplot(gs[3:5, -1])
    fig.colorbar(im_err, cax=cax_err)

    cb_sol = fig.colorbar(im_sol, cax=cax_sol)
    cb_sol.ax.tick_params(labelsize=28)
    
    cb_err = fig.colorbar(im_err, cax=cax_err)
    cb_err.ax.tick_params(labelsize=28)

    # original PDF (no interpolation)
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"Saved {pdf_path}")

    # HD PDF with bilinear interpolation
    for im_obj in field_ims:
        im_obj.set_interpolation('bilinear')
    pdf_hd_path = save_path.replace('.png', '_hd.pdf')
    plt.savefig(pdf_hd_path, dpi=400, bbox_inches='tight')
    print(f"Saved {pdf_hd_path}")

    plt.close(fig)


def plot_ns_ablation(fields_true, fields_rom_hyco, fields_rom_abl,
                     vis_times, nu_hyco, nu_abl, nu_true,
                     history_hyco, history_abl,
                     save_path='figures/ns/ablation.png'):
    """
    Ablation figure: True | HYCO phys | Ablation phys, errors, and nu histories.

    Field rows 0-2 share one vorticity colorbar.
    Error rows 3-4 share one error colorbar.
    Bottom panel: nu convergence for HYCO vs ablation.

    Args:
        fields_true (list[np.ndarray]): True vorticity fields, each (N, N).
        fields_rom_hyco (list[np.ndarray]): HYCO ROM-decoded fields, each (N, N).
        fields_rom_abl (list[np.ndarray]): Ablation ROM-decoded fields, each (N, N).
        vis_times (list[float]): Visualisation times.
        nu_hyco (float): HYCO recovered viscosity.
        nu_abl (float): Ablation recovered viscosity.
        nu_true (float): Ground-truth viscosity.
        history_hyco (dict): HYCO training history with key 'nu'.
        history_abl (dict): Ablation training history with key 'nu'.
        save_path (str): Output file path.
    """
    vmin     = min(f.min() for f in fields_true)
    vmax     = max(f.max() for f in fields_true)
    err_hyco = [np.abs(ft - fr) for ft, fr in zip(fields_true, fields_rom_hyco)]
    err_abl  = [np.abs(ft - fr) for ft, fr in zip(fields_true, fields_rom_abl)]
    emax     = max(max(e.max() for e in err_hyco), max(e.max() for e in err_abl))

    n_cols = len(vis_times)
    fig    = plt.figure(figsize=(4 * n_cols + 1, 18))
    outer  = GridSpec(2, 1, figure=fig, height_ratios=[5, 1.2], hspace=0.2)
    gs_f   = GridSpecFromSubplotSpec(5, n_cols + 1, subplot_spec=outer[0],
                                     width_ratios=[1] * n_cols + [0.05],
                                     hspace=0.04, wspace=0.04)
    gs_h   = GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    row_labels  = ['True $u$',
                   f'HYCO phys\n$\\nu$={nu_hyco:.5f}',
                   f'Abl phys\n$\\nu$={nu_abl:.5f}',
                   'HYCO |error|', 'Abl |error|']
    fields_rows = [fields_true, fields_rom_hyco, fields_rom_abl, err_hyco, err_abl]
    vmins       = [vmin, vmin, vmin,    0,    0]
    vmaxs       = [vmax, vmax, vmax, emax, emax]
    cmaps       = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'Reds', 'Reds']

    im_sol = im_err = None
    for row, (label, fields, vlo, vhi, cmap) in enumerate(
            zip(row_labels, fields_rows, vmins, vmaxs, cmaps)):
        for col, (field, tv) in enumerate(zip(fields, vis_times)):
            ax = fig.add_subplot(gs_f[row, col])
            im = ax.imshow(field.T, origin='lower', cmap=cmap,
                           vmin=vlo, vmax=vhi, extent=[0, 2 * np.pi, 0, 2 * np.pi])
            if row == 0:
                ax.set_title(f't = {tv:.1f}', fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row in (3, 4) and col == 0:
                ax.set_xlabel(f'RMS={float(np.std(field)):.3f}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 2 and col == n_cols - 1:
                im_sol = im
            if row == 4 and col == n_cols - 1:
                im_err = im

    cax_sol = fig.add_subplot(gs_f[0:3, -1])
    fig.colorbar(im_sol, cax=cax_sol, label='vorticity')
    cax_err = fig.add_subplot(gs_f[3:5, -1])
    fig.colorbar(im_err, cax=cax_err, label='|error|')

    ax_nu = fig.add_subplot(gs_h[0, 0])
    ax_nu.plot(history_hyco['nu'], color='tab:blue',
               label=f'HYCO ($\\nu$={nu_hyco:.5f})')
    ax_nu.plot(history_abl['nu'],  color='tab:orange',
               label=f'Ablation ($\\nu$={nu_abl:.5f})')
    ax_nu.axhline(nu_true, color='r', linestyle='--', label=f'True $\\nu$={nu_true}')
    ax_nu.set_xlabel('Epoch'); ax_nu.set_ylabel('$\\nu$')
    ax_nu.set_title('$\\nu$ convergence: HYCO vs Ablation')
    ax_nu.legend(); ax_nu.grid(True, alpha=0.3)

    fig.suptitle(
        f'Ablation — 2D NS  $\\nu_{{true}}$={nu_true}  '
        f'HYCO: {nu_hyco:.5f}  Abl: {nu_abl:.5f}',
        fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ns_comparison_all(fields_true, vis_times, nu_true, comp_methods,
                           save_path='figures/ns/comparison.png'):
    """
    Multi-method field comparison + nu convergence.

    One field row per method below True; all rows share one vorticity colorbar.
    Bottom panel: nu convergence for all methods.

    Args:
        fields_true (list[np.ndarray]): True vorticity fields, each (N, N).
        vis_times (list[float]): Visualisation times.
        nu_true (float): Ground-truth viscosity.
        comp_methods (list[tuple]): Each entry is
            (label, fields, nu_found, hist_nu, e_nu, t_s, color) where
            fields is list[np.ndarray], hist_nu is list[float] or None,
            e_nu is the |log(nu/nu*)| metric, t_s is wall-clock or None.
        save_path (str): Output file path.
    """
    n_methods = len(comp_methods)
    n_rows_f  = 1 + n_methods   # True row + one row per method
    n_cols    = len(vis_times)

    vmin = min(f.min() for f in fields_true)
    vmax = max(f.max() for f in fields_true)

    fig   = plt.figure(figsize=(4 * n_cols + 1, 4 * n_rows_f + 4))
    outer = GridSpec(2, 1, figure=fig,
                     height_ratios=[n_rows_f, 1.2], hspace=0.2)
    gs_f  = GridSpecFromSubplotSpec(n_rows_f, n_cols + 1, subplot_spec=outer[0],
                                    width_ratios=[1] * n_cols + [0.05],
                                    hspace=0.04, wspace=0.04)
    gs_h  = GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    im_sol = None
    # True row
    for col, (field, tv) in enumerate(zip(fields_true, vis_times)):
        ax = fig.add_subplot(gs_f[0, col])
        im = ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, extent=[0, 2 * np.pi, 0, 2 * np.pi])
        ax.set_title(f't = {tv:.1f}', fontsize=10)
        if col == 0:
            ax.set_ylabel('True $u$', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if col == n_cols - 1:
            im_sol = im

    # One row per method
    for row_idx, (label, fields, nu_found, hist_nu, e_nu, t_s, color) in \
            enumerate(comp_methods, start=1):
        for col, (field, tv) in enumerate(zip(fields, vis_times)):
            ax = fig.add_subplot(gs_f[row_idx, col])
            ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                      vmin=vmin, vmax=vmax, extent=[0, 2 * np.pi, 0, 2 * np.pi])
            if col == 0:
                row_lbl = f'{label}\n$\\nu$={nu_found:.5f}'
                if not np.isnan(e_nu):
                    row_lbl += f'\n|log|={e_nu:.3f}'
                ax.set_ylabel(row_lbl, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    cax = fig.add_subplot(gs_f[:, -1])
    fig.colorbar(im_sol, cax=cax, label='vorticity')

    ax_nu = fig.add_subplot(gs_h[0, 0])
    for label, _, nu_found, hist_nu, e_nu, t_s, color in comp_methods:
        if hist_nu is not None:
            lbl = f'{label} ($\\nu$={nu_found:.5f}'
            if t_s is not None:
                lbl += f', {t_s:.0f}s'
            lbl += ')'
            ax_nu.plot(np.array(hist_nu), color=color, label=lbl)
    ax_nu.axhline(nu_true, color='r', linestyle='--', label=f'True $\\nu$={nu_true}')
    ax_nu.set_xlabel('Epoch'); ax_nu.set_ylabel('$\\nu$')
    ax_nu.set_title('$\\nu$ convergence'); ax_nu.legend(); ax_nu.grid(True, alpha=0.3)

    fig.suptitle(f'Method comparison — 2D NS  $\\nu_{{true}}$={nu_true}', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Conservation law ──────────────────────────────────────────────────────────

from matplotlib.patches import Rectangle


def _st_field(ax, x_grid, t_grid, field, vmin, vmax, cmap, title=None):
    """
    Render a space-time field on a structured (t, x) grid as a smooth pcolormesh.

    Args:
        ax (plt.Axes): Target axes.
        x_grid (np.ndarray): (N_X,) spatial grid.
        t_grid (np.ndarray): (N_T,) time grid.
        field (np.ndarray): (N_T, N_X) field values.
        vmin, vmax (float): Shared colour-scale limits.
        cmap (str): Colormap name.
        title (str | None): Panel title.

    Returns:
        matplotlib.collections.QuadMesh: The mesh (for the colorbar).
    """
    pcm = ax.pcolormesh(x_grid, t_grid, field, shading='gouraud',
                        cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return pcm


def _overlay_data(ax, tx_data=None, data_region=None):
    """
    Overlay observations and the data window on a space-time (x, t) panel.

    Args:
        ax (plt.Axes): Target axes (the Reference panel).
        tx_data (np.ndarray | None): (M,2) observation coordinates [t, x],
            drawn as white x markers.
        data_region (tuple | None): (t_lo, t_hi, x_lo, x_hi) window, drawn as a
            white dashed rectangle.

    Returns:
        None
    """
    if data_region is not None:
        t_lo, t_hi, x_lo, x_hi = data_region
        # Lightly shaded fill makes the observation window read at a glance,
        # with the dashed outline kept on top for a crisp boundary.
        ax.add_patch(Rectangle((x_lo, t_lo), x_hi - x_lo, t_hi - t_lo,
                               facecolor='white', alpha=0.18, edgecolor='none',
                               zorder=3))
        ax.add_patch(Rectangle((x_lo, t_lo), x_hi - x_lo, t_hi - t_lo,
                               fill=False, edgecolor='white', lw=1.2, ls='--',
                               zorder=4))
    if tx_data is not None:
        txd = np.asarray(tx_data)
        ax.scatter(txd[:, 1], txd[:, 0], marker='x', c='white',
                   s=12, linewidths=0.6, zorder=5)


def _param_axis(ax, curves, true_val, init_val, ylabel, hlines=None):
    """
    Plot parameter-convergence curves against true/init reference lines.

    y-limits are fixed to [0, 1]. Methods without an epoch history (e.g. EKI)
    are drawn as dashed horizontal lines at their converged value via `hlines`.

    Args:
        ax (plt.Axes): Target axes.
        curves (list[tuple]): (label, values, color) per method; values is 1D.
        true_val (float): Ground-truth value (black dashed line).
        init_val (float): Initial value (grey dotted line).
        ylabel (str): y-axis label / title stem.
        hlines (list[tuple] | None): (label, value, color) drawn as dashed lines.

    Returns:
        None
    """
    ax.axhline(true_val, color='k', ls='--', lw=1.5, label='True')
    ax.axhline(init_val, color='gray', ls=':', lw=1.0, label='Init')
    for label, values, color in curves:
        ax.plot(np.arange(len(values)), values, color=color, label=label)
    for label, value, color in (hlines or []):
        ax.axhline(value, color=color, ls='--', lw=1.5, label=label)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel('epoch'); ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} convergence', fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)


def plot_claw_basic(x_grid, t_grid, u_ref_grid, u_phy_grid, u_syn_grid,
                    hist_hyco, true_params, init_params,
                    tx_data=None, data_region=None,
                    cmap='jet', save_path='figures/claw/basic.png'):
    """
    LWR basic experiment figure.

    Row 1: Reference, HYCO u_phy, HYCO u_syn (one shared colorbar).
    Row 2: v_max convergence, rho_max convergence (both y in [0, 1]), and the
           physical solution error |u_phy - u_ref| (own colorbar).

    Args:
        x_grid (np.ndarray): (N_X,) spatial grid.
        t_grid (np.ndarray): (N_T,) time grid.
        u_ref_grid (np.ndarray): (N_T, N_X) reference solution.
        u_phy_grid (np.ndarray): (N_T, N_X) HYCO physical solution.
        u_syn_grid (np.ndarray): (N_T, N_X) HYCO synthetic solution.
        hist_hyco (dict): HYCO history; hist_hyco['params'] is a list of
            {'v_max', 'rho_max'} dicts, one per recorded epoch.
        true_params (dict): Ground-truth {'v_max', 'rho_max'}.
        init_params (dict): Initial {'v_max', 'rho_max'}.
        cmap (str): Colormap for the solution row.
        save_path (str): Output PNG path.

    Returns:
        None
    """
    fields = [u_ref_grid, u_phy_grid, u_syn_grid]
    titles = ['Reference', r'HYCO $u_{phy}$', r'HYCO $u_{syn}$']
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: solution fields, one shared colorbar.
    pcms = [_st_field(ax, x_grid, t_grid, f, vmin, vmax, cmap, title=t)
            for ax, f, t in zip(axes[0], fields, titles)]
    _overlay_data(axes[0, 0], tx_data, data_region)
    fig.colorbar(pcms[0], ax=list(axes[0]), shrink=0.85)

    # Row 2: parameter convergence + physical-solution error.
    v_hist = np.array([p['v_max']   for p in hist_hyco['params']])
    r_hist = np.array([p['rho_max'] for p in hist_hyco['params']])
    _param_axis(axes[1, 0], [('HYCO', v_hist, 'tab:blue')],
                float(true_params['v_max']), float(init_params['v_max']),
                r'$v_{\max}$')
    _param_axis(axes[1, 1], [('HYCO', r_hist, 'tab:blue')],
                float(true_params['rho_max']), float(init_params['rho_max']),
                r'$\rho_{\max}$')

    err = np.abs(u_phy_grid - u_ref_grid)
    pcm_e = _st_field(axes[1, 2], x_grid, t_grid, err, 0.0, float(err.max()),
                      'Reds', title=r'$|u_{phy} - u_{ref}|$')
    fig.colorbar(pcm_e, ax=axes[1, 2], shrink=0.85)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_claw_comparison(x_grid, t_grid, row1_fields, param_curves,
                         true_params, init_params, param_hlines=None,
                         tx_data=None, data_region=None,
                         cmap='jet', clim=None, save_path='figures/claw/comparison.png'):
    """
    LWR method-comparison figure.

    Row 1: one space-time panel per entry in row1_fields (Reference, HYCO u_phy,
           HYCO u_syn, PINN, EKI, HYCO-abl u_syn), all sharing one colorbar.
    Row 2: v_max convergence (left half) and rho_max convergence (right half),
           column-aligned with the field panels above; overlaying every method
           with an epoch history; methods in param_hlines (e.g. EKI) appear as
           dashed horizontal lines.

    Args:
        x_grid (np.ndarray): (N_X,) spatial grid.
        t_grid (np.ndarray): (N_T,) time grid.
        row1_fields (list[tuple]): (title, grid) per panel, in display order;
            grid has shape (N_T, N_X). The first entry sets the default colour
            scale.
        param_curves (list[tuple]): (label, v_hist, rho_hist, color) per method
            with an epoch history.
        true_params (dict): Ground-truth {'v_max', 'rho_max'}.
        init_params (dict): Initial {'v_max', 'rho_max'}.
        param_hlines (list[tuple] | None): (label, v_val, rho_val, color) for
            methods drawn as dashed converged-value lines (e.g. EKI).
        cmap (str): Colormap for row 1.
        clim (tuple | None): (vmin, vmax) for row 1. If None, taken from the
            Reference field (row1_fields[0]); out-of-range panels saturate.
        save_path (str): Output PNG path.

    Returns:
        None
    """
    n = len(row1_fields)
    ref_field = row1_fields[0][1]
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = float(ref_field.min()), float(ref_field.max())
    all_min = min(f.min() for _, f in row1_fields)
    all_max = max(f.max() for _, f in row1_fields)
    extend = 'both' if (all_min < vmin or all_max > vmax) else 'neither'

    fig = plt.figure(figsize=(3.2 * n + 0.6, 8))
    # Dedicated colorbar column so row 2 aligns under the field columns.
    gs = GridSpec(2, n + 1, figure=fig, width_ratios=[1] * n + [0.05],
                  height_ratios=[1.0, 0.9], hspace=0.35, wspace=0.3)

    # Row 1: fields, one shared colorbar in its own column.
    pcm = None
    for j, (title, field) in enumerate(row1_fields):
        ax = fig.add_subplot(gs[0, j])
        pcm = _st_field(ax, x_grid, t_grid, field, vmin, vmax, cmap, title=title)
        if j == 0:
            _overlay_data(ax, tx_data, data_region)
        # Highlight the HYCO predictions (physical and synthetic), but leave the
        # ablated HYCO-abl panel unframed.
        if 'HYCO' in title and 'abl' not in title.lower():
            _frame_axes(ax)
    cax = fig.add_subplot(gs[0, n])
    fig.colorbar(pcm, cax=cax, extend=extend)

    # Row 2: parameter convergence, aligned with the field columns.
    hlines = param_hlines or []
    half = n // 2
    ax_v = fig.add_subplot(gs[1, :half])
    ax_r = fig.add_subplot(gs[1, half:n])
    _param_axis(ax_v, [(lbl, v, c) for lbl, v, _, c in param_curves],
                float(true_params['v_max']), float(init_params['v_max']),
                r'$v_{\max}$',
                hlines=[(lbl, v, c) for lbl, v, _, c in hlines])
    _param_axis(ax_r, [(lbl, r, c) for lbl, _, r, c in param_curves],
                float(true_params['rho_max']), float(init_params['rho_max']),
                r'$\rho_{\max}$',
                hlines=[(lbl, r, c) for lbl, _, r, c in hlines])

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")










# ── Helmholtz ─────────────────────────────────────────────────────────────────


"""
Helmholtz plotting functions for utils/plotting.py.

These do not exist in the current utils/plotting.py (confirmed by the
ImportError on plot_helmholtz_basic) — append them to that file, alongside
its existing plot_helmholtz_params / plot_helmholtz_coefficients /
plot_helmholtz_solutions etc. Do not use this as a full replacement for
utils/plotting.py, it only contains what's below.
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from physics.helmholtz_fem import kappa, eta

TITLE_FONTSIZE = 15  # bumped up from default for per-panel titles


def _blob_field(ax, fem_data, values, vmin, vmax, cmap='RdBu_r', title=None):
    """
    Render a nodal field on the blob mesh as a smooth tripcolor plot.

    Args:
        ax (plt.Axes): Target axes.
        fem_data (FEMData): Mesh data (nodes, elements).
        values (np.ndarray): (n_nodes,) nodal field to plot.
        vmin, vmax (float): Shared color scale limits.
        cmap (str): Colormap name.
        title (str | None): Panel title.

    Returns:
        matplotlib.collections.TriMesh: The plotted mesh collection (for colorbar).
    """
    triang = mtri.Triangulation(np.array(fem_data.nodes[:, 0]),
                                np.array(fem_data.nodes[:, 1]),
                                np.array(fem_data.elements))
    tpc = ax.tripcolor(triang, np.array(values), shading='gouraud',
                       cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return tpc


def plot_helmholtz_basic(fem_data, u_ref, u_hyco_phy, u_hyco_syn, sensor_xy,
                         phy_hyco, true_params, hist_hyco, save_path):
    """
    Plot the Helmholtz basic experiment: recovered solution fields and
    recovered kappa/eta coefficients against ground truth.

    Laid out in (True, HYCO, |error|) columns. The solution occupies two rows,
    the physical branch (u_phy) then the synthetic branch (u_syn), with the
    unused (synthetic-row, True) cell left blank; kappa and eta follow on one row
    each. Each row shares a single colorbar (the two solution rows share one), so
    every error map is drawn on the exact same scale and colormap as the fields
    it is compared against.

    Args:
        fem_data (FEMData): Fine mesh data used for plotting.
        u_ref (np.ndarray): (n_nodes,) reference solution.
        u_hyco_phy (np.ndarray): (n_nodes,) HYCO physical prediction.
        u_hyco_syn (np.ndarray): (n_nodes,) HYCO synthetic prediction.
        sensor_xy (np.ndarray): (M,2) sensor locations.
        phy_hyco (dict): Recovered {'alpha1','c1','alpha2','c2'}.
        true_params (dict): Ground-truth {'alpha1','c1','alpha2','c2'}.
        hist_hyco (dict): Training history, unused here (kept for API symmetry).
        save_path (str): Output PNG path.

    Returns:
        None
    """
    u_err_phy = np.abs(u_hyco_phy - u_ref)
    u_err_syn = np.abs(u_hyco_syn - u_ref)
    nodes = np.array(fem_data.nodes)
    kappa_true = np.array(kappa(nodes, true_params['alpha1'], true_params['c1']))
    kappa_hyco = np.array(kappa(nodes, phy_hyco['alpha1'], phy_hyco['c1']))
    eta_true   = np.array(eta(nodes, true_params['alpha2'], true_params['c2']))
    eta_hyco   = np.array(eta(nodes, phy_hyco['alpha2'], phy_hyco['c2']))
    kappa_err  = np.abs(kappa_hyco - kappa_true)
    eta_err    = np.abs(eta_hyco - eta_true)

    # 4 rows x 3 panel columns (True | HYCO | |error|) plus a thin colorbar
    # column. The solution spans rows 0-1 (physical then synthetic branch); the
    # (row 1, col 0) cell is left empty. Equal-width panel columns keep the
    # panels uniform and aligned.
    fig = plt.figure(figsize=(12, 15))
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1, 1, 1, 0.06],
                  wspace=0.08, hspace=0.2)

    # Rows 0-1: solution, one shared colorbar spanning both rows.
    u_vmax = max(np.max(np.abs(f)) for f in (u_ref, u_hyco_phy, u_hyco_syn))
    ax_u_true = fig.add_subplot(gs[0, 0])
    ax_u_phy  = fig.add_subplot(gs[0, 1])
    ax_u_syn  = fig.add_subplot(gs[1, 1])
    tpc_u = _blob_field(ax_u_true, fem_data, u_ref, -u_vmax, u_vmax, title=r'True $u$')
    _blob_field(ax_u_phy, fem_data, u_hyco_phy, -u_vmax, u_vmax, title=r'HYCO $u_{phy}$')
    _blob_field(fig.add_subplot(gs[0, 2]), fem_data, u_err_phy, -u_vmax, u_vmax,
                title=r'$|u_{phy} - u^\star|$')
    _blob_field(ax_u_syn, fem_data, u_hyco_syn, -u_vmax, u_vmax, title=r'HYCO $u_{syn}$')
    _blob_field(fig.add_subplot(gs[1, 2]), fem_data, u_err_syn, -u_vmax, u_vmax,
                title=r'$|u_{syn} - u^\star|$')
    for ax in (ax_u_true, ax_u_phy, ax_u_syn):
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], marker='x', color='k', s=20)
    fig.colorbar(tpc_u, cax=fig.add_subplot(gs[0:2, 3]))

    # Row 2: kappa; the error map reuses the field scale and colormap.
    k_vmin = min(kappa_true.min(), kappa_hyco.min())
    k_vmax = max(kappa_true.max(), kappa_hyco.max())
    tpc_k = None
    for c, (f, t) in enumerate([(kappa_true, r'True $\kappa$'),
                                (kappa_hyco, r'HYCO $\kappa$'),
                                (kappa_err, r'$|\kappa - \kappa^\star|$')]):
        tpc_k = _blob_field(fig.add_subplot(gs[2, c]), fem_data, f, k_vmin, k_vmax,
                            cmap='viridis', title=t)
    fig.colorbar(tpc_k, cax=fig.add_subplot(gs[2, 3]))

    # Row 3: eta; the error map reuses the field scale and colormap.
    e_vmin = min(eta_true.min(), eta_hyco.min())
    e_vmax = max(eta_true.max(), eta_hyco.max())
    tpc_e = None
    for c, (f, t) in enumerate([(eta_true, r'True $\eta$'),
                                (eta_hyco, r'HYCO $\eta$'),
                                (eta_err, r'$|\eta - \eta^\star|$')]):
        tpc_e = _blob_field(fig.add_subplot(gs[3, c]), fem_data, f, e_vmin, e_vmax,
                            cmap='plasma', title=t)
    fig.colorbar(tpc_e, cax=fig.add_subplot(gs[3, 3]))

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_helmholtz_comparison(fem_data, u_ref, true_params, sensor_xy,
                              comp_methods, save_path,
                              u_bounds=(-0.2,0.2), kappa_bounds=None, eta_bounds=None):
    """
    Plot the Helmholtz comparison: recovered solution and coefficient fields
    for HYCO against all competing methods.

    Each row (u, kappa, eta) shares a single colorbar across all panels,
    including the ground truth.

    Args:
        fem_data (FEMData): Fine mesh data used for plotting.
        u_ref (np.ndarray): (n_nodes,) reference solution.
        true_params (dict): Ground-truth {'alpha1','c1','alpha2','c2'}.
        sensor_xy (np.ndarray): (M,2) sensor locations.
        comp_methods (list[tuple]): (label, u_pred, phy_params, e_s, e_p, t_s) per method.
        save_path (str): Output PNG path.
        u_bounds (tuple | None): (vmin, vmax) for row 1; defaults to
            symmetric ±max(|u|) over all panels if None.
        kappa_bounds (tuple | None): (vmin, vmax) for row 2; defaults to
            (min, max) over all panels if None.
        eta_bounds (tuple | None): (vmin, vmax) for row 3; defaults to
            (min, max) over all panels if None.

    Returns:
        None
    """
    nodes = np.array(fem_data.nodes)
    kappa_true = np.array(kappa(nodes, true_params['alpha1'], true_params['c1']))
    eta_true   = np.array(eta(nodes, true_params['alpha2'], true_params['c2']))
    u_fields = [u_ref] + [u_pred for _, u_pred, *_ in comp_methods]
    kappas = [kappa_true] + [np.array(kappa(nodes, phy['alpha1'], phy['c1']))
                             for _, _, phy, *_ in comp_methods]
    etas = [eta_true] + [np.array(eta(nodes, phy['alpha2'], phy['c2']))
                         for _, _, phy, *_ in comp_methods]

    n_cols = 1 + len(comp_methods)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 13))

    # Row 1: solution fields, one shared colorbar.
    if u_bounds is not None:
        u_vmin, u_vmax = u_bounds
    else:
        u_vmax = max(np.max(np.abs(f)) for f in u_fields)
        u_vmin = -u_vmax
    titles = [r'True $u^\star$'] + [
        rf"{label}" + "\n" + rf'$e_s$={es:.3e}' for label, _, _, es, _, _ in comp_methods
    ]
    tpcs_u = [_blob_field(ax, fem_data, f, u_vmin, u_vmax, title=t)
             for ax, f, t in zip(axes[0], u_fields, titles)]
    axes[0, 0].scatter(sensor_xy[:, 0], sensor_xy[:, 1], marker='x', color='k', s=20)
    fig.colorbar(tpcs_u[0], ax=list(axes[0]), shrink=0.85, pad=0.01)

    # Row 2: kappa fields, one shared colorbar.
    k_vmin, k_vmax = kappa_bounds if kappa_bounds is not None else \
        (min(f.min() for f in kappas), max(f.max() for f in kappas))
    k_titles = [r'True $\kappa$'] + [rf'{label} $\kappa$' for label, *_ in comp_methods]
    tpcs_k = [_blob_field(ax, fem_data, f, k_vmin, k_vmax, cmap='viridis', title=t)
             for ax, f, t in zip(axes[1], kappas, k_titles)]
    fig.colorbar(tpcs_k[0], ax=list(axes[1]), shrink=0.85, pad=0.01)

    # Row 3: eta fields, one shared colorbar.
    e_vmin, e_vmax = eta_bounds if eta_bounds is not None else \
        (min(f.min() for f in etas), max(f.max() for f in etas))
    e_titles = [r'True $\eta$'] + [rf'{label} $\eta$' for label, *_ in comp_methods]
    tpcs_e = [_blob_field(ax, fem_data, f, e_vmin, e_vmax, cmap='plasma', title=t)
             for ax, f, t in zip(axes[2], etas, e_titles)]
    fig.colorbar(tpcs_e[0], ax=list(axes[2]), shrink=0.85, pad=0.01)

    # Highlight the HYCO column (offset by 1 for the leading ground-truth column).
    for k, (label, *_) in enumerate(comp_methods):
        if label == 'HYCO':
            for r in range(3):
                _frame_axes(axes[r, 1 + k])

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_helmholtz_ablation(hist_hyco, hist_abl, save_path):
    """
    Plot the Helmholtz ablation: training loss curves for coupled HYCO vs.
    decoupled HYCO-abl (gamma_int=0), split into physical (L1) and synthetic
    (L2) loss panels, matching the phy/syn split used elsewhere.

    Field/coefficient panels are intentionally not shown here — HYCO-abl is
    now a column in plot_helmholtz_comparison, so this figure only shows
    what's unique to the ablation: the effect of the interaction loss on
    convergence.

    Args:
        hist_hyco (dict): Coupled training history, keys L_syn/L_phy/L_int/L1/L2/params.
        hist_abl (dict): Decoupled training history, same keys.
        save_path (str): Output PNG path.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(hist_hyco['L1'], label='HYCO', color='tab:blue')
    axes[0].plot(hist_abl['L1'], label='HYCO-abl', color='tab:orange')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('$L_1$ (phy)')
    axes[0].set_title('Physical loss', fontsize=TITLE_FONTSIZE)
    axes[0].legend(fontsize=9)

    axes[1].plot(hist_hyco['L2'], label='HYCO', color='tab:blue')
    axes[1].plot(hist_abl['L2'], label='HYCO-abl', color='tab:orange')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('$L_2$ (syn)')
    axes[1].set_title('Synthetic loss', fontsize=TITLE_FONTSIZE)
    axes[1].legend(fontsize=9)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)













    
# ── Poisson ───────────────────────────────────────────────────────────────────

def plot_poisson_results(fem, u_ref, u_hyco, f_true_nodes, f_rec_nodes,
                         sensor_xy,
                         save_path='figures/poisson_results.png'):
    """
    2×2 panel: true u | HYCO u | true f | recovered f.

    All fields are nodal (evaluated at fem.nodes) and rendered with tripcolor.
    Sensor locations are overlaid on the solution panels.

    Args:
        fem (FEMData): FEM mesh data.
        u_ref (jnp.ndarray): (n_nodes,) reference Poisson solution.
        u_hyco (jnp.ndarray): (n_nodes,) HYCO-recovered Poisson solution.
        f_true_nodes (jnp.ndarray): (n_nodes,) true forcing evaluated at nodes.
        f_rec_nodes (jnp.ndarray): (n_nodes,) recovered forcing evaluated at nodes.
        sensor_xy (jnp.ndarray): (M, 2) sensor coordinates.
        save_path (str): Output file path.
    """
    nodes = np.array(fem.nodes)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], np.array(fem.elements))

    u_ref  = np.array(u_ref)
    u_hyco = np.array(u_hyco)
    f_true_nodes = np.array(f_true_nodes)
    f_rec_nodes  = np.array(f_rec_nodes)

    vm_u = max(abs(u_ref.min()), abs(u_ref.max()))
    vm_f = max(abs(f_true_nodes.min()), abs(f_true_nodes.max()))

    kw_u = dict(shading='gouraud', cmap='RdBu_r', vmin=-vm_u, vmax=vm_u)
    kw_f = dict(shading='gouraud', cmap='viridis',
                vmin=f_true_nodes.min(), vmax=f_true_nodes.max())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # ── row 0: solutions ──
    for ax, u, title in [
        (axes[0, 0], u_ref,  'True $u$'),
        (axes[0, 1], u_hyco, 'HYCO $u$'),
    ]:
        tpc = ax.tripcolor(tri, u, **kw_u)
        ax.set_title(title)
        fig.colorbar(tpc, ax=ax)
        ax.set_aspect('equal')
        ax.scatter(np.array(sensor_xy[:, 0]), np.array(sensor_xy[:, 1]),
                   c='black', s=18, marker='x', zorder=5, linewidths=0.8)

    # error annotation below the HYCO panel
    u_err = float(np.linalg.norm(u_ref - u_hyco) / (np.linalg.norm(u_ref) + 1e-12))
    axes[0, 1].set_xlabel(f'rel. error = {u_err:.3e}', fontsize=9)

    # ── row 1: forcing ──
    for ax, f, title in [
        (axes[1, 0], f_true_nodes, 'True $f$'),
        (axes[1, 1], f_rec_nodes,  'Recovered $f$'),
    ]:
        tpc = ax.tripcolor(tri, f, **kw_f)
        ax.set_title(title)
        fig.colorbar(tpc, ax=ax)
        ax.set_aspect('equal')

    f_err = float(np.linalg.norm(f_true_nodes - f_rec_nodes)
                  / (np.linalg.norm(f_true_nodes) + 1e-12))
    axes[1, 1].set_xlabel(f'rel. error = {f_err:.3e}', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")

# ── Ablation comparison (HYCO vs uncoupled) ──────────────────────────────────

def plot_ablation(metric_pairs, history_pairs, save_path, title):
    """
    Ablation figure: bar chart of paired metrics + (optional) history-curve overlay.

    Panel layout: 1 panel (bars only) if history_pairs is empty, else 2 panels
      (bars + curves). HYCO is solid, uncoupled is dashed.

    Args:
        metric_pairs (list[tuple]): Each entry (label, value_hyco, value_uncoupled).
            Values are floats, plotted on a log scale.
        history_pairs (list[tuple]): Each entry (label, hyco_array, uncoupled_array).
            Arrays are 1D, length = n_epochs (or any length; x-axis is the index).
            Pass [] to skip the curves panel.
        save_path (str): Output file path.
        title (str): Figure suptitle.
    """
    if history_pairs:
        fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax_bar = plt.subplots(1, 1, figsize=(7, 5))
        ax_hist = None

    # ── Panel 1: paired bar chart ─────────────────────────────────────────────
    n      = len(metric_pairs)
    labels = [p[0] for p in metric_pairs]
    h_vals = np.array([abs(float(p[1])) for p in metric_pairs])
    u_vals = np.array([abs(float(p[2])) for p in metric_pairs])

    # avoid log(0): clamp to a tiny floor
    floor  = max(1e-16, min(h_vals[h_vals > 0].min() if (h_vals > 0).any() else 1e-16,
                            u_vals[u_vals > 0].min() if (u_vals > 0).any() else 1e-16) * 1e-3)
    h_vals = np.clip(h_vals, floor, None)
    u_vals = np.clip(u_vals, floor, None)

    x     = np.arange(n)
    width = 0.38
    ax_bar.bar(x - width / 2, h_vals, width, label='HYCO',      color='#4C72B0')
    ax_bar.bar(x + width / 2, u_vals, width, label='Uncoupled', color='#C44E52')
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax_bar.set_ylabel('|metric|  (log scale)')
    ax_bar.set_title('Final metrics')
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis='y', alpha=0.3, which='both')

    # numeric labels on top of bars (signed original values for clarity)
    for xi, (_, h, u) in zip(x, metric_pairs):
        ax_bar.text(xi - width / 2, h_vals[xi] * 1.05, f'{float(h):.2e}',
                    ha='center', va='bottom', fontsize=7, rotation=90)
        ax_bar.text(xi + width / 2, u_vals[xi] * 1.05, f'{float(u):.2e}',
                    ha='center', va='bottom', fontsize=7, rotation=90)

    # ── Panel 2: history curves (optional) ────────────────────────────────────
    if ax_hist is not None:
        cmap = plt.cm.tab10
        for i, (lbl, h_hist, u_hist) in enumerate(history_pairs):
            c = cmap(i % 10)
            h_arr = np.asarray(h_hist, dtype=float)
            u_arr = np.asarray(u_hist, dtype=float)
            if h_arr.size > 0:
                ax_hist.plot(np.arange(h_arr.size), np.abs(h_arr),
                             color=c, ls='-',  lw=1.5, label=f'HYCO {lbl}')
            if u_arr.size > 0:
                ax_hist.plot(np.arange(u_arr.size), np.abs(u_arr),
                             color=c, ls='--', lw=1.5, label=f'Uncoupled {lbl}')
        ax_hist.set_yscale('log')
        ax_hist.set_xlabel('epoch')
        ax_hist.set_ylabel('|value|  (log scale)')
        ax_hist.set_title('Training history')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3, which='both')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")

# ── Gray-Scott ────────────────────────────────────────────────────────────────
# Append to utils/plotting.py.

def plot_gs_comparison(snap_times, row_fields, save_path, component='u'):
    """
    Gray-Scott comparison grid: one row per method, one column per time snapshot.

    Mirrors the paper figure (fig:GS): rows are HYCO physical, HYCO synthetic,
    NN, PINN, ground truth; columns are time snapshots of the chosen component.

    Args:
        snap_times (sequence[float]): Snapshot times (column labels).
        row_fields (list[tuple[str, np.ndarray]]): Ordered (label, fields) pairs,
            each fields of shape (K, n, n) with K == len(snap_times). Rows with
            fields=None are skipped.
        save_path (str): Output PNG path.
        component (str): Label for the colourbar (e.g. 'u').
    """
    rows = [(lbl, f) for lbl, f in row_fields if f is not None]
    n_rows = len(rows)
    n_cols = len(snap_times)

    # Shared colour scale from the ground-truth row if present, else all rows.
    ref = next((f for lbl, f in rows if lbl.lower().startswith('ground')), None)
    scale_src = ref if ref is not None else np.concatenate([f for _, f in rows])
    vmin, vmax = float(np.min(scale_src)), float(np.max(scale_src))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows),
                             squeeze=False)
    im = None
    for i, (label, fields) in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            im = ax.imshow(np.asarray(fields[j]).T, origin='lower',
                           extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax,
                           cmap='viridis', aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t = {snap_times[j]:.0f}", fontsize=10)
        axes[i][0].set_ylabel(label, fontsize=10)

    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01,
                 label=f"${component}$")
    fig.suptitle("Gray-Scott: model comparison", fontsize=12)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


# ── LWR: alternating vs simultaneous updates (paper Section 7.4) ─────────────
CLAW_ALTERNATION_METRICS = ("e_p", "e_s_phy", "e_s_syn", "objective")


def plot_claw_alternation(runs, seeds, save_path):
    """
    Paired alternating/simultaneous histories over several seeds.

    Faint curves: individual seeds; bold curves: arithmetic means. Panel titles
    report the reduction of the terminal mean (alternating vs simultaneous).

    Args:
        runs (dict): seed -> method -> {'cycle': (n,), metric: (n,)} for
            methods 'alternating' and 'simultaneous' (see alternation_claw.py).
        seeds (sequence[int]): Seeds to plot.
        save_path (str | Path): Output path without extension; .png and .pdf
            are written.

    Returns:
        dict: method -> metric -> (n,) mean history.
    """
    methods = ("alternating", "simultaneous")
    labels = {
        "e_p": r"Parameter error $\mathsf{e}_p$",
        "e_s_phy": r"Physical field error $\mathsf{e}_s^{\mathrm{phy}}$",
        "e_s_syn": r"Synthetic field error $\mathsf{e}_s^{\mathrm{syn}}$",
        "objective": r"Full Loss $\mathsf{L}$",
    }
    colors = {"alternating": "tab:blue", "simultaneous": "tab:orange"}
    cycles = runs[seeds[0]]["alternating"]["cycle"]
    mean = {m: {k: np.mean([runs[s][m][k] for s in seeds], axis=0)
                for k in CLAW_ALTERNATION_METRICS} for m in methods}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for ax, metric in zip(axes.flat, CLAW_ALTERNATION_METRICS):
        for method in methods:
            for seed in seeds:
                ax.plot(runs[seed][method]["cycle"], runs[seed][method][metric],
                        color=colors[method], alpha=0.20, linewidth=0.9)
            ax.plot(cycles, mean[method][metric], color=colors[method], linewidth=2.4,
                    label=method.capitalize())
        reduction = 100 * (1 - mean["alternating"][metric][-1]
                           / mean["simultaneous"][metric][-1])
        ax.set_title(f"{labels[metric]}\nMean reduction: {reduction:+.2f}%")
        ax.set_yscale("log")
        ax.set_ylabel("Relative error" if metric != "objective" else "Full Loss")
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{save_path}.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}.png")
    return mean
