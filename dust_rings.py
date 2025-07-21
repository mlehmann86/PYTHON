import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters, reconstruct_grid

def read_single_snapshot(path, snapshot, read_dust1dens=False, read_gasenergy=False, read_gasdens=False):
    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    data_arrays = {}
    if read_dust1dens:
        data_arrays['dust1dens'] = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)
    if read_gasenergy:
        data_arrays['gasenergy'] = read_field_file(path, 'gasenergy', snapshot, nx, ny, nz)
    if read_gasdens:
        data_arrays['gasdens'] = read_field_file(path, 'gasdens', snapshot, nx, ny, nz)

    return data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, parameters

def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2,1,0)
    else:
        return np.zeros((ny, nx, nz))

def load_initial_gasenergy(path, nx, ny, nz):
    initial_snapshot = 0
    initial_gasenergy = read_field_file(path, 'gasenergy', initial_snapshot, nx, ny, nz)
    return initial_gasenergy

def extract_metallicity(simulation_name):
    if "Z1dm4" in simulation_name:
        return r'$Z=0.0001$'
    elif "Z1dm3" in simulation_name:
        return r'$Z=0.001$'
    elif "Z1dm2" in simulation_name:
        return r'$Z=0.01$'
    elif "Z2dm2" in simulation_name:
        return r'$Z=0.02$'
    elif "Z3dm2" in simulation_name:
        return r'$Z=0.03$'
    elif "Z5dm2" in simulation_name:
        return r'$Z=0.05$'
    else:
        return r'$Z=?$'

def plot_epsilon_with_vertical_cuts_and_pressure(simulations, snapshots, rmin=0.9, rmax=1.2, fontsize=20):
    # Increase figure size (height) so labels and color bars are not cut off
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 3], hspace=0.1, wspace=0.4)

    # Example list of horizontal shifts (in figure coordinates) for each simulation
    horizontal_shifts = [-0.03, 0.007, 0.047]

    for i, (sim_name, snapshot) in enumerate(zip(simulations, snapshots)):
        base_path = determine_base_path(sim_name)
        data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, parameters = read_single_snapshot(
            base_path, snapshot, read_dust1dens=True, read_gasenergy=True, read_gasdens=True
        )
        dust1dens = data_arrays['dust1dens']
        gasenergy = data_arrays['gasenergy']
        gasdens   = data_arrays['gasdens']
        initial_gasenergy = load_initial_gasenergy(base_path, nx, ny, nz)

        epsilon = dust1dens / gasdens
        epsilon_azimuthal_avg = np.mean(epsilon, axis=0)
        radial_mask = (xgrid >= rmin) & (xgrid <= rmax)
        xgrid_masked = xgrid[radial_mask]
        eps_azim_masked = epsilon_azimuthal_avg[radial_mask, :]

        # Max epsilon location
        y_idx_max, x_idx_max, z_idx_max = np.unravel_index(np.argmax(epsilon), epsilon.shape)
        z_cut = zgrid[z_idx_max]
        eps_planar_cut = epsilon[:, radial_mask, z_idx_max]

        # Color range
        vmin = -4
        vmax = np.log10(np.max(eps_planar_cut))

        # === TOP PANEL ===
        ax1 = fig.add_subplot(gs[0, i])
        im_v = ax1.imshow(
            np.log10(eps_azim_masked.T),
            extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
            aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax
        )
        ax1.set_ylabel(r'$z/r_0$', fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize)
        ax1.set_xticks([])

        # Metallicty label
        ax1.text(0.05, 0.95, extract_metallicity(sim_name),
                 transform=ax1.transAxes, fontsize=fontsize,
                 color='white', ha='left', va='top',
                 bbox=dict(facecolor='black', alpha=0.7))

        # --- Add vertical tick marks at r/r_0=1.0 and 1.1 ---
        from matplotlib.transforms import blended_transform_factory
        tick_positions = [1.0, 1.1]
        trans = blended_transform_factory(ax1.transData, ax1.transAxes)

        tick_positions = [1.0, 1.1]
        for tp in tick_positions:
        # Make the line thick and black, and ensure it draws above other elements
            ax1.plot(
                [tp, tp],            # x from tp to tp
                [-0.02, 0],          # y from -0.15 to 0 in axes coordinates
                transform=trans,
                color='black',
                lw=1,                # thicker line
                clip_on=False,
                zorder=10            # draw on top
            )



        # --- Create a colorbar axis ABOVE ax1 ---
        pos1 = ax1.get_position()  # [x0, y0, width, height] in figure coordinates
        cbar_gap = 0.03            # vertical gap between ax1 and the colorbar
        cbar_height = 0.02         # height of the colorbar
        shift = horizontal_shifts[i]  # per-simulation horizontal shift
        cbar_x = pos1.x0 + shift
        cbar_w = pos1.width
        cbar_y = pos1.y1 + cbar_gap

        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_height])
        cbar_v  = plt.colorbar(im_v, cax=cbar_ax, orientation='horizontal')
        cbar_v.ax.xaxis.set_ticks_position('top')
        cbar_v.ax.xaxis.set_label_position('top')
        cbar_v.set_label(r'$\log_{10}(\epsilon)$', fontsize=fontsize)
        cbar_v.ax.tick_params(labelsize=fontsize)

        # === BOTTOM PANEL ===
        ax2 = fig.add_subplot(gs[1, i])
        im = ax2.imshow(
            np.log10(eps_planar_cut),
            extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
            aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax
        )
        ax2.set_xlabel(r'$r/r_0$', fontsize=fontsize)
        ax2.set_ylabel(r'$\varphi$', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)

        # Overplot pressure gradients
        ax3 = ax2.twinx()
        current_grad = np.gradient(np.mean(gasenergy[:, radial_mask, :], axis=(0, 2)), xgrid_masked)
        initial_grad = np.gradient(np.mean(initial_gasenergy[:, radial_mask, :], axis=(0, 2)), xgrid_masked)
        scaling_factor = 1 / initial_gasenergy[ny // 2, nx // 2, nz // 2]

        ax3.plot(xgrid_masked, scaling_factor*current_grad, color='white', lw=3)
        ax3.plot(xgrid_masked, scaling_factor*current_grad, color='red', lw=2, linestyle='--',
                 label=r'$\mathrm{d}P/\mathrm{d}r$')
        ax3.plot(xgrid_masked, scaling_factor*initial_grad, color='white', lw=3)
        ax3.plot(xgrid_masked, scaling_factor*initial_grad, color='black', lw=2, linestyle='--',
                 label=r'$\mathrm{d}P/\mathrm{d}r$ (initial)')
        ax3.tick_params(axis='y', labelsize=fontsize, colors='red')
        ax3.set_ylim(-10, 0)

    # Legend in the first bottom panel
    handles, labels = ax3.get_legend_handles_labels()
    fig.axes[2].legend(handles, labels, fontsize=fontsize, loc='upper left')

    # Adjust spacing so color bars and labels aren't cut off
    plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.10)

    output_filename = "epsilon_with_vertical_cuts_and_pressure.pdf"
    plt.savefig(output_filename)
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")

# Example usage:
simulations = [
    "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
    "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
    "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
]
snapshots = [35, 38, 121]

plot_epsilon_with_vertical_cuts_and_pressure(simulations, snapshots)
