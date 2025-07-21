#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import subprocess
from matplotlib.gridspec import GridSpec

from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters, reconstruct_grid

def read_single_snapshot(path, snapshot, read_dust1dens=False, read_gasenergy=False, read_gasdens=False):
    """Reads one snapshot of dust/gas fields, returning them plus grid/parameters."""
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
    """Reads a binary .dat file for a single field at a given snapshot."""
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        print(f"Reading file: {file}")
        data = np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny))
        # Transpose so data_arrays['dust1dens'][y, x, z] is consistent:
        return data.transpose(2, 1, 0)
    else:
        print(f"File not found: {file} (returning zeros)")
        return np.zeros((ny, nx, nz))

def load_initial_gasenergy(path, nx, ny, nz):
    """Load the initial (snapshot=0) gasenergy file for reference in dP/dr plots."""
    initial_snapshot = 0
    initial_gasenergy = read_field_file(path, 'gasenergy', initial_snapshot, nx, ny, nz)
    return initial_gasenergy

def extract_metallicity(simulation_name):
    """Returns a LaTeX string for the metallicity, based on the sim name."""
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

def compute_logepsilon_range_for_final_snapshot(sim_name, final_snap, rmin, rmax):
    """
    Read the *final* snapshot for this simulation to determine
    the min and max of log10(epsilon) in the same plane cuts
    used in the plotting. Return (vmin, vmax) for that column.
    """
    base_path = determine_base_path(sim_name)
    # read final snapshot data
    data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, params = read_single_snapshot(
        base_path, final_snap, read_dust1dens=True, read_gasenergy=True, read_gasdens=True
    )
    dust1dens = data_arrays['dust1dens']
    gasdens   = data_arrays['gasdens']
    # If gasdens is missing or zero, fallback:
    if np.all(gasdens == 0):
        print(f"Warning: no valid gasdens in final snapshot {final_snap} for {sim_name}")
        return -4.0, 0.0

    epsilon = dust1dens / gasdens

    # 1) Azimuthally averaged vertical cut
    epsilon_azimuthal_avg = np.mean(epsilon, axis=0)  # shape (nx, nz)
    radial_mask = (xgrid >= rmin) & (xgrid <= rmax)
    eps_vert = epsilon_azimuthal_avg[radial_mask, :]

    # 2) Planar cut at the max-epsilon height
    y_idx, x_idx, z_idx = np.unravel_index(np.argmax(epsilon), epsilon.shape)
    eps_planar = epsilon[:, radial_mask, z_idx]

    # Combine these two arrays to find overall min/max
    combined = np.concatenate([eps_vert.ravel(), eps_planar.ravel()])
    eps_min = np.min(combined)
    eps_max = np.max(combined)

    # Avoid negative or zero in log10
    eps_min = max(eps_min, 1e-12)
    vmin = np.log10(eps_min)
    if eps_max < 1e-12:
        vmax = -12
    else:
        vmax = np.log10(eps_max)

    return vmin, vmax

def plot_epsilon_movie(
    simulations, snapshots,
    rmin=0.9, rmax=1.2, fontsize=20,
    fps=10, movie_filename="movie_output.mp4"
):
    """
    Create a movie by looping over frames from 0 up to max(snapshots).
    For panel i, the snapshot used is min(current_frame, snapshots[i]),
    so once the designated snapshot is reached, it stays fixed.

    The color bar for each column is fixed to the same range [-4,2].

    Finally, calls ffmpeg to combine PNG frames into an MP4 with '-y'
    so it automatically overwrites any existing file, and then 
    scp-transfers it to a remote directory.
    """

    # (Optional) You can still compute the individual ranges from final snapshots,
    # but we will override them below to use the fixed range.
    color_ranges = []
    for sim_name, final_snap in zip(simulations, snapshots):
        vmin, vmax = compute_logepsilon_range_for_final_snapshot(
            sim_name, final_snap, rmin, rmax
        )
        color_ranges.append((vmin, vmax))

    # 2) Determine how many frames we will produce
    max_frame = max(snapshots)
    print(f"Creating {max_frame + 1} frames (0 through {max_frame}).")

    # 3) Generate each frame
    for frame_idx in range(max_frame + 1):
        print(f"=== Rendering frame {frame_idx} / {max_frame} ===")
        fig = plt.figure(figsize=(18, 10))
        gs  = GridSpec(
            3, 3, 
            height_ratios=[1, 0.1, 3], 
            hspace=0.1, wspace=0.4
        )

        last_ax3 = None

        # Loop over columns
        for i, (sim_name, final_snap) in enumerate(zip(simulations, snapshots)):
            # Snap to display: once we exceed final_snap, stay at final_snap
            current_snapshot = min(frame_idx, final_snap)

            # Load data
            base_path = determine_base_path(sim_name)
            data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, params = read_single_snapshot(
                base_path, current_snapshot,
                read_dust1dens=True, read_gasenergy=True, read_gasdens=True
            )
            dust1dens = data_arrays['dust1dens']
            gasenergy = data_arrays['gasenergy']
            gasdens   = data_arrays['gasdens']

            if np.all(gasdens == 0):
                print(f"Missing data for {sim_name} at snapshot={current_snapshot}")
                continue

            epsilon = dust1dens / gasdens

            # --- FIXED COLOR RANGE ---
            # Instead of using the computed range, we override it for all panels.
            col_vmin, col_vmax = -4, 2

            # -- Vertical cut (R–z) (top panel) --
            epsilon_azimuthal_avg = np.mean(epsilon, axis=0)  # shape (nx, nz)
            radial_mask = (xgrid >= rmin) & (xgrid <= rmax)
            eps_vert = epsilon_azimuthal_avg[radial_mask, :]  # shape (nr, nz)

            ax1 = fig.add_subplot(gs[0, i])
            im_v = ax1.imshow(
                np.log10(eps_vert.T),
                extent=[xgrid[radial_mask].min(), xgrid[radial_mask].max(),
                        zgrid.min(), zgrid.max()],
                origin='lower', 
                aspect='auto', 
                cmap='inferno',
                vmin=col_vmin, vmax=col_vmax  # fixed color limits
            )
            # Add a title showing the simulation time (8 * snapshot number) in orbits
            ax1.set_title(f'$t = {8*current_snapshot}\\,\\mathrm{{orbits}}$', fontsize=fontsize)

            ax1.set_ylabel(r'$z/r_0$', fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize)
            ax1.set_xticks([])

            # Label metallicity
            zlabel = extract_metallicity(sim_name)
            ax1.text(
                0.05, 0.95, zlabel,
                transform=ax1.transAxes,
                fontsize=fontsize, 
                color='white',
                ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.7)
            )

            # Horizontal color bar in top panel with fixed ticks
            fixed_ticks = np.linspace(col_vmin, col_vmax, 5)
            cbar_v = fig.colorbar(
                im_v, ax=ax1,
                orientation='horizontal',
                fraction=0.1, pad=0.1,
                ticks=fixed_ticks
            )
            cbar_v.set_label(r'$\log_{10}(\epsilon)$', fontsize=fontsize)
            cbar_v.ax.tick_params(labelsize=fontsize)

            # -- Planar cut (R–phi) at max-epsilon height (bottom panel) --
            y_idx, x_idx, z_idx = np.unravel_index(np.argmax(epsilon), epsilon.shape)
            eps_planar = epsilon[:, radial_mask, z_idx]

            ax2 = fig.add_subplot(gs[2, i])
            im = ax2.imshow(
                np.log10(eps_planar),
                extent=[xgrid[radial_mask].min(), xgrid[radial_mask].max(),
                        ygrid.min(), ygrid.max()],
                origin='lower', 
                aspect='auto', 
                cmap='inferno',
                vmin=col_vmin, vmax=col_vmax  # fixed color limits here as well
            )
            ax2.set_xlabel(r'$r/r_0$', fontsize=fontsize)
            ax2.set_ylabel(r'$\varphi$', fontsize=fontsize)
            ax2.tick_params(axis='both', labelsize=fontsize)

            # Overplot radial pressure gradients on a twin y-axis
            ax3 = ax2.twinx()

            initial_gas = load_initial_gasenergy(base_path, nx, ny, nz)
            curr_grad   = np.gradient(
                np.mean(gasenergy[:, radial_mask, :], axis=(0, 2)),
                xgrid[radial_mask]
            )
            init_grad   = np.gradient(
                np.mean(initial_gas[:, radial_mask, :], axis=(0, 2)),
                xgrid[radial_mask]
            )
            sf = 1.0 / max(1e-12, initial_gas[ny//2, nx//2, nz//2])

            ax3.plot(
                xgrid[radial_mask], sf * curr_grad,
                color='white', lw=3
            )
            ax3.plot(
                xgrid[radial_mask], sf * curr_grad,
                color='red', lw=2, ls='--',
                label=r'$\mathrm{d}P/\mathrm{d}r$'
            )
            ax3.plot(
                xgrid[radial_mask], sf * init_grad,
                color='white', lw=3
            )
            ax3.plot(
                xgrid[radial_mask], sf * init_grad,
                color='black', lw=2, ls='--',
                label=r'$\mathrm{d}P/\mathrm{d}r$ (initial)'
            )

            ax3.tick_params(axis='y', labelsize=fontsize, colors='red')
            ax3.set_ylim(-10, 0)
            last_ax3 = ax3

        # Put the legend in the lower-left panel if available
        if last_ax3 is not None:
            handles, labels = last_ax3.get_legend_handles_labels()
            # The leftmost lower subplot is typically fig.axes[2]
            fig.axes[2].legend(handles, labels, fontsize=fontsize, loc='upper left')

        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08)

        # Save this frame
        frame_filename = f"frame_{frame_idx:04d}.png"
        plt.savefig(frame_filename, dpi=150)
        plt.close(fig)

    # -- Combine into a movie with ffmpeg, using '-y' so it overwrites automatically --
    print("Combining frames into a movie with ffmpeg...")
    subprocess.run([
        "ffmpeg",
        "-y",                # <--- overwrite without asking
        "-framerate", str(fps),
        "-i", "frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        movie_filename
    ], check=True)

    print(f"Movie saved as '{movie_filename}'.")

    # -- Transfer the movie via SCP --
    scp_transfer(movie_filename, "/Users/mariuslehmann/Downloads/Movies", "mariuslehmann")
    print("SCP transfer completed.")


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    simulations = [
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]
    # For production you might use:
    snapshots = [35, 38, 121]
    #snapshots = [3, 4, 5]

    plot_epsilon_movie(
        simulations, snapshots,
        rmin=0.9, rmax=1.2, fontsize=20,
        fps=10, movie_filename="dust_rings.mp4"
    )
