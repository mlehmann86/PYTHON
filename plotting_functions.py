import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import subprocess
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import uniform_filter, uniform_filter1d
from scipy.spatial import Delaunay
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from matplotlib.animation import FuncAnimation


# Custom imports
from data_reader import read_parameters
from data_storage import save_simulation_quantities, scp_transfer
from planet_data import read_torque_data, extract_planet_mass_and_migration
from Gaussian import gaussian

def create_simulation_movie_noz(data_arrays, xgrid, ygrid, zgrid, time_array, output_path,
                                planet=False, r_min=None, r_max=None, plot_mode="standard", IDEFIX=False):
    """
    Creates a movie of the simulation data with flexible radial range.
    (Plot mode is forced to "vorticity" for this example.)
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm

    # Force plot_mode to vorticity
    plot_mode = "vorticity"

    # Set default radial limits
    if r_min is None:
        r_min = np.min(xgrid)
    if r_max is None:
        r_max = np.max(xgrid)

    # Get current array shape
    ny, nx, nz, nsteps = data_arrays['gasdens'].shape
    bytes_per_element = 8  # float64

    # Define threshold based on known working case (safe config)
    ny_ref, nx_ref, nz_ref, nt_ref = 628, 1650, 76, 125
    max_allowed_bytes = ny_ref * nx_ref * nz_ref * nt_ref * bytes_per_element
    max_allowed_gb = max_allowed_bytes / 1024**3

    # Initial estimate
    thinning_factor = 1
    current_bytes = ny * nx * nz * nsteps * bytes_per_element
    current_gb = current_bytes / 1024**3

    print(f"Estimated array size: {current_gb:.2f} GiB (per field)")
    print(f"Threshold based on shape ({ny_ref}, {nx_ref}, {nz_ref}, {nt_ref}) = {max_allowed_gb:.2f} GiB")

    # Adaptive thinning
    while current_bytes > max_allowed_bytes:
        thinning_factor += 1
        thinned_nsteps = nsteps // thinning_factor
        current_bytes = ny * nx * nz * thinned_nsteps * bytes_per_element
        current_gb = current_bytes / 1024**3

    if thinning_factor > 1:
        print(f"⚠️  Estimated size exceeds threshold. Thinning time steps by factor {thinning_factor}")
        for key in data_arrays:
            data_arrays[key] = data_arrays[key][..., ::thinning_factor]
        time_array = time_array[::thinning_factor]
        nsteps = data_arrays['gasdens'].shape[-1]

    # Final printout
    print(f"Radial domain: r_min = {r_min}, r_max = {r_max}")
    print(f"Plot mode: {plot_mode} (co-rotating frame)")
    print(f"Number of time steps (after thinning): {nsteps}")

    # Set matplotlib font sizes locally
    plt.rc('font', size=8)
    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('figure', titlesize=12)

    # Create radial mask (static) and masked grid arrays
    radial_mask = (xgrid >= r_min) & (xgrid <= r_max)
    xgrid_masked = xgrid[radial_mask]

    # For IDEFIX simulations, load parameters from log file
    if IDEFIX:
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)

    # --- Extract or compute data arrays based on simulation dimensionality ---
    if len(zgrid) == 1:  # 2D simulation
        gasdens = np.squeeze(data_arrays['gasdens'])
        if 'gasenergy' in data_arrays:
            gasenergy = np.squeeze(data_arrays['gasenergy'])
        elif IDEFIX:
            print("Computing isothermal gasenergy from h0 and flaringIndex for IDEFIX...")
            h0 = float(parameters.get("h0", 0.05))
            flaring_index = float(parameters.get("flaringIndex", 1.0))
            gamma = float(parameters.get("gamma", 1.6667))
            cs2_profile = h0**2 * xgrid**(2 * flaring_index - 1)
            gasenergy = gasdens * cs2_profile[np.newaxis, :] / (gamma - 1)
        else:
            raise ValueError("❌ Missing 'gasenergy' field and not running IDEFIX.")

        gasvy = np.squeeze(data_arrays['gasvy'])
        if 'gasvx' in data_arrays:
            gasvx = np.squeeze(data_arrays['gasvx'])
            gasvx = gasvx - gasvx[..., 0][..., None]
        else:
            gasvx = None
        if 'gasvz' in data_arrays:
            gasvz = np.squeeze(data_arrays['gasvz'])
        else:
            gasvz = None

    else:  # 3D simulation
        gasdens = np.squeeze(data_arrays['gasdens'][:, :, len(zgrid)//2, :])
        if 'gasenergy' in data_arrays:
            gasenergy = np.squeeze(data_arrays['gasenergy'][:, :, len(zgrid)//2, :])
        elif IDEFIX:
            print("Computing isothermal gasenergy from h0 and flaringIndex for IDEFIX...")
            h0 = float(parameters.get("h0", 0.05))
            flaring_index = float(parameters.get("flaringIndex", 1.0))
            gamma = float(parameters.get("gamma", 1.6667))
            cs2_profile = h0**2 * xgrid**(2 * flaring_index - 1)
            gasenergy = gasdens * cs2_profile[np.newaxis, :] / (gamma - 1)
        else:
            raise ValueError("❌ Missing 'gasenergy' field and not running IDEFIX.")

        gasvy_full = data_arrays['gasvy']
        gasvy = np.squeeze(gasvy_full[:, :, len(zgrid)//2, :])
        if 'gasvx' in data_arrays:
            gasvx_full = data_arrays['gasvx']
            gasvx_full = gasvx_full - gasvx_full[..., 0][..., None]
            gasvx = np.squeeze(gasvx_full[:, :, len(zgrid)//2, :])
        else:
            gasvx = None
            gasvx_full = None
        if 'gasvz' in data_arrays:
            gasvz_full = data_arrays['gasvz']
            gasvz = np.squeeze(gasvz_full[:, :, len(zgrid)//2, :])
        else:
            gasvz = None
            gasvz_full = None

    # Initial gas density and pressure gradient for reference (static)
    initial_gas_density = np.mean(np.squeeze(gasdens[..., 0]), axis=0)
    initial_scaled_rhog = initial_gas_density / initial_gas_density[len(xgrid)//2]
    scal_pg = gasenergy[len(ygrid)//2, len(xgrid)//2, 0]
    initial_pressure_gradient_full = np.mean(np.gradient(gasenergy[..., 0], xgrid, axis=1), axis=0)
    initial_scaled_pg = initial_pressure_gradient_full / scal_pg

    # Load saved quantities (e.g. for time evolution)
    quantities_file = os.path.join(output_path, f"{os.path.basename(output_path)}_quantities.npz")
    loaded_data = np.load(quantities_file)
    alpha_r = loaded_data['alpha_r']
    rms_vr = loaded_data['rms_vr']

    # Read disk aspect ratio (H_g)
    if IDEFIX:
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)
        h0 = float(parameters.get("h0", 0.05))
        H_g = h0
    else:
        summary_file = os.path.join(output_path, "summary0.dat")
        parameters = read_parameters(summary_file)
        aspectratio = float(parameters['ASPECTRATIO'])
        H_g = aspectratio

    # --- Precompute a meshgrid for coordinate derivatives (static) ---
    # Here r corresponds to xgrid and phi to ygrid.
    R_full, _ = np.meshgrid(xgrid, ygrid, indexing='xy')

    # --- Define vectorized vorticity calculation ---
    def calculate_vorticity(vr, vphi, vz, r, phi, z, time_idx, R):
        """
        Vectorized calculation of vorticity components in cylindrical coordinates.
        Uses precomputed R (meshgrid of r and phi).
        """
        if len(z) > 1:
            vr_mid = vr[:, :, len(z)//2, time_idx]
            vphi_mid = vphi[:, :, len(z)//2, time_idx]
            vz_mid = vz[:, :, len(z)//2, time_idx]
        else:
            vr_mid = vr[:, :, time_idx]
            vphi_mid = vphi[:, :, time_idx]
            vz_mid = vz[:, :, time_idx]
        # Compute d(vr)/d(phi)
        dvr_dphi = np.gradient(vr_mid, phi, axis=0)
        # Compute derivative of (r * vphi) with respect to r
        rvphi = R * vphi_mid
        drvphi_dr = np.gradient(rvphi, r, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            omega_z = (1.0 / R) * drvphi_dr - (1.0 / R) * dvr_dphi
            omega_z[R <= 0] = 0
        # For 3D, compute additional components
        if len(z) > 1:
            dvz_dphi = np.gradient(vz_mid, phi, axis=0)
            mid_z = len(z) // 2
            # Choose a small slice around the midplane for z derivatives
            z_slice = slice(max(0, mid_z - 1), min(len(z), mid_z + 2))
            z_subset = zgrid[z_slice]
            # Compute gradients along z in a vectorized way for vphi and vr
            dvphi_dz_full = np.gradient(vphi[:, :, z_slice, time_idx], z_subset, axis=2)
            dvr_dz_full = np.gradient(vr[:, :, z_slice, time_idx], z_subset, axis=2)
            mid_idx = len(z_subset) // 2
            dvphi_dz = dvphi_dz_full[:, :, mid_idx]
            dvr_dz = dvr_dz_full[:, :, mid_idx]
            # Derivative of vz with respect to r at the midplane
            dvz_dr = np.gradient(vz[:, :, mid_z, time_idx], r, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                omega_r = (1.0 / R) * dvz_dphi - dvphi_dz
                omega_r[R <= 0] = 0
            omega_phi = dvr_dz - dvz_dr
        else:
            omega_r = np.zeros_like(omega_z)
            omega_phi = np.zeros_like(omega_z)
        return omega_r, omega_phi, omega_z

    # --- Create figure, axes, and pre-allocate plot objects ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.25)

    # Prepare a secondary y-axis on ax3 (for pressure gradient)
    ax3_right = ax3.twinx()

    # Set fixed positions for colorbars (create axes once)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("top", size="5%", pad=0.3)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("top", size="5%", pad=0.3)

    # --- Precompute initial images / line objects for frame 0 ---
    # For vorticity mode (forced)
    if len(zgrid) > 1 and (gasvx_full is not None) and (gasvz_full is not None):
        try:
            omega_r_init, omega_phi_init, omega_z_init = calculate_vorticity(
                gasvy_full, gasvx_full, gasvz_full, xgrid, ygrid, zgrid, 0, R_full)
            # Apply azimuthal shift (roll) to align features
            omega_z_init_shifted = np.roll(omega_z_init, shift=len(ygrid)//2, axis=0)
            omega_rp_init = np.sqrt(omega_r_init**2 + omega_phi_init**2)
            omega_rp_init_shifted = np.roll(omega_rp_init, shift=len(ygrid)//2, axis=0)
            # Apply the radial mask
            omega_z_mask_init = omega_z_init_shifted[:, radial_mask]
            omega_rp_mask_init = omega_rp_init_shifted[:, radial_mask]
            oz_max_init = max(abs(np.percentile(omega_z_mask_init, 1)),
                              abs(np.percentile(omega_z_mask_init, 99)))
            orp_max_init = np.percentile(omega_rp_mask_init, 99)
        except Exception as e:
            print(f"❌ Initial 3D vorticity calculation error: {e}")
            omega_z_mask_init = np.zeros((len(ygrid), np.sum(radial_mask)))
            omega_rp_mask_init = np.zeros_like(omega_z_mask_init)
            oz_max_init = 1
            orp_max_init = 1
        # Create image objects (created once; will be updated in animate)
        im1 = ax1.imshow(omega_z_mask_init,
                         extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                         aspect='auto', origin='lower', cmap='coolwarm',
                         vmin=-oz_max_init, vmax=oz_max_init)
        im2 = ax2.imshow(omega_rp_mask_init,
                         extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                         aspect='auto', origin='lower', cmap='viridis',
                         vmin=0, vmax=orp_max_init)
        # Add static text labels
        ax1.text(0.02, 0.98, r'$\omega_{z}$ (co-rotating)', transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top', color='black',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax2.text(0.02, 0.98, r'$\sqrt{\omega_{r}^2+\omega_{\phi}^2}$ (co-rotating)', transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', color='black',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    else:
        # 2D simulation branch for vorticity mode
        try:
            omega_r_init, omega_phi_init, omega_z_init = calculate_vorticity(
                gasvy, gasvx, gasvz, xgrid, ygrid, zgrid, 0, R_full)
            omega_z_init_full = np.roll(omega_z_init, shift=len(ygrid)//2, axis=0)
            omega_z_mask_init = omega_z_init_full[:, radial_mask]
            radial_velocity_init_full = np.roll(gasvy[..., 0] / H_g, shift=len(ygrid)//2, axis=0)
            radial_velocity_init_mask = radial_velocity_init_full[:, radial_mask]
            oz_max_init = max(abs(np.percentile(omega_z_mask_init, 1)),
                              abs(np.percentile(omega_z_mask_init, 99)))
        except Exception as e:
            print(f"❌ Initial 2D vorticity calculation error: {e}")
            omega_z_mask_init = np.zeros((len(ygrid), np.sum(radial_mask)))
            radial_velocity_init_mask = np.zeros_like(omega_z_mask_init)
            oz_max_init = 1
        im1 = ax1.imshow(omega_z_mask_init,
                         extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                         aspect='auto', origin='lower', cmap='coolwarm',
                         vmin=-oz_max_init, vmax=oz_max_init)
        im2 = ax2.imshow(radial_velocity_init_mask,
                         extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                         aspect='auto', origin='lower', cmap='coolwarm')
    
    # Create colorbars once (they will be updated via set_clim if necessary)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')

    # --- Precompute initial profiles for bottom-left panel (radial gas density and pressure gradient) ---
    xgrid_for_ax3 = xgrid[radial_mask]
    radial_gas_density_full_init = np.mean(gasdens[..., 0], axis=0)
    radial_gas_density_init = radial_gas_density_full_init[radial_mask]
    scaled_rhog_init = radial_gas_density_init / initial_gas_density[len(xgrid)//2]
    # For pressure gradient, use the precomputed scal_pg from the initial gasenergy slice
    pressure_gradient_full_init = np.mean(np.gradient(gasenergy[..., 0], xgrid, axis=1), axis=0)
    pressure_gradient_init = pressure_gradient_full_init[radial_mask]
    scaled_pg_init = pressure_gradient_init / scal_pg

    # Create line objects for the bottom-left panel
    line_rhog, = ax3.plot(xgrid_for_ax3, scaled_rhog_init, color='red', lw=2, label=r"Current $\rho_{g}$")
    line_rhog_init, = ax3.plot(xgrid_for_ax3, initial_scaled_rhog[radial_mask], color='black', lw=2,
                               linestyle='--', label=r"Initial $\rho_{g}$")
    line_pg, = ax3_right.plot(xgrid_for_ax3, scaled_pg_init, color='blue', lw=2, label=r"$\frac{dP}{dr}$")
    line_pg_init, = ax3_right.plot(xgrid_for_ax3, initial_scaled_pg[radial_mask], color='black', lw=2,
                                   linestyle='--', label=r"Initial $\frac{dP}{dr}$")
    ax3.set_xlim([r_min, r_max])
    ax3.set_ylim([0, 3])
    ax3.set_xlabel(r"$r/r_{0}$")
    ax3.legend(loc='center right', fontsize='small')

    # --- Pre-create line objects for the bottom-right panel (time evolution) ---
    line_alpha, = ax4.plot(time_array[:1], alpha_r[:1], label=r'$\alpha_r(t)$')
    line_rms, = ax4.plot(time_array[:1], rms_vr[:1], label=r'RMS(Radial Velocity)')
    ax4.set_yscale('log')
    ax4.set_xlim([0, np.max(time_array)])
    ax4.set_ylim([1e-5, 1e3])
    ax4.set_xlabel("Time [Orbits]")
    ax4.legend(loc='upper left', fontsize='small')

    # Create a text object for time display
    time_text = fig.text(0.51, 0.95, '', ha='center', va='center', fontsize=14)

    # Create a tqdm progress bar (this will update with each frame)
    progress_bar = tqdm(total=nsteps, desc="Creating Movie", ncols=100)

    # --- Define the animation update function ---
    def animate(i):
        # Update vorticity images without clearing axes
        if plot_mode == "vorticity":
            if len(zgrid) > 1 and (gasvx_full is not None) and (gasvz_full is not None):
                # 3D branch: update using full 3D velocity fields
                try:
                    omega_r, omega_phi, omega_z = calculate_vorticity(
                        gasvy_full, gasvx_full, gasvz_full, xgrid, ygrid, zgrid, i, R_full)
                    omega_z_shifted = np.roll(omega_z, shift=len(ygrid)//2, axis=0)
                    omega_rp = np.sqrt(omega_r**2 + omega_phi**2)
                    omega_rp_shifted = np.roll(omega_rp, shift=len(ygrid)//2, axis=0)
                    omega_z_mask = omega_z_shifted[:, radial_mask]
                    omega_rp_mask = omega_rp_shifted[:, radial_mask]
                    # Update color limits dynamically
                    oz_max = max(abs(np.percentile(omega_z_mask, 1)),
                                 abs(np.percentile(omega_z_mask, 99)))
                    orp_max = np.percentile(omega_rp_mask, 99)
                    im1.set_clim(-oz_max, oz_max)
                    im2.set_clim(0, orp_max)
                    im1.set_data(omega_z_mask)
                    im2.set_data(omega_rp_mask)
                except Exception as e:
                    print(f"❌ Error calculating 3D vorticity at frame {i}: {e}")
            else:
                # 2D branch: update using 2D velocity fields
                try:
                    omega_r, omega_phi, omega_z = calculate_vorticity(
                        gasvy, gasvx, gasvz, xgrid, ygrid, zgrid, i, R_full)
                    omega_z_full = np.roll(omega_z, shift=len(ygrid)//2, axis=0)
                    omega_z_mask = omega_z_full[:, radial_mask]
                    radial_velocity_full = np.roll(gasvy[..., i] / H_g, shift=len(ygrid)//2, axis=0)
                    radial_velocity_mask = radial_velocity_full[:, radial_mask]
                    oz_max = max(abs(np.percentile(omega_z_mask, 1)),
                                 abs(np.percentile(omega_z_mask, 99)))
                    im1.set_clim(-oz_max, oz_max)
                    im1.set_data(omega_z_mask)
                    im2.set_data(radial_velocity_mask)
                except Exception as e:
                    print(f"❌ Error calculating 2D vorticity at frame {i}: {e}")

        # Update bottom-left panel: radial gas density and pressure gradient
        radial_gas_density_full = np.mean(gasdens[..., i], axis=0)
        radial_gas_density = radial_gas_density_full[radial_mask]
        # Note: here we assume the reference density is the initial density at the center
        scaled_rhog = radial_gas_density / gasdens[len(ygrid)//2, len(xgrid)//2, 0]
        pressure_gradient_full = np.mean(np.gradient(gasenergy[..., i], xgrid, axis=1), axis=0)
        pressure_gradient = pressure_gradient_full[radial_mask]
        scaled_pg = pressure_gradient / scal_pg
        line_rhog.set_data(xgrid_for_ax3, scaled_rhog)
        line_pg.set_data(xgrid_for_ax3, scaled_pg)
        # (Initial curves remain unchanged)

        # Update bottom-right panel: time evolution curves
        line_alpha.set_data(time_array[:i+1], alpha_r[:i+1])
        line_rms.set_data(time_array[:i+1], rms_vr[:i+1])

        # Update time annotation text
        time_text.set_text(f"{time_array[i]:.2f} Orbits")
        progress_bar.update(1)
        # Return a tuple of all artists that are updated (for blitting if enabled)
        return im1, im2, line_rhog, line_pg, line_alpha, line_rms, time_text

    # Create the animation (blit can be enabled if all artists are properly returned)
    ani = animation.FuncAnimation(fig, animate, frames=nsteps, blit=False, repeat=False)

    mode_suffix = "STANDARD" if plot_mode == "standard" else "VORTICITY"
    movie_filename = f"movie_{os.path.basename(output_path)}_{mode_suffix}.mp4"
    output_filepath = os.path.join(output_path, movie_filename)

    # Save the movie using ffmpeg writer
    ani.save(output_filepath, writer='ffmpeg', dpi=150, fps=8)

    progress_bar.close()
    plt.close(fig)
    print("#######################################")
    print(f"✅ Simulation movie saved to {output_filepath}")
    print("#######################################")

    # Transfer the movie via SCP (assumes scp_transfer is implemented)
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Movies", "mariuslehmann")

    return output_filepath


def create_combined_movie(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, IDEFIX=False):
    """
    Creates a movie with two panels:
    1) Top panel: precomputed Cartesian-transformed imshow of gas density deviation.
       (with a "ramp" at the outer boundary).
    2) Bottom panel: radial density + torque vs. time on twin axes.
       (Lines are created once and only updated each frame, to avoid slowdowns.)

    Parameters
    ----------
    data_arrays : dict
        Dictionary of data arrays (e.g., gasdens).
    xgrid, ygrid, zgrid : arrays
        Simulation grid.
    time_array : array
        Time array for frames (from data).
    output_path : str
        Path to the simulation output directory.
    IDEFIX : bool
        Whether this is an IDEFIX simulation (no summary0.dat, tqwk0.dat instead).
    """
    import os
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    from data_reader import read_parameters
    from planet_data import extract_planet_mass_and_migration, read_alternative_torque  # update module name

    nsteps = data_arrays['gasdens'].shape[-1]
    time_array = time_array[:nsteps]

    # ---------------------------
    # 1) LOAD DATA
    # ---------------------------
    gasdens = data_arrays['gasdens']
    initial_gas_density = np.mean(np.mean(gasdens[:, :, :, 0], axis=0), axis=1)

    if IDEFIX:
        tqwk_file = os.path.join(output_path, "tqwk0.dat")
        idefix_logfile = os.path.join(output_path, "idefix.0.log")
        date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file, IDEFIX=True)
        params = read_parameters(idefix_logfile, IDEFIX=True)
        qp = float(params.get("planetToPrimary", 0.0))
        migration = None  # Not used for IDEFIX
    else:
        tqwk_file = os.path.join(output_path, "tqwk0.dat")
        date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file, IDEFIX=False)
        summary_file = os.path.join(output_path, "summary0.dat")
        qp, migration = extract_planet_mass_and_migration(summary_file)

    time_in_orbits = date_torque / (2 * np.pi)

    # --- Filter torque signal ---
    rolling_window_size = min(100, len(time_in_orbits))
    time_averaged_torque = uniform_filter1d(torque, size=rolling_window_size, mode='nearest')

    valid_global_indices = (time_in_orbits <= max(time_array))
    global_torque_min = np.min(qp * time_averaged_torque[valid_global_indices])
    global_torque_max = np.max(qp * time_averaged_torque[valid_global_indices])

    print(f"Global Torque Min (Filtered): {global_torque_min:.2e}")
    print(f"Global Torque Max (Filtered): {global_torque_max:.2e}")

    # (Plotting and movie code continues...)

    # ---------------------------
    # 2) GEOMETRY & DELAUNAY
    # ---------------------------
    dr = xgrid[1] - xgrid[0]
    r_padded = np.insert(xgrid, 0, np.arange(xgrid[0] - dr, 0, -dr)[::-1])

    r_mesh, phi_mesh = np.meshgrid(r_padded, ygrid, indexing='ij')
    x_polar = r_mesh * np.cos(phi_mesh)
    y_polar = r_mesh * np.sin(phi_mesh)

    points_polar = np.column_stack([x_polar.ravel(), y_polar.ravel()])
    tri = Delaunay(points_polar)

    NPX = 500
    x_cartesian = np.linspace(x_polar.min(), x_polar.max(), NPX)
    y_cartesian = np.linspace(y_polar.min(), y_polar.max(), NPX)
    x_grid, y_grid = np.meshgrid(x_cartesian, y_cartesian)

    x_out = x_grid.ravel()
    y_out = y_grid.ravel()

    # ---------------------------
    # 3) PRECOMPUTE FRAMES
    # ---------------------------
    def compute_cartesian_for_frame(f):
        gas_density_final = gasdens[:, :, :, f]
        gd_dev_3d = (gas_density_final - gasdens[:, :, :, 0]) / gasdens[0, len(xgrid)//2, len(zgrid)//2, 0]
        del gas_density_final  # Free memory
        gd_dev_avg = np.mean(gd_dev_3d, axis=2)
        del gd_dev_3d  # Free memory
        zero_padding = np.zeros((gd_dev_avg.shape[0], len(r_padded) - len(xgrid)))
        gd_dev_padded = np.hstack((zero_padding, gd_dev_avg))
        # No ramping at the disk boundary
        gd_dev_ramped = gd_dev_padded
        values = gd_dev_ramped.T.flatten()
        interp_fn = LinearNDInterpolator(tri, values, fill_value=np.nan)
        cartesian_data = interp_fn(x_out, y_out).reshape(x_grid.shape)
        # Clip values > 1
        cartesian_data = np.clip(cartesian_data, None, 1)
        return cartesian_data

    # Try precomputation with decreasing number of jobs
    def try_parallel_computation(n_jobs):
        try:
            print(f"Attempting precomputation with {n_jobs} threads...")
            with tqdm_joblib(tqdm(desc=f"Frames ({n_jobs} threads)", total=nsteps)) as progress_bar:
                result = Parallel(n_jobs=n_jobs, timeout=300)(
                    delayed(compute_cartesian_for_frame)(f) for f in range(nsteps)
                )
            print(f"✅ ...successfully completed with {n_jobs} threads.")
            return result, True
        except Exception as e:
            print(f"❌ Error with {n_jobs} threads: {str(e)}")
            return None, False

    print("Precomputing Cartesian data...")
    success = False
    # Try with different numbers of threads
    for n_jobs in [8, 6, 4, 2, 1]:
        precomputed_cartesian, success = try_parallel_computation(n_jobs)
        if success:
            break
        print(f"Retrying with {n_jobs-2} threads...")

    if not success:
        print("All parallel attempts failed. Computing sequentially...")
        precomputed_cartesian = []
        for f in tqdm(range(nsteps), desc="Computing frames sequentially"):
            precomputed_cartesian.append(compute_cartesian_for_frame(f))
        print("Sequential computation complete.")

    print("...done precomputing Cartesian data.")

    # Compute global min and max across all frames
    global_min = min(np.nanmin(frame) for frame in precomputed_cartesian)
    global_max = max(np.nanmax(frame) for frame in precomputed_cartesian)

    # Fixed colorbar range based on qp:
    if qp < 2e-5:
        vmin = global_min
        vmax = global_max
    else:
        vmin = -0.5
        vmax = 1.0

    # ---------------------------
    # 4) FIGURE SETUP (ONCE)
    # ---------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    im = axs[0].imshow(
        precomputed_cartesian[0],
        origin='lower',
        extent=[x_cartesian.min(), x_cartesian.max(), y_cartesian.min(), y_cartesian.max()],
        aspect='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Gas Density Deviation (z-averaged)')

    axs[0].axhline(0, color='white', linestyle='--', linewidth=0.5)
    axs[0].axvline(0, color='white', linestyle='--', linewidth=0.5)
    circle = plt.Circle((0, 0), 1.0, color='white', linestyle='--', fill=False)
    axs[0].add_artist(circle)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title(f"Time = {time_array[0]:.1f} orbits")

    # Bottom Panel: Density and Torque Plot
    ax_density = axs[1]
    ax_density.set_xlabel("Radial Distance (r)")
    ax_density.set_ylabel("Gas Density (Azimuth & Z-Averaged)")
    ax_density.set_xlim([xgrid[0], xgrid[-1]])
    ax_density.grid(True)
    ax_density.spines["top"].set_visible(False)
    ax_density.spines["right"].set_visible(False)

    (line_init,) = ax_density.plot(
        xgrid, initial_gas_density, label="Initial Gas Density", color='blue'
    )
    (line_final,) = ax_density.plot(
        xgrid, initial_gas_density, label="Final Gas Density", color='red'
    )

    # Top axis for time on density plot
    ax_time = ax_density.twiny()
    ax_time.set_xlabel("Time (orbits)")
    ax_time.spines["top"].set_visible(True)
    ax_time.spines["bottom"].set_visible(False)
    num_ticks = 5
    tmax = max(time_array)
    tvals = np.linspace(0, tmax, num_ticks)
    ax_time.set_xlim(0, tmax)
    ax_time.set_xticks(tvals)
    ax_time.set_xticklabels([f"{t:.1f}" for t in tvals])

    # Right axis for torque
    ax_torque = ax_time.twinx()
    ax_torque.set_ylabel("Torque (q_p x averaged)")
    ax_torque.set_ylim(global_torque_min, global_torque_max)
    ax_torque.spines["right"].set_visible(True)
    ax_torque.spines["left"].set_visible(False)

    # Replace line with scatter plot for torque
    torque_scatter = ax_torque.scatter([], [], s=3, color='green', label="Torque", zorder=3)

    # Dummy line for legend (since scatter doesn't work well in some legend implementations)
    dummy_torque_line, = ax_torque.plot([], [], color='green', label="Torque")
    dummy_torque_line.set_visible(False)  # Hide the line, we just need it for the legend

    lines_main, labels_main = ax_density.get_legend_handles_labels()
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    lines_torq, labels_torq = ax_torque.get_legend_handles_labels()
    ax_density.legend(
        lines_main + lines_time + lines_torq,
        labels_main + labels_time + labels_torq,
        loc='upper right', fontsize='small'
    )

    anim_progress = tqdm(total=nsteps, desc="Creating Movie", ncols=100)

    # ---------------------------
    # 5) UPDATE FUNCTION
    # ---------------------------
    def update(frame):
        # Update the top imshow with the precomputed data for the current frame
        cart_data = precomputed_cartesian[frame]
        im.set_data(cart_data)
        axs[0].set_title(f"Time = {time_array[frame]:.1f} orbits")

        # Update bottom panel: density profile
        radial_gas_density = np.mean(np.mean(gasdens[:, :, :, frame], axis=0), axis=1)
        line_final.set_ydata(radial_gas_density)

        # Update torque scatter plot
        current_time = time_array[frame]
        valid_indices = (time_in_orbits <= current_time)
        
        if np.any(valid_indices):
            x_torque = time_in_orbits[valid_indices]
            y_torque = qp * time_averaged_torque[valid_indices]
            
            # Downsample if there are too many points (optional)
            if len(x_torque) > 1000:
                step = len(x_torque) // 1000
                x_torque = x_torque[::step]
                y_torque = y_torque[::step]
                
            # Update the scatter plot data
            torque_scatter.set_offsets(np.column_stack([x_torque, y_torque]))
        else:
            torque_scatter.set_offsets(np.empty((0, 2)))  # Empty scatter plot
        
        anim_progress.update(1)
        return (im, line_final, torque_scatter)
    
    # ---------------------------
    # 6) RUN ANIMATION
    # ---------------------------
    ani = FuncAnimation(
        fig, update, frames=nsteps, repeat=False, blit=False
    )

    subdir_name = os.path.basename(output_path)
    output_filename = f"{subdir_name}_combined_density_movie.mp4"
    output_filepath = os.path.join(output_path, output_filename)

    ani.save(output_filepath, writer='ffmpeg', fps=8, dpi=100)
    anim_progress.close()

    print("#######################################")
    print(f"✅ simulation movie saved to {output_filepath}")
    print("#######################################")

    # Transfer the movie to a local directory
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Movies/planet_evolution", "mariuslehmann")



def create_simulation_movie_axi(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, dust=False, planet=False, IDEFIX=False):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm

    # Extract data
    gasdens = data_arrays['gasdens']
    nsteps = data_arrays['gasdens'].shape[-1]
    time_array = time_array[:nsteps]
    gasvz = data_arrays['gasvz']
    if dust:
        dust1dens = data_arrays['dust1dens']
    else:
        gasvx = data_arrays['gasvx']

    # Handle gasenergy field
    if 'gasenergy' in data_arrays:
        gasenergy = data_arrays['gasenergy']
    elif IDEFIX:
        print("⚠️ 'gasenergy' not found — computing isothermal pressure from h0 and flaringIndex (IDEFIX)")
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)
        h0 = float(parameters.get("h0", 0.05))
        flaring_index = float(parameters.get("flaringIndex", 1.0))
        gamma = float(parameters.get("gamma", 1.6667))
        cs2_profile = h0**2 * xgrid**(2 * flaring_index - 1)
        gasenergy = data_arrays['gasdens'] * cs2_profile[np.newaxis, :, np.newaxis] / (gamma - 1)
    else:
        raise KeyError("❌ 'gasenergy' is missing in data_arrays and not running IDEFIX, cannot proceed.")

    if not dust:
        gasvx0 = gasvx[:, :, :, 0]
        delta_v_phi = gasvx - gasvx0[:, :, :, np.newaxis]

    initial_rho_value = gasdens[0, len(xgrid) // 2, len(zgrid) // 2, 0]
    initial_pg_value = gasenergy[0, len(xgrid) // 2, len(zgrid) // 2, 0]
    initial_gas_density = np.mean(np.mean(gasdens[:, :, :, 0], axis=0), axis=1)
    initial_pressure_gradient = np.gradient(np.mean(gasenergy[:, :, :, 0], axis=0), xgrid, axis=0)
    initial_scaled_pg = initial_pressure_gradient / initial_pg_value

    quantities_file = os.path.join(output_path, f"{os.path.basename(output_path)}_quantities.npz")
    loaded_data = np.load(quantities_file)
    alpha_r = loaded_data['alpha_r']
    alpha_r_HS = loaded_data['alpha_r_HS']
    rms_vz = loaded_data['rms_vz']
    if dust:
        max_epsilon = loaded_data['max_epsilon']
        H_d_array = loaded_data['H_d']

    if IDEFIX:
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)
        h0 = float(parameters.get("h0", 0.05))
        H_g = h0
    else:
        summary_file = os.path.join(output_path, "summary0.dat")
        parameters = read_parameters(summary_file)
        aspectratio = float(parameters['ASPECTRATIO'])
        H_g = aspectratio

    if dust:
        roche_density = (9 / (4 * np.pi * np.power(xgrid, 3)))[np.newaxis, :, np.newaxis]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.25)
    ax3_right = ax3.twinx()

    # Set colorbar axes first!
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("top", size="5%", pad=0.3)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("top", size="5%", pad=0.3)

    # Prepare initial data for imshow and colorbars
    radial_mask = (xgrid >= 0.6) & (xgrid <= 1.5)
    xgrid_masked = xgrid[radial_mask]
    if dust:
        plotar_az_avg = np.mean(dust1dens[:, :, :, 0] / gasdens[:, :, :, 0], axis=0)
        im1_data = np.sqrt(plotar_az_avg[radial_mask, :].T)
    else:
        # Compute azimuthal + vertical average at t=0 → shape (nx,)
        base_vphi = np.mean(gasvx[:, :, :, 0], axis=(0, 2), keepdims=False)  # shape: (nx,)
        # Reshape for broadcasting: (1, nx, 1, 1)
        base_vphi = base_vphi[np.newaxis, :, np.newaxis, np.newaxis]
        delta_v_phi = gasvx - base_vphi  # full shape: (ny, nx, nz, nt)
        plotar_az_avg = np.mean(delta_v_phi[:, :, :, 0], axis=0)
        im1_data = plotar_az_avg[radial_mask, :].T

    gasvz_az_avg = np.mean(gasvz[:, :, :, 0], axis=0)
    im2_data = gasvz_az_avg[radial_mask, :].T

    im1 = ax1.imshow(im1_data, extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
                     aspect='auto', origin='lower', cmap='hot')
    im2 = ax2.imshow(im2_data, extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
                     aspect='auto', origin='lower', cmap='hot')
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')

    # Add static labels ONCE (not in animate)
    if dust:
        ax1.text(0.025, 0.95, r"$\epsilon$", transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    else:
        ax1.text(0.025, 0.95, r"$\delta v_{g\varphi}$", transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    ax2.text(0.025, 0.95, r"$v_{gz}$", transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    ax1.set_ylabel(r"$z/r_{0}$")
    ax2.set_ylabel(r"$z/r_{0}$")
    ax1.set_xlabel(r"$r/r_{0}$")
    ax2.set_xlabel(r"$r/r_{0}$")

    time_text = fig.text(0.51, 0.95, '', ha='center', va='center', fontsize=14)
    if dust:
        roche_exceeded_times = []

    # Frame thinning
    def get_sparse_indices(rms_vz, threshold=0.03, max_skip=6):
        indices = [0]
        last_idx = 0
        for i in range(1, len(rms_vz)):
            prev = rms_vz[last_idx] if rms_vz[last_idx] > 0 else 1e-10
            now = rms_vz[i] if rms_vz[i] > 0 else 1e-10
            log_diff = abs(np.log10(now) - np.log10(prev))
            if log_diff > threshold or (i - last_idx) >= max_skip:
                indices.append(i)
                last_idx = i
        if indices[-1] != len(rms_vz) - 1:
            indices.append(len(rms_vz) - 1)
        return indices

    frame_indices = get_sparse_indices(rms_vz, threshold=0.03, max_skip=6)
    print(f"Thinning movie: using {len(frame_indices)} frames (from {len(rms_vz)} available)")

    def animate(i):
        # Calculate epsilon averages without masking first
        if dust:
            plotar_az_avg = np.mean(dust1dens[:, :, :, i] / gasdens[:, :, :, i], axis=0)
        else:
            plotar_az_avg = np.mean(delta_v_phi[:, :, :, i], axis=0)
        gasvz_az_avg = np.mean(gasvz[:, :, :, i], axis=0)

        # Apply radial mask AFTER averaging
        plotar_az_avg_masked = plotar_az_avg[radial_mask, :]
        gasvz_az_avg_masked = gasvz_az_avg[radial_mask, :]

        # Update the images
        if dust:
            im1.set_data(np.sqrt(plotar_az_avg_masked.T))
        else:
            im1.set_data(plotar_az_avg_masked.T)
        im2.set_data(gasvz_az_avg_masked.T)

        # Remove this entire block!
        # if dust:
        #     cbar1.set_ticks(np.linspace(np.sqrt(np.min(plotar_az_avg_masked)), np.sqrt(np.max(plotar_az_avg_masked)), 5))
        #     cbar2.set_ticks(np.linspace(np.min(gasvz_az_avg_masked), np.max(gasvz_az_avg_masked), 5))
        # else:
        #     cbar1.set_ticks(np.linspace(np.min(plotar_az_avg_masked), np.max(plotar_az_avg_masked), 5))
        #     cbar2.set_ticks(np.linspace(np.min(gasvz_az_avg_masked), np.max(gasvz_az_avg_masked), 5))

        # Instead, just update colormap limits if you want dynamic scaling:
        if dust:
            im1.set_clim(np.sqrt(np.min(plotar_az_avg_masked)), np.sqrt(np.max(plotar_az_avg_masked)))
        else:
            im1.set_clim(np.min(plotar_az_avg_masked), np.max(plotar_az_avg_masked))
        im2.set_clim(np.min(gasvz_az_avg_masked), np.max(gasvz_az_avg_masked))

        # Update the lower panels (clear and redraw)
        ax3.clear()
        ax3_right.clear()

        radial_gas_density = np.mean(np.mean(gasdens[:, :, :, i], axis=0), axis=1)
        pressure_gradient = np.gradient(np.mean(gasenergy[:, :, :, i], axis=0), xgrid, axis=0)
        scaled_pg = pressure_gradient / initial_pg_value

        line1, = ax3.plot(xgrid, radial_gas_density / initial_rho_value, color='red', lw=2, label=r"Current $\rho_{g}$")
        line2, = ax3.plot(xgrid, initial_gas_density / initial_rho_value, color='black', lw=2, linestyle='--', label=r"Initial $\rho_{g}$")
        line3 = ax3_right.plot(xgrid, scaled_pg, color='blue', lw=2, label=r"$\frac{dP}{dr}$")[0]
        line4 = ax3_right.plot(xgrid, initial_scaled_pg, color='black', lw=2, linestyle='--', label=r"Initial $\frac{dP}{dr}$")[0]

        ax3.set_xlabel(r"$r/r_{0}$")
        #ax3.set_xlim([0.8, 1.2])
        ax3.set_ylim([0, 3])
        ax3_right.set_ylim([-40, 5])

        # Combined legend
        lines = [line1, line2, line3, line4]
        labels = [line.get_label() for line in lines]
        ax3.legend(lines, labels, loc='center right', fontsize='small')

        if dust:
            # Check if the Roche density is exceeded and store the time index
            # (Note: If Roche is ever exceeded, append ONCE per frame)
            radial_mask_check = (xgrid > 0.6) & (xgrid < 1.5)
            if np.any((gasdens[:, :, :, i] + dust1dens[:, :, :, i] >= roche_density) & radial_mask_check[np.newaxis, :, np.newaxis]):
                roche_exceeded_times.append(i)

        ax4.clear()
        ax4.plot(time_array[:i + 1], alpha_r[:i + 1], label=r'$\alpha_r(t)$')
        ax4.plot(time_array[:i + 1], alpha_r_HS[:i + 1], label=r'$\alpha_{r,HS}(t)$')
        ax4.plot(time_array[:i + 1], rms_vz[:i + 1] / H_g, label=r'RMS(Vertical Velocity)')
        if dust:
            ax4.plot(time_array[:i + 1], max_epsilon[:i + 1], label=r'Max $\epsilon$', color='green', lw=2)
            # Highlight Roche-exceeded points
            ax4.plot(np.array(time_array)[roche_exceeded_times], np.array(max_epsilon)[roche_exceeded_times],
                     label=None, color='green', marker='o', linestyle='None')
            ax4.plot(time_array[:i + 1], H_d_array[:i + 1] / H_g, label=r'Dust Scale Height $H_d/H_g$')
        ax4.set_xlim([0, np.max(time_array)])
        # SET Y RANGE
        # Gather all y-data into a single list

        # Prepare alpha data, ignoring tiny values for scaling
        alpha_r_plot = alpha_r[:i + 1]
        alpha_max = np.nanmax(alpha_r_plot)
        alpha_cutoff = 0.01 * alpha_max

        # Only include alpha values above cutoff for determining y-axis limits
        alpha_for_ylim = alpha_r_plot[alpha_r_plot > alpha_cutoff]

        # Gather other y-data (as before)
        y_data = [alpha_for_ylim, rms_vz[:i + 1] / H_g]
        if dust:
            y_data.append(max_epsilon[:i + 1])
            y_data.append(H_d_array[:i + 1] / H_g)

        # Concatenate, flatten, and keep only positive values (for log scale)
        y_values = np.concatenate([np.atleast_1d(arr) for arr in y_data])
        y_values = y_values[y_values > 0]

        # Apply global y-min floor (set here, e.g., 1e-9)
        global_ymin_floor = 1e-9
        if len(y_values) > 0:
            y_min = np.nanmin(y_values)
            y_max = np.nanmax(y_values)
            y_min_plot = max(y_min * 0.7, global_ymin_floor)
            y_max_plot = y_max * 1.5 if y_max > 0 else 1e3
            if y_min_plot == y_max_plot:
                y_min_plot *= 0.5
                y_max_plot *= 2.0
            ax4.set_ylim([y_min_plot, y_max_plot])
        else:
            ax4.set_ylim([global_ymin_floor, 1e3])


        ax4.set_yscale('log')
        ax4.set_xlabel("Time [Orbits]")
        ax4.set_ylabel(" ")
        ax4.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax4.legend(loc='upper left', fontsize='small')

        # Update the time annotation
        time_text.set_text(f"{time_array[i]:.2f} Orbits")

    # Create the movie with tqdm progress bar on frames
    ani = animation.FuncAnimation(
        fig, animate,
        frames=tqdm(frame_indices, desc="Creating Movie", ncols=100),
        repeat=False
    )
    subdir_name = os.path.basename(output_path)
    movie_filename = f"movie_{subdir_name}.mp4"
    output_filepath = os.path.join(output_path, movie_filename)
    ani.save(output_filepath, writer='ffmpeg', dpi=300, fps=15)

    plt.close(fig)
    print(f"#######################################")
    print(f"✅ simulation movie saved to {output_filepath}")
    print(f"#######################################")

    # Call the scp_transfer function to transfer the movie to your laptop
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Movies", "mariuslehmann")





def create_simulation_movie(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, dust=False, planet=False, IDEFIX=False):
    # Extract necessary data for the plots
    #gasdens = data_arrays['gasdens']
    #gasenergy = data_arrays['gasenergy']
    #if dust:
    #    dust1dens = data_arrays['dust1dens']
    #else:
    #    gasvz = data_arrays['gasvz']

    # Set font size locally for this function
    plt.rc('font', size=8)         # Default font size for text
    plt.rc('axes', titlesize=10)   # Font size for axis titles
    plt.rc('axes', labelsize=8)    # Font size for axis labels
    plt.rc('xtick', labelsize=8)   # Font size for x tick labels
    plt.rc('ytick', labelsize=8)   # Font size for y tick labels
    plt.rc('legend', fontsize=8)   # Font size for legends
    plt.rc('figure', titlesize=12) # Font size for figure title

    nsteps = data_arrays['gasdens'].shape[-1]
    time_array = time_array[:nsteps]

    if 'gasenergy' in data_arrays:
        gasenergy = data_arrays['gasenergy']
    elif IDEFIX:
        print("⚠️ 'gasenergy' not found — computing isothermal pressure from h0 and flaringIndex (IDEFIX)")
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)
        h0 = float(parameters.get("h0", 0.05))
        flaring_index = float(parameters.get("flaringIndex", 1.0))
        gamma = float(parameters.get("gamma", 1.6667))
        cs2_profile = h0**2 * xgrid**(2 * flaring_index - 1)
        gasenergy = data_arrays['gasdens'] * cs2_profile[np.newaxis, :, np.newaxis] / (gamma - 1)

    # Define initial values for scaling
    initial_rho_value = data_arrays['gasdens'][0, len(xgrid) // 2, len(zgrid) // 2, 0]
    initial_pg_value = gasenergy[0, len(xgrid) // 2, len(zgrid) // 2, 0]

    # Initial gas density profile for reference
    initial_gas_density = np.mean(np.mean(data_arrays['gasdens'][:, :, :, 0], axis=0), axis=1)  # Azimuthal and vertical average

    # Load saved quantities
    quantities_file = os.path.join(output_path, f"{os.path.basename(output_path)}_quantities.npz")
    loaded_data = np.load(quantities_file)
    alpha_r = loaded_data['alpha_r']
    alpha_r_HS = loaded_data['alpha_r_HS']
    rms_vz = loaded_data['rms_vz']
    max_vz = loaded_data['max_vz']
    min_vz = loaded_data['min_vz']
    if dust:
        max_epsilon = loaded_data['max_epsilon']
        H_d_array = loaded_data['H_d']

    # Read disk aspect ratio (H_g = h0 * r)
    if IDEFIX:
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)
        h0 = float(parameters.get("h0", 0.05))  # Default to 0.05 if not found
        H_g = h0  # This is used as h0; actual H_g varies with r
    else:
        summary_file = os.path.join(output_path, "summary0.dat")
        parameters = read_parameters(summary_file)
        aspectratio = float(parameters['ASPECTRATIO'])  # e.g., h = H/r
        H_g = aspectratio

    if dust:
        # Roche density calculation, reshaped to match gasdens and dust1dens
        roche_density = (9 / (4 * np.pi * np.power(xgrid, 3)))[np.newaxis, :, np.newaxis]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # Adjust horizontal and vertical spacing between plots
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.25)

    # Create the secondary y-axis for the radial plot
    ax3_right = ax3.twinx()

    # Set fixed positions for colorbars
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("top", size="5%", pad=0.3)  # Adjusted for better placement
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("top", size="5%", pad=0.3)  # Adjusted for better placement

    cbar1, cbar2 = None, None
    time_text = fig.text(0.51, 0.95, '', ha='center', va='center', fontsize=14)  # Add time annotation

    if dust:
        # Initialize a list to store indices where Roche density is exceeded
        roche_exceeded_times = []

    # Create a tqdm progress bar
    progress_bar = tqdm(total=nsteps, desc="Creating Movie", ncols=100)

    def animate(i):
        nonlocal cbar1, cbar2

        # Apply radial mask
        if planet:
            #radial_mask = (xgrid >= np.min(xgrid)+0.1) & (xgrid <= np.max(xgrid)-0.1)
            radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
        else:
            radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
            #radial_mask = (xgrid >= np.min(xgrid)+0.1) & (xgrid <= np.max(xgrid)-0.1)

        xgrid_masked = xgrid[radial_mask]

        nphi = len(ygrid)  # Number of azimuthal points
        shift = nphi // 2  # Shift by pi

        # Calculate epsilon averages without masking first
        if dust:
            plotar_az_avg = np.mean(data_arrays['dust1dens'][:, :, :, i] / data_arrays['gasdens'][:, :, :, i], axis=0)  # Shape: (radial, z)
            plotar_z_avg = np.mean(data_arrays['dust1dens'][:, :, :, i] / data_arrays['gasdens'][:, :, :, i], axis=2)  # Shape: (azimuthal, radial)
        else:
            plotar_az_avg = np.mean(data_arrays['gasvz'][:, :, :, i] , axis=0)  # Shape: (radial, z)
            plotar_z_avg = np.mean(data_arrays['gasvz'][:, :, :, i] , axis=2)  # Shape: (azimuthal, radial)

            plotar_z_avg = np.roll(plotar_z_avg, shift, axis=0)


        # Apply radial mask AFTER averaging
        plotar_az_avg_masked = plotar_az_avg[radial_mask, :]  # Now mask the radial axis
        plotar_z_avg_masked = plotar_z_avg[:, radial_mask]  # Mask the radial axis here as well

        # Plot using the masked arrays
        #IMSHOW
        if dust:
            im1 = ax1.imshow(np.sqrt(plotar_az_avg_masked.T), extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
                     aspect='auto', origin='lower', cmap='hot')
            im2 = ax2.imshow(np.sqrt(plotar_z_avg_masked), extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                     aspect='auto', origin='lower', cmap='hot')
        else:
            im1 = ax1.imshow(plotar_az_avg_masked.T, extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
                     aspect='auto', origin='lower', cmap='hot')
            im2 = ax2.imshow(plotar_z_avg_masked, extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                     aspect='auto', origin='lower', cmap='hot')

        # Add labels to the upper panels with the desired LaTeX formatting
        if dust:
            ax1.text(0.025, 0.95, r"$\langle \epsilon \rangle_{\varphi}$", transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            ax2.text(0.025, 0.95, r"$\langle \epsilon \rangle_{z}$", transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        else:
            ax1.text(0.025, 0.95, r"$\langle v_{gz} \rangle_{\varphi}$", transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            
            ax2.text(0.025, 0.95, r"$\langle v_{gz} \rangle_{z}$", transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        if cbar1 is None and cbar2 is None:
            cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
        else:
            for c in cbar1.ax.collections:
                c.remove()
            for c in cbar2.ax.collections:
                c.remove()
            cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')

        # Update tick marks and labels
        if dust:
            cbar1.set_ticks(np.linspace(np.sqrt(np.min(plotar_az_avg_masked)), np.sqrt(np.max(plotar_az_avg_masked)), 5))
            cbar2.set_ticks(np.linspace(np.sqrt(np.min(plotar_z_avg_masked)), np.sqrt(np.max(plotar_z_avg_masked)), 5))
        else:
            cbar1.set_ticks(np.linspace(np.min(plotar_az_avg_masked), np.max(plotar_az_avg_masked), 5))
            cbar2.set_ticks(np.linspace(np.min(plotar_z_avg_masked), np.max(plotar_z_avg_masked), 5))

        # Update the plots with relevant data for this time step
        ax3.clear()
        ax3_right.clear()

        radial_gas_density = np.mean(np.mean(data_arrays['gasdens'][:, :, :, i], axis=0), axis=1)
        pressure_gradient = np.gradient(np.mean(gasenergy[:, :, :, i], axis=0), xgrid, axis=0)
        initial_pressure_gradient = np.gradient(np.mean(gasenergy[:, :, :, 0], axis=0), xgrid, axis=0)
        scaled_pg = pressure_gradient / initial_pg_value
        initial_scaled_pg = initial_pressure_gradient / initial_pg_value

        line1, = ax3.plot(xgrid, radial_gas_density / initial_rho_value, color='red', lw=2, label=r"Current $\rho_{g}$")
        line2, = ax3.plot(xgrid, initial_gas_density / initial_rho_value, color='black', lw=2, linestyle='--', label=r"Initial $\rho_{g}$")
        line3 = ax3_right.plot(xgrid, scaled_pg, color='blue', lw=2, label=r"$\frac{dP}{dr}$")[0]
        line3 = ax3_right.plot(xgrid, initial_scaled_pg, color='black', lw=2, linestyle='--', label=r"Initial $\frac{dP}{dr}$")[0]

        ax3.set_xlabel(r"$r/r_{0}$")  # Updated horizontal axis label
        if planet:
            ax3.set_xlim([np.min(xgrid)+0.1, np.max(xgrid)-0.1])
        else:
            ax3.set_xlim([0.8, 1.2])
            #ax3.set_xlim([np.min(xgrid)+0.1, np.max(xgrid)-0.1])

        ax3.set_ylim([0, 3])  # Fixed y-axis range for the left axis
        ax3_right.set_ylim([-40, 5])  # Fixed y-axis range for the right axis

        # Use a combined legend for both axes
        lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax3.legend(lines, labels, loc='center right', fontsize='small')

        if dust:
            # Check if the Roche density is exceeded and store the time index
            radial_mask = (xgrid > 0.8) & (xgrid < 1.2)
            if np.any((data_arrays['gasdens'][:, :, :, i] + data_arrays['dust1dens'][:, :, :, i] >= roche_density) & radial_mask[np.newaxis, :, np.newaxis]):
                roche_exceeded_times.append(i)
    
        ax4.clear()

        # --- Determine averaging interval for alpha_r ---
        time_cutoff = time_array[i] - 200 if time_array[i] > 200 else 0
        valid_indices = np.where(time_array[:i + 1] >= time_cutoff)[0]
        alpha_r_avg = np.mean(alpha_r[valid_indices])
        avg_label = fr'$\langle \alpha_r \rangle_{{{int(time_array[i] - time_cutoff):d}}} = {alpha_r_avg:.1e}$'

        # --- Plotting ---
        ax4.plot(time_array[:i + 1], alpha_r[:i + 1], label=fr'$\alpha_r(\pm 0.5 H_g)$')

        # Add invisible legend entry for average
        ax4.plot([], [], label=avg_label, linewidth=0)

        ax4.plot(time_array[:i + 1], rms_vz[:i + 1], label=r'RMS(Vertical Velocity)')
        ax4.plot(time_array[:i + 1], max_vz[:i + 1] - min_vz[:i + 1], label=r'max($v_z$)-min($v_z$)', color='red')

        if dust:
            ax4.plot(time_array[:i + 1], max_epsilon[:i + 1], label=r'Max $\epsilon$', color='green', lw=2)
            ax4.plot(np.array(time_array)[roche_exceeded_times], np.array(max_epsilon)[roche_exceeded_times],
                     label=None, color='green', marker='o', linestyle='None')
            ax4.plot(time_array[:i + 1], H_d_array[:i + 1] / H_g, label=r'Dust Scale Height $H_d/H_g$')

        ax4.set_xlim([0, np.max(time_array)])

        # ==== Set Y-axis limits ====
        y_values = [alpha_r[:i + 1], rms_vz[:i + 1], max_vz[:i + 1] - min_vz[:i + 1]]
        if dust:
            y_values.extend([max_epsilon[:i + 1], H_d_array[:i + 1] / H_g])

        all_y = np.concatenate(y_values)
        all_y = all_y[np.isfinite(all_y)]

        if all_y.size > 0:
            y_max = np.max(all_y)
            y_max = max(y_max * 1.2, 1e-6)
            ax4.set_ylim([1e-7, y_max])
        else:
            ax4.set_ylim([1e-7, 1e-3])

        ax4.set_yscale('log')
        ax4.set_xlabel("Time [Orbits]")
        ax4.set_ylabel(" ")
        ax4.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax4.legend(loc='upper left', fontsize='small')

        # Add labels to the upper panels
        ax1.set_ylabel(r"$z/r_{0}$")  # Added vertical axis label
        ax2.set_ylabel(r"$\varphi$")  # Added vertical axis label
        ax1.set_xlabel(r"$r/r_{0}$")  # Added horizontal axis label
        ax2.set_xlabel(r"$r/r_{0}$")  # Added horizontal axis label

        # Update the time annotation
        time_text.set_text(f"{time_array[i]:.2f} Orbits")

        # Update the progress bar
        progress_bar.update(1)

    # Create the movie
    ani = animation.FuncAnimation(fig, animate, frames=nsteps, repeat=False)
    subdir_name = os.path.basename(output_path)  # Extract the subdirectory name from the output path
    movie_filename = f"movie_{subdir_name}.mp4"
    output_filepath = os.path.join(output_path, movie_filename)  # Full path to the movie file
    ani.save(output_filepath, writer='ffmpeg', dpi=150, fps=8)

    # Close the progress bar
    progress_bar.close()

    plt.close(fig)
    print(f"#######################################")
    print(f"✅ simulation movie saved to {output_filepath}")
    print(f"#######################################")

    # Call the scp_transfer function to transfer the movie to your laptop
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Movies", "mariuslehmann")





def plot_metallicity(data_arrays, xgrid, time_array, output_path, planet=False):
    """
    Plot the space-time contour of the metallicity Z across the entire azimuth at the disk midplane.

    Parameters:
    - xgrid (array): The grid array for the radial coordinate.
    - time_array (array): The array of time steps.
    - gasdens (array): The gas density array (ny, nx, nz, nts).
    - dust1dens (array): The dust density array (ny, nx, nz, nts).
    - output_path (str): Path to save the plot.
    """

    # Set global font size and other styling properties using rcParams
    plt.rcParams.update({
        'font.size': 16,        # Set default font size for all text
        'axes.titlesize': 20,   # Set font size for the title of each subplot
        'axes.labelsize': 18,   # Set font size for x and y labels
        'xtick.labelsize': 16,  # Set font size for x-axis tick labels
        'ytick.labelsize': 16,  # Set font size for y-axis tick labels
        'legend.fontsize': 14,  # Set font size for the legend
        'figure.titlesize': 24  # Set font size for the overall figure title
    })

    # Calculate surface mass densities by integrating over the vertical direction (z-axis)
    sigma_gas = np.sum(data_arrays['gasdens'], axis=2)  # Shape (ny, nx, nts)
    sigma_dust = np.sum(data_arrays['dust1dens'], axis=2)  # Shape (ny, nx, nts)

    # Calculate the metallicity Z = sigma_dust / sigma_gas
    Z = sigma_dust / sigma_gas  # Shape (ny, nx, nts)

    # Take the maximum value across the azimuthal direction (y-axis)
    Z_max = np.max(Z, axis=0)  # Shape (nx, nts)

    # Apply radial mask
    radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
    xgrid_masked = xgrid[radial_mask]
    Z_max_masked = Z_max[radial_mask, :]  # Shape (masked nx, nts)

    # Create the contour plot
    plt.figure(figsize=(10, 6))
    cp = plt.imshow(np.sqrt(Z_max_masked.T), extent=[xgrid_masked.min(), xgrid_masked.max(), time_array.min(), time_array.max()],
                    aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(cp, label=r'$Z^{1/2}_{\max}(\varphi)$')
    plt.xlabel('Disk Radius')
    plt.ylabel('Orbits')
    #plt.title(r'Space-Time Plot of Metallicity $Z$ at Disk Midplane (Max Azimuthal Value)')
    plt.title(r' ')

    # Extract the subdirectory name
    subdir_name = os.path.basename(output_path)

    # Check if the current simulation matches the specific one where dust feedback is turned off
    if subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_fboff":
        feedback_off_time = 726
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5, label='Feedback Off')
        plt.legend()

    elif subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150":
        feedback_off_time = 0
        # Draw the horizontal line (this line will not appear in the legend)
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5)
        # Create a custom legend entry for "No Feedback" without any line, using an invisible marker
        plt.legend([plt.Line2D([0], [0], linestyle='None', marker='None', label='Feedback')], ['Feedback'])

    elif subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_nfb":
        feedback_off_time = 0
        # Draw the horizontal line (this line will not appear in the legend)
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5)
        # Create a custom legend entry for "No Feedback" without any line, using an invisible marker
        plt.legend([plt.Line2D([0], [0], linestyle='None', marker='None', label='No Feedback')], ['No Feedback'])

    elif subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_nfb_fbon":
        feedback_off_time = 400
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5, label='Feedback On')
        plt.legend()

    # Save the plot with the subdirectory name in the file name
    pdf_filename = f"{subdir_name}_space_time_plot_metallicity_Z.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(os.path.join(output_path, pdf_filename))
    plt.close()
    print(f"#######################################")
    print(f"✅ Metallicity space-time plot saved to {os.path.join(output_path, pdf_filename)}")
    print(f"#######################################")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")








def plot_vorticity_difference(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, planet=False):
    """
    Plot the space-time contour of the difference in z-component vorticity: 
    <omega_z>_z - <omega_z>_zphi.

    Parameters:
    - xgrid (array): The radial grid array.
    - ygrid (array): The azimuthal grid array.
    - zgrid (array): The vertical grid array.
    - time_array (array): The array of time steps.
    - gasvx (array): Azimuthal velocity (v_phi) array.
    - gasvy (array): Radial velocity (v_r) array.
    - output_path (str): Path to save the plot.
    """
    print("Computing vorticity difference...")

    ny = len(ygrid)

    # Calculate the cylindrical vorticity component omega_z
    r = xgrid[np.newaxis, :, np.newaxis, np.newaxis]  # Radial grid expanded to shape (1, nx, 1, 1)

    # Derivatives for vorticity
    d_vphi_dr = np.gradient(r * data_arrays['gasvx'], xgrid, axis=1) / r  # First term: 1/r * d(r * v_phi)/dr
    d_vr_dphi = np.gradient(data_arrays['gasvy'], ygrid, axis=0) if ny > 1 else 0   # Second term: d(v_r)/dphi

    # Compute omega_z
    omega_z = d_vphi_dr - d_vr_dphi

    # Calculate <omega_z>_z (vertical average)
    omega_z_avg_z = np.mean(omega_z, axis=2)  # Shape (ny, nx, nts)

    # Calculate <omega_z>_zphi (vertical and azimuthal average)
    omega_z_avg_zphi = np.mean(omega_z_avg_z, axis=0)  # Shape (nx, nts)

    # Calculate the difference <omega_z>_z - <omega_z>_zphi
    vorticity_diff = omega_z_avg_z - omega_z_avg_zphi[np.newaxis, :, :]  # Shape (ny, nx, nts)

    # Compute the minimum across the azimuthal direction
    vorticity_diff_min = np.min(vorticity_diff, axis=0)  # Shape (nx, nts)

    print("Creating contour plot...")

    # Create the contour plot
    plt.figure(figsize=(10, 6))
    cp = plt.contourf(xgrid, time_array, vorticity_diff_min.T, cmap='viridis', levels=100)  
    plt.colorbar(cp, label=r'$ \min \left[ \langle \omega_z \rangle_z - \langle \omega_z \rangle_{z \phi} \right]$')
    plt.xlabel('Disk Radius')
    plt.ylabel('Orbits')
    plt.title(r'Space-Time Plot of $ \min \left[ \langle \omega_z \rangle_z - \langle \omega_z \rangle_{z \phi} \right]$')

    # Extract the subdirectory name
    subdir_name = os.path.basename(output_path)

    # Save the plot with the subdirectory name in the file name
    pdf_filename = f"{subdir_name}_space_time_plot_vorticity_diff.pdf"
    save_path = os.path.join(output_path, pdf_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Vorticity difference space-time plot saved to {save_path}")

    scp_transfer(save_path, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")





# CODE BLOCK 1: Add this new function to the top of your file.

# REPLACEMENT CODE BLOCK: Use this updated worker function.

# FINAL REPLACEMENT: This worker function fixes the transposition error.

def worker_process_timestep(t_idx, time_val, data_slice, initial_slice, static_params):
    """
    Independent worker function to process a single time step.
    This version includes a fix for the transposed gasvx0 array.
    """
    # Unpack static parameters
    xgrid = static_params['xgrid']
    ygrid = static_params['ygrid']
    zgrid = static_params['zgrid']
    radial_mask = static_params['radial_mask']
    hs_mask = static_params['hs_mask']
    nz = static_params['nz']
    ny = static_params['ny']
    h0 = static_params['h0']
    flaringIndex = static_params['flaringIndex']
    m_gas_initial = static_params['m_gas_initial']
    m_dust_initial = static_params['m_dust_initial']
    roche_density_masked = static_params.get('roche_density_masked')
    zgrid_std = static_params['zgrid_std']
    dust = static_params['dust']

    # Unpack the data for the current time step
    gasdens_t = data_slice['gasdens_t']
    gasvx_t = data_slice['gasvx_t']
    gasvy_t = data_slice['gasvy_t']
    gasvz_t = data_slice['gasvz_t']
    gasenergy_t = data_slice.get('gasenergy_t')

    # Unpack initial condition data
    gasvx0 = initial_slice['gasvx0']
    gasvx0_HS = initial_slice['gasvx0_HS']

    # ==================================================================
    # 🚨 FIX: Check for and correct the transposed shape of gasvx0 🚨
    if gasvx0.shape[0] != ny:
        gasvx0 = gasvx0.transpose(1, 0, 2)
    if gasvx0_HS.shape[0] != ny:
        gasvx0_HS = gasvx0_HS.transpose(1, 0, 2)
    # ==================================================================

    # --- Create masked versions of arrays ONCE for consistency ---
    gasdens_t_masked = gasdens_t[:, radial_mask, :]
    gasvx_t_masked = gasvx_t[:, radial_mask, :]
    gasvy_t_masked = gasvy_t[:, radial_mask, :]

    # === Start of calculations ===
    m_gas_val = np.sum(gasdens_t_masked) / m_gas_initial
    roche_reached = False

    if dust:
        dust1dens_t = data_slice.get('dust1dens_t')
        dust1vx_t = data_slice.get('dust1vx_t')
        dust1vy_t = data_slice.get('dust1vy_t')
        dust1vz_t = data_slice.get('dust1vz_t')
        dust1dens_t_masked = dust1dens_t[:, radial_mask, :]
        m_dust_val = np.sum(dust1dens_t_masked) / m_dust_initial
        total_density = gasdens_t_masked + dust1dens_t_masked
        sigma_gas = np.sum(gasdens_t_masked, axis=2)
        sigma_dust = np.sum(dust1dens_t_masked, axis=2)
        metallicity = sigma_dust / sigma_gas
        avg_metallicity_val = np.mean(metallicity)
        roche_exceed_mask = total_density >= roche_density_masked[:, np.newaxis, np.newaxis]
        roche_reached = np.any(roche_exceed_mask)
        epsilon = dust1dens_t_masked / gasdens_t_masked
        max_epsilon_val = np.max(epsilon)
        dust_profile = np.mean(dust1dens_t_masked, axis=(0, 1))
        p0 = [np.max(dust_profile), 0, zgrid_std, np.min(dust_profile)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(gaussian, zgrid, dust_profile, p0=p0, maxfev=1000)
            H_d_val = abs(popt[2])
        except RuntimeError:
            H_d_val = h0
    else:
        m_dust_val, max_epsilon_val, H_d_val, avg_metallicity_val = None, 1e-15, 1e-15, 0.0

    if gasenergy_t is not None:
        pressure = gasenergy_t[:, radial_mask, :]
    else:
        xgrid_masked = xgrid[radial_mask]
        cs2_profile = h0**2 * xgrid_masked**(2 * flaringIndex - 1)
        pressure = gasdens_t_masked * np.broadcast_to(cs2_profile[np.newaxis, :, np.newaxis], gasdens_t_masked.shape)

    alpha_r_val = np.mean(gasdens_t_masked * gasvy_t_masked * (gasvx_t_masked - gasvx0)) / np.mean(pressure)

    # --- Alpha_r_HS ---
    gasdens_HS = gasdens_t[:, hs_mask, :]
    gasvy_HS = gasvy_t[:, hs_mask, :]
    gasvx_HS = gasvx_t[:, hs_mask, :]
    if gasenergy_t is not None:
        pressure_HS = gasenergy_t[:, hs_mask, :]
        T_HS = pressure_HS / gasdens_HS
    else:
        xgrid_HS = xgrid[hs_mask]
        cs2_HS = h0**2 * xgrid_HS**(2 * flaringIndex - 1)
        pressure_HS = gasdens_HS * np.broadcast_to(cs2_HS[np.newaxis, :, np.newaxis], gasdens_HS.shape)
        T_HS = np.broadcast_to(cs2_HS[np.newaxis, :, np.newaxis], gasdens_HS.shape)
    alpha_r_HS_val = np.mean(gasdens_HS * gasvy_HS * (gasvx_HS - gasvx0_HS)) / np.mean(pressure_HS)

    # --- p_HS and q_HS ---
    gasdens_HS_sq, T_HS_sq, r_HS = np.squeeze(gasdens_HS), np.squeeze(T_HS), xgrid[hs_mask]
    p_HS_val, q_HS_val = np.nan, np.nan
    axis_candidates = [i for i, s in enumerate(gasdens_HS_sq.shape) if s == r_HS.shape[0]]
    if len(axis_candidates) == 1 and r_HS.shape[0] >= 2:
        axis_rad = axis_candidates[0]
        axes_to_avg = tuple(i for i in range(gasdens_HS_sq.ndim) if i != axis_rad)
        rho_HS_1d, T_HS_1d = np.mean(gasdens_HS_sq, axis=axes_to_avg), np.mean(T_HS_sq, axis=axes_to_avg)
        if r_HS.shape[0] == rho_HS_1d.shape[0]:
            dln_rho_dlnr, dln_T_dlnr = np.gradient(np.log(rho_HS_1d), np.log(r_HS)), np.gradient(np.log(T_HS_1d), np.log(r_HS))
            p_HS_val, q_HS_val = np.mean(dln_rho_dlnr), np.mean(dln_T_dlnr)

    # --- RMS Velocities ---
    if dust:
        dust1vx_t_masked_mid = dust1vx_t[:, radial_mask, nz // 2-3 : nz // 2+3]
        dust1vy_t_masked_mid = dust1vy_t[:, radial_mask, nz // 2-3 : nz // 2+3]
        dust1vz_t_masked_mid = dust1vz_t[:, radial_mask, nz // 2-3 : nz // 2+3]
        gasvx0_mid = gasvx0[:, :, nz // 2-3 : nz // 2+3]
        rms_vr_val = np.sqrt(np.mean(dust1vy_t_masked_mid**2))
        rms_vphi_val = np.sqrt(np.mean((dust1vx_t_masked_mid - gasvx0_mid)**2))
        rms_vz_val = np.sqrt(np.mean(dust1vz_t_masked_mid**2))
    else:
        gasvz_t_masked = gasvz_t[:, radial_mask, :] if nz > 1 else None
        rms_vr_val = np.sqrt(np.mean(gasvy_t_masked**2))
        rms_vphi_val = np.sqrt(np.mean((gasvx_t_masked - gasvx0)**2))
        rms_vz_val = np.sqrt(np.mean(gasvz_t_masked**2)) if nz > 1 else 0

    if nz > 1:
        gasvz_t_masked = gasvz_t[:, radial_mask, :]
        max_vz_val, min_vz_val = np.max(gasvz_t_masked), np.min(gasvz_t_masked)
        rms_vr_z = np.sqrt(np.mean(np.mean(gasvy_t_masked**2, axis=0), axis=0))
        rms_vphi_z = np.sqrt(np.mean(np.mean(gasvx_t_masked**2, axis=0), axis=0))
        rms_vz_z = np.sqrt(np.mean(np.mean(gasvz_t_masked**2, axis=0), axis=0))
    else:
        max_vz_val, min_vz_val, rms_vr_z, rms_vphi_z, rms_vz_z = 0, 0, 0, 0, 0

    # --- Vorticity ---
    r_3d = xgrid[np.newaxis, :, np.newaxis]
    vx_temp = gasvx_t * r_3d
    d_vphi_dr = np.gradient(vx_temp, xgrid, axis=1) / r_3d
    d_vr_dphi = np.gradient(gasvy_t, ygrid, axis=0) if ny > 1 else 0
    omega_z = d_vphi_dr - d_vr_dphi
    vortz_avg_val = np.mean(omega_z)

    nsx, nsy = 100, 10
    omega_z_avg_z = omega_z[:, :, nz // 2]
    omega_z_smoothed = uniform_filter(omega_z_avg_z, size=(nsy, nsx))
    omega_z_avg_zphi = np.mean(omega_z_smoothed, axis=0)
    vorticity_diff = omega_z_smoothed - omega_z_avg_zphi[np.newaxis, :]
    vort_min_val = np.min(vorticity_diff[:, radial_mask])

    if nz > 1:
        d_vz_dphi = np.gradient(gasvz_t, ygrid, axis=0) / r_3d if ny > 1 else 0
        d_vphi_dz = np.gradient(gasvx_t, zgrid, axis=2)
        omega_r = d_vz_dphi - d_vphi_dz
        vortr_avg_val = np.mean(omega_r)
    else:
        vortr_avg_val = 0

    return (alpha_r_val, alpha_r_HS_val, p_HS_val, q_HS_val, rms_vr_val, rms_vphi_val, 
            rms_vz_val, max_vz_val, min_vz_val, max_epsilon_val, H_d_val, m_gas_val, 
            m_dust_val, avg_metallicity_val, rms_vr_z, rms_vphi_z, rms_vz_z, 
            vort_min_val, vortr_avg_val, vortz_avg_val, roche_reached)


def plot_alpha_r(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, nsteps, dust=False, planet=False, IDEFIX=False):
    """
    Calculate and plot the turbulent alpha parameter (alpha_r), RMS of vertical dust velocity, 
    maximum value of epsilon, the dust scale height (H_d), total gas mass (m_gas), 
    total dust mass (m_dust) scaled by initial values, average metallicity <Z>,
    highlight times when the Roche density is achieved during the simulation, 
    and compute the vertical profiles of time, radius, and azimuth averaged RMS velocities.

    Parameters:
    - data_arrays: Dictionary containing the simulation data.
    - xgrid: Radial grid.
    - zgrid: Vertical grid.
    - time_array: Array of time steps.
    - output_path: Directory to save the plot.
    """

    # Apply radial masks
    radial_mask = (xgrid >= 0.9) & (xgrid <= 1.2)
    radial_mask_2 = (xgrid >= 0.9) & (xgrid <= 1.2)

    nz = len(zgrid)
    ny = len(ygrid)


    if dust:
        # Calculate the Roche density at each radial location
        roche_density_masked = 9 / (4 * np.pi * xgrid[radial_mask]**3)

    # Initialize arrays to hold the computed values
    nsteps = data_arrays['gasdens'].shape[-1]
    time_array = time_array[:nsteps]

    vort_min = np.zeros(nsteps)  
    alpha_r = np.zeros(nsteps)
    p_HS = np.zeros(nsteps) # NEW: Power law slopes near r=1
    q_HS = np.zeros(nsteps)
    alpha_r_HS = np.zeros(nsteps)
    rms_vr = np.zeros(nsteps)
    rms_vphi = np.zeros(nsteps)
    rms_vz = np.zeros(nsteps)
    max_vz = np.zeros(nsteps)
    min_vz = np.zeros(nsteps)
    max_epsilon = np.zeros(nsteps)
    H_d_array = np.zeros(nsteps)
    m_gas = np.zeros(nsteps)
    m_dust = np.zeros(nsteps) if dust else None
    avg_metallicity = np.zeros(nsteps)
    vortr_avg = np.zeros(nsteps)  # New
    vortz_avg = np.zeros(nsteps)  # New

    # New arrays for vertical profiles
    rms_vr_profile = np.zeros(nz)
    rms_vphi_profile = np.zeros(nz)
    rms_vz_profile = np.zeros(nz)

    roche_times = []

    # Read disk aspect ratio (H_g = h0 * r)
    if IDEFIX:
        log_file = os.path.join(output_path, "idefix.0.log")
        parameters = read_parameters(log_file, IDEFIX=True)
        h0 = float(parameters.get("h0", 0.05))  # Default to 0.05 if not found
        flaringIndex = float(parameters.get("flaringindex", 0.0))
        H_g = h0  # This is used as h0; actual H_g varies with r
        aspectratio = h0
    else:
        summary_file = os.path.join(output_path, "summary0.dat")
        parameters = read_parameters(summary_file)
        aspectratio = float(parameters['ASPECTRATIO'])  # e.g., h = H/r
        flaringIndex = float(parameters['FLARINGINDEX'])  
        H_g = aspectratio

    # Precompute the std of zgrid if it's constant
    zgrid_std = np.std(zgrid)

    r = xgrid[np.newaxis, :, np.newaxis]  

    # Smoothing in (azimuthal, radial) directions
    nsx = 100  
    nsy = 10   

    # Compute the initial gas and dust mass at t_idx=0 for scaling
    gasdens_initial = data_arrays['gasdens'][:, :, :, 0]
    gasdens_initial = gasdens_initial[:, radial_mask, :]
    m_gas_initial = np.sum(gasdens_initial)
    m_dust_initial = None
    if dust:
        dust1dens_initial = data_arrays['dust1dens'][:, :, :, 0]
        dust1dens_initial = dust1dens_initial[:, radial_mask, :]
        m_dust_initial = np.sum(dust1dens_initial)

    
# CODE BLOCK 2: Replace the corresponding section in your plot_alpha_r function.

    print('COMPUTING TIME EVOLUTION OF VARIOUS QUANTITIES')

    # --- Prepare data for parallel processing ---
    # Pack static parameters that are the same for all workers
    static_params = {
        'xgrid': xgrid, 'ygrid': ygrid, 'zgrid': zgrid,
        'radial_mask': radial_mask, 'hs_mask': (xgrid >= 1.0 - 0.5 * H_g) & (xgrid <= 1.0 + 0.5 * H_g),
        'nz': nz, 'ny': ny, 'h0': aspectratio, 'flaringIndex': flaringIndex,
        'm_gas_initial': m_gas_initial, 'm_dust_initial': m_dust_initial,
        'zgrid_std': zgrid_std, 'dust': dust,
    }
    if dust:
        static_params['roche_density_masked'] = roche_density_masked
    
    # Pack initial condition slices (t=0)
    initial_slice = {
        'gasvx0': data_arrays['gasvx'][:, radial_mask, :, 0],
        'gasvx0_HS': data_arrays['gasvx'][:, static_params['hs_mask'], :, 0]
    }

# REPLACEMENT CODE BLOCK: Use this updated helper function.

    # I've added 'import joblib' and 'import traceback' for more specific error handling.
    # Make sure these imports are at the top of your main script.
    import joblib
    import traceback

    def run_parallel_with_fallback(tasks, nsteps):
        """
        Tries parallel processing. If a non-memory error occurs, it stops and
        runs sequentially to provide a clear traceback for debugging.
        """
        try:
            for n_jobs in [16, 8, 4, 2, 1]:
                if n_jobs > os.cpu_count(): continue
                try:
                    print(f"Trying with {n_jobs} parallel workers...")
                    with parallel_backend("loky", inner_max_num_threads=1):
                        with tqdm_joblib(tqdm(desc=f"Processing ({n_jobs} workers)", total=nsteps)):
                            # The 'return' will exit the function upon success
                            return Parallel(n_jobs=n_jobs, timeout=600)(tasks)

                # Catch ONLY memory/timeout errors to retry
                except OSError as e:
                     print(f"❌ Worker error with {n_jobs} workers: {e}. Retrying with fewer...")
                     time.sleep(2)
                     continue
        except Exception as e:
            print("\n❌ A critical, non-recoverable error occurred during parallel processing.")
            print("The full traceback from the parallel failure is:")
            traceback.print_exc()
            print("\nFalling back to a sequential loop to pinpoint the exact error location...")
            # Fall through to the sequential run for a cleaner traceback
            pass

        # If all parallel attempts failed or a critical error occurred, run sequentially for debugging.
        print("⚠️ Running sequentially to identify the point of failure...")
        results = []
        for i, delayed_func in enumerate(tqdm(tasks, desc="Sequential Debug Run")):
            try:
                # delayed_func is a tuple of (function, args, kwargs)
                func, args, kwargs = delayed_func
                results.append(func(*args, **kwargs))
            except Exception:
                # When an exception happens, print context and re-raise to stop everything.
                print(f"\n\n🛑 FAILURE ON TIME STEP INDEX: {i} 🛑")
                print("The error occurred inside the 'worker_process_timestep' function.")
                print("The traceback below points to the exact line of the bug:")
                print("------------------- TRACEBACK -------------------")
                raise # Re-raise the exception to get the full traceback and stop the script

        return results


    # Create a list of delayed tasks. Each task gets only the data it needs.
    tasks = []
    for t_idx in range(nsteps):
        data_slice = {
            'gasdens_t': data_arrays['gasdens'][:, :, :, t_idx],
            'gasvx_t':   data_arrays['gasvx'][:, :, :, t_idx],
            'gasvy_t':   data_arrays['gasvy'][:, :, :, t_idx],
            'gasvz_t':   data_arrays['gasvz'][:, :, :, t_idx],
        }
        if 'gasenergy' in data_arrays:
            data_slice['gasenergy_t'] = data_arrays['gasenergy'][:, :, :, t_idx]
        if dust:
            data_slice['dust1dens_t'] = data_arrays['dust1dens'][:, :, :, t_idx]
            data_slice['dust1vx_t'] = data_arrays['dust1vx'][:, :, :, t_idx]
            data_slice['dust1vy_t'] = data_arrays['dust1vy'][:, :, :, t_idx]
            data_slice['dust1vz_t'] = data_arrays['dust1vz'][:, :, :, t_idx]

        # Arguments for the worker function
        args = (t_idx, time_array[t_idx], data_slice, initial_slice, static_params)
        tasks.append(delayed(worker_process_timestep)(*args))

    results = run_parallel_with_fallback(tasks, nsteps)


    # --- Unpack results ---
    roche_times = []
    for t_idx, result_tuple in enumerate(results):
        (alpha_r_val, alpha_r_HS_val, p_HS_val, q_HS_val, rms_vr_val, rms_vphi_val, 
         rms_vz_val, max_vz_val, min_vz_val, max_epsilon_val, H_d_val, m_gas_val, 
         m_dust_val, avg_metallicity_val, rms_vr_z, rms_vphi_z, rms_vz_z, 
         vort_min_val, vortr_avg_val, vortz_avg_val, roche_reached) = result_tuple

        alpha_r[t_idx] = alpha_r_val
        alpha_r_HS[t_idx] = alpha_r_HS_val
        p_HS[t_idx] = p_HS_val
        q_HS[t_idx] = q_HS_val
        rms_vr[t_idx] = rms_vr_val
        rms_vphi[t_idx] = rms_vphi_val
        rms_vz[t_idx] = rms_vz_val
        max_vz[t_idx] = max_vz_val
        min_vz[t_idx] = min_vz_val
        max_epsilon[t_idx] = max_epsilon_val
        H_d_array[t_idx] = H_d_val
        m_gas[t_idx] = m_gas_val
        vort_min[t_idx] = vort_min_val
        vortr_avg[t_idx] = vortr_avg_val
        vortz_avg[t_idx] = vortz_avg_val

        if dust:
            m_dust[t_idx] = m_dust_val
            avg_metallicity[t_idx] = avg_metallicity_val

        if roche_reached:
            roche_times.append(time_array[t_idx])

        # Accumulate vertical profiles for time-averaging
        rms_vr_profile += rms_vr_z
        rms_vphi_profile += rms_vphi_z
        rms_vz_profile += rms_vz_z

    # Normalize the profiles by the number of time steps
    if nsteps > 0:
        rms_vr_profile /= nsteps
        rms_vphi_profile /= nsteps
        rms_vz_profile /= nsteps

    save_simulation_quantities(output_path, time_array, alpha_r, alpha_r_HS, p_HS, q_HS, rms_vr, rms_vphi, rms_vz, max_vz, min_vz, max_epsilon, H_d_array, roche_times, m_gas, m_dust, avg_metallicity, vort_min, rms_vr_profile, rms_vphi_profile, rms_vz_profile, vortr_avg, vortz_avg)


    import matplotlib.pyplot as plt

    # Create figure and two vertically stacked subplots (shared x-axis)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

    def safe_plot(ax, x, y, label, log=False):
        mask = y > 0 if log else np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], label=label)
        else:
            print(f"⚠️  Skipping {label}: no valid values to plot.")

    # --- Panel 1: Turbulence and Dust Quantities ---
    ax1 = axs[0]
    safe_plot(ax1, time_array, alpha_r, r'$\alpha_r(t)$', log=True)
    safe_plot(ax1, time_array, alpha_r_HS, r'$\alpha_{r,HS}(t)$', log=True)
    safe_plot(ax1, time_array, rms_vz / H_g, r'RMS$(v_z)/c_0$', log=True)

    if dust:
        safe_plot(ax1, time_array, max_epsilon, r'Max $\epsilon$', log=True)
        safe_plot(ax1, time_array, H_d_array / H_g, r'Dust Scale Height $H_d/H_g$', log=True)

    ax1.set_yscale('log')
    ax1.set_ylabel('Turbulence / Dust Values')
    ax1.set_title(r'Turbulence $\alpha_r(t)$, RMS$(v_z)$, Max $\epsilon$, $H_d/H_g$')
    ax1.grid(True)
    ax1.legend()

    # --- Panel 2: Slope Parameters ---
    ax2 = axs[1]
    safe_plot(ax2, time_array, p_HS, r'$p_{HS}(t)$', log=False)
    safe_plot(ax2, time_array, q_HS, r'$q_{HS}(t)$', log=False)

    ax2.set_xlabel('Time [Orbits]')
    ax2.set_ylabel(r'Pressure / Density Slopes')
    ax2.set_title(r'Radial Slopes $p_{HS}(t)$ and $q_{HS}(t)$')
    ax2.grid(True)
    ax2.legend()

    # Save the figure
    pdf_filename = f"{output_path}_alpha_r_panels_over_time.pdf"
    plt.savefig(os.path.join(output_path, pdf_filename))
    plt.close()

    output_filename = os.path.join(output_path, pdf_filename)

    print(f"✅ Alpha_r, RMS dust velocity, max epsilon, and scale height plot saved to {pdf_filename}")

    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")





def plot_pressure_gradient_deviation(data_arrays, xgrid, time_array, output_path, planet=False):
    """
    Plot the space-time contour of the radial pressure gradient deviation from its initial value.
    """
    import matplotlib.pyplot as plt
    from data_reader import read_parameters
    import os

    # === Font settings ===
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 24
    })

    # === Check for gasenergy ===
    if "gasenergy" in data_arrays:
        pressure = data_arrays["gasenergy"]
    else:
        # --- Construct pressure manually ---
        gasdens = data_arrays["gasdens"]
        idefix_logfile = os.path.join(output_path, "idefix.0.log")
        params = read_parameters(idefix_logfile, IDEFIX=True)
        h0 = float(params.get("h0", 0.05))
        flaring_index = float(params.get("flaringIndex", 0.0))
        cs2_profile = h0**2 * xgrid**(2 * flaring_index - 1.0)  # Shape: [nx]

        # Expand cs2 to match dimensions of gasdens: [ny, nx, nz, nt]
        cs2_full = cs2_profile[np.newaxis, :, np.newaxis, np.newaxis]
        pressure = gasdens * cs2_full

    # === Radial pressure gradient ===
    dp_dr = np.gradient(pressure, xgrid, axis=1)

    # (Continue with the rest of the function...)
    
    # Calculate the initial radial pressure gradient
    initial_dp_dr = dp_dr[..., 0]  # Initial radial pressure gradient snapshot (ny, nx, nz)

    # Calculate scaling factor \rho_g c_0^2 from initial pressure at nx/2, nz/2
    rho_g_c0_sq = pressure[pressure.shape[0] // 2, pressure.shape[1] // 2, pressure.shape[2] // 2, 0] 

    # Calculate deviation of radial pressure gradient from initial value, scaled by \rho_g c_0^2
    dp_dr_deviation = (dp_dr - initial_dp_dr[..., np.newaxis]) / rho_g_c0_sq  # Shape (ny, nx, nz, nts)

    # Integrate over the azimuthal and vertical directions (y and z axes)
    dp_dr_deviation_avg = np.mean(dp_dr_deviation, axis=(0, 2))  # Shape (nx, nts)

    # Apply radial mask
    radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
    xgrid_masked = xgrid[radial_mask]
    dp_dr_dev_masked = np.cbrt(dp_dr_deviation_avg[radial_mask, :])  # Shape (masked nx, nts)

    # Create the contour plot
    plt.figure(figsize=(10, 6))
    cp = plt.imshow(dp_dr_dev_masked.T, extent=[xgrid_masked.min(), xgrid_masked.max(), time_array.min(), time_array.max()],
                    aspect='auto', origin='lower', cmap='seismic', vmin=np.min(dp_dr_dev_masked),
                    vmax=np.max(dp_dr_dev_masked))
    plt.colorbar(cp, label=r'$(\Delta (\partial P / \partial r))^{1/3}$')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel('Orbits')
    #plt.title(r'Space-Time Plot of Scaled Radial Pressure Gradient Deviation from Initial Value')

    # Extract the subdirectory name for saving the plot and npz file
    subdir_name = os.path.basename(output_path)
    
    # Save the plot
    pdf_filename = f"{subdir_name}_space_time_plot_pressure_gradient_deviation.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"#######################################")
    print(f"✅ Pressure gradient deviation space-time plot saved to {output_filepath}")
    print(f"#######################################")

    # Save the required data in an npz file
    npz_filename = f"{subdir_name}_pressure_gradient_deviation_data.npz"
    npz_filepath = os.path.join(output_path, npz_filename)
    np.savez(npz_filepath, pressure_dev=dp_dr_dev_masked.T, time_array=time_array, xgrid=xgrid_masked)
    print(f"#######################################")
    print(f"✅ Pressure gradient deviation data saved to {npz_filepath}")
    print(f"#######################################")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")



def plot_vertically_averaged_gas_density(data_arrays, xgrid, time_array, output_path):
    """
    Plot the space-time contour of the vertically averaged gas density.

    Parameters:
    - data_arrays (dict): Dictionary containing data arrays including 'gasdens' (gas density).
    - xgrid (array): The grid array for the radial coordinate.
    - time_array (array): The array of time steps.
    - output_path (str): Path to save the plot.
    """

    # Set global font size and styling
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 24
    })

    # Retrieve gas density (gasdens) data
    gas_density = data_arrays['gasdens']  # Shape (ny, nx, nz, nts)

    # Calculate the vertically averaged gas density
    vertically_averaged_gas_density = np.mean(gas_density, axis=2)  # Average over z-axis (Shape: ny, nx, nts)

    # Apply radial mask
    radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
    xgrid_masked = xgrid[radial_mask]
    gas_density_masked = vertically_averaged_gas_density[:, radial_mask, :]  # Masked gas density (Shape: ny, masked nx, nts)

    # Further average over azimuthal direction (y-axis)
    gas_density_avg = np.mean(gas_density_masked, axis=0)  # Shape: (masked nx, nts)

    # Create the contour plot
    plt.figure(figsize=(10, 6))
    cp = plt.imshow(gas_density_avg.T, extent=[xgrid_masked.min(), xgrid_masked.max(), time_array.min(), time_array.max()],
                    aspect='auto', origin='lower', cmap='viridis', vmin=np.min(gas_density_avg),
                    vmax=np.max(gas_density_avg))
    plt.colorbar(cp, label=r'$\langle \rho_\mathrm{gas} \rangle_z$')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel('Orbits')
    plt.title(r'Space-Time Plot of Vertically Averaged Gas Density')

    # Extract the subdirectory name for saving the plot and npz file
    subdir_name = os.path.basename(output_path)
    
    # Save the plot
    pdf_filename = f"{subdir_name}_space_time_plot_gas_density.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"#######################################")
    print(f"✅ Gas density space-time plot saved to {output_filepath}")
    print(f"#######################################")

    # Save the required data in an npz file
    npz_filename = f"{subdir_name}_gas_density_data.npz"
    npz_filepath = os.path.join(output_path, npz_filename)
    np.savez(npz_filepath, gas_density_avg=gas_density_avg.T, time_array=time_array, xgrid=xgrid_masked)
    print(f"#######################################")
    print(f"✅ Gas density data saved to {npz_filepath}")
    print(f"#######################################")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")



import os
import matplotlib.pyplot as plt

def plot_debugging_profiles(data_arrays, xgrid, ygrid, zgrid, time_array, output_path):
    """Plot debugging profiles to understand the issue with the data."""
    # Get the index for the initial time step: time=0
    gasdens = data_arrays['gasdens']
    nx = gasdens.shape[1]
    ny = gasdens.shape[0]
    nz = gasdens.shape[2]

    # Set the initial time index
    time_index = 0

    # Create a figure for the plots
    plt.figure(figsize=(12, 6))

    # Plot xgrid versus gasdens(0, :, nz/2, time_index)
    plt.subplot(1, 3, 1)
    plt.plot(xgrid, gasdens[0, :, nz//2, time_index], label=f'Time = {time_array[time_index]}')
    plt.xlabel('Radial Location (x)')
    plt.ylabel('gasdens(y=0, z=mid-plane)')
    plt.title('Raw Profile: xgrid vs gasdens(y=0, :, nz/2)')
    plt.legend()
    plt.grid(True)

    # Plot ygrid versus gasdens(:, nx/2, nz/2, time_index) if ny > 1
    if ny > 1:
        plt.subplot(1, 3, 2)
        plt.plot(ygrid, gasdens[:, nx//2, nz//2, time_index], label=f'Time = {time_array[time_index]}')
        plt.xlabel('Azimuthal Location (y)')
        plt.ylabel('gasdens(:, x=nx/2, z=mid-plane)')
        plt.title('Raw Profile: ygrid vs gasdens(:, x=nx/2, nz/2)')
        plt.legend()
        plt.grid(True)

    # Plot zgrid versus gasdens(0, nx/2, :, time_index)
    plt.subplot(1, 3, 3)
    plt.plot(zgrid, gasdens[0, nx//2, :, time_index], label=f'Time = {time_array[time_index]}')
    plt.xlabel('Vertical Location (z)')
    plt.ylabel('gasdens(y=0, x=mid-radial)')
    plt.title('Raw Profile: zgrid vs gasdens(y=0, x=nx/2, :)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Extract the subdirectory name
    subdir_name = os.path.basename(output_path)

    # Save the plot with the subdirectory name in the file name
    pdf_filename = f"{subdir_name}_debugging_profiles_gasdens.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"#######################################")
    print(f"✅ Debug plot saved to {output_filepath}")
    print(f"#######################################")

    # Call the scp_transfer function
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


import numpy as np
import matplotlib.pyplot as plt
import os

def plot_radial_profiles_over_time(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, parameters=None, IDEFIX=False):
    """
    Plot azimuthally averaged radial profiles of v_r, v_phi (as deviation), v_z (if 3D), density, pressure,
    and N_r^2 (only if NOT isothermal) for multiple times.
    """
    vx = data_arrays['gasvx']  # azimuthal
    vy = data_arrays['gasvy']  # radial
    rho = data_arrays['gasdens']
    press = data_arrays['gasenergy']  # assuming gasenergy = pressure

    import re

    isothermal = False
    def_file = os.path.join(output_path, "definitions.hpp")
    if os.path.exists(def_file):
        with open(def_file, 'r') as f:
            for line in f:
                line_clean = re.sub(r'//.*', '', line).strip()
                if re.fullmatch(r'#\s*define\s+ISOTHERMAL', line_clean):
                    isothermal = True
                    print("✅ ISOTHERMAL mode detected via definitions.hpp")
                    break

    GAMMA = parameters['GAMMA'] if parameters and 'GAMMA' in parameters else 1.4
    aspectratio = parameters['ASPECTRATIO'] if parameters and 'ASPECTRATIO' in parameters else 0.05

    nt = vx.shape[3]
    ny, nx, nz = vx.shape[:3]
    has_vz_panel = nz > 0
    include_Nr2 = not isothermal

    if nt < 10:
        time_indices = np.arange(nt)
    else:
        time_indices = np.linspace(0, nt-1, 10, dtype=int)

    # --- Determine number of panels ---
    n_panels = 4  # vr, vphi deviation, rho, press
    if has_vz_panel:
        n_panels += 1
    if include_Nr2:
        n_panels += 1

    fig, axs = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)

    # Precompute initial vphi for deviation
    vphi_init = np.mean(np.mean(vx[:, :, :, 0], axis=0), axis=1)

    for t in time_indices:
        vr_avg    = np.mean(np.mean(vy[:, :, :, t], axis=0), axis=1)
        vphi_avg  = np.mean(np.mean(vx[:, :, :, t], axis=0), axis=1)
        vphi_dev  = vphi_avg - vphi_init
        rho_avg   = np.mean(np.mean(rho[:, :, :, t], axis=0), axis=1)
        press_avg = np.mean(np.mean(press[:, :, :, t], axis=0), axis=1)
        label = f"{time_array[t]:.1f} orbits"

        axs[0].plot(xgrid, vr_avg, label=label)
        axs[1].plot(xgrid, vphi_dev, label=label)

        panel_idx = 2
        if has_vz_panel:
            vz = data_arrays['gasvz']
            vz_avg = np.mean(np.mean(vz[:, :, :, t], axis=0), axis=1)
            axs[panel_idx].plot(xgrid, vz_avg, label=label)
            panel_idx += 1

        axs[panel_idx].plot(xgrid, rho_avg, label=label)
        panel_idx += 1
        axs[panel_idx].plot(xgrid, press_avg, label=label)
        panel_idx += 1

        if include_Nr2:
            P_t = press[:, :, :, t] * (GAMMA - 1)
            rho_t = rho[:, :, :, t]
            S_t = np.log(P_t / (rho_t ** GAMMA))
            dP_dr_t = np.gradient(P_t, xgrid, axis=1)
            dS_dr_t = np.gradient(S_t, xgrid, axis=1)
            N2_full = -(1.0 / GAMMA) * (1.0 / rho_t) * dP_dr_t * dS_dr_t
            N2_avg = np.mean(np.mean(N2_full, axis=0), axis=1)
            axs[panel_idx].plot(xgrid, N2_avg, label=label)

    # --- Set panel labels and titles ---
    axs[0].set_ylabel(r"$\langle v_r \rangle_{\phi,z}$")
    axs[0].set_title("Radial Velocity Profile Over Time")

    axs[1].set_ylabel(r"$\Delta v_\phi$")
    axs[1].set_title("Azimuthal Velocity Deviation from Initial")

    panel_idx = 2
    if has_vz_panel:
        axs[panel_idx].set_ylabel(r"$\langle v_z \rangle_{\phi,z}$")
        axs[panel_idx].set_title("Vertical Velocity Profile Over Time")
        panel_idx += 1

    axs[panel_idx].set_ylabel(r"$\langle \rho \rangle_{\phi,z}$")
    axs[panel_idx].set_title("Gas Density Profile Over Time")
    panel_idx += 1

    axs[panel_idx].set_ylabel(r"$\langle P \rangle_{\phi,z}$")
    axs[panel_idx].set_title("Gas Pressure Profile Over Time")
    panel_idx += 1

    if include_Nr2:
        axs[panel_idx].set_ylabel(r"$N_r^2$")
        axs[panel_idx].set_title(r"Radial Buoyancy Frequency $N_r^2$ Profile Over Time")
        axs[panel_idx].set_xlabel(r"Radial position $r/r_0$")
    else:
        axs[panel_idx - 1].set_xlabel(r"Radial position $r/r_0$")

    # Finalize all axes
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

    # Set N_r^2 y-limits if included
    if include_Nr2:
        nr2_ax = axs[-1]
        y_data_all = [line.get_ydata() for line in nr2_ax.get_lines()]
        y_data_concat = np.concatenate(y_data_all)
        if not np.all(np.isnan(y_data_concat)):
            y_min, y_max = np.nanmin(y_data_concat), np.nanmax(y_data_concat)
            if y_min < -0.5 or y_max > 0.5:
                nr2_ax.set_ylim(-0.5, 0.5)
            else:
                nr2_ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    subdir_name = os.path.basename(output_path)
    pdf_filename = f"{subdir_name}_radial_profiles.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"✅ Radial profiles saved to: {output_filepath}")

    if IDEFIX:
        scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles/IDEFIX", "mariuslehmann")
    else:
        scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")
