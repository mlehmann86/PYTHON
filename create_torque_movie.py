import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Assuming these custom modules are in your PYTHONPATH
from data_reader import read_single_snapshot, read_parameters, reconstruct_grid, determine_nt
from data_storage import determine_base_path, scp_transfer
from planet_data import read_alternative_torque, extract_planet_mass_and_migration

# --- Self-Contained, Correct Functions ---
def compute_torque_gravity(gasdens, xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index):
    ny, nx, nz = gasdens.shape
    r_p = 1.0; phi_p = 0.0
    torque_density = np.zeros(nx)
    dphi = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0] if nz > 1 else 1.0
    
    for j in range(ny):
        phi = ygrid[j]
        delta_phi = (phi - phi_p + np.pi) % (2 * np.pi) - np.pi
        for i in range(nx):
            r_cell = xgrid[i]
            smoothing = h0 * (r_cell**flaring_index) * r_cell * thick_smooth
            smoothing_sq = smoothing**2
            dx = r_cell - r_p
            dy = r_cell * delta_phi
            for k in range(nz):
                z_cell = zgrid[k]
                dist_to_planet_sq = dx**2 + dy**2 + (z_cell - 0.0)**2
                denom = (dist_to_planet_sq + smoothing_sq)**1.5
                fphi = -qp * dy / (denom + 1e-30)
                torque_density[i] += gasdens[j, i, k] * r_cell**2 * fphi * dphi * dz
    return torque_density

def get_torque_map(gasdens, xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index):
    ny, nx, nz = gasdens.shape
    r_3d = xgrid[np.newaxis, :, np.newaxis]; phi_3d = ygrid[:, np.newaxis, np.newaxis]
    z_3d = zgrid[np.newaxis, np.newaxis, :]
    dx = r_3d - 1.0
    dy = r_3d * ((phi_3d - 0.0 + np.pi) % (2*np.pi) - np.pi)
    smoothing = h0 * (r_3d**flaring_index) * r_3d * thick_smooth
    denom = (dx**2 + dy**2 + (z_3d-0.0)**2 + smoothing**2)**1.5
    f_phi = -qp * dy / (denom + 1e-30)
    torque_integrand_3d = gasdens * (r_3d**2) * f_phi
    return np.mean(torque_integrand_3d, axis=2)

def calculate_vorticity(gasvx, gasvy, xgrid, ygrid, nz):
    gasvx_2d = np.mean(gasvx, axis=2) if nz > 1 else gasvx.squeeze(axis=2)
    gasvy_2d = np.mean(gasvy, axis=2) if nz > 1 else gasvy.squeeze(axis=2)
    r = xgrid[np.newaxis, :]
    v_phi_inertial = gasvx_2d + r
    grad_r_v_phi = np.gradient(r * v_phi_inertial, xgrid, axis=1)
    grad_v_r = np.gradient(gasvy_2d, ygrid, axis=0)
    return (1.0 / (r + 1e-12)) * (grad_r_v_phi - grad_v_r)

def calculate_torque_for_frame(args):
    # Unpack arguments
    snapshot_idx, base_path, params, grid_info, idefix, qp = args
    xgrid, ygrid, zgrid, _, _, _ = grid_info

    # --- SIMPLIFIED: A single call to the unified reader ---
    data_dict = read_single_snapshot(
        base_path, snapshot_idx, read_gasdens=True,
        IDEFIX=idefix, params=params, grid_info=grid_info
    )[0]

    # Parameter extraction logic remains the same
    if idefix:
        h0 = float(params.get("h0"))
        aspect_ratio = float(params.get("h0"))
        thick_smooth = float(params.get("smoothing")) / aspect_ratio
        flaring_index = float(params.get("flaringindex"))
    else:
        h0 = float(params.get("ASPECTRATIO"))
        thick_smooth = float(params.get("THICKNESSSMOOTHING"))
        flaring_index = float(params.get("FLARINGINDEX"))

    radial_torque_profile = compute_torque_gravity(data_dict["gasdens"], xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index)
    dr = xgrid[1] - xgrid[0]
    torque_on_disk = np.sum(radial_torque_profile) * dr
    return snapshot_idx, -torque_on_disk



def load_equilibrium_state(base_path, params, grid_info, idefix=False):
    xgrid, ygrid, zgrid, ny, nx, nz = grid_info
    
    # For IDEFIX, or as a fallback for FARGO, use snapshot 0
    if idefix:
        print("ℹ️  IDEFIX mode: Using snapshot 0 for the initial equilibrium state.")
        return read_single_snapshot(
            base_path, 0, read_gasvx=True, read_gasvy=True, read_gasdens=True,
            IDEFIX=True, params=params, grid_info=grid_info
        )[0]

    # FARGO-specific logic for I0 files
    i0_path = os.path.join(base_path, 'gasdensI0.dat')
    if os.path.exists(i0_path):
        print("✅ Found 'I0' files. Using them for the initial equilibrium state.")
        try:
            data0 = {
                'gasdens': np.fromfile(os.path.join(base_path, 'gasdensI0.dat'), dtype=np.float64).reshape((ny, nx, nz), order='F'),
                'gasvx': np.fromfile(os.path.join(base_path, 'gasvxI0.dat'), dtype=np.float64).reshape((ny, nx, nz), order='F'),
                'gasvy': np.fromfile(os.path.join(base_path, 'gasvyI0.dat'), dtype=np.float64).reshape((ny, nx, nz), order='F')
            }
            return data0
        except Exception as e:
            print(f"❌ Warning: Failed to read 'I0' files: {e}. Falling back to snapshot 0.")
    else:
        print("ℹ️  'I0' files not found. Using snapshot 0 for the initial equilibrium state.")

    # Fallback for FARGO if I0 files are not found/readable
    return read_single_snapshot(
        base_path, 0, read_gasvx=True, read_gasvy=True, read_gasdens=True,
        IDEFIX=False, params=params, grid_info=grid_info
    )[0]


# USE THIS NEW BLOCK INSTEAD
def process_and_save_frame(args):
    # Unpack the new, slightly different set of arguments
    (frame_idx, snapshot_idx, base_path, params, grid_info, idefix, qp,
     initial_vorticity, initial_gas_density_norm, torque_map_equilibrium,
     time_array, map_torque_history, tqwk_time_full, tqwk_torque_norm_full,
     tqwk_torque_smoothed_full, tmp_dir, Gamma0, plot_range_r,
     radial_mask_indices, ylim_ax4, avg_source) = args

    xgrid, ygrid, zgrid, ny, nx, nz = grid_info

    # A single, unified call to read the snapshot data
    data_dict = read_single_snapshot(
        base_path, snapshot_idx, read_gasvx=True, read_gasvy=True, read_gasdens=True,
        IDEFIX=idefix, params=params, grid_info=grid_info
    )[0]

    # The if/else is still needed here because the parameter NAMES are different
    if idefix:
        h0 = float(params.get("h0"))
        aspect_ratio = float(params.get("h0"))
        thick_smooth = float(params.get("smoothing")) / aspect_ratio
        flaring_index = float(params.get("flaringindex"))
    else:
        h0 = float(params.get("ASPECTRATIO"))
        thick_smooth = float(params.get("THICKNESSSMOOTHING"))
        flaring_index = float(params.get("FLARINGINDEX"))
    
    gasdens = data_dict["gasdens"]
    nz = zgrid.shape[0]; ny = ygrid.shape[0]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.4, hspace=0.5)
    ax3_right = ax3.twinx()
    
    current_time = time_array[snapshot_idx]
    current_avg_torque = np.nan
    
    if avg_source == 'tqwk' and tqwk_time_full is not None:
        source_data_avg, source_time_avg, avg_label = tqwk_torque_norm_full, tqwk_time_full, 'Cumulative Avg (tqwk0)'
    else:
        source_data_avg, source_time_avg, avg_label = map_torque_history, time_array, 'Cumulative Avg (Snapshots)'

    plot_mask_avg = ~np.isnan(source_data_avg) & (source_time_avg <= current_time)
    if np.any(plot_mask_avg):
        history_points = source_data_avg[plot_mask_avg]
        cumulative_avg = np.cumsum(history_points) / np.arange(1, len(history_points) + 1)
        current_avg_torque = cumulative_avg[-1]

    title_text = f"Time: {current_time:.2f} Orbits   |   Cumulative Avg Torque: {current_avg_torque:.3f}"
    fig.suptitle(title_text, fontsize=14)

    # Panel 1
    current_vorticity = calculate_vorticity(data_dict['gasvx'], data_dict['gasvy'], xgrid, ygrid, nz)
    vorticity_dev = current_vorticity - initial_vorticity
    vorticity_dev_shifted = np.roll(vorticity_dev, shift=(ny // 2), axis=0)
    vmax = np.percentile(np.abs(vorticity_dev_shifted[:, radial_mask_indices]), 99.5)
    im1 = ax1.imshow(vorticity_dev_shifted, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
                     aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_title(r"Vorticity Deviation $(\Delta \omega_z)$"); ax1.set_xlabel(r"$r/r_0$"); ax1.set_ylabel(r"$\phi$ (shifted)")
    ax1.set_xlim(plot_range_r)
    cax1 = make_axes_locatable(ax1).append_axes("top", size="5%", pad=0.35)
    fig.colorbar(im1, cax=cax1, orientation='horizontal'); cax1.xaxis.set_ticks_position('top')

    # Panel 2
    torque_map_current = get_torque_map(gasdens, xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index)
    torque_deviation_on_planet = -(torque_map_current - torque_map_equilibrium)
    torque_deviation_shifted = np.roll(torque_deviation_on_planet, shift=(ny // 2), axis=0)
    scaled_torque_map = torque_deviation_shifted / Gamma0 if Gamma0 > 0 else torque_deviation_shifted
    vmax_map = np.percentile(np.abs(scaled_torque_map[:, radial_mask_indices]), 98)
    im2 = ax2.imshow(scaled_torque_map, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
               aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax_map, vmax=vmax_map)
    ax2.set_title(r"Torque Deviation on Planet $\delta \Gamma / \Gamma_0$"); ax2.set_xlabel(r"$r/r_0$"); ax2.set_ylabel(r"$\phi$ (shifted)")
    ax2.set_xlim(plot_range_r)
    cax2 = make_axes_locatable(ax2).append_axes("top", size="5%", pad=0.35)
    fig.colorbar(im2, cax=cax2, orientation='horizontal'); cax2.xaxis.set_ticks_position('top')
    
    # Panel 3
    current_gas_density = np.mean(gasdens, axis=(0, 2))
    l1, = ax3.plot(xgrid, current_gas_density, color='red', lw=2, label=r"Current $\rho_g$")
    l2, = ax3.plot(xgrid, initial_gas_density_norm, color='black', lw=1.5, ls=':', label=r"Initial $\rho_g$")
    ax3.set_xlabel(r"$r/r_0$"); ax3.set_ylabel(r"Density", color='red')
    ax3.set_xlim(plot_range_r)
    ax3.axvline(1.0, color='grey', linestyle='--', lw=1)
    plot_indices = np.where((xgrid >= plot_range_r[0]) & (xgrid <= plot_range_r[1]))
    visible_density_data = current_gas_density[plot_indices]
    min_rho, max_rho = np.min(visible_density_data), np.max(visible_density_data)
    padding = (max_rho - min_rho) * 0.15
    if padding < 1e-9:
        padding = 0.1 * max_rho if max_rho > 1e-9 else 0.01
    ax3.set_ylim(min_rho - padding, max_rho + padding)
    
    radial_torque_profile = compute_torque_gravity(gasdens, xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index)
    scaled_radial_profile_planet = (-radial_torque_profile / Gamma0) if Gamma0 > 0 else -radial_torque_profile
    l3, = ax3_right.plot(xgrid, scaled_radial_profile_planet, color='darkviolet', lw=2, label=r"$d\Gamma_{\mathrm{planet}}/dr$")
    ax3_right.set_ylabel(r"$\frac{d\Gamma}{dr} / \Gamma_0$", color='darkviolet')
    ax3_right.axhline(0, color='black', lw=1, ls='--')
    ax3.legend([l1, l2, l3], [l.get_label() for l in [l1,l2,l3]], loc='upper left', fontsize='small')

    # Panel 4
    map_plot_mask = ~np.isnan(map_torque_history) & (time_array <= current_time)
    ax4.plot(time_array[map_plot_mask], map_torque_history[map_plot_mask], 'x', markersize=6, color='red', label='Torque (Snapshots)', zorder=5)
    if tqwk_time_full is not None:
        ax4.plot(tqwk_time_full, tqwk_torque_smoothed_full, label='Torque (tqwk0)', ls='--', lw=2, color='deepskyblue', zorder=10)
    if np.any(plot_mask_avg):
        ax4.plot(source_time_avg[plot_mask_avg], cumulative_avg, color='black', lw=2, ls='-', label=avg_label, zorder=50)
    ax4.set_xlim(0, min(time_array.max(), 1000)); ax4.set_xlabel("Time [Orbits]"); ax4.set_ylabel(r"$\Gamma_{\mathrm{planet}} / \Gamma_0$")
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5); ax4.legend(loc='upper right', fontsize='small')
    ax4.set_yscale('linear'); ax4.set_ylim(ylim_ax4)
    ax4.axhline(0, color='black', lw=1, ls='--', zorder=100)

    frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:05d}.png")
    fig.savefig(frame_path, dpi=150); plt.close(fig)

def create_advanced_movie(simname, test_mode=False, parallel_cores=None, avg_source='tqwk', idefix=False):
    plot_range_r = [0.6, 1.4]
    
    print(f"🎬 Starting movie generation for: {simname} (IDEFIX mode: {idefix})")
    base_path = determine_base_path(simname, IDEFIX=idefix)
    
    param_file = os.path.join(base_path, "idefix.0.log" if idefix else "summary0.dat")
        
    params = read_parameters(param_file, IDEFIX=idefix)
    
    # --- FIX #1: Create the grid_info tuple right after reconstructing the grid ---
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(params, IDEFIX=idefix)
    grid_info = (xgrid, ygrid, zgrid, ny, nx, nz) # This line was missing

    nsteps = determine_nt(base_path, IDEFIX=idefix)
    
    if idefix:
        interval = float(params.get("vtk", 50.265))
        time_array = np.arange(nsteps) * interval / (2.0 * np.pi)
    else:
        time_array = np.arange(nsteps) * (float(params['NINTERM']) / 20.0)

    num_test_frames = 5
    snapshot_indices = np.linspace(0, nsteps - 1, num_test_frames, dtype=int) if test_mode else np.arange(nsteps)

    qp, _ = extract_planet_mass_and_migration(param_file, IDEFIX=idefix)

    if idefix:
        h0 = float(params.get("h0"))
        SIGMA0 = float(params.get("sigma0"))
    else:
        h0 = float(params.get("ASPECTRATIO"))
        SIGMA0 = float(params.get("SIGMA0"))
        
    Gamma0 = (qp / h0)**2 * SIGMA0
    print(f"⚙️  Using qp={qp:.2e}, h0={h0:.3f}, Sigma0={SIGMA0:.3f} -> Gamma_0 = {Gamma0:.3e}")

    tqwk_torque_norm = None; tqwk_torque_smoothed = None; tqwk_time = None
    try:
        tqwk_file = os.path.join(base_path, "tqwk0.dat")
        tqwk_time_raw, torque_on_planet_raw, _ = read_alternative_torque(tqwk_file, IDEFIX=idefix)
        tqwk_time = tqwk_time_raw if idefix else tqwk_time_raw / (2.0 * np.pi)
        
        norm_factor_tqwk = qp / Gamma0 if Gamma0 > 0 else 0.0
        tqwk_torque_norm = torque_on_planet_raw * norm_factor_tqwk
        dt = np.mean(np.diff(tqwk_time))
        smoothing_window = max(1, int(round(1.0 / dt)))
        tqwk_torque_smoothed = uniform_filter1d(tqwk_torque_norm, size=smoothing_window, mode='nearest')
        print("✅ Loaded & normalized torque from tqwk0.dat.")
    except Exception as e:
        print(f"❌ Warning: Could not load tqwk0.dat: {e}")

    print(f"\n-- Pass 1: Calculating torque for {len(snapshot_indices)} snapshots --")
    map_torque_history = np.full(nsteps, np.nan)
    
    torque_tasks = [(snap_idx, base_path, params, grid_info, idefix, qp) for snap_idx in snapshot_indices]
    
    with ProcessPoolExecutor(max_workers=parallel_cores) as executor:
        results = list(tqdm(executor.map(calculate_torque_for_frame, torque_tasks), total=len(torque_tasks), desc="Calculating Torques"))
    
    for idx, torque_val in results:
        map_torque_history[idx] = torque_val * (1.0 / Gamma0 if Gamma0 > 0 else 0.0)

    all_y_data = []
    if tqwk_torque_smoothed is not None: all_y_data.append(tqwk_torque_smoothed)
    all_y_data.append(map_torque_history[~np.isnan(map_torque_history)])
    if all_y_data and np.any([a.size > 0 for a in all_y_data]):
        combined_data = np.concatenate([arr for arr in all_y_data if arr.size > 0])
        y_min, y_max = np.min(combined_data), np.max(combined_data)
        padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 1e-9 else 0.1
        ylim_ax4 = (y_min - padding, y_max + padding)
    else:
        ylim_ax4 = (-1, 1)

    print("\n-- Pass 2: Generating movie frames --")

    # --- FIX #2: Correct the arguments being passed to load_equilibrium_state ---
    data0 = load_equilibrium_state(base_path, params, grid_info, idefix=idefix)
    
    initial_vorticity = calculate_vorticity(data0['gasvx'], data0['gasvy'], xgrid, ygrid, nz)
    initial_gas_density_norm = np.mean(data0['gasdens'], axis=(0, 2))
    
    if idefix:
        thick_smooth = float(params.get("smoothing")) / float(params.get("h0"))
        flaring_index = float(params.get("flaringindex"))
    else:
        thick_smooth = float(params.get("THICKNESSSMOOTHING"))
        flaring_index = float(params.get("FLARINGINDEX"))

    torque_map_equilibrium = get_torque_map(data0['gasdens'], xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index)
    radial_mask_indices = np.where((xgrid >= plot_range_r[0]) & (xgrid <= plot_range_r[1]))[0]

    tmp_dir = os.path.join(base_path, "temp_movie_frames")
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    frame_tasks = []
    for frame_idx, snapshot_idx in enumerate(snapshot_indices):
        worker_args = (frame_idx, snapshot_idx, base_path, params, grid_info, idefix, qp,
                       initial_vorticity, initial_gas_density_norm, torque_map_equilibrium,
                       time_array, map_torque_history, 
                       tqwk_time, tqwk_torque_norm, tqwk_torque_smoothed,
                       tmp_dir, Gamma0, plot_range_r, radial_mask_indices, ylim_ax4, avg_source)
        frame_tasks.append(worker_args)

    with ProcessPoolExecutor(max_workers=parallel_cores) as executor:
        list(tqdm(executor.map(process_and_save_frame, frame_tasks), total=len(frame_tasks), desc="🖼️  Processing Frames"))

    print("🎬 Stitching frames into movie with ffmpeg...")
    movie_filename = f"advanced_movie_{simname}{'_test' if test_mode else ''}.mp4"
    output_filepath = os.path.join(base_path, movie_filename)
    ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
                  '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264',
                  '-pix_fmt', 'yuv420p', '-crf', '20', output_filepath]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed.\n--- stderr ---\n{e.stderr}")
        return

    shutil.rmtree(tmp_dir)
    print(f"\n✅ Movie saved to: {output_filepath}")
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Movies", "mariuslehmann")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an advanced simulation movie with vorticity and torque maps.")
    parser.add_argument("simname", type=str, help="Name of the simulation subdirectory.")
    parser.add_argument("--test", action="store_true", help="Run in test mode on a few snapshots.")
    parser.add_argument("--idefix", action="store_true", help="Flag to indicate an IDEFIX simulation.")
    parser.add_argument("--parallel", type=int, nargs='?', const=os.cpu_count(), default=None, 
                        help="Enable parallel processing. Optionally specify number of cores.")
    parser.add_argument("--avg-source", type=str, default='tqwk', choices=['tqwk', 'map'],
                        help="Data source for cumulative average torque: 'tqwk' (default) or 'map' (snapshots).")
    args = parser.parse_args()
    create_advanced_movie(args.simname, test_mode=args.test, parallel_cores=args.parallel, avg_source=args.avg_source, idefix=args.idefix)
