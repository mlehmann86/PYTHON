# plot_torque_evolution_v6.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import multiprocessing

# Assuming your custom modules are in the python path
from data_reader import (
    read_single_snapshot,
    read_parameters,
    reconstruct_grid,
    determine_nt,
)
from data_storage import determine_base_path, scp_transfer
from planet_data import (
    compute_theoretical_torques_PK11,
    extract_planet_mass_and_migration,
)
from create_torque_movie import compute_torque_gravity


def boxcar_average(data, box_size):
    """Computes the boxcar average of a 1D array."""
    box_size = int(box_size)
    if box_size > 1:
        return uniform_filter1d(data, size=box_size, mode='nearest')
    return data

def compute_local_profiles(gasdens, gasenergy, gasvx, gasvy, xgrid, ygrid, zgrid, gamma):
    """
    Computes azimuthally-averaged radial profiles of density slope (p),
    temperature slope (q), and alpha viscosity.
    """
    if len(zgrid) > 1:
        gasdens_2d = np.mean(gasdens, axis=2)
        gasenergy_2d = np.mean(gasenergy, axis=2)
        gasvx_2d = np.mean(gasvx, axis=2)
        gasvy_2d = np.mean(gasvy, axis=2)
    else:
        gasdens_2d = gasdens.squeeze(axis=2)
        gasenergy_2d = gasenergy.squeeze(axis=2)
        gasvx_2d = gasvx.squeeze(axis=2)
        gasvy_2d = gasvy.squeeze(axis=2)

    sigma = np.mean(gasdens_2d, axis=0)
    pressure = np.mean(gasenergy_2d, axis=0) * (gamma - 1.0)
    temp = pressure / sigma
    vx = np.mean(gasvx_2d, axis=0)
    vy = np.mean(gasvy_2d, axis=0)

    log_r = np.log(xgrid)
    d_log_sigma = np.gradient(np.log(sigma), log_r)
    d_log_temp = np.gradient(np.log(temp), log_r)
    p = -d_log_sigma
    q = -d_log_temp

    cs = np.sqrt(temp)
    d_omega_dr = np.gradient(vx / xgrid, xgrid)
    alpha = (vy * d_omega_dr) / (cs**2 * xgrid**-1.5)

    return p, q, alpha


def process_snapshot(snapshot_index, base_path, simname, summary_file, xgrid, ygrid, zgrid, params, qp, gam0, h0, thick_smooth, flaring_index):
    """Processes a single snapshot to calculate torques and disk properties."""
    gamma = float(params.get("GAMMA"))
    nu_laminar = float(params.get("NU", 0.0))
    chi_laminar = float(params.get("CHI", 0.0))

    data = read_single_snapshot(
        base_path, snapshot_index,
        read_gasdens=True, read_gasenergy=True, read_gasvx=True, read_gasvy=True
    )[0]

    torque_profile = compute_torque_gravity(
        gasdens=data["gasdens"], xgrid=xgrid, ygrid=ygrid, zgrid=zgrid,
        qp=qp, h0=h0, thick_smooth=thick_smooth, flaring_index=flaring_index
    )
    torque_profile /= (gam0 / gamma)

    x_s = 1.1 * np.sqrt(qp / h0)
    dr = xgrid[1] - xgrid[0]
    corotation_mask = (xgrid >= 1.0 - x_s) & (xgrid <= 1.0 + x_s)
    lindblad_mask = ~corotation_mask

    sim_corotation = np.sum(torque_profile[corotation_mask]) * dr
    sim_lindblad = np.sum(torque_profile[lindblad_mask]) * dr

    p, q, alpha_turb = compute_local_profiles(
        data["gasdens"], data["gasenergy"], data["gasvx"], data["gasvy"], xgrid, ygrid, zgrid, gamma
    )
    
    p_disk_avg = np.mean(p)
    q_disk_avg = np.mean(q)
    alpha_disk_avg = np.mean(alpha_turb)
    
    nu_turb_avg = alpha_disk_avg * h0**2
    
    temp_params = params.copy()
    temp_params['SIGMASLOPE'] = -p_disk_avg
    temp_params['FLARINGINDEX'] = (1.0 - q_disk_avg) / 2.0
    temp_params['NU'] = nu_laminar + nu_turb_avg
    temp_params['CHI'] = chi_laminar

    # CORRECTED: Pass the summary_file path to correctly detect thermal diffusion
    _, pred_lindblad, pred_corotation, _, _, _ = compute_theoretical_torques_PK11(
        temp_params, qp, simname, summary_file=summary_file, avg_start_orbit=0, avg_end_orbit=1
    )
    
    return {
        "torque_profile": torque_profile,
        "sim_lindblad": sim_lindblad,
        "sim_corotation": sim_corotation,
        "pred_lindblad": pred_lindblad,
        "pred_corotation": pred_corotation,
        "p": p_disk_avg,
        "q": q_disk_avg,
        "alpha": alpha_disk_avg,
    }

def plot_average_torque_profile(time_avg_profile, xgrid, x_s, simname):
    """Plots the time-averaged radial torque profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xgrid, time_avg_profile, color='black', label='Time-Averaged Torque')
    ax.axvspan(1.0 - x_s, 1.0 + x_s, color='red', alpha=0.2, label='Corotation Region')
    ax.axvspan(xgrid.min(), 1.0 - x_s, color='blue', alpha=0.2, label='Lindblad Region')
    ax.axvspan(1.0 + x_s, xgrid.max(), color='blue', alpha=0.2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Radius [r/r_p]")
    ax.set_ylabel(r"Time-Averaged Torque Density d$\Gamma$/dr")
    ax.set_title(f"Time-Averaged Torque Profile for {simname}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    output_filename = f"torque_profile_avg_{simname}.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Average torque profile plot saved to {output_filename}")
    return output_filename

def main(simname, start_snap, end_snap, num_cores):
    # --- Hardcoded averaging periods ---
    LINDBLAD_SMOOTHING_ORBITS = 10.0

    base_path = determine_base_path(simname)
    summary_file = os.path.join(base_path, "summary0.dat")
    params = read_parameters(summary_file)

    qp, _ = extract_planet_mass_and_migration(summary_file)
    h0 = float(params.get("ASPECTRATIO"))
    gamma = float(params.get("GAMMA"))
    sigma0 = float(params.get("SIGMA0", 1.0))
    thick_smooth = float(params.get("THICKNESSSMOOTHING"))
    flaring_index = float(params.get("FLARINGINDEX", 0.0))
    gam0 = (qp / h0)**2 * sigma0

    xgrid, ygrid, zgrid, _, _, _ = reconstruct_grid(params)
    nt_max = determine_nt(base_path)
    end_snap = min(end_snap, nt_max)
    snapshot_range = range(start_snap, end_snap)
    
    orbits_per_snap = float(params['NINTERM']) / (2.0 * np.pi)
    time_array = np.array(snapshot_range) * orbits_per_snap

    # CORRECTED: Added summary_file to the list of arguments passed to each worker
    task_args = [(i, base_path, simname, summary_file, xgrid, ygrid, zgrid, params, qp, gam0, h0, thick_smooth, flaring_index) for i in snapshot_range]

    print(f"Starting parallel processing on {num_cores} cores...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.starmap(process_snapshot, task_args), total=len(task_args)))
    print("Processing complete.")
    
    torques = {key: [r[key] for r in results] for key in results[0] if key != 'torque_profile'}
    all_torque_profiles = np.array([r["torque_profile"] for r in results])
    
    time_avg_profile = np.mean(all_torque_profiles, axis=0)
    x_s_avg = 1.1 * np.sqrt(qp / h0)
    profile_pdf_path = plot_average_torque_profile(time_avg_profile, xgrid, x_s_avg, simname)

    # --- Physically Motivated Averaging ---
    libration_period_orbits = (8.0 * np.pi) / (3.0 * (x_s_avg**2))
    libration_width_snaps = libration_period_orbits / orbits_per_snap
    print(f"Planet libration period: {libration_period_orbits:.2f} orbits. Averaging corotation torque over this period.")
    
    lindblad_width_snaps = LINDBLAD_SMOOTHING_ORBITS / orbits_per_snap
    print(f"Using boxcar average for Lindblad torque over {LINDBLAD_SMOOTHING_ORBITS} orbits.")

    sim_corotation_smooth = boxcar_average(torques['sim_corotation'], libration_width_snaps)
    sim_lindblad_smooth = boxcar_average(torques['sim_lindblad'], lindblad_width_snaps)

    # --- Multi-Panel Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Torques
    ax1.plot(time_array, sim_lindblad_smooth, label=f"Sim. Lindblad (Smoothed {LINDBLAD_SMOOTHING_ORBITS:.0f} orbits)", color="blue")
    ax1.plot(time_array, sim_corotation_smooth, label=f"Sim. Corotation (Smoothed {libration_period_orbits:.1f} orbits)", color="red")
    ax1.plot(time_array, torques['pred_lindblad'], label="Predicted Lindblad Torque", color="cyan", linestyle="--")
    ax1.plot(time_array, torques['pred_corotation'], label="Predicted Corotation Torque", color="magenta", linestyle="--")
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel(r"Torque / ($\Gamma_0 / \gamma$)")
    ax1.set_title(f"Time Evolution of Torque Components for {simname}")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7)

    # Panel 2: Disk Properties
    ax2.plot(time_array, torques['alpha'], label=r"Disk-Avg $\alpha$", color="green")
    ax2.plot(time_array, torques['p'], label="Disk-Avg p (Density Slope)", color="orange")
    ax2.plot(time_array, torques['q'], label="Disk-Avg q (Temp. Slope)", color="purple")
    ax2.set_xlabel("Time [Orbits]")
    ax2.set_ylabel("Value")
    ax2.set_title("Time Evolution of Disk-Averaged Properties")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    main_plot_filename = f"torque_evolution_{simname}.pdf"
    plt.savefig(main_plot_filename)
    plt.close()
    print(f"\nMain plot saved to {main_plot_filename}")
    
    # --- SCP Transfer using imported function ---
    print("Initiating SCP transfer...")
    scp_transfer(main_plot_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")
    scp_transfer(profile_pdf_path, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the time evolution of torque components and disk properties.")
    parser.add_argument("simname", type=str, help="Simulation subdirectory name.")
    parser.add_argument("--start", type=int, default=0, help="Starting snapshot number.")
    parser.add_argument("--end", type=int, default=10000, help="Ending snapshot number.")
    parser.add_argument("--cores", type=int, default=16, help="Number of CPU cores to use.")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
        
    main(args.simname, args.start, args.end, args.cores)
