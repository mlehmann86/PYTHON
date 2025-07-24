# plot_torque_evolution_v24.py

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import multiprocessing
from scipy.integrate import cumulative_trapezoid

# Custom modules from your repository
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
    read_alternative_torque,
)
from create_torque_movie import compute_torque_gravity
from corotation_torque_pnu import extract_xs


def boxcar_average(data, box_size):
    """Computes the boxcar average of a 1D array."""
    box_size = int(box_size)
    if box_size > 1:
        return uniform_filter1d(data, size=box_size, mode='nearest')
    return np.asarray(data)

def compute_local_profiles(gasdens_3d, gasenergy_3d, gasvx_3d, gasvy_3d, xgrid, gasvx0_1d, gamma, h0, qp):
    """
    Computes INSTANTANEOUS, azimuthally-averaged p, q, and alpha over the horseshoe region.
    """
    # Fargo3D convention: gasvx is azimuthal (v_phi), gasvy is radial (v_r)
    if gasdens_3d.ndim == 3 and gasdens_3d.shape[2] > 1:
        gasdens_2d = np.mean(gasdens_3d, axis=2)
        gasenergy_2d = np.mean(gasenergy_3d, axis=2)
        gasvx_2d = np.mean(gasvx_3d, axis=2)
        gasvy_2d = np.mean(gasvy_3d, axis=2)
    else:
        gasdens_2d = np.squeeze(gasdens_3d, axis=2) if gasdens_3d.ndim == 3 else gasdens_3d
        gasenergy_2d = np.squeeze(gasenergy_3d, axis=2) if gasenergy_3d.ndim == 3 else gasenergy_3d
        gasvx_2d = np.squeeze(gasvx_3d, axis=2) if gasvx_3d.ndim == 3 else gasvx_3d
        gasvy_2d = np.squeeze(gasvy_3d, axis=2) if gasvy_3d.ndim == 3 else gasvy_3d

    sigma_profile = np.mean(gasdens_2d, axis=0)
    # CORRECTED: gasenergy from read_single_snapshot already includes (gamma-1)
    pressure_profile = np.mean(gasenergy_2d, axis=0)
    temp_profile = pressure_profile / sigma_profile

    log_r = np.log(xgrid)
    d_log_sigma = np.gradient(np.log(sigma_profile), log_r)
    d_log_temp = np.gradient(np.log(temp_profile), log_r)
    
    p_profile = -d_log_sigma
    q_profile = -d_log_temp
    
    x_s_local = 1.1 * np.sqrt(qp / h0)
    hs_mask = (xgrid >= 1.0 - x_s_local) & (xgrid <= 1.0 + x_s_local)
    
    gasvx0_2d = np.broadcast_to(gasvx0_1d, (gasvx_2d.shape[0], gasvx_2d.shape[1]))
    denominator = np.mean(pressure_profile[hs_mask])
    numerator = np.mean(gasdens_2d[:, hs_mask] * gasvy_2d[:, hs_mask] * (gasvx_2d[:, hs_mask] - gasvx0_2d[:, hs_mask]))
    alpha_hs_avg = numerator / denominator if denominator != 0 else 0.0

    p_avg = np.mean(p_profile[hs_mask])
    q_avg = np.mean(q_profile[hs_mask])

    return p_avg, q_avg, alpha_hs_avg


def process_snapshot(snapshot_index, base_path, xgrid, ygrid, zgrid, gasvx0, params, qp, gam0, h0, thick_smooth, flaring_index, x_s):
    """
    Stage 1: Extracts raw simulation torques and instantaneous disk properties.
    """
    gamma = float(params.get("GAMMA"))
    data = read_single_snapshot(
        base_path, snapshot_index,
        read_gasdens=True, read_gasenergy=True, read_gasvx=True, read_gasvy=True
    )[0]

    torque_profile = compute_torque_gravity(
        gasdens=data["gasdens"], xgrid=xgrid, ygrid=ygrid, zgrid=zgrid,
        qp=qp, h0=h0, thick_smooth=thick_smooth, flaring_index=flaring_index
    )
    
    torque_profile_norm = torque_profile / (gam0 / gamma)

    dr = xgrid[1] - xgrid[0]
    corotation_mask = (xgrid >= 1.0 - x_s) & (xgrid <= 1.0 + x_s)
    
    sim_corotation = np.sum(torque_profile_norm[corotation_mask]) * dr
    sim_lindblad = np.sum(torque_profile_norm[~corotation_mask]) * dr

    p_avg, q_avg, alpha_hs_avg = compute_local_profiles(
        data["gasdens"], data["gasenergy"], data["gasvx"], data["gasvy"], xgrid, gasvx0, gamma, h0, qp
    )
    
    return {
        "torque_profile": torque_profile_norm,
        "sim_lindblad": sim_lindblad,
        "sim_corotation": sim_corotation,
        "p": p_avg,
        "q": q_avg,
        "alpha": alpha_hs_avg,
    }

def plot_average_torque_profile(time_avg_profile, xgrid, x_s, simname, h0, gamma_eff):
    """
    Plots the time-averaged radial torque profile with several enhancements.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    fontsize = 16
    
    r_p = 1.0
    x_axis_H = (xgrid - r_p) / h0

    torque_density_scaled = time_avg_profile * -gamma_eff
    
    cumulative_torque = cumulative_trapezoid(torque_density_scaled, xgrid, initial=0)

    ax.plot(x_axis_H, torque_density_scaled, color='black', label='Time-Averaged Torque Density', linewidth=2.5)

    ax2 = ax.twinx()
    ax2.plot(x_axis_H, cumulative_torque, color='purple', linestyle=':', label='Cumulative Torque', linewidth=2.5)
    
    m_high_order = 20
    for m in [m_high_order]:
        r_L_outer = (1 + 1/m)**(2/3)
        r_L_inner = (1 - 1/m)**(2/3)
        x_L_outer_H = (r_L_outer - r_p) / h0
        x_L_inner_H = (r_L_inner - r_p) / h0
        ax.axvline(x_L_outer_H, color='gray', linestyle='--', lw=1.5, label=f'High-Order OLR')
        ax.axvline(x_L_inner_H, color='gray', linestyle='-.', lw=1.5, label=f'High-Order ILR')
        
    x_s_H = x_s / h0
    ax.axvspan(-x_s_H, x_s_H, color='red', alpha=0.2, label=f'Corotation Region (xs={x_s:.3f})')
        
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel(r"$(r - r_p) / H_p$", fontsize=fontsize)
    ax.set_ylabel(r"Torque Density $(-\gamma_{eff} d\Gamma/dr) / (\Gamma_0/\gamma)$", fontsize=fontsize)
    ax2.set_ylabel(r"Cumulative Torque $(-\gamma_{eff} \Gamma) / (\Gamma_0/\gamma)$", color='purple', fontsize=fontsize)
    ax.set_title(f"Time-Averaged Torque Profile for {simname}", fontsize=fontsize + 2)
    
    ax.set_xlim(x_axis_H.min(), x_axis_H.max())
    
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.tick_params(axis='y', labelcolor='purple', labelsize=fontsize-2)
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=fontsize - 2)

    ax.grid(True, linestyle=":", alpha=0.6)
    
    output_filename = f"torque_profile_avg_{simname}.pdf"
    plt.savefig(output_filename)
    plt.close()
    return output_filename


def main(simname, start_snap, end_snap, num_cores):
    LINDBLAD_SMOOTHING_ORBITS = 50.0

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

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(params)
    
    gasvx0_path = os.path.join(base_path, "gasvxI0.dat")
    try:
        gasvx0_flat = np.fromfile(gasvx0_path, dtype=np.float64)
        gasvx0_3d = gasvx0_flat.reshape(ny, nx, nz)
        gasvx0 = np.mean(gasvx0_3d, axis=(0, 2))
    except (FileNotFoundError, ValueError):
        gasvx0 = xgrid**(-0.5)

    nt_max = determine_nt(base_path)
    end_snap = min(end_snap, nt_max)
    snapshot_range = range(start_snap, end_snap)
    
    orbits_per_snap = (float(params['NINTERM']) * float(params['DT'])) / (2.0 * np.pi)
    time_array = np.array(snapshot_range) * orbits_per_snap

    uses_td = "-DTHERMALDIFFUSION" in open(summary_file).read()
    x_s = extract_xs(simname, chi_simulation=uses_td, beta_simulation=not uses_td)

    x_s *=1.3 #empirical correction

    task_args = [(i, base_path, xgrid, ygrid, zgrid, gasvx0, params, qp, gam0, h0, thick_smooth, flaring_index, x_s) for i in snapshot_range]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.starmap(process_snapshot, task_args), total=len(task_args)))
    
    raw_data = {key: np.array([r[key] for r in results]) for key in results[0]}
    
    _, _, _, gamma_eff, _, _ = compute_theoretical_torques_PK11(params, qp, simname, summary_file=summary_file)

    time_avg_profile = np.mean(raw_data['torque_profile'], axis=0)
    profile_pdf_path = plot_average_torque_profile(time_avg_profile, xgrid, x_s, simname, h0, gamma_eff)

    libration_period_orbits = (4.0 * np.pi) / (3.0 * x_s)
    libration_width_snaps = libration_period_orbits / orbits_per_snap
    lindblad_width_snaps = LINDBLAD_SMOOTHING_ORBITS / orbits_per_snap
    
    sim_corotation_smooth = boxcar_average(raw_data['sim_corotation'], libration_width_snaps)
    sim_lindblad_smooth = boxcar_average(raw_data['sim_lindblad'], lindblad_width_snaps)

    alpha_smooth = boxcar_average(raw_data['alpha'], libration_width_snaps)
    p_smooth = boxcar_average(raw_data['p'], lindblad_width_snaps)
    q_smooth = boxcar_average(raw_data['q'], lindblad_width_snaps)

    pred_lindblad_series = []
    pred_corotation_series = []
    nu_laminar = float(params.get("NU", 0.0))
    chi_laminar = float(params.get("CHI", 0.0))

    for i in tqdm(range(len(time_array)), desc="Calculating predicted torques"):
        temp_params = params.copy()
        temp_params['SIGMASLOPE'] = p_smooth[i]
        temp_params['FLARINGINDEX'] = (1.0 - q_smooth[i]) / 2.0
        
        nu_turb = alpha_smooth[i] * h0**2 if alpha_smooth[i] > 0 else 0.0
        
        temp_params['NU'] = nu_laminar + nu_turb
        temp_params['CHI'] = chi_laminar

        total_torque, lindblad_torque, _, _, _, _ = compute_theoretical_torques_PK11(
            temp_params, qp, simname, summary_file=summary_file
        )
        corotation_torque = total_torque - lindblad_torque if alpha_smooth[i] > 0 else np.nan
        
        pred_lindblad_series.append(lindblad_torque)
        pred_corotation_series.append(corotation_torque)
    
    try:
        tqwk_file = os.path.join(base_path, "tqwk0.dat")
        tqwk_time_raw, tqwk_torque_raw, _ = read_alternative_torque(tqwk_file)
        tqwk_time_orbits = tqwk_time_raw / (2.0 * np.pi)
        tqwk_torque_norm = (tqwk_torque_raw * qp) / gam0 # Torque ON PLANET
    except Exception:
        tqwk_time_orbits = time_array
        tqwk_torque_norm = np.full_like(time_array, np.nan)
    
    # CORRECTED: Cumulative average calculation based on plot_cos_torque_comparison.py
    cumulative_tqwk_torque = np.cumsum(tqwk_torque_norm) / (np.arange(len(tqwk_torque_norm)) + 1)
    
    sim_total_torque_on_planet = - (raw_data['sim_lindblad'] + raw_data['sim_corotation'])
    cumulative_sim_torque = np.cumsum(sim_total_torque_on_planet) / (np.arange(len(time_array)) + 1)

    fontsize = 16
    linewidth = 2.5
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 25), sharex=True)
    ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    # Torque on planet = -Torque on disk
    ax1.plot(time_array, -sim_lindblad_smooth * gamma_eff, label=f"Sim. Lindblad", color="blue", linewidth=linewidth)
    ax1.plot(time_array, np.array(pred_lindblad_series) * gamma_eff, label="Pred. Lindblad", color="blue", linestyle="--", linewidth=linewidth)
    ax1.set_ylabel(r"$\Gamma_L / \Gamma_0$", fontsize=fontsize)
    ax1.legend(fontsize=fontsize-2)

    ax2.plot(time_array, -sim_corotation_smooth * gamma_eff, label=f"Sim. Corotation", color="red", linewidth=linewidth)
    ax2.plot(time_array, np.array(pred_corotation_series) * gamma_eff, label="Pred. Corotation", color="red", linestyle="--", linewidth=linewidth)
    ax2.set_ylabel(r"$\Gamma_C / \Gamma_0$", fontsize=fontsize)
    ax2.legend(fontsize=fontsize-2)
    
    # Plotting cumulative torque ON PLANET
    ax3.plot(time_array, cumulative_sim_torque * gamma_eff, label="Cumulative Sim. Torque (L+C)", color="green", linewidth=linewidth)
    ax3.plot(tqwk_time_orbits, cumulative_tqwk_torque * gamma_eff, label="Cumulative Torque (from tqwk0.dat)", color="black", linewidth=linewidth)
    ax3.axhline(0, color='gray', linestyle=':', lw=1.5)
    ax3.set_ylabel(r"Cumulative $\Gamma_{tot} / \Gamma_0$", fontsize=fontsize)
    ax3.legend(fontsize=fontsize-2)

    # Dynamically set y-limits based on data
    all_vals = np.concatenate([
        cumulative_sim_torque * gamma_eff,
        cumulative_tqwk_torque * gamma_eff
    ])
    y_min = max(-10, np.min(all_vals))
    y_max = min(10, np.max(all_vals))
    ax3.set_ylim(y_min, y_max)

    ax4.plot(time_array, raw_data['p'], color='orange', alpha=0.3)
    ax4.plot(time_array, p_smooth, label=f"p (Smoothed)", color="orange", linewidth=linewidth)
    ax4.plot(time_array, raw_data['q'], color='purple', alpha=0.3)
    ax4.plot(time_array, q_smooth, label=f"q (Smoothed)", color="purple", linewidth=linewidth)
    ax4.legend(fontsize=fontsize-2)
    ax4.set_ylabel("Slopes", fontsize=fontsize)

    raw_alpha_pos = np.where(raw_data['alpha'] > 0, raw_data['alpha'], np.nan)
    alpha_smooth_pos = np.where(alpha_smooth > 0, alpha_smooth, np.nan)
    ax5.plot(time_array, raw_alpha_pos, color='green', alpha=0.3, label=r"Raw Positive $\alpha$")
    ax5.plot(time_array, alpha_smooth_pos, label=r"Smoothed Positive $\alpha$", color="green", linewidth=linewidth)
    
    alpha_laminar = nu_laminar / (h0**2)
    if alpha_laminar > 0:
        ax5.axhline(alpha_laminar, color='gray', linestyle='--', label=f'Laminar $\\alpha = {alpha_laminar:.2e}$', linewidth=linewidth)
    ax5.set_yscale('log')
    ax5.set_ylim(bottom=1e-7)
    ax5.legend(fontsize=fontsize-2)
    ax5.set_ylabel(r"Turbulent $\alpha$", fontsize=fontsize)
    ax5.set_xlabel("Time [Orbits]", fontsize=fontsize)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

    plt.tight_layout(pad=2.0)
    main_plot_filename = f"torque_evolution_{simname}.pdf"
    plt.savefig(main_plot_filename)
    plt.close()
    
    scp_transfer(main_plot_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")
    scp_transfer(profile_pdf_path, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the time evolution of torque components and disk properties.")
    parser.add_argument("simname", type=str, help="Simulation subdirectory name.")
    parser.add_argument("--start", type=int, default=0, help="Starting snapshot number.")
    parser.add_argument("--end", type=int, default=10000, help="Ending snapshot number.")
    parser.add_argument("--parallel", type=int, default=16, help="Number of CPU cores to use.")
    args = parser.parse_args()
        
    main(args.simname, args.start, args.end, args.parallel)
