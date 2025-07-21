# source_term_hist_deltapv_contours.py

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
# Assuming these modules are in the same directory or PYTHONPATH
from data_reader import read_parameters, reconstruct_grid, read_single_snapshot, read_single_snapshot_idefix
from data_storage import determine_base_path, scp_transfer
import matplotlib.colors as colors # For SymLogNorm
try:
    from gamma_eff import extract_beta_value
except ImportError:
    print("ERROR: Could not import 'extract_beta_value' from 'gamma_eff.py'.")
    def extract_beta_value(simulation_path):
        print("WARNING: Using dummy extract_beta_value function.")
        return float('inf')

# --- compute_vortensity_source function ---
def compute_vortensity_source(gasdens, gasenergy, xgrid, ygrid, gamma):
    ny, nx = gasdens.shape
    pressure = gasenergy * (gamma - 1.0)
    sigma_floor = np.maximum(1e-12 * np.mean(gasdens), 1e-15)
    pressure_floor = sigma_floor * (gamma - 1.0) * 1e-12 * np.mean(gasenergy)
    gasdens_floor = np.maximum(gasdens, sigma_floor)
    pressure_floor = np.maximum(pressure, pressure_floor)
    enthalpy = (gamma / (gamma - 1.0)) * pressure_floor / gasdens_floor
    entropy = np.log(pressure_floor) - gamma * np.log(gasdens_floor)
    grad_H_phi = np.gradient(enthalpy, ygrid, axis=0)
    grad_H_r   = np.gradient(enthalpy, xgrid, axis=1)
    grad_S_phi = np.gradient(entropy, ygrid, axis=0)
    grad_S_r   = np.gradient(entropy, xgrid, axis=1)
    if xgrid.ndim == 1: r = xgrid[np.newaxis, :]
    else: r = xgrid
    r = np.maximum(r, 1e-10)
    term1 = grad_H_r * (1.0 / r) * grad_S_phi
    term2 = (1.0 / r) * grad_H_phi * grad_S_r
    cross_product_z = term1 - term2
    source_term = cross_product_z / gasdens_floor**2
    source_term = np.nan_to_num(source_term, nan=0.0, posinf=0.0, neginf=0.0)
    return source_term

# --- compute_potential_vorticity function ---
def compute_potential_vorticity(gasdens, gasvx, gasvy, xgrid, ygrid):
    ny, nx = gasdens.shape
    sigma_floor = np.maximum(1e-12 * np.mean(gasdens), 1e-15)
    gasdens_floor = np.maximum(gasdens, sigma_floor)
    if xgrid.ndim == 1: r = xgrid[np.newaxis, :]
    else: r = xgrid
    r = np.maximum(r, 1e-10)
    v_r_inertial = gasvy; v_phi_inertial = gasvx + r
    r_v_phi_inertial = r * v_phi_inertial
    grad_r_v_phi_inertial_r = np.gradient(r_v_phi_inertial, xgrid, axis=1)
    grad_v_r_phi = np.gradient(v_r_inertial, ygrid, axis=0)
    omega_z_inertial = (1.0 / r) * grad_r_v_phi_inertial_r - (1.0 / r) * grad_v_r_phi
    pv = omega_z_inertial / gasdens_floor
    pv = np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0)
    return pv

# --- Function to plot histogram ---
def plot_histogram(data_subset, bin_edges, hist, chosen_levels, hist_range, simname, snapshot):
    """Plots the histogram used for contour level selection."""
    if data_subset is None or bin_edges is None or hist is None or chosen_levels is None:
        print("Histogram plotting skipped: Missing data.")
        return None

    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Plot histogram bars
    ax_hist.bar(bin_centers, hist, width=bin_width*0.9, align='center', label=f'Frequency (Bins={len(hist)})')

    # Mark chosen contour levels
    # Use a contrasting colormap for level markers
    level_colors = plt.cm.get_cmap('autumn', len(chosen_levels) + 1)
    for i, level in enumerate(chosen_levels):
        ax_hist.axvline(level, color=level_colors(i / len(chosen_levels)), linestyle='--',
                        label=f'Level {i+1}: {level:.2e}')

    ax_hist.set_xlabel("Delta PV Value")
    ax_hist.set_ylabel("Number of Grid Cells in Bin")
    ax_hist.set_title(f"Histogram of Delta PV in range [{hist_range[0]:.2e}, {hist_range[1]:.2e}]\n"
                      f"({simname}, Snapshot {snapshot})")
    ax_hist.legend(fontsize='small')
    ax_hist.grid(True, alpha=0.5)
    # Use log scale for counts? Optional, might depend on distribution.
    # ax_hist.set_yscale('log')
    # ax_hist.set_ylim(bottom=0.8)

    hist_outname = f"delta_pv_histogram_{simname}_{snapshot:04d}.pdf"
    plt.tight_layout()
    try:
        plt.savefig(hist_outname)
        print(f"Saved histogram plot: {hist_outname}")
        return hist_outname # Return filename for potential transfer
    except Exception as e:
        print(f"Error saving histogram plot: {e}")
        return None
    finally:
        plt.close(fig_hist) # Close histogram figure


def main():
    parser = argparse.ArgumentParser(description="Plot 2D Source Term S* with Delta PV contours chosen via histogram near minimum.")
    # Added hist_bins, num_hist_levels, plot_histogram
    parser.add_argument("snapshot", type=int, help="Snapshot number.")
    parser.add_argument("simname", type=str, help="Simulation name prefix.")
    parser.add_argument("--IDEFIX", action="store_true", help="Flag for IDEFIX simulation.")
    parser.add_argument("--r_min", type=float, default=0.9, help="Minimum radius for plot (x-axis).")
    parser.add_argument("--r_max", type=float, default=1.1, help="Maximum radius for plot (x-axis).")
    parser.add_argument("--phi_min", type=float, default=None, help="Minimum azimuth for plot (y-axis).")
    parser.add_argument("--phi_max", type=float, default=None, help="Maximum azimuth for plot (y-axis).")
    parser.add_argument("--cmap", type=str, default="RdBu_r", help="Colormap for S*.")
    parser.add_argument("--clim_perc", type=float, default=99.5, help="Percentile for S* color limits.")
    parser.add_argument("--log_scale", action="store_true", help="Use symmetric log scale for S* color.")
    parser.add_argument("--shading", type=str, default="gouraud", choices=['flat', 'nearest', 'gouraud', 'auto'], help="Shading for pcolormesh.")
    # Contour arguments
    parser.add_argument("--num_hist_levels", type=int, default=3, help="Number of contour levels to plot based on histogram peaks.")
    parser.add_argument("--hist_bins", type=int, default=100, help="Number of bins for histogram.")
    parser.add_argument("--contour_color", type=str, default='lime', help="Color for the Delta PV contours.")
    parser.add_argument("--contour_lw", type=float, default=1.5, help="Linewidth for Delta PV contours.")
    parser.add_argument("--contour_alpha", type=float, default=0.9, help="Alpha for Delta PV contours.")
    parser.add_argument("--clabel_fontsize", type=str, default='small', help="Font size for contour labels.")
    parser.add_argument("--clabel_bg", action="store_true", help="Add white background to contour labels.")
    parser.add_argument("--plot_histogram", action="store_true", help="Additionally plot the histogram used for level selection.") # New arg
    # SCP arguments
    parser.add_argument("--scp_user", type=str, default="mariuslehmann", help="Username for SCP transfer.")
    parser.add_argument("--scp_dir", type=str, default="/Users/mariuslehmann/Downloads/Profiles/", help="Target directory for SCP transfer.")


    args = parser.parse_args()

    # --- Setup, Parameter Reading, Extract Beta ---
    # (Parameter reading remains the same)
    beta_val = float('inf'); alpha = None
    try:
        base_path = determine_base_path(args.simname, IDEFIX=args.IDEFIX); print(f"Base path: {base_path}")
        beta_val = extract_beta_value(base_path); print(f"Extracted beta: {beta_val}")
        summary_file = f"{base_path}/{'idefix.0.log' if args.IDEFIX else 'summary0.dat'}"
        parameters = read_parameters(summary_file, IDEFIX=args.IDEFIX)
        if args.IDEFIX: alpha = parameters.get("SigmaExponent")
        else: alpha = parameters.get("SigmaSlope"); alpha = parameters.get("alpha") if alpha is None else alpha
        if alpha is None: print("ERROR: Could not get alpha."); alpha = 0.0; print("WARNING: Assuming alpha=0.")
        else: print(f"Using alpha = {alpha}")
    except FileNotFoundError as e: print(f"Error finding files: {e}"); return
    except Exception as e: print(f"Error reading params/beta: {e}"); return
    gamma = parameters.get("gamma") if args.IDEFIX else parameters.get("GAMMA")

    # --- Estimate Time ---
    # (Time estimation remains the same)
    try: # Estimate Time
        if args.IDEFIX:
             output_config = parameters.get("output", {}); vtk_config = output_config.get("vtk", {})
             dt_vtk = vtk_config.get("dt")
             if dt_vtk: time_snapshot = args.snapshot * dt_vtk
             else: output_interval = 50.265; time_snapshot = args.snapshot * output_interval
             time_orbit = time_snapshot / (2 * np.pi)
        else: # FARGO
             ninterm = parameters.get("NINTERM", 20); outputs_per_orbit = ninterm
             dt_param = parameters.get("DT")
             if dt_param:
                 try: nstep = parameters.get("NSTEP")
                 except: nstep = None
                 if nstep: time_per_output = nstep * dt_param; time_snapshot = args.snapshot * time_per_output; time_orbit = time_snapshot / (2 * np.pi)
                 else: time_orbit = args.snapshot / outputs_per_orbit if outputs_per_orbit > 0 else float('nan')
             else: time_orbit = args.snapshot / outputs_per_orbit if outputs_per_orbit > 0 else float('nan')
        time_orbit_str = f"{time_orbit:.1f} orbits" if not np.isnan(time_orbit) else f"Snapshot {args.snapshot}"
    except Exception as e: print(f"Warning: Could not get time: {e}"); time_orbit_str = f"Snapshot {args.snapshot}"

    # --- Load Data ---
    # (Loading remains the same)
    print(f"Loading snapshot {args.snapshot} and snapshot 0...")
    fields_to_read = ["gasdens", "gasenergy", "gasvx", "gasvy"]
    try:
        if args.IDEFIX: data_snap = read_single_snapshot_idefix(base_path, args.snapshot, **{f"read_{f}": True for f in fields_to_read})[0]
        else: data_snap = read_single_snapshot(base_path, args.snapshot, **{f"read_{f}": True for f in fields_to_read})[0]
        if args.IDEFIX: data_init = read_single_snapshot_idefix(base_path, 0, **{f"read_{f}": True for f in fields_to_read})[0]
        else: data_init = read_single_snapshot(base_path, 0, **{f"read_{f}": True for f in fields_to_read})[0]
        gasdens = data_snap["gasdens"]; gasenergy = data_snap["gasenergy"]; gasvx = data_snap["gasvx"]; gasvy = data_snap["gasvy"]
        gasdens0 = data_init["gasdens"]; gasvx0 = data_init["gasvx"]; gasvy0 = data_init["gasvy"]
    except FileNotFoundError as e: print(f"Error: Snapshot file not found. {e}"); return
    except KeyError as e: print(f"Error: Field {e} not found. Check data_reader."); return
    except Exception as e: print(f"Error reading snapshot data: {e}"); return

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=args.IDEFIX)
    phi_plot = ygrid; r_plot = xgrid

    if gasdens.ndim == 3: # Select midplane
        idx = gasdens.shape[2] // 2 if gasdens.shape[2] > 1 else 0
        gasdens = gasdens[:, :, idx]; gasenergy = gasenergy[:, :, idx]
        gasvx = gasvx[:, :, idx]; gasvy = gasvy[:, :, idx]
        gasdens0 = gasdens0[:, :, idx]; gasvx0 = gasvx0[:, :, idx]; gasvy0 = gasvy0[:, :, idx]

    # --- Calculate S*, PV, PV0, and Delta PV ---
    # (Calculations remain the same)
    source_term = compute_vortensity_source(gasdens, gasenergy, xgrid, ygrid, gamma)
    pv_field = compute_potential_vorticity(gasdens, gasvx, gasvy, xgrid, ygrid)
    pv0_field = compute_potential_vorticity(gasdens0, gasvx0, gasvy0, xgrid, ygrid)
    delta_pv = pv_field - pv0_field
    delta_pv = np.nan_to_num(delta_pv, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Calculated Delta PV array. Min: {np.min(delta_pv):.2e}, Max: {np.max(delta_pv):.2e}")

    # --- Shift Data Azimuthally ---
    shift_index = ny // 2
    print(f"Shifting data azimuthally by {shift_index} cells.")
    source_term_rolled = np.roll(source_term, shift_index, axis=0)
    delta_pv_rolled = np.roll(delta_pv, shift_index, axis=0)

    # --- Plotting ---
    print("Generating plot (Radius on X, Azimuth on Y)...")
    fig, ax = plt.subplots(figsize=(8, 10))

    # Plot Source Term S* (Background)
    # (Remains the same)
    plot_data_bg = source_term_rolled
    abs_vals_bg = np.abs(plot_data_bg[np.isfinite(plot_data_bg)])
    max_abs_val_bg = np.percentile(abs_vals_bg, args.clim_perc) if len(abs_vals_bg) > 0 else 1e-9
    if max_abs_val_bg == 0: max_abs_val_bg = 1e-9
    vmin_bg, vmax_bg = -max_abs_val_bg, max_abs_val_bg
    norm_bg = None
    if args.log_scale:
        linthresh_bg = max_abs_val_bg / 1000.0; linscale_bg = 0.5
        if linthresh_bg <= 0: linthresh_bg = 1e-12
        try: norm_bg = colors.SymLogNorm(linthresh=linthresh_bg, linscale=linscale_bg, vmin=vmin_bg, vmax=vmax_bg, base=10)
        except ValueError as e: print(f"Warning: SymLogNorm failed ({e}), falling back to linear scale.")
    if norm_bg is None: norm_bg = colors.Normalize(vmin=vmin_bg, vmax=vmax_bg)
    im = ax.pcolormesh(r_plot, phi_plot, plot_data_bg, cmap=args.cmap, norm=norm_bg, shading=args.shading)
    cbar = plt.colorbar(im, ax=ax, extend='both' if args.log_scale else 'neither', label=r"$S^* \propto (\nabla H \times \nabla s)_z / \Sigma^2$")


    # --- Plot Delta PV Contours (Histogram Method near Minimum) ---
    plot_data_cont = delta_pv_rolled
    cont_finite = plot_data_cont[np.isfinite(plot_data_cont)]

    # Determine min/max delta PV in the plotted view
    min_dpv_global = np.min(delta_pv_rolled)
    min_dpv_in_view = min_dpv_global
    max_dpv_in_view = np.max(delta_pv_rolled) # Need max for positive range calc
    phi_min_plot = args.phi_min if args.phi_min is not None else ygrid.min()
    phi_max_plot = args.phi_max if args.phi_max is not None else ygrid.max()
    r_indices = np.where((r_plot >= args.r_min) & (r_plot <= args.r_max))[0]
    phi_indices = np.where((phi_plot >= phi_min_plot) & (phi_plot <= phi_max_plot))[0]
    dpv_view = None
    if len(r_indices) > 0 and len(phi_indices) > 0:
        idx_r_min, idx_r_max = r_indices.min(), r_indices.max()
        idx_phi_min, idx_phi_max = phi_indices.min(), phi_indices.max()
        idx_phi_max = min(idx_phi_max, delta_pv_rolled.shape[0]-1)
        idx_r_max = min(idx_r_max, delta_pv_rolled.shape[1]-1)
        if idx_phi_min <= idx_phi_max and idx_r_min <= idx_r_max:
             dpv_view = delta_pv_rolled[idx_phi_min:idx_phi_max+1, idx_r_min:idx_r_max+1]
             if dpv_view.size > 0:
                  min_dpv_in_view = np.min(dpv_view); max_dpv_in_view = np.max(dpv_view)
                  print(f"Min/Max Delta PV within plot limits: {min_dpv_in_view:.3e} / {max_dpv_in_view:.3e}")
             else: print("Warning: No Delta PV data found within plot limits.")
        else: print("Warning: Invalid indices for plot limits.")
    else: print("Warning: Could not determine indices for plot limits.")
    min_dpv_to_display = min_dpv_in_view

    # --- Define contour levels using histogram method (with FAC fallback) --- 
    contour_levels = None
    hist = None
    bin_edges = None
    data_subset = None
    hist_range = None
    mode_dpv_value = None  # For use in title or legend

    FAC_values = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]  # Try decreasing FAC values

    if len(cont_finite) > 0 and dpv_view is not None and dpv_view.size > args.hist_bins:
        print(f"DEBUG: min_dpv_in_view for histogram = {min_dpv_in_view:.3e}")

        for FAC in FAC_values:
            level_min_range = None
            level_max_range = None
            hist_range_ok = False

            if min_dpv_in_view < -1e-9:
                level_min_range = min_dpv_in_view
                level_max_range = min_dpv_in_view * FAC
                if level_max_range > level_min_range:
                    hist_range_ok = True
                else:
                    print(f"FAC={FAC:.3f}: Hist range [min, FAC*min] invalid.")
            elif min_dpv_in_view >= -1e-9:
                data_range_view = max_dpv_in_view - min_dpv_in_view
                if data_range_view > 1e-9:
                    level_min_range = min_dpv_in_view
                    level_max_range = min_dpv_in_view + 0.20 * data_range_view * FAC
                    hist_range_ok = True
                else:
                    print(f"FAC={FAC:.3f}: Data range too small for positive histogram.")

            if not hist_range_ok:
                continue

            hist_range = (level_min_range, level_max_range)
            data_subset = dpv_view[(dpv_view >= level_min_range) & (dpv_view <= level_max_range)]
            if data_subset.size <= args.hist_bins:
                print(f"FAC={FAC:.3f}: Not enough data points ({data_subset.size}) for histogram.")
                continue

            print(f"FAC={FAC:.3f}: Found {data_subset.size} points in range [{level_min_range:.2e}, {level_max_range:.2e}] for histogram.")
            try:
                hist, bin_edges = np.histogram(data_subset, bins=args.hist_bins, range=hist_range)
                max_bin_index = np.argmax(hist)
                mode_dpv_value = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])
                valid_bins = np.where(hist > 0)[0]
                if len(valid_bins) > 0:
                    sorted_indices = valid_bins[np.argsort(-hist[valid_bins])]
                    num_levels_actual = min(args.num_hist_levels, len(valid_bins))
                    top_indices = sorted_indices[:num_levels_actual]
                    top_levels = (bin_edges[top_indices] + bin_edges[top_indices + 1]) / 2.0
                    contour_levels = np.unique(np.sort(top_levels))
                    print(f"Selected {len(contour_levels)} contour levels via histogram peaks.")
                    break  # Stop after successful extraction
                else:
                    print(f"FAC={FAC:.3f}: No bins with counts > 0 found.")
            except Exception as e:
                print(f"FAC={FAC:.3f}: Histogram analysis failed - {e}.")
    else:
        print("Warning: Histogram skipped due to insufficient data or bad input.")

    if contour_levels is None:
        print("Warning: No contour levels determined or plotted.")

    # Plot contours if levels were determined
    if contour_levels is not None and len(contour_levels) > 0:
        print(f"Plotting {len(contour_levels)} Delta PV contours at levels: {contour_levels}")
        cont = ax.contour(r_plot, phi_plot, plot_data_cont, levels=contour_levels, colors=args.contour_color, linewidths=args.contour_lw, alpha=args.contour_alpha)
        clabel_kwargs = {'inline': True, 'fontsize': args.clabel_fontsize, 'fmt': '%.1e'}
        if args.clabel_bg:
             try: # Add background for visibility
                 import matplotlib
                 major, minor, *_ = map(int, matplotlib.__version__.split('.'))
                 if (major > 3) or (major == 3 and minor >= 4):
                     clabel_kwargs['inline_backgroundcolor'] = 'white'; clabel_kwargs['inline_background_alpha'] = 0.7
                     print("Applying inline background to clabels.")
                 else: print("Warning: Matplotlib < 3.4, inline background for clabel not supported.")
             except Exception as e: print(f"Warning: Could not check Matplotlib version or apply clabel background: {e}")
        # CONTOUR LABELS
        #ax.clabel(cont, **clabel_kwargs)
    else:
        print("Warning: No contour levels determined or plotted.")

    # --- Plot Histogram (Optional) ---
    hist_outname = None # Initialize filename
    if args.plot_histogram and hist is not None and bin_edges is not None and contour_levels is not None:
        try:
            hist_outname = plot_histogram(
                data_subset, bin_edges, hist, contour_levels, hist_range, args.simname, args.snapshot
            )
        except Exception as e:
            print(f"Failed to plot histogram: {e}")


    # --- Finalize Plot ---
    # (Setting limits, labels remain the same)
    ax.set_ylim(phi_min_plot, phi_max_plot)
    ax.set_xlim(args.r_min, args.r_max)
    ax.set_ylabel(r"Azimuth $\phi$ (rad)")
    ax.set_xlabel(r"Radius $r$")

    # --- Construct Title showing Min Delta PV ---
    # (Title construction remains the same)
    title_parts = []
    if np.isfinite(beta_val):
        if abs(beta_val) < 1e-2 or abs(beta_val) >= 1e3: beta_str = f"{beta_val:.1e}"
        else: beta_str = f"{beta_val:.4g}"
        title_parts.append(rf"$\beta={beta_str}$")
    else: title_parts.append(f"{args.simname}")
    title_parts.append(f"{time_orbit_str}")
    title_parts.append(f"Min $\delta PV$ (view): {min_dpv_to_display:.2e}")
    if 'mode_dpv_value' in locals():
        #title_parts.append(f"Contours show $\delta PV \\approx {mode_dpv_value:.2e}$")
        if contour_levels is not None and len(contour_levels) > 0:
            dpv_min = np.min(contour_levels)
            dpv_max = np.max(contour_levels)
            title_parts.append(
                rf"Contours: $\delta PV \in [{dpv_min:.2e},\, {dpv_max:.2e}]$"
            )
    title_string = f"Source Term $S^*$ & Frequent $\delta PV$ Contours\n({', '.join(title_parts)})" # Updated title
    ax.set_title(title_string, fontsize=10)

    # Planet marker, legend, grid remain the same
    planet_r = parameters.get("PlanetPosR", 1.0)
    plot_planet_phi = np.pi
    ax.plot(planet_r, plot_planet_phi, 'm*', markersize=12, markeredgecolor='k', label=r'Planet Loc ($\phi \approx \pi$)')
    ax.axhline(plot_planet_phi, color='grey', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3, linestyle=':')

    # --- Output & SCP ---
    outname = f"source_term_hist_deltapv_contours_{args.simname}_{args.snapshot:04d}.pdf" # New filename
    plt.tight_layout()
    try: plt.savefig(outname); print(f"Saved plot: {outname}")
    except Exception as e: print(f"Error saving plot: {e}"); return

    try: # SCP transfer main plot
        target_user = args.scp_user; target_dir = args.scp_dir
        print(f"Attempting SCP transfer of main plot to {target_user}:{target_dir}...")
        scp_transfer(outname, target_dir, target_user); print("SCP transfer successful.")
    except FileNotFoundError: print(f"Could not SCP transfer file: 'scp' command not found or script missing.")
    except Exception as e: print(f"Could not SCP transfer file: {e}"); print("Please transfer manually if needed.")

    # SCP transfer histogram plot if it was created
    if args.plot_histogram and hist_outname:
         try:
             print(f"Attempting SCP transfer of histogram plot to {target_user}:{target_dir}...")
             scp_transfer(hist_outname, target_dir, target_user); print("SCP transfer successful.")
         except FileNotFoundError: print(f"Could not SCP transfer file: 'scp' command not found or script missing.")
         except Exception as e: print(f"Could not SCP transfer histogram: {e}"); print("Please transfer manually if needed.")


if __name__ == "__main__":
    main()
#example prompt
#python3 vortensity_source_plotter.py 1 cos_bet1d1_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D --r_min 0.95 --r_max 1.05 --phi_min 1.5707964 --phi_max 4.7123890 --plot_histogram
