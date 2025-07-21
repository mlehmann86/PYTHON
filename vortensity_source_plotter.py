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
import matplotlib.gridspec as gridspec # For colorbar positioning

# --- Dummy gamma_eff.extract_beta_value (if not found) ---
try:
    from gamma_eff import extract_beta_value
except ImportError:
    print("ERROR: Could not import 'extract_beta_value' from 'gamma_eff.py'.")
    def extract_beta_value(simulation_path):
        print("WARNING: Using dummy extract_beta_value function, returning inf.")
        # Try to extract beta from simname as a fallback
        match = re.search(r'bet(\d+d\d+|\d+)', simulation_path)
        if match:
            beta_str = match.group(1).replace('d', '.')
            try:
                return float(beta_str)
            except ValueError:
                pass
        return float('inf') # Return infinity if extraction fails

# --- compute_vortensity_source function ---
def compute_vortensity_source_enthalpy(gasdens, gasenergy, xgrid, ygrid, gamma):
    ny, nx = gasdens.shape
    pressure = gasenergy * (gamma - 1.0)
    sigma_floor = np.maximum(1e-12 * np.mean(gasdens), 1e-15)
    pressure_floor = sigma_floor * (gamma - 1.0) * 1e-12 * np.mean(gasenergy)
    # Use np.maximum to apply floor element-wise
    gasdens_floor = np.maximum(gasdens, sigma_floor)
    pressure_floor = np.maximum(pressure, pressure_floor)

    # Avoid division by zero or very small numbers in enthalpy calculation
    safe_gasdens = np.maximum(gasdens_floor, 1e-20)
    enthalpy = (gamma / (gamma - 1.0)) * pressure_floor / safe_gasdens

    # Avoid log(0) or log(negative)
    safe_pressure = np.maximum(pressure_floor, 1e-20)
    safe_density_for_log = np.maximum(gasdens_floor, 1e-20)
    entropy = np.log(safe_pressure) - gamma * np.log(safe_density_for_log)

    # Use edge_order=2 for potentially smoother gradients
    grad_H_phi = np.gradient(enthalpy, ygrid, axis=0, edge_order=2)
    grad_H_r   = np.gradient(enthalpy, xgrid, axis=1, edge_order=2)
    grad_S_phi = np.gradient(entropy, ygrid, axis=0, edge_order=2)
    grad_S_r   = np.gradient(entropy, xgrid, axis=1, edge_order=2)

    if xgrid.ndim == 1: r = xgrid[np.newaxis, :]
    else: r = xgrid
    r = np.maximum(r, 1e-10) # Avoid division by zero radius

    term1 = grad_H_r * (1.0 / r) * grad_S_phi
    term2 = (1.0 / r) * grad_H_phi * grad_S_r
    cross_product_z = term1 - term2

    # Avoid division by zero or very small numbers in the final step
    safe_gasdens_sq = np.maximum(gasdens_floor**2, 1e-20)
    source_term = cross_product_z / safe_gasdens_sq

    # Final cleanup of potential numerical issues
    source_term = np.nan_to_num(source_term, nan=0.0, posinf=0.0, neginf=0.0)
    return source_term

import numpy as np

def compute_vortensity_source(gasdens, gasenergy, xgrid, ygrid, gamma):
    """
    Compute the baroclinic source term S = (nabla Sigma x nabla P) / Sigma^3 directly.
    
    Parameters:
    - gasdens: 2D array of surface density (Sigma)
    - gasenergy: 2D array of internal energy
    - xgrid: 1D or 2D array of radial coordinates (r)
    - ygrid: 1D or 2D array of azimuthal coordinates (phi)
    - gamma: Adiabatic index
    
    Returns:
    - source_term: 2D array of the vortensity source term
    """
    ny, nx = gasdens.shape
    pressure = gasenergy * (gamma - 1.0)  # P = (gamma - 1) * energy

    # Compute gradients of Sigma and P
    grad_Sigma_phi = np.gradient(gasdens, ygrid, axis=0, edge_order=2)
    grad_Sigma_r   = np.gradient(gasdens, xgrid, axis=1, edge_order=2)
    grad_P_phi     = np.gradient(pressure, ygrid, axis=0, edge_order=2)
    grad_P_r       = np.gradient(pressure, xgrid, axis=1, edge_order=2)

    # Adjust for radius in polar coordinates
    if xgrid.ndim == 1:
        r = xgrid[np.newaxis, :]
    else:
        r = xgrid
    r_safe = np.maximum(r, 1e-10)  # Avoid division by zero

    # Compute the z-component of the cross product: (∂Σ/∂r * ∂P/∂φ - ∂Σ/∂φ * ∂P/∂r) / r
    cross_product_z = (grad_Sigma_r * grad_P_phi - grad_Sigma_phi * grad_P_r) / r_safe

    #cross_product_z = (grad_Sigma_r * grad_P_phi) / r_safe

    # Compute source term S = cross_product_z / Sigma^3
    epsilon = 1e-20
    safe_Sigma_cube = np.maximum(gasdens**3, epsilon)  # Prevent division by zero
    source_term = cross_product_z / safe_Sigma_cube

    # Mask low-density regions to reduce numerical noise
    sigma_threshold = 1e-6 * np.mean(gasdens)
    source_term = np.where(gasdens > sigma_threshold, source_term, 0.0)

    # Handle NaNs and infinities
    source_term = np.nan_to_num(source_term, nan=0.0, posinf=0.0, neginf=0.0)
    return source_term


# --- compute_potential_vorticity function ---
def compute_potential_vorticity(gasdens, gasvx, gasvy, xgrid, ygrid):
    ny, nx = gasdens.shape
    sigma_floor = np.maximum(1e-12 * np.mean(gasdens), 1e-15)
    gasdens_floor = np.maximum(gasdens, sigma_floor)

    if xgrid.ndim == 1: r = xgrid[np.newaxis, :]
    else: r = xgrid
    r = np.maximum(r, 1e-10) # Avoid division by zero radius

    # Ensure gasvx and gasvy have the same shape as r
    if gasvx.ndim == 1: gasvx = gasvx[np.newaxis, :]
    if gasvy.ndim == 1: gasvy = gasvy[np.newaxis, :]
    if gasvx.shape != r.shape: gasvx = np.tile(gasvx, (r.shape[0],1)) # Basic broadcasting attempt
    if gasvy.shape != r.shape: gasvy = np.tile(gasvy, (r.shape[0],1))

    v_r_inertial = gasvy
    v_phi_inertial = gasvx + r 
    r_v_phi_inertial = r * v_phi_inertial

    # Use edge_order=2 for potentially smoother gradients
    grad_r_v_phi_inertial_r = np.gradient(r_v_phi_inertial, xgrid, axis=1, edge_order=2)
    grad_v_r_phi = np.gradient(v_r_inertial, ygrid, axis=0, edge_order=2)

    # Avoid division by zero radius
    safe_r = np.maximum(r, 1e-10)
    omega_z_inertial = (1.0 / safe_r) * grad_r_v_phi_inertial_r - (1.0 / safe_r) * grad_v_r_phi

    # Avoid division by zero or very small numbers
    safe_gasdens = np.maximum(gasdens_floor, 1e-20)
    pv = omega_z_inertial / safe_gasdens

    # Final cleanup
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
    cmap_levels = plt.get_cmap('viridis', len(chosen_levels) + 1) # Changed colormap
    for i, level in enumerate(chosen_levels):
        ax_hist.axvline(level, color=cmap_levels(i / len(chosen_levels)), linestyle='--',
                        label=f'Level {i+1}: {level:.2e}')

    ax_hist.set_xlabel("Delta PV Value")
    ax_hist.set_ylabel("Number of Grid Cells in Bin")
    ax_hist.set_title(f"Histogram of Delta PV in range [{hist_range[0]:.2e}, {hist_range[1]:.2e}]\n"
                      f"({simname}, Snapshot {snapshot})")
    ax_hist.legend(fontsize='small')
    ax_hist.grid(True, axis='y', alpha=0.5) # Grid on y-axis only
    ax_hist.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # Scientific notation for x-axis

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

# --- Function to load and process data for one simulation ---
def load_and_process_sim_data(simname, snapshot, args):
    """Loads data, calculates S*, dPV, and other necessary fields for a single simulation."""
    print(f"\n--- Processing {simname}, Snapshot {snapshot} ---")
    output_data = {"simname": simname, "snapshot": snapshot, "error": None}

    # --- Setup, Parameter Reading, Extract Beta ---
    beta_val = float('inf'); alpha = None
    try:
        base_path = determine_base_path(simname, IDEFIX=args.IDEFIX); print(f"Base path: {base_path}")
        output_data["base_path"] = base_path
        beta_val = extract_beta_value(base_path); print(f"Extracted beta: {beta_val}")
        summary_file = f"{base_path}/{'idefix.0.log' if args.IDEFIX else 'summary0.dat'}"
        parameters = read_parameters(summary_file, IDEFIX=args.IDEFIX)
        if args.IDEFIX: alpha = parameters.get("problem", {}).get("SigmaExponent") # More robust IDEFIX param access
        else: alpha = parameters.get("SigmaSlope"); alpha = parameters.get("alpha") if alpha is None else alpha
        if alpha is None: print("WARNING: Could not get alpha. Assuming alpha=0."); alpha = 0.0
        else: print(f"Using alpha = {alpha}")
        gamma = parameters.get("physics", {}).get("gamma") if args.IDEFIX else parameters.get("GAMMA") # More robust IDEFIX param access
        if gamma is None: raise ValueError("Gamma could not be determined.")
        output_data["parameters"] = parameters
        output_data["beta_val"] = beta_val
        output_data["gamma"] = gamma


    except FileNotFoundError as e: output_data["error"] = f"Error finding files: {e}"; return output_data
    except Exception as e: output_data["error"] = f"Error reading params/beta: {e}"; return output_data

    # --- Estimate Time ---
    try:
        time_orbit = float('nan')
        if args.IDEFIX:
            output_config = parameters.get("output", {}); vtk_config = output_config.get("vtk", [{}])[0] # Assume first vtk output
            dt_vtk = vtk_config.get("dt")
            if dt_vtk: time_snapshot = snapshot * dt_vtk
            else: time_snapshot = float('nan') # Or provide a default?
        else: # FARGO
             ninterm = parameters.get("NINTERM", 1); outputs_per_orbit = ninterm # Default NINTERM=1
             time_orbit = snapshot / outputs_per_orbit if outputs_per_orbit > 0 else float('nan')

        if not np.isnan(time_orbit): time_orbit_str = f"{time_orbit:.1f} orbits"
        elif not np.isnan(time_snapshot) and not args.IDEFIX: time_orbit = time_snapshot / (2 * np.pi); time_orbit_str = f"{time_orbit:.1f} orbits"
        elif not np.isnan(time_snapshot) and args.IDEFIX: time_orbit_str = f"t={time_snapshot:.2f}" # Use code time for IDEFIX if dt known
        else: time_orbit_str = f"Snap {snapshot}"

        output_data["time_orbit_str"] = time_orbit_str

    except Exception as e:
        print(f"Warning: Could not get time: {e}")
        output_data["time_orbit_str"] = f"Snap {snapshot}"

    # --- Load Data ---
    print(f"Loading snapshot {snapshot} and snapshot 0...")
    fields_to_read = ["gasdens", "gasenergy", "gasvx", "gasvy"]
    try:
        read_func = read_single_snapshot_idefix if args.IDEFIX else read_single_snapshot
        data_snap = read_func(base_path, snapshot, **{f"read_{f}": True for f in fields_to_read})[0]
        data_init = read_func(base_path, 0, **{f"read_{f}": True for f in fields_to_read})[0]

        gasdens = data_snap["gasdens"]; gasenergy = data_snap["gasenergy"]; gasvx = data_snap["gasvx"]; gasvy = data_snap["gasvy"]
        gasdens0 = data_init["gasdens"]; gasvx0 = data_init["gasvx"]; gasvy0 = data_init["gasvy"] # Only need these from init

        # Read gasenergy0 if needed for initial PV calculation refinement (optional)
        # read_init_energy = True
        # if read_init_energy:
        #    data_init_en = read_func(base_path, 0, read_gasenergy=True)[0]
        #    gasenergy0 = data_init_en["gasenergy"]
        # else:
        #    gasenergy0 = None # Or estimate based on initial conditions if needed

    except FileNotFoundError as e: output_data["error"] = f"Snapshot file not found: {e}"; return output_data
    except KeyError as e: output_data["error"] = f"Field {e} not found. Check data_reader."; return output_data
    except Exception as e: output_data["error"] = f"Error reading snapshot data: {e}"; return output_data

    # --- Reconstruct Grid ---
    try:
        xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=args.IDEFIX)
        output_data["xgrid"] = xgrid; output_data["ygrid"] = ygrid
        output_data["nx"] = nx; output_data["ny"] = ny
    except Exception as e: output_data["error"] = f"Error reconstructing grid: {e}"; return output_data

    # --- Select Midplane if 3D ---
    if gasdens.ndim == 3:
        idx = gasdens.shape[2] // 2 if gasdens.shape[2] > 1 else 0
        print(f"Selecting midplane index: {idx}")
        gasdens = gasdens[:, :, idx]; gasenergy = gasenergy[:, :, idx]
        gasvx = gasvx[:, :, idx]; gasvy = gasvy[:, :, idx]
        gasdens0 = gasdens0[:, :, idx]; gasvx0 = gasvx0[:, :, idx]; gasvy0 = gasvy0[:, :, idx]
        # if read_init_energy and gasenergy0 is not None and gasenergy0.ndim == 3:
        #     gasenergy0 = gasenergy0[:, :, idx]


    # --- Calculate S*, PV, PV0, and Delta PV ---
    print("Calculating Source Term (S*)...")
    source_term = compute_vortensity_source(gasdens, gasenergy, xgrid, ygrid, gamma)
    print("Calculating Potential Vorticity (PV) for snapshot...")
    pv_field = compute_potential_vorticity(gasdens, gasvx, gasvy, xgrid, ygrid)
    print("Calculating Initial Potential Vorticity (PV0)...")
    # If PV0 calculation needs gasenergy0, pass it here:
    # pv0_field = compute_potential_vorticity(gasdens0, gasvx0, gasvy0, xgrid, ygrid) # Needs update if PV depends on energy/pressure
    pv0_field = compute_potential_vorticity(gasdens0, gasvx0, gasvy0, xgrid, ygrid) # Assuming PV func only needs dens, vels


    delta_pv = pv_field - pv0_field
    delta_pv = np.nan_to_num(delta_pv, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Calculated Delta PV array. Min: {np.min(delta_pv):.2e}, Max: {np.max(delta_pv):.2e}")

    # --- Shift Data Azimuthally ---
    shift_index = ny // 2
    print(f"Shifting data azimuthally by {shift_index} cells.")
    source_term_rolled = np.roll(source_term, shift_index, axis=0)
    delta_pv_rolled = np.roll(delta_pv, shift_index, axis=0)

    output_data["source_term_rolled"] = source_term_rolled
    output_data["delta_pv_rolled"] = delta_pv_rolled
    output_data["r_plot"] = xgrid # Radial coordinate (usually xgrid)
    output_data["phi_plot"] = ygrid # Azimuthal coordinate (usually ygrid)

    # --- Calculate Contour Levels ---
    plot_data_cont = delta_pv_rolled
    cont_finite = plot_data_cont[np.isfinite(plot_data_cont)]

    # Determine min/max delta PV in the specified plot view
    min_dpv_global = np.min(delta_pv_rolled)
    max_dpv_global = np.max(delta_pv_rolled)
    phi_min_plot = args.phi_min if args.phi_min is not None else ygrid.min()
    phi_max_plot = args.phi_max if args.phi_max is not None else ygrid.max()

    # Ensure r_plot and phi_plot are 1D or 2D as expected
    r_coord = xgrid[0, :] if xgrid.ndim == 2 else xgrid # Assume radial coord varies along axis 1
    phi_coord = ygrid[:, 0] if ygrid.ndim == 2 else ygrid # Assume phi coord varies along axis 0

    r_indices = np.where((r_coord >= args.r_min) & (r_coord <= args.r_max))[0]
    phi_indices = np.where((phi_coord >= phi_min_plot) & (phi_coord <= phi_max_plot))[0]

    dpv_view = None
    min_dpv_in_view = min_dpv_global
    max_dpv_in_view = max_dpv_global # Needed for positive range calculation fallback

    if len(r_indices) > 0 and len(phi_indices) > 0:
        idx_r_min, idx_r_max = r_indices.min(), r_indices.max()
        idx_phi_min, idx_phi_max = phi_indices.min(), phi_indices.max()
        # Ensure indices are within bounds
        idx_phi_max = min(idx_phi_max, delta_pv_rolled.shape[0]-1)
        idx_r_max = min(idx_r_max, delta_pv_rolled.shape[1]-1)

        if idx_phi_min <= idx_phi_max and idx_r_min <= idx_r_max:
            dpv_view = delta_pv_rolled[idx_phi_min:idx_phi_max+1, idx_r_min:idx_r_max+1]
            if dpv_view.size > 0:
                 min_dpv_in_view = np.min(dpv_view); max_dpv_in_view = np.max(dpv_view)
                 print(f"Min/Max Delta PV within plot limits: {min_dpv_in_view:.3e} / {max_dpv_in_view:.3e}")
            else: print("Warning: No Delta PV data found within plot limits.")
        else: print(f"Warning: Invalid indices for plot limits (phi: {idx_phi_min}-{idx_phi_max}, r: {idx_r_min}-{idx_r_max}).")
    else: print("Warning: Could not determine indices for plot limits.")

    output_data["min_dpv_to_display"] = min_dpv_in_view

    # --- Define contour levels using histogram method (with FAC fallback) ---
    contour_levels = None
    hist = None
    bin_edges = None
    data_subset = None
    hist_range = None

    FAC_values = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01] # Try decreasing FAC values

    if len(cont_finite) > 0 and dpv_view is not None and dpv_view.size > args.hist_bins:
        print(f"Attempting histogram analysis for contours. Min dPV in view: {min_dpv_in_view:.3e}")

        for FAC in FAC_values:
            level_min_range = None
            level_max_range = None
            hist_range_ok = False

            # Define histogram range based on minimum dPV in view
            if min_dpv_in_view < -1e-9: # If minimum is significantly negative
                level_min_range = min_dpv_in_view
                level_max_range = min_dpv_in_view * FAC # Range between min and FAC*min
                if level_max_range > level_min_range: # Ensure max > min
                     hist_range_ok = True
                else:
                     print(f"FAC={FAC:.3f}: Hist range [min={level_min_range:.2e}, FAC*min={level_max_range:.2e}] invalid.")
            elif min_dpv_in_view >= -1e-9: # If minimum is close to zero or positive
                 data_range_view = max_dpv_in_view - min_dpv_in_view
                 if data_range_view > 1e-9:
                     # Focus on the lower 20% * FAC of the range starting from the minimum
                     level_min_range = min_dpv_in_view
                     level_max_range = min_dpv_in_view + 0.20 * data_range_view * FAC
                     hist_range_ok = True
                 else:
                     print(f"FAC={FAC:.3f}: Data range [{min_dpv_in_view:.2e}, {max_dpv_in_view:.2e}] too small for positive histogram.")

            if not hist_range_ok:
                continue # Try next FAC

            hist_range = (level_min_range, level_max_range)
            # Select data within the calculated range for the histogram
            data_subset = dpv_view[(dpv_view >= hist_range[0]) & (dpv_view <= hist_range[1])]

            if data_subset.size <= args.hist_bins:
                print(f"FAC={FAC:.3f}: Not enough data points ({data_subset.size}) in range [{hist_range[0]:.2e}, {hist_range[1]:.2e}] for histogram (need > {args.hist_bins}).")
                continue # Try next FAC

            print(f"FAC={FAC:.3f}: Found {data_subset.size} points in range [{hist_range[0]:.2e}, {hist_range[1]:.2e}] for histogram.")
            try:
                # Calculate histogram
                hist, bin_edges = np.histogram(data_subset, bins=args.hist_bins, range=hist_range)
                valid_bins = np.where(hist > 0)[0] # Indices of bins with counts > 0

                if len(valid_bins) > 0:
                    # Find indices of top N bins by count
                    sorted_indices = valid_bins[np.argsort(-hist[valid_bins])] # Sort valid bins by count descending
                    num_levels_actual = min(args.num_hist_levels, len(valid_bins))
                    top_indices = sorted_indices[:num_levels_actual]

                    # Calculate bin centers corresponding to these top bins
                    top_levels = (bin_edges[top_indices] + bin_edges[top_indices + 1]) / 2.0
                    contour_levels = np.unique(np.sort(top_levels)) # Sort and remove duplicates

                    print(f"Selected {len(contour_levels)} contour levels via histogram peaks using FAC={FAC:.3f}.")
                    break # Successfully found levels, exit FAC loop
                else:
                    print(f"FAC={FAC:.3f}: No bins with counts > 0 found in the histogram range.")

            except Exception as e:
                print(f"FAC={FAC:.3f}: Histogram analysis failed - {e}.")
                # Continue trying other FAC values maybe? Or break if error seems fatal.
                # break # Let's break on error for now.

        # Store histogram data for potential plotting later
        output_data["hist_results"] = {
            "data_subset": data_subset,
            "bin_edges": bin_edges,
            "hist": hist,
            "chosen_levels": contour_levels,
            "hist_range": hist_range,
        }

    else: # Fallback if not enough data for histogram
        print("Warning: Histogram skipped due to insufficient data or invalid view range.")
        output_data["hist_results"] = None # Indicate histogram was not run

    if contour_levels is None or len(contour_levels) == 0:
        print("Warning: No contour levels determined.")
        output_data["contour_levels"] = None
    else:
        print(f"Final contour levels: {contour_levels}")
        output_data["contour_levels"] = contour_levels

    print(f"--- Finished processing {simname} ---")
    return output_data


# === MAIN SCRIPT EXECUTION ===
def main():
    # ... (ArgumentParser setup remains the same) ...
    parser = argparse.ArgumentParser(description="Plot 2D Source Term S* with Delta PV contours.")
    parser.add_argument("snapshot", type=int, help="Snapshot number.")
    parser.add_argument("simname", type=str, nargs='?', default=None, help="Simulation name prefix (required if --comparison is not set).") # Optional if comparison is used
    # --- Comparison Flag ---
    parser.add_argument("--comparison", action="store_true", help="Plot comparison for three specific beta values.")
    # --- Other Arguments ---
    parser.add_argument("--IDEFIX", action="store_true", help="Flag for IDEFIX simulation.")
    parser.add_argument("--r_min", type=float, default=0.9, help="Minimum radius for plot (x-axis).")
    parser.add_argument("--r_max", type=float, default=1.1, help="Maximum radius for plot (x-axis).")
    parser.add_argument("--phi_min", type=float, default=None, help="Minimum azimuth for plot (y-axis). Auto = 0")
    parser.add_argument("--phi_max", type=float, default=None, help="Maximum azimuth for plot (y-axis). Auto = 2*pi")
    parser.add_argument("--cmap", type=str, default="RdBu_r", help="Colormap for S*.")
    parser.add_argument("--clim_perc", type=float, default=99.5, help="Percentile for S* color limits (used for individual plots OR global range in comparison).")
    parser.add_argument("--log_scale", action="store_true", help="Use symmetric log scale for S* color.")
    parser.add_argument("--shading", type=str, default="gouraud", choices=['flat', 'nearest', 'gouraud', 'auto'], help="Shading for pcolormesh.")
    # Contour arguments
    parser.add_argument("--num_hist_levels", type=int, default=3, help="Max number of contour levels to plot based on histogram peaks.")
    parser.add_argument("--hist_bins", type=int, default=100, help="Number of bins for histogram.")
    parser.add_argument("--contour_color", type=str, default='lime', help="Color for the Delta PV contours.")
    parser.add_argument("--contour_lw", type=float, default=1.5, help="Linewidth for Delta PV contours.")
    parser.add_argument("--contour_alpha", type=float, default=0.9, help="Alpha for Delta PV contours.")
    parser.add_argument("--clabel_fontsize", type=str, default='small', help="Font size for contour labels.")
    parser.add_argument("--clabel_bg", action="store_true", help="Add white background to contour labels.")
    parser.add_argument("--plot_histogram", action="store_true", help="Additionally plot the histogram used for level selection (creates separate files).")
    # SCP arguments
    parser.add_argument("--scp_user", type=str, default="mariuslehmann", help="Username for SCP transfer.")
    parser.add_argument("--scp_dir", type=str, default="/Users/mariuslehmann/Downloads/Profiles/", help="Target directory for SCP transfer.")


    args = parser.parse_args()

    # --- Validate arguments ---
    if not args.comparison and args.simname is None:
        parser.error("Argument 'simname' is required when --comparison is not set.")
    if args.comparison and args.simname is not None:
        print("WARNING: 'simname' argument is ignored when --comparison is used.")

    # --- Determine simulations to run ---
    if args.comparison:
        sims_to_run = [
            'cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D',
            'cos_bet1d1_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D',
            'cos_bet1d4_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D',
        ]
        if len(sims_to_run) != 3:
             print("ERROR: Comparison mode currently expects exactly 3 simulations.")
             return # Or adjust plotting code
        print(f"Running comparison mode for simulations: {sims_to_run}")
    else:
        sims_to_run = [args.simname]
        print(f"Running single simulation mode for: {args.simname}")


    # --- Load and process data for all required simulations ---
    all_sim_data = []
    for sim_name in sims_to_run:
        data = load_and_process_sim_data(sim_name, args.snapshot, args)
        if data.get("error"):
            print(f"ERROR processing {sim_name}: {data['error']}")
            # Decide whether to continue or exit
            # return # Exit if one fails
            continue # Skip this sim and try others
        all_sim_data.append(data)

    if not all_sim_data:
        print("No simulation data successfully processed. Exiting.")
        return

    # --- Plotting ---
    hist_filenames = [] # Collect histogram filenames for SCP
    # Define base font size for easier adjustment
    base_fontsize = 22 # Increased base font size

    # --- Single Simulation Plot ---
    if not args.comparison:
        if not all_sim_data: return # Should not happen if validation passed, but check anyway
        data = all_sim_data[0]
        simname = data["simname"]
        snapshot = data["snapshot"]
        r_plot = data["r_plot"]
        phi_plot = data["phi_plot"]
        source_term_rolled = data["source_term_rolled"]
        delta_pv_rolled = data["delta_pv_rolled"]
        contour_levels = data["contour_levels"]
        parameters = data["parameters"]
        beta_val = data["beta_val"]
        time_orbit_str = data["time_orbit_str"]
        min_dpv_to_display = data["min_dpv_to_display"]

        print("Generating single plot...")
        # Slightly larger figure size for single plot if needed
        fig, ax = plt.subplots(figsize=(9, 11))

        # Plot Source Term S* (Background)
        # ... (calculation of vmin_bg, vmax_bg, norm_bg as before) ...
        plot_data_bg = source_term_rolled
        abs_vals_bg = np.abs(plot_data_bg[np.isfinite(plot_data_bg)])
        max_abs_val_bg = np.percentile(abs_vals_bg, args.clim_perc) if len(abs_vals_bg) > 0 else 1e-9
        if max_abs_val_bg == 0: max_abs_val_bg = 1e-9
        vmin_bg, vmax_bg = -max_abs_val_bg, max_abs_val_bg
        norm_bg = None
        if args.log_scale:
            linthresh_bg = max_abs_val_bg / 1000.0; linscale_bg = 0.5
            if linthresh_bg <= 0: linthresh_bg = 1e-12 # Ensure linthresh is positive
            try: norm_bg = colors.SymLogNorm(linthresh=linthresh_bg, linscale=linscale_bg, vmin=vmin_bg, vmax=vmax_bg, base=10)
            except ValueError as e: print(f"Warning: SymLogNorm failed ({e}), falling back to linear scale.")
        if norm_bg is None: norm_bg = colors.Normalize(vmin=vmin_bg, vmax=vmax_bg)

        im = ax.pcolormesh(r_plot, phi_plot, plot_data_bg, cmap=args.cmap, norm=norm_bg, shading=args.shading, rasterized=True)
        cbar = plt.colorbar(im, ax=ax, extend='both' if args.log_scale else 'neither')
        cbar.set_label(r"$S^* \propto (\nabla H \times \nabla s)_z / \Sigma^2$", size=base_fontsize) # Increased label size
        cbar.ax.tick_params(labelsize=base_fontsize - 2) # Increased tick label size


        # Plot Delta PV Contours
        if contour_levels is not None and len(contour_levels) > 0:
            print(f"Plotting {len(contour_levels)} Delta PV contours...")
            cont = ax.contour(r_plot, phi_plot, delta_pv_rolled, levels=contour_levels, colors=args.contour_color, linewidths=args.contour_lw, alpha=args.contour_alpha)
            # ... (clabel code remains the same, font size controlled by args.clabel_fontsize) ...
        else:
            print("Skipping Delta PV contours.")

        # Plot Histogram (Optional)
        # ... (histogram plotting code remains the same) ...
        if args.plot_histogram and data.get("hist_results"):
            hr = data["hist_results"]
            if hr.get("hist") is not None and hr.get("bin_edges") is not None and hr.get("chosen_levels") is not None:
                 try:
                     hist_outname = plot_histogram(
                         hr["data_subset"], hr["bin_edges"], hr["hist"], hr["chosen_levels"],
                         hr["hist_range"], simname, snapshot
                     )
                     if hist_outname: hist_filenames.append(hist_outname)
                 except Exception as e:
                     print(f"Failed to plot histogram: {e}")
            else:
                print("Histogram data incomplete, skipping histogram plot.")


        # Finalize Plot
        phi_min_plot = args.phi_min if args.phi_min is not None else phi_plot.min()
        phi_max_plot = args.phi_max if args.phi_max is not None else phi_plot.max()
        ax.set_ylim(phi_min_plot, phi_max_plot)
        ax.set_xlim(args.r_min, args.r_max)
        ax.set_ylabel(r"$\phi$", fontsize=base_fontsize) # Increased font size
        ax.set_xlabel(r"$r/r_0$", fontsize=base_fontsize) # Increased font size
        ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 2) # Increased tick label size

        # Construct Title
        title_parts = []
        if np.isfinite(beta_val):
            if abs(beta_val) < 1e-2 or abs(beta_val) >= 1e3: beta_str = f"{beta_val:.1e}"
            else: beta_str = f"{beta_val:.4g}"
            title_parts.append(rf"$\beta={beta_str}$")
        else: title_parts.append(f"{simname}")
        title_parts.append(f"{time_orbit_str}")
        title_parts.append(f"Min $\delta PV$ (view): {min_dpv_to_display:.2e}")
        if contour_levels is not None and len(contour_levels) > 0:
             dpv_min = np.min(contour_levels); dpv_max = np.max(contour_levels)
             if dpv_min == dpv_max: title_parts.append(rf"Contour: $\delta PV = {dpv_min:.2e}$")
             else: title_parts.append(rf"Contours: $\delta PV \in [{dpv_min:.2e},\, {dpv_max:.2e}]$")
        title_string = f"Source Term $S^*$ & Frequent $\delta PV$ Contours\n({', '.join(title_parts)})"
        ax.set_title(title_string, fontsize=base_fontsize - 2) # Adjusted title font size

        # Planet marker, legend, grid
        planet_r = parameters.get("PlanetPosR", 1.0)
        plot_planet_phi = np.pi
        ax.plot(planet_r, plot_planet_phi, 'm*', markersize=12, markeredgecolor='k', label=r'Planet Loc ($\phi \approx \pi$)')
        ax.axhline(plot_planet_phi, color='grey', linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=base_fontsize - 4) # Adjusted legend font size
        ax.grid(True, alpha=0.3, linestyle=':')

        # Output Filename
        outname = f"source_term_hist_deltapv_contours_{simname}_{snapshot:04d}.pdf"

        # Save and SCP
        plt.tight_layout()
        try: plt.savefig(outname, dpi=150); print(f"Saved plot: {outname}") # Optional: dpi for rasterized elements
        except Exception as e: print(f"Error saving plot: {e}"); return
        finally: plt.close(fig)

        files_to_scp = [outname] + hist_filenames


    # --- Comparison Plot ---
    elif args.comparison:
        if len(all_sim_data) != 3:
            print("Error: Cannot create comparison plot, expected 3 simulations.")
            return

        print("Generating 3-panel comparison plot...")
        # --- Adjusted Figure Size and Spacing ---
        fig, axes = plt.subplots(1, 3,
                                 figsize=(20, 8),        # Increased figure size
                                 sharey=True,            # Shared Y axis
                                 gridspec_kw={'wspace': 0.08}) # Reduced horizontal space

        # --- Determine Global Color Limits for S* (as before) ---
        all_source_terms = np.concatenate([d["source_term_rolled"].ravel() for d in all_sim_data])
        abs_vals_global = np.abs(all_source_terms[np.isfinite(all_source_terms)])
        max_abs_val_global = np.percentile(abs_vals_global, args.clim_perc) if len(abs_vals_global) > 0 else 1e-9
        if max_abs_val_global == 0: max_abs_val_global = 1e-9
        vmin_global, vmax_global = -max_abs_val_global, max_abs_val_global

        norm_global = None
        if args.log_scale:
            linthresh_global = max_abs_val_global / 1000.0; linscale_global = 0.5
            if linthresh_global <= 0: linthresh_global = 1e-12
            try: norm_global = colors.SymLogNorm(linthresh=linthresh_global, linscale=linscale_global, vmin=vmin_global, vmax=vmax_global, base=10)
            except ValueError as e: print(f"Warning: Global SymLogNorm failed ({e}), falling back to linear.")
        if norm_global is None: norm_global = colors.Normalize(vmin=vmin_global, vmax=vmax_global)

        # --- Loop through simulations and plot on each axis ---
        pcm = None
        for i, data in enumerate(all_sim_data):
            ax = axes[i]
            simname = data["simname"]
            snapshot = data["snapshot"]
            r_plot = data["r_plot"]
            phi_plot = data["phi_plot"]
            source_term_rolled = data["source_term_rolled"]
            delta_pv_rolled = data["delta_pv_rolled"]
            contour_levels = data["contour_levels"]
            parameters = data["parameters"]
            beta_val = data["beta_val"]
            # time_orbit_str = data["time_orbit_str"] # Not used in label anymore
            # min_dpv_to_display = data["min_dpv_to_display"] # Not used in label anymore

            # Plot Source Term S* (Background) using GLOBAL norm
            pcm = ax.pcolormesh(r_plot, phi_plot, source_term_rolled, cmap=args.cmap, norm=norm_global, shading=args.shading, rasterized=True)

            # Plot Delta PV Contours (using INDIVIDUAL levels)
            if contour_levels is not None and len(contour_levels) > 0:
                cont = ax.contour(r_plot, phi_plot, delta_pv_rolled, levels=contour_levels, colors=args.contour_color, linewidths=args.contour_lw, alpha=args.contour_alpha)
            else:
                print(f"Skipping contours for sim {i+1} ({simname}) as none were determined.")


            # --- Set Limits and Labels with Increased Font Size ---
            phi_min_plot = args.phi_min if args.phi_min is not None else phi_plot.min()
            phi_max_plot = args.phi_max if args.phi_max is not None else phi_plot.max()
            ax.set_ylim(phi_min_plot, phi_max_plot)
            ax.set_xlim(args.r_min, args.r_max)
            ax.set_xlabel(r"$r/r_0$", fontsize=base_fontsize) # Increased font size
            ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 2) # Increased tick label size

            if i == 0: # Only label Y axis on the first plot
                ax.set_ylabel(r"$\phi$", fontsize=base_fontsize) # Increased font size
            else:
                 ax.tick_params(axis='y', which='major', labelleft=False) # Hide y-tick labels for middle/right plots

            # --- Construct Label Text for Top-Left Corner ---
            label_parts = []
            if np.isfinite(beta_val):
                if abs(beta_val) < 1e-2 or abs(beta_val) >= 1e3: beta_str = f"{beta_val:.1e}"
                else: beta_str = f"{beta_val:.2g}" # Use fewer sig figs for label
                label_parts.append(rf"$\beta={beta_str}$")
            else: label_parts.append(f"Sim {i+1}") # Fallback label

            if contour_levels is not None and len(contour_levels) > 0:
                 dpv_min = np.min(contour_levels); dpv_max = np.max(contour_levels)
                 # Simplify label format
                 if dpv_min == dpv_max: label_parts.append(rf"$\delta PV \approx {dpv_min:.1e}$")
                 else: label_parts.append(rf"$\delta PV \in [{dpv_min:.1e},{dpv_max:.1e}]$")
            label_text = "\n".join(label_parts)

            # --- Add Text Label in Top-Left Corner ---
            ax.text(0.04, 0.96, label_text, # x, y position in axes coords
                    transform=ax.transAxes,
                    fontsize=base_fontsize - 4, # Slightly smaller font for label
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75)) # White background


            # Planet marker
            planet_r = parameters.get("PlanetPosR", 1.0)
            plot_planet_phi = np.pi
            ax.plot(planet_r, plot_planet_phi, 'm*', markersize=10, markeredgecolor='k')
            ax.axhline(plot_planet_phi, color='grey', linestyle='--', alpha=0.6)
            ax.grid(True, alpha=0.3, linestyle=':')

            # Plot Histogram (Optional) - Generates separate files
            # ... (histogram plotting code remains the same) ...
            if args.plot_histogram and data.get("hist_results"):
                hr = data["hist_results"]
                if hr.get("hist") is not None and hr.get("bin_edges") is not None and hr.get("chosen_levels") is not None:
                     try:
                         hist_outname = plot_histogram(
                             hr["data_subset"], hr["bin_edges"], hr["hist"], hr["chosen_levels"],
                             hr["hist_range"], simname, snapshot
                         )
                         if hist_outname: hist_filenames.append(hist_outname)
                     except Exception as e:
                         print(f"Failed to plot histogram for {simname}: {e}")
                else:
                    print(f"Histogram data incomplete for {simname}, skipping histogram plot.")


        # --- Add Shared Horizontal Colorbar Above ---
        if pcm:
             P=0.01
             VAL1=0.87+P
             VAL2=0.88-P
             # Adjust layout slightly more if needed (gridspec_kw handles spacing, this fine-tunes margins)
             fig.subplots_adjust(top=VAL1, bottom=0.1, left=0.07, right=0.98) # Adjusted margins

             # Create axes for the colorbar
             cbar_ax = fig.add_axes([0.15, VAL2, 0.7, 0.035]) # Slightly thicker bar

             cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', extend='both' if args.log_scale else 'neither')
             # --- Updated Colorbar Label and Font Size ---
             cbar.set_label(r"$S^*$", size=base_fontsize) # Changed label to S*, increased size
             cbar.ax.tick_params(labelsize=base_fontsize - 2) # Increased tick label size
             cbar.ax.xaxis.set_ticks_position('top')
             cbar.ax.xaxis.set_label_position('top')
        else:
             plt.tight_layout() # Fallback layout

        # --- Removed Overall Title ---
        # fig.suptitle(...) removed

        # Output Filename for comparison plot
        outname = f"source_term_hist_deltapv_comparison_snap{args.snapshot:04d}.pdf"

        # Save and SCP
        try: plt.savefig(outname, dpi=150); print(f"Saved comparison plot: {outname}") # Optional: dpi for rasterized elements
        except Exception as e: print(f"Error saving comparison plot: {e}"); return
        finally: plt.close(fig)

        files_to_scp = [outname] + hist_filenames # Main plot + any histograms

    # --- SCP Transfer ---
    # ... (SCP code remains the same) ...
    if files_to_scp:
        target_user = args.scp_user
        target_dir = args.scp_dir
        print(f"\nAttempting SCP transfer to {target_user}:{target_dir}...")
        transferred_count = 0
        for f_to_scp in files_to_scp:
            if os.path.exists(f_to_scp):
                try:
                    scp_transfer(f_to_scp, target_dir, target_user)
                    print(f"  Successfully transferred: {f_to_scp}")
                    transferred_count += 1
                except FileNotFoundError: print(f"  SCP Error: 'scp' command not found or script missing for {f_to_scp}.")
                except Exception as e: print(f"  SCP Error transferring {f_to_scp}: {e}")
            else:
                print(f"  Skipping SCP for non-existent file: {f_to_scp}")
        print(f"SCP transfer summary: {transferred_count}/{len(files_to_scp)} files transferred.")
        if transferred_count < len(files_to_scp):
             print("Please transfer remaining files manually if needed.")


if __name__ == "__main__":
    # Make sure helper functions (load_and_process_sim_data, compute_*, plot_histogram)
    # are defined before this point or imported correctly.
    main()

# Example usage:
# python3 vortensity_source_plotter.py 10 cos_bet1d1_gam53_ss15_q1_r0516_nu1dm11_COR_HR1502D_INIT --r_min 0.95 --r_max 1.05 --phi_min 1.57 --phi_max 4.72
# python3 vortensity_source_plotter.py 1 dummy --comparison --r_min 0.95 --r_max 1.05 --phi_min 1.57 --phi_max 4.72
