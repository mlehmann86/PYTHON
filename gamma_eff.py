import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Custom modules
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters
from planet_data import (
    compute_theoretical_torques_PK11,
    read_alternative_torque,
    extract_planet_mass_and_migration
)


# --- UTILITY FUNCTION ---
def get_parameter(simulation_name, parameter_key, IDEFIX=False):
    """
    Reads the summary file (FARGO) or idefix.ini (IDEFIX) for a simulation and 
    returns the value of a specific parameter.

    Parameters:
        simulation_name : str
            The name of the simulation (subdirectory).
        parameter_key : str
            The parameter name to look up (case-insensitive).
        idefix : bool
            Whether the simulation is an IDEFIX run (default: False).
    """
    base_path = determine_base_path(simulation_name, IDEFIX=IDEFIX)
    config_file = os.path.join(base_path, "idefix.ini" if IDEFIX else "summary0.dat")

    if not os.path.exists(config_file):
        print(f"WARNING: Parameter file not found for {simulation_name} at {config_file}")
        return None

    try:
        parameters = read_parameters(config_file, IDEFIX=IDEFIX)
    except Exception as e:
        print(f"WARNING: Error reading parameters from {config_file}: {e}")
        return None

    key_upper = parameter_key.upper()

    if key_upper in parameters:
        return parameters[key_upper]
    else:
        print(f"WARNING: Parameter '{key_upper}' not found in {config_file}")
        print(f"Available keys: {list(parameters.keys())}")
        return None
# --- END OF UTILITY FUNCTION ---



def extract_beta_value(simulation_path):
    """
    Extract the beta cooling time value from the IDEFIX log file or FARGO3D summary file.

    Parameters
    ----------
    simulation_path : str
        Full path to the simulation output directory (not just the name).

    Returns
    -------
    float
        Extracted beta value, or np.inf if not found.
    """
    # Check for IDEFIX log file
    idefix_log = os.path.join(simulation_path, "idefix.0.log")
    if os.path.exists(idefix_log):
        with open(idefix_log, "r") as f:
            for line in f:
                # Match line like: "    beta        0.31622777"
                match = re.search(r'^\s*beta\s+([0-9.eE+-]+)', line)
                if match:
                    return float(match.group(1))

    # Check for FARGO summary file
    fargo_summary = os.path.join(simulation_path, "summary0.dat")
    if os.path.exists(fargo_summary):
        with open(fargo_summary, "r") as f:
            for line in f:
                # Match line like: "BETA          1.000000e-03"
                match = re.search(r'BETA\s+([0-9.eE+-]+)', line)
                if match:
                    return float(match.group(1))

    print(f"WARNING: Could not extract beta value from files in {simulation_path}")
    return float('inf')



def read_simulation_torque(simulation_name, avg_orbits=200, IDEFIX=False):
    """
    Read torque data from a simulation and compute the averaged value over the specified
    number of orbits at the end of the simulation. Returns essential parameters.
    """
    print(f"\n--- Processing Simulation: {simulation_name} ---")
    base_path = determine_base_path(simulation_name, IDEFIX=IDEFIX)
    tqwk_file = os.path.join(base_path, "tqwk0.dat")

    if not IDEFIX:
        summary_file = os.path.join(base_path, "summary0.dat")
        if not os.path.exists(summary_file):
            print(f"  ERROR: Missing summary file: {summary_file}")
            return None
    else:
        summary_file = os.path.join(base_path, "idefix.0.log")

    # Read torque data
    torque_data_read = False
    if os.path.exists(tqwk_file):
        print(f"  Reading torque data from tqwk0.dat")
        try:
            date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file, IDEFIX=IDEFIX)
            torque_data_read = True
        except Exception as e:
            print(f"  ERROR reading tqwk0.dat: {e}")
            return None
    else:
        print(f"  ERROR: No torque file found (tqwk0.dat)")
        return None

    if not torque_data_read or len(torque) == 0:
        print("  ERROR: Failed to read valid torque data.")
        return None

    # Read parameters, planet mass
    print("  Reading simulation parameters...")
    try:
        parameters = read_parameters(summary_file, IDEFIX=IDEFIX)
        if IDEFIX:
            qp = float(parameters.get("planetToPrimary", 0.0))
            migration = None
        else:
            qp, migration = extract_planet_mass_and_migration(summary_file) # qp is planet-to-star mass ratio
        print(f"  Planet mass ratio qp = {qp}") # Debug print for qp
    except Exception as e:
        print(f"  ERROR reading parameters or planet mass: {e}")
        return None

    # Extract beta value from simulation name
    beta = extract_beta_value(base_path)
    if beta == float('inf'):
        print("  ERROR: Could not extract beta value.")
        return None

    # Convert time to orbits
    time_in_orbits = date_torque / (2 * np.pi)

    # Determine avg_torque
    avg_torque = np.nan
    if len(time_in_orbits) < 2:
        print(f"  Warning: Not enough data points (<2) for averaging. Using last torque value.")
        if len(torque) > 0:
            avg_torque = torque[-1]
        else:
             print("  ERROR: Torque array is empty even after reading.")
             return None
    else:
        # Determine averaging interval
        if time_in_orbits[-1] >= 1000:
            start_orbit = 800
            end_orbit = 1000
            print("  Averaging over fixed interval: 800–1000 orbits.")
        else:
            start_orbit = time_in_orbits[-1] - avg_orbits
            end_orbit = time_in_orbits[-1]
            print(f"  Averaging over last {avg_orbits} orbits (from orbit {start_orbit:.1f} to {end_orbit:.1f}).")

        start_idx = np.searchsorted(time_in_orbits, start_orbit)
        end_idx = np.searchsorted(time_in_orbits, end_orbit)

        if start_idx >= len(torque):
            print("  ERROR: Start index for torque averaging is beyond available data.")
            return None
        if end_idx > len(torque):
            end_idx = len(torque)

        avg_torque = np.mean(torque[start_idx:end_idx])
        print(f"  Averaging torque from index {start_idx} (orbit {time_in_orbits[start_idx]:.2f}) "
              f"to {end_idx} (orbit {time_in_orbits[end_idx-1]:.2f}).")
        if start_idx >= len(time_in_orbits):
            print(f"  Warning: Cutoff time {cutoff_time} is after last data point {time_in_orbits[-1]}. Averaging all points.")
            start_idx = 0
        if start_idx < len(torque):
             avg_torque = np.mean(torque[start_idx:])
             print(f"  Averaging torque from index {start_idx} (time {time_in_orbits[start_idx]:.2f} orbits) to end.")
        else:
             print("  ERROR: Could not determine valid start index for torque averaging.")
             return None

    print(f"  Calculated avg_torque (from file, needs scaling interpretation) = {avg_torque}")

    if not np.isfinite(avg_torque):
         print(f"  ERROR: Calculated avg_torque is not finite ({avg_torque}).")
         return None

    # Compute theoretical Lindblad torque
    predicted_torque_lindblad = np.nan
    GAM0 = np.nan
    try:
        print("  Computing theoretical Lindblad torque...")
        # Assuming compute_theoretical_torques returns normalized torque Gamma_L_adi / GAM0
        _, predicted_torque_lindblad, GAM0, gameff, _, _ = compute_theoretical_torques_PK11(parameters, qp, simulation_name, IDEFIX=IDEFIX, summary_file=summary_file, avg_start_orbit=None, avg_end_orbit=None, nu_threshold=1e-15)
        predicted_torque_lindblad = predicted_torque_lindblad*GAM0*gameff



        print(f"  Predicted Lindblad Torque (normalized, adi) = {predicted_torque_lindblad}")
        print(f"  GAM0 scaling factor = {GAM0}")
    except Exception as e:
        print(f"  ERROR computing theoretical torques: {e}")
        return None

    if not np.isfinite(predicted_torque_lindblad):
        print(f"  ERROR: Predicted Lindblad torque is not finite ({predicted_torque_lindblad}).")
        return None

    # Get gamma and aspect ratio
    gamma = parameters.get('GAMMA', None) if not IDEFIX else parameters.get('gamma', None)
    h_sim = parameters.get('ASPECTRATIO', None) if not IDEFIX else parameters.get('h0', None)

    if gamma is None or h_sim is None:
        print(f"  ERROR: GAMMA or ASPECTRATIO missing in parameters.")
        return None

    # --- CORRECTED gamma_eff CALCULATION ---
    gamma_eff = np.nan
    if avg_torque == 0:
        print(f"  Warning: Average measured torque is zero. Cannot calculate gamma_eff via ratio.")
        return None # Return None if avg_torque is zero
    else:
        gamma_eff = (predicted_torque_lindblad / (avg_torque*qp)) 
        print(f"  Corrected gamma_eff = {gamma:.3f} * ({predicted_torque_lindblad:.6e} / {avg_torque:.6e}) * {qp:.2e} = {gamma_eff:.4f}")

    if not np.isfinite(gamma_eff):
        print(f"  ERROR: Calculated gamma_eff is not finite ({gamma_eff}).")
        return None

    result = {
        'simulation_name': simulation_name,
        'beta': beta,
        'gamma': gamma,
        'h': h_sim,
        'avg_torque': avg_torque,
        'qp': qp,
        'GAM0': GAM0,
        'predicted_torque_lindblad': predicted_torque_lindblad,
        'gamma_eff': gamma_eff,
    }
    print(f"--- Successfully processed {simulation_name} ---")
    return result


# ===== NEW: Alternative Torque File Reader =====
def read_simulation_torque_alternative(simulation_name, avg_orbits=200, IDEFIX=False):
    """
    Read alternative torque data from a simulation.
    The alternative file is expected to be named:
       torque_evolution_<simulation_name>.npz
    and contain two 1D arrays: "grav_total" and "time_array".
    """
    print(f"\n--- Processing Simulation (Alternative Torque): {simulation_name} ---")
    base_path = determine_base_path(simulation_name, IDEFIX=IDEFIX)
    alt_file = os.path.join(base_path, f"torque_evolution_{simulation_name}.npz")
    
    if not os.path.exists(alt_file):
        print(f"  ERROR: Alternative torque file not found: {alt_file}")
        return None

    try:
        data = np.load(alt_file)
        torque = data["grav_total"]
        date_torque = data["time_array"]
        print(f"  Reading alternative torque data from {alt_file}")
    except Exception as e:
        print(f"  ERROR reading alternative torque file: {e}")
        return None

    if len(torque) == 0:
        print("  ERROR: Alternative torque data is empty.")
        return None

    if not IDEFIX:
        summary_file = os.path.join(base_path, "summary0.dat")
        if not os.path.exists(summary_file):
            print(f"  ERROR: Missing summary file: {summary_file}")
            return None
    else:
        summary_file = os.path.join(base_path, "idefix.0.log")

    # Read simulation parameters and extract planet mass
    print("  Reading simulation parameters...")
    try:
        parameters = read_parameters(summary_file, IDEFIX=IDEFIX)
        if IDEFIX:
            qp = float(parameters.get("planetToPrimary", 0.0))
            migration = None
        else:
            qp, migration = extract_planet_mass_and_migration(summary_file)
        print(f"  Planet mass ratio qp = {qp}")
    except Exception as e:
        print(f"  ERROR reading parameters or planet mass: {e}")
        return None

    beta = extract_beta_value(base_path)
    if beta == float('inf'):
        print("  ERROR: Could not extract beta value.")
        return None

    time_in_orbits = date_torque 

    # Average torque calculation (using alternative data)
    if len(time_in_orbits) < 2:
        print(f"  Warning: Not enough data points (<2) for averaging. Using last torque value.")
        avg_torque = torque[-1] if len(torque) > 0 else np.nan
    else:
        cutoff_time = time_in_orbits[-1] - avg_orbits
        start_idx = np.searchsorted(time_in_orbits, cutoff_time)
        if start_idx >= len(time_in_orbits):
            print(f"  Warning: Cutoff time {cutoff_time} is after last data point {time_in_orbits[-1]}. Averaging all points.")
            start_idx = 0
        if start_idx < len(torque):
            avg_torque = np.mean(torque[start_idx:])
            print(f"  Averaging alternative torque from index {start_idx} (time {time_in_orbits[start_idx]:.2f} orbits) to end.")
        else:
            print("  ERROR: Could not determine valid start index for alternative torque averaging.")
            return None

    print(f"  Calculated avg_torque (alternative from file) = {avg_torque}")

    if not np.isfinite(avg_torque):
         print(f"  ERROR: Calculated alternative avg_torque is not finite ({avg_torque}).")
         return None

    # Compute theoretical Lindblad torque
    try:
        print("  Computing theoretical Lindblad torque (for alternative torque comparison)...")
        #predicted_torque_lindblad, _, GAM0 = compute_theoretical_torques(parameters, qp, eq_label="Paardekooper2", IDEFIX=IDEFIX)

        _, predicted_torque_lindblad, GAM0, gameff, _, _ = compute_theoretical_torques_PK11(parameters, qp, simulation_name, IDEFIX=IDEFIX,summary_file=summary_file, avg_start_orbit=None, avg_end_orbit=None, nu_threshold=1e-15)
        #rescaling to be compatible with this routine
        predicted_torque_lindblad = predicted_torque_lindblad*GAM0*gameff


        print(f"  Predicted Lindblad Torque = {predicted_torque_lindblad}")
        print(f"  GAM0 scaling factor = {GAM0}")
    except Exception as e:
        print(f"  ERROR computing theoretical torques: {e}")
        return None

    if not np.isfinite(predicted_torque_lindblad):
        print(f"  ERROR: Predicted Lindblad torque is not finite ({predicted_torque_lindblad}).")
        return None

    gamma = parameters.get('GAMMA', None) if not IDEFIX else parameters.get('gamma', None)
    h_sim = parameters.get('ASPECTRATIO', None) if not IDEFIX else parameters.get('h0', None)
    if gamma is None or h_sim is None:
        print(f"  ERROR: GAMMA or ASPECTRATIO missing in parameters.")
        return None

    if avg_torque == 0:
        print(f"  Warning: Average measured alternative torque is zero. Cannot calculate gamma_eff.")
        return None
    else:
        gamma_eff = (predicted_torque_lindblad / (avg_torque * qp))
        print(f"  Calculated alternative gamma_eff = {gamma:.3f} * ({predicted_torque_lindblad:.6e} / {avg_torque:.6e}) * {qp:.2e} = {gamma_eff:.4f}")

    if not np.isfinite(gamma_eff):
        print(f"  ERROR: Calculated alternative gamma_eff is not finite ({gamma_eff}).")
        return None

    result = {
        'simulation_name': simulation_name,
        'beta': beta,
        'gamma': gamma,
        'h': h_sim,
        'avg_torque': avg_torque,
        'qp': qp,
        'GAM0': GAM0,
        'predicted_torque_lindblad': predicted_torque_lindblad,
        'gamma_eff': gamma_eff,
    }
    print(f"--- Successfully processed alternative torque for {simulation_name} ---")
    return result



# --- Corrected Y-Limit Helper (assumed same) ---
def calculate_plot_ylimits(data_list_1, data_list_2, fixed_val):
    all_finite_data = [g for g in data_list_1 + data_list_2 if np.isfinite(g)]
    if not all_finite_data: return fixed_val - 0.2, fixed_val + 0.2
    min_data, max_data = min(all_finite_data), max(all_finite_data)
    overall_min, overall_max = min(min_data, fixed_val), max(max_data, fixed_val)
    data_range = overall_max - overall_min; y_padding = max(0.1 * data_range, 0.05)
    return overall_min - y_padding, overall_max + y_padding

# ==============================================================================
# Plotting Function 1: vs Beta (Corrected Theory Curve)
# ==============================================================================
# --- Plotting Function ---
def create_gamma_eff_plot(simulations_2d, simulations_3d, avg_orbits=200, output_path="plots", IDEFIX=False):
    """
    Create a plot of gamma_eff vs beta cooling time for both 2D and 3D simulations.
    """
    print("\n=== Starting Plot Creation ===")
    # Process simulations
    print("Processing 2D simulations...")
    results_2d = [read_simulation_torque(sim, avg_orbits, IDEFIX=IDEFIX) for sim in simulations_2d]
    print("\nProcessing 3D simulations...")
    results_3d = [read_simulation_torque(sim, avg_orbits, IDEFIX=IDEFIX) for sim in simulations_3d]

    # Filter out None results
    print("\nFiltering results...")
    results_2d_filtered = [r for r in results_2d if r is not None]
    results_3d_filtered = [r for r in results_3d if r is not None]
    print(f"  Valid 2D results: {len(results_2d_filtered)} out of {len(simulations_2d)}")
    print(f"  Valid 3D results: {len(results_3d_filtered)} out of {len(simulations_3d)}")


    if not results_2d_filtered and not results_3d_filtered:
        print("Error: No valid simulation results to plot after processing.")
        return None

    # Sort results by beta value
    results_2d_filtered.sort(key=lambda x: x['beta'])
    results_3d_filtered.sort(key=lambda x: x['beta'])

    # Extract beta and gamma_eff values for plotting
    beta_values_2d = [r['beta'] for r in results_2d_filtered]
    gamma_eff_values_2d = [r['gamma_eff'] for r in results_2d_filtered]

    beta_values_3d = [r['beta'] for r in results_3d_filtered]
    gamma_eff_values_3d = [r['gamma_eff'] for r in results_3d_filtered]

    # --- DEBUG PRINTS ---
    print("\n--- Plotting Data ---")
    print("2D beta values:", beta_values_2d)
    print("2D gamma_eff values:", gamma_eff_values_2d)
    print("3D beta values:", beta_values_3d)
    print("3D gamma_eff values:", gamma_eff_values_3d) # CHECK THESE VALUES
    #-----------------------

    if not beta_values_3d: # Check specifically if 3D data exists before proceeding
         print("WARNING: No valid 3D data points to plot.")
         # Optionally decide whether to proceed with just the theory curve or exit
         # return None # Uncomment this to stop if no 3D data

    # Get gamma_adiabatic and h from the first valid simulation
    first_valid_result = results_2d_filtered[0] if results_2d_filtered else results_3d_filtered[0]
    gamma_adiabatic = first_valid_result['gamma']
    h = first_valid_result['h']
    first_sim_name = first_valid_result['simulation_name']

    print(f"\nUsing parameters from simulation {first_sim_name} for theoretical curve:")
    print(f"  gamma_adiabatic = {gamma_adiabatic}")
    print(f"  h (Aspect Ratio) = {h}\n")

    gamma_iso = 1.0
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    line_style = '-'

    def gamma_eff_theory(gamma, beta, h_val):
        beta = np.asarray(beta, dtype=np.float64)
        Q = (2.0 * beta) / (3.0 * h_val)
        with np.errstate(divide='ignore', invalid='ignore'):
            sqrt_term_inner = (1.0 + Q**2) / (1.0 + gamma**2 * Q**2)
            sqrt_term = np.sqrt(np.maximum(0.0, sqrt_term_inner))
            numerator = 2.0
            denominator = (1.0 + gamma * Q**2) / (1.0 + gamma**2 * Q**2) + sqrt_term
            gamma_eff_val = numerator / denominator
        return gamma_eff_val
    # --------------------------------------------

    # Theoretical curve range
    beta_range = np.logspace(-4, 5, 300)

    # Calculate theoretical curve
    gamma_eff_theory_curve = gamma_eff_theory(gamma_adiabatic, beta_range, h)

    # Plot theoretical curve
    plt.plot(beta_range, gamma_eff_theory_curve, linestyle=line_style, color='black', linewidth=2.5, label=r'Theory $\gamma_{\mathrm{eff}}$')

    # Plot simulation data
    # --- 3D simulations ---
    if beta_values_3d:
        print(f"Attempting to plot {len(beta_values_3d)} 3D points...")

        beta_3d_pos = [b for b, g in zip(beta_values_3d, gamma_eff_values_3d) if g >= 0]
        gamma_3d_pos = [g for g in gamma_eff_values_3d if g >= 0]

        beta_3d_neg = [b for b, g in zip(beta_values_3d, gamma_eff_values_3d) if g < 0]
        gamma_3d_neg = [-g for g in gamma_eff_values_3d if g < 0]

        # Positive gamma_eff: solid red squares
        #plt.scatter(beta_3d_pos, gamma_3d_pos, marker='s', s=100,
        #            facecolors='red', edgecolors='red', linewidth=2.5, label='3D Simulations', zorder=5)
        # Negative gamma_eff: open red squares
        #plt.scatter(beta_3d_neg, gamma_3d_neg, marker='s', s=100,
        #            facecolors='none', edgecolors='red', linewidth=2.5, zorder=5)

        # Connect *all* points (original values) with dotted red line
        #plt.plot(beta_values_3d, [abs(g) for g in gamma_eff_values_3d],
        #         linestyle=':', color='red', linewidth=1, zorder=4)

    # --- 2D simulations ---
    if beta_values_2d:
        print(f"Attempting to plot {len(beta_values_2d)} 2D points...")

        beta_2d_pos = [b for b, g in zip(beta_values_2d, gamma_eff_values_2d) if g >= 0]
        gamma_2d_pos = [g for g in gamma_eff_values_2d if g >= 0]

        beta_2d_neg = [b for b, g in zip(beta_values_2d, gamma_eff_values_2d) if g < 0]
        gamma_2d_neg = [-g for g in gamma_eff_values_2d if g < 0]

        # Positive gamma_eff: solid blue circles
        plt.scatter(beta_2d_pos, gamma_2d_pos, marker='o', s=100,
                    facecolors='blue', edgecolors='blue', linewidth=2, label='2D Simulations', zorder=5)
        # Negative gamma_eff: open blue circles
        plt.scatter(beta_2d_neg, gamma_2d_neg, marker='o', s=100,
                    facecolors='none', edgecolors='blue', linewidth=2, zorder=5)

        # Connect *all* points (original values) with dotted blue line
        plt.plot(beta_values_2d, [abs(g) for g in gamma_eff_values_2d],
                 linestyle=':', color='blue', linewidth=1, zorder=4)

    # Add horizontal lines without showing them in the legend
    plt.axhline(gamma_iso, color='grey', linestyle='--', linewidth=1.5, label='_nolegend_')
    plt.axhline(gamma_adiabatic, color='grey', linestyle=':', linewidth=1.5, label='_nolegend_')

    # Set up plot details
    plt.xscale('log')
    plt.xlabel(r'$\beta = t_{\mathrm{cool}}\Omega_K$', fontsize=16)
    plt.ylabel(r'$\gamma_{\mathrm{eff}}$ (Derived from Torque)', fontsize=16)
    plt.title('Effective Adiabatic Index vs. Cooling Time', fontsize=18)
    plt.grid(True, which="major", axis='both', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.grid(True, which="minor", axis='both', linestyle=':', linewidth=0.3, alpha=0.4)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', length=3)

    # --- DEBUG Y-LIMITS ---
    y_min_plot = np.nan # Initialize
    y_max_plot = np.nan
    all_gamma_eff = gamma_eff_values_2d + gamma_eff_values_3d
    if all_gamma_eff and np.any(np.isfinite(all_gamma_eff)): # Check if any finite data exists
        finite_gamma_eff = [g for g in all_gamma_eff if np.isfinite(g)]
        if finite_gamma_eff:
            min_sim_gamma = min(finite_gamma_eff)
            max_sim_gamma = max(finite_gamma_eff)
            y_min_plot = min(gamma_iso, min_sim_gamma) - 0.1 * (gamma_adiabatic - gamma_iso)
            y_max_plot = max(gamma_adiabatic, max_sim_gamma) + 0.1 * (gamma_adiabatic - gamma_iso)
            y_min_plot = max(0.8, y_min_plot)
            # Let's temporarily remove the upper clamp to see if that's hiding points
            # y_max_plot = min(y_max_plot, gamma_adiabatic + 0.3)
        else: # Handle case where data exists but is all NaN/Inf
             print("WARNING: Simulation gamma_eff data contains no finite values.")
             y_min_plot = gamma_iso - 0.2
             y_max_plot = gamma_adiabatic + 0.2
    else: # Fallback if no sim data
        print("WARNING: No valid simulation data for setting y-limits dynamically.")
        y_min_plot = gamma_iso - 0.2 # Slightly larger default margin
        y_max_plot = gamma_adiabatic + 0.2
    print(f"Setting Y Limits: ({y_min_plot:.4f}, {y_max_plot:.4f})")
    ax = plt.gca()
    plt.yscale('log')
    plt.ylim(0.8, 2)

    # Set clean, uniform y-ticks without ugly font/notation issues
    ytick_vals = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([f"{y:.1f}" for y in ytick_vals], fontsize=14)
    ax.set_yticks([], minor=True)

    plt.legend(fontsize=14, loc='upper left')


    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save the figure
    if IDEFIX:
        output_file = os.path.join(output_path, "gamma_eff_vs_beta_IDEFIX.pdf")
    else:
        output_file = os.path.join(output_path, "gamma_eff_vs_beta_FARGO.pdf")
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    plt.close()

    local_directory = "/Users/mariuslehmann/Downloads/Profiles/"
    username = "mariuslehmann" # Or get from config/environment
    print(f"\nAttempting to transfer {output_file} to {local_directory} for user {username}...")
    scp_transfer(output_file, local_directory, username)

    return output_file

# ===== NEW: Plotting Function 1 Alternative: vs Beta (Using Alternative Torque Files) =====
def create_gamma_eff_plot_alternative(simulations_2d, simulations_3d, avg_orbits=200, output_path="plots", IDEFIX=False):
    """
    Create a plot of gamma_eff vs beta cooling time for both 2D and 3D simulations,
    using the alternative torque files computed from the density field snapshots.
    """
    print("\n=== Starting Alternative Plot Creation (Using Hand-Computed Torque Files) ===")
    print("Processing 2D simulations with alternative torque...")
    results_2d = [read_simulation_torque_alternative(sim, avg_orbits, IDEFIX=IDEFIX) for sim in simulations_2d]
    print("\nProcessing 3D simulations with alternative torque...")
    results_3d = [read_simulation_torque_alternative(sim, avg_orbits, IDEFIX=IDEFIX) for sim in simulations_3d]

    results_2d_filtered = [r for r in results_2d if r is not None]
    results_3d_filtered = [r for r in results_3d if r is not None]
    print(f"  Valid 2D alternative results: {len(results_2d_filtered)} out of {len(simulations_2d)}")
    print(f"  Valid 3D alternative results: {len(results_3d_filtered)} out of {len(simulations_3d)}")

    if not results_2d_filtered and not results_3d_filtered:
        print("Error: No valid alternative simulation results to plot after processing.")
        return None

    results_2d_filtered.sort(key=lambda x: x['beta'])
    results_3d_filtered.sort(key=lambda x: x['beta'])

    beta_values_2d = [r['beta'] for r in results_2d_filtered]
    gamma_eff_values_2d = [r['gamma_eff'] for r in results_2d_filtered]
    beta_values_3d = [r['beta'] for r in results_3d_filtered]
    gamma_eff_values_3d = [r['gamma_eff'] for r in results_3d_filtered]

    print("\n--- Alternative Plotting Data ---")
    print("2D beta values:", beta_values_2d)
    print("2D gamma_eff values:", gamma_eff_values_2d)
    print("3D beta values:", beta_values_3d)
    print("3D gamma_eff values:", gamma_eff_values_3d)

    if not beta_values_3d:
         print("WARNING: No valid 3D alternative data points to plot.")

    first_valid_result = results_2d_filtered[0] if results_2d_filtered else results_3d_filtered[0]
    gamma_adiabatic = first_valid_result['gamma']
    h = first_valid_result['h']
    first_sim_name = first_valid_result['simulation_name']

    print(f"\nUsing parameters from simulation {first_sim_name} for theoretical curve (alternative torque):")
    print(f"  gamma_adiabatic = {gamma_adiabatic}")
    print(f"  h (Aspect Ratio) = {h}\n")

    gamma_iso = 1.0
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    line_style = '-'

    def gamma_eff_theory(gamma, beta, h_val):
        beta = np.asarray(beta, dtype=np.float64)
        Q = (2.0 * beta) / (3.0 * h_val)
        with np.errstate(divide='ignore', invalid='ignore'):
            sqrt_term_inner = (1.0 + Q**2) / (1.0 + gamma**2 * Q**2)
            sqrt_term = np.sqrt(np.maximum(0.0, sqrt_term_inner))
            numerator = 2.0
            denominator = (1.0 + gamma * Q**2) / (1.0 + gamma**2 * Q**2) + sqrt_term
            gamma_eff_val = numerator / denominator
        return gamma_eff_val

    beta_range = np.logspace(-4, 5, 300)
    gamma_eff_theory_curve = gamma_eff_theory(gamma_adiabatic, beta_range, h)

    plt.plot(beta_range, gamma_eff_theory_curve, linestyle=line_style, color='black', linewidth=2.5,
             label=r'Theory $\gamma_{\mathrm{eff}}$')

    if beta_values_3d:
        plt.scatter(beta_values_3d, gamma_eff_values_3d, marker='s', s=100,
                    facecolors='none', edgecolors='red', linewidth=2.5, label='3D Simulations (Alt)', zorder=5)
        plt.plot(beta_values_3d, gamma_eff_values_3d, linestyle=':', color='red', linewidth=1, zorder=4)
    if beta_values_2d:
        plt.scatter(beta_values_2d, gamma_eff_values_2d, marker='o', s=100,
                    facecolors='none', edgecolors='blue', linewidth=2, label='2D Simulations (Alt)', zorder=5)
        plt.plot(beta_values_2d, gamma_eff_values_2d, linestyle=':', color='blue', linewidth=1, zorder=4)

    plt.axhline(gamma_iso, color='grey', linestyle='--', linewidth=1.5, label='_nolegend_')
    plt.axhline(gamma_adiabatic, color='grey', linestyle=':', linewidth=1.5, label='_nolegend_')

    plt.xscale('log')
    plt.xlabel(r'$\beta = t_{\mathrm{cool}}\Omega_K$', fontsize=16)
    plt.ylabel(r'$\gamma_{\mathrm{eff}}$ (Derived from Alternative Torque)', fontsize=16)
    plt.title('Effective Adiabatic Index vs. Cooling Time (Alternative Torque)', fontsize=18)
    plt.grid(True, which="major", axis='both', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.grid(True, which="minor", axis='both', linestyle=':', linewidth=0.3, alpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', length=3)
    #plt.yscale('log')
    plt.ylim(-5, 20)
    plt.legend(fontsize=14, loc='center right')

    os.makedirs(output_path, exist_ok=True)
    if IDEFIX:
        output_file = os.path.join(output_path, "gamma_eff_vs_beta_alternative_IDEFIX.pdf")
    else:
        output_file = os.path.join(output_path, "gamma_eff_vs_beta_alternative_FARGO.pdf")
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nAlternative Torque Plot saved to {output_file}")
    except Exception as e:
        print(f"\nError saving alternative plot: {e}")
    plt.close()

    local_directory = "/Users/mariuslehmann/Downloads/Profiles/"
    username = "mariuslehmann"
    print(f"\nAttempting to transfer alternative plot {output_file} to {local_directory} for user {username}...")
    scp_transfer(output_file, local_directory, username)
    return output_file


# ==============================================================================
# Plotting Function 2: vs SigmaSlope (Corrected y-limits)
# ==============================================================================
def create_gamma_eff_vs_sigmaslope_plot(simulations_sigmaslope_lr, simulations_sigmaslope_hr, avg_orbits=200, output_path="plots", IDEFIX=False):
    """ Plots gamma_eff vs sigmaslope, compares resolutions, with corrected y-limit logic. """
    print("\n=== Starting Plot Creation: Gamma_eff vs SigmaSlope (LR vs HR) ===")
    def process_sim_list(sim_list, sim_label): # Inner helper function
        print(f"\n--- Processing {sim_label} Simulations ---")
        results = []
        for sim in sim_list:
            sim_result = read_simulation_torque(sim, avg_orbits, IDEFIX=IDEFIX)
            if sim_result:
                sigmaslope_val = get_parameter(sim, 'SIGMASLOPE', IDEFIX=IDEFIX)
                if sigmaslope_val is not None:
                    print(f"  {sim_label} Sim {sim}: SIGMASLOPE = {sigmaslope_val}, gamma_eff = {sim_result['gamma_eff']:.4f}")
                    sim_result['sigmaslope'] = float(sigmaslope_val) # Ensure float
                    results.append(sim_result)
                else: print(f"  Skipping {sim_label} {sim}: No SIGMASLOPE.")
            else: print(f"  Skipping {sim_label} {sim}: Failed processing.")
        print(f"Filtered {sim_label} results: {len(results)} out of {len(sim_list)}")
        results.sort(key=lambda x: x['sigmaslope'])
        return results

    results_lr = process_sim_list(simulations_sigmaslope_lr, "Std Res")
    results_hr = process_sim_list(simulations_sigmaslope_hr, "High Res")

    if not results_lr and not results_hr: print("Error: No valid sim results."); return None

    sigmaslope_values_lr = [r['sigmaslope'] for r in results_lr]; gamma_eff_values_lr = [r['gamma_eff'] for r in results_lr]
    sigmaslope_values_hr = [r['sigmaslope'] for r in results_hr]; gamma_eff_values_hr = [r['gamma_eff'] for r in results_hr]

    print("\n--- Plotting Data (vs SigmaSlope) ---")
    print("LR SigmaSlopes:", sigmaslope_values_lr); print("LR Gamma_effs:", gamma_eff_values_lr)
    print("HR SigmaSlopes:", sigmaslope_values_hr); print("HR Gamma_effs:", gamma_eff_values_hr)

    first_valid = results_lr[0] if results_lr else results_hr[0]; gamma_adiabatic = first_valid['gamma']

    plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(8, 6))
    if results_lr: plt.plot(sigmaslope_values_lr, gamma_eff_values_lr, marker='o', linestyle='-', color='blue', markersize=8, linewidth=2, label='Std Res Simulations (Adiabatic)')
    if results_hr: plt.plot(sigmaslope_values_hr, gamma_eff_values_hr, marker='s', linestyle='--', color='red', markersize=8, linewidth=2, label='High Res Simulations (Adiabatic)')
    plt.axhline(gamma_adiabatic, color='grey', linestyle=':', linewidth=1.5, label=rf'Expected $\gamma_{{adi}}={gamma_adiabatic:.2f}$')

    plt.xlabel(r'Surface Density Slope ($\Sigma \propto r^{\mathrm{sigmaslope}}$)', fontsize=16); plt.ylabel(r'$\gamma_{\mathrm{eff}}$ (Derived from Torque)', fontsize=16)
    plt.title(r'Effective $\gamma$ vs. $\Sigma$ Slope (Resolution Comparison)', fontsize=18); plt.grid(True, which="major", linestyle='-', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Corrected Y-Limits
    ymin, ymax = calculate_plot_ylimits(gamma_eff_values_lr, gamma_eff_values_hr, gamma_adiabatic)
    print(f"Setting Y Limits (vs SigmaSlope): ({ymin:.4f}, {ymax:.4f})")
    plt.ylim(ymin, ymax)

    plt.legend(fontsize=14, loc='best');
    os.makedirs(output_path, exist_ok=True); 
    if IDEFIX:
        output_file = os.path.join(output_path, "gamma_eff_vs_sigmaslope_res_comp_IDEFIX.pdf")
    else:
        output_file = os.path.join(output_path, "gamma_eff_vs_sigmaslope_res_comp_FARGO.pdf")
    try: plt.savefig(output_file, dpi=300, bbox_inches='tight'); print(f"\nPlot saved: {output_file}")
    except Exception as e: print(f"\nError saving plot: {e}")
    plt.close()
    try: local_directory = "/Users/mariuslehmann/Downloads/Profiles/"; username = "mariuslehmann"; scp_transfer(output_file, local_directory, username)
    except Exception as e: print(f"WARNING: SCP transfer failed: {e}")
    return output_file

# ==============================================================================
# Plotting Function 3: vs TempSlope (Corrected y-limits)
# ==============================================================================
def create_gamma_eff_vs_tempslope_plot(simulations_tempslope, avg_orbits=200, output_path="plots", IDEFIX=False):
    """ Plots gamma_eff vs tempslope (beta_T), with corrected y-limit logic. """
    print("\n=== Starting Plot Creation: Gamma_eff vs TempSlope ===")
    results = []
    for sim in simulations_tempslope:
        sim_result = read_simulation_torque(sim, avg_orbits, IDEFIX=IDEFIX)
        if sim_result:
            tempslope_val = get_parameter(sim, 'TEMPSLOPE', IDEFIX=IDEFIX)
            if tempslope_val is None:
                 print(f"  WARN: TEMPSLOPE not found for {sim}. Trying FLARINGINDEX.")
                 fi_val = get_parameter(sim, 'FLARINGINDEX', IDEFIX=IDEFIX)
                 if fi_val is not None:
                     try: tempslope_val = 1.0 - 2.0 * float(fi_val); print(f"  Calculated beta_T={tempslope_val:.2f} from FLARINGINDEX={fi_val}")
                     except (ValueError, TypeError) as e_fi: print(f"  Skipping {sim}: Bad FLARINGINDEX '{fi_val}' - {e_fi}"); continue
                 else: print(f"  Skipping {sim}: No TEMPSLOPE or FLARINGINDEX."); continue
            else:
                try: tempslope_val = float(tempslope_val); print(f"  Found TEMPSLOPE = {tempslope_val:.2f}")
                except (ValueError, TypeError): print(f"  Skipping {sim}: Bad TEMPSLOPE value '{tempslope_val}'."); continue

            sim_result['tempslope'] = tempslope_val; results.append(sim_result)
        else: print(f"  Skipping {sim}: Failed processing.")

    print(f"\nFiltered results for TempSlope plot: {len(results)} out of {len(simulations_tempslope)}")
    if not results: print("Error: No valid sim results."); return None

    results.sort(key=lambda x: x['tempslope']);
    tempslope_values = [r['tempslope'] for r in results]; gamma_eff_values = [r['gamma_eff'] for r in results]
    print("\n--- Plotting Data (vs TempSlope) ---")
    print("TempSlope (beta_T):", tempslope_values); print("Gamma_eff:", gamma_eff_values)
    gamma_adiabatic = results[0]['gamma']

    plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(8, 6))
    plt.plot(tempslope_values, gamma_eff_values, marker='o', linestyle='-', color='green', markersize=8, linewidth=2, label='2D Simulations (Adiabatic Limit)')
    plt.axhline(gamma_adiabatic, color='grey', linestyle=':', linewidth=1.5, label=rf'Expected $\gamma_{{adi}}={gamma_adiabatic:.2f}$')
    plt.xlabel(r'Temperature Slope ($\beta_T$ where $T \propto r^{-\beta_T}$)', fontsize=16); plt.ylabel(r'$\gamma_{\mathrm{eff}}$ (Derived from Torque)', fontsize=16)
    plt.title(r'Effective $\gamma$ in Adiabatic Limit vs. $T$ Slope', fontsize=18); plt.grid(True, which="major", linestyle='-', alpha=0.5); plt.tick_params(axis='both', which='major', labelsize=14)

    # Corrected Y-Limits
    ymin, ymax = calculate_plot_ylimits(gamma_eff_values, [], gamma_adiabatic) # Pass empty list for second dataset
    print(f"Setting Y Limits (vs TempSlope): ({ymin:.4f}, {ymax:.4f})")
    plt.ylim(ymin, ymax)

    plt.legend(fontsize=14, loc='best');
    os.makedirs(output_path, exist_ok=True); 
    if IDEFIX:
        output_file = os.path.join(output_path, "gamma_eff_vs_tempslope_IDEFIX.pdf")
    else:
        output_file = os.path.join(output_path, "gamma_eff_vs_tempslope_FARGO.pdf")
    try: plt.savefig(output_file, dpi=300, bbox_inches='tight'); print(f"\nPlot saved: {output_file}")
    except Exception as e: print(f"\nError saving plot: {e}")
    plt.close()
    try: local_directory = "/Users/mariuslehmann/Downloads/Profiles/"; username = "mariuslehmann"; scp_transfer(output_file, local_directory, username)
    except Exception as e: print(f"WARNING: SCP transfer failed: {e}")
    return output_file


# ==============================================================================
# Plotting Function 4: vs Viscosity
# ==============================================================================
def create_gamma_eff_vs_viscosity_plot(simulations_viscosity, avg_orbits=200, output_path="plots", IDEFIX=False):
    """Plots gamma_eff vs. viscosity nu for given simulations."""
    print("\n=== Starting Plot Creation: Gamma_eff vs Viscosity ===")
    results = []
    for sim in simulations_viscosity:
        r = read_simulation_torque(sim, avg_orbits, IDEFIX=IDEFIX)
        if not r: continue
        nu_val = get_parameter(sim, 'NU', IDEFIX=IDEFIX)
        if nu_val is None:
            print(f"  WARN: NU not found for {sim}.")
            continue
        try:
            nu_val = float(nu_val)
        except ValueError:
            print(f"  WARN: Bad NU value '{nu_val}' for {sim}.")
            continue
        r['nu'] = nu_val
        results.append(r)

    print(f"\nFiltered results for Viscosity plot: {len(results)} of {len(simulations_viscosity)}")
    if not results:
        print("Error: No valid sim results for viscosity plot.")
        return None

    results.sort(key=lambda x: x['nu'])
    nus = [r['nu'] for r in results]
    gammas = [r['gamma_eff'] for r in results]
    gamma_adi = results[0]['gamma']
    gamma_iso = 1.0  # Isothermal limit

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8,6))
    plt.plot(nus, gammas, marker='o', linestyle='-', linewidth=2, markersize=8, label='Simulations')
    # Horizontal lines for isothermal and adiabatic limits
    plt.axhline(gamma_iso, color='grey', linestyle='--', linewidth=1.5,
                label=r'Isothermal $\gamma=1$')
    plt.axhline(gamma_adi, color='grey', linestyle=':', linewidth=1.5,
                label=rf'Adiabatic $\gamma_{{adi}}={gamma_adi:.2f}$')

    plt.xscale('log')
    plt.xlabel(r'Viscosity $\nu$', fontsize=16)
    plt.ylabel(r'$\gamma_{{eff}}$ (Derived from Torque)', fontsize=16)
    plt.title('Effective Adiabatic Index vs. Viscosity', fontsize=18)
    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Y-limits include both gamma_iso and gamma_adi
    ymin = min(min(gammas), gamma_iso) * 0.9
    ymax = max(max(gammas), gamma_adi) * 1.1
    print(f"Setting Y Limits (vs Viscosity): ({ymin:.4f}, {ymax:.4f})")
    plt.ylim(ymin, ymax)

    plt.legend(fontsize=14, loc='best')
    os.makedirs(output_path, exist_ok=True)
    fname = f"gamma_eff_vs_viscosity_{'IDEFIX' if IDEFIX else 'FARGO'}.pdf"
    out = os.path.join(output_path, fname)
    try:
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved viscosity plot to {out}")
    except Exception as e:
        print(f"Error saving viscosity plot: {e}")
    plt.close()

    # Transfer
    try:
        scp_transfer(out, "/Users/mariuslehmann/Downloads/Profiles/", "mariuslehmann")
    except Exception as e:
        print(f"WARNING: SCP transfer failed: {e}")

    return out

# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":

    import sys
    IDEFIX = "--IDEFIX" in sys.argv

    if IDEFIX:
        simulations_2d_beta = [
            "cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1dm3_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1dm2_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1dm1_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1dm05_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1d0_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1d05_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1d1_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            #"cos_bet1d15_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1d2_gam53_ss15_q1_r0516_nu1dm11_HR150_2D_IDEFIX",
            "cos_bet1d3_gam53_ss15_q1_r0615_nu1dm11_LR_2D_IDEFIX",
            "cos_bet1d5_gam53_ss15_q1_r0615_nu1dm11_LR_2D_IDEFIX",
        ]

        #simulations_2d_beta = [
        #    "cos_bet1dm2_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1dm1_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1dm05_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1d0_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1d05_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1d1_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1d15_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX",
        #    "cos_bet1d2_gam53_ss15_q1_r0615_nu1dm11_HR150_2D_IDEFIX"
        #]

        simulations_3d_beta = []
    else:
        # Define simulations (Ensure these paths/names are correct)
        #OLD LIST
        #simulations_2d_beta = [
        #    "cos_bet1dm4_gam53_ss15_q1_r0416_z05_nu1dm11_COR_HR_2D",
        #    "cos_bet1dm3_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet4dm2_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet1dm1_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet4dm1_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet4d0_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet1d1_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet4d1_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet1d3_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    #"cos_bet1d4_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        #    "cos_bet1d4_gam53_ss15_q1_r0516_z05_nu1dm11_COR_LR_2D",
        #    #"cos_bet1d4_gam53_ss15_q1_r0416_z05_nu1dm11_COR_UHR_2D",
        #    "cos_bet1d5_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D"
        #]
#---------------------------------------------------------------------
        #HR150 INVISCID
        simulations_2d_beta = [
            "cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            #"cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm6_COR_HR1502D",
            #"cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm5_COR_HR1502D",
            "cos_bet1dm3_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1dm2_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1dm1_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1dm05_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1d0_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1d05_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1d1_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            #"cos_bet1d1_gam53_ss15_q1_r0516_nu1dm5_COR_HR1502D",
            "cos_bet1d15_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1d2_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1d3_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            "cos_bet1d4_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_2D",
            #"cos_bet1d4_gam53_ss15_q1_r0516_nu1dm5_COR_HR1502D",
        ]
        #HR150 NU=1dm7
        simulations_2d_beta = [
            "cos_bet1dm3_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1dm2_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1dm1_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1d0_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1d1_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1d2_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1d2_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1d3_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D",
            "cos_bet1d4_gam53_ss15_q1_r0516_nu1dm11_COR_HR150_nu1dm7_2D"
        ]


#---------------------------------------------------------------------
        #HR150
        simulations_3d_beta = [
            "cos_bet1dm4_gam53_p15_q1_r0516_z05_nu1dm11_COR_HR150_3D",
            "cos_bet1dm1_gam53_p15_q1_r0516_z05_nu1dm11_COR_HR150_3D"#,
            #"cos_bet1d0_gam53_p15_q1_r0516_z05_nu1dm11_COR_HR150_3D",
            #"cos_bet1d1_gam53_p15_q1_r0516_z05_nu1dm11_COR_HR150_3D",
            #"cos_bet1d4_gam53_p15_q1_r0516_z05_nu1dm11_COR_HR150_3D"
        ]
#---------------------------------------------------------------------
    simulations_sigmaslope_lr = [
        "cos_bet1d4_gam53_ss0_q1_r0615_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss05_q1_r0615_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss10_q1_r0615_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss20_q1_r0615_z05_nu1dm11_COR_LR_2D"
    ]
#---------------------------------------------------------------------
    simulations_sigmaslope_hr = [
        "cos_bet1d4_gam53_ss0_q1_r0416_z05_nu1dm11_COR_UHR_2D",
        "cos_bet1d4_gam53_ss05_q1_r0416_z05_nu1dm11_COR_UHR_2D",
        # "cos_bet1d4_gam53_ss10_q1_r0416_z05_nu1dm11_COR_UHR_2D", # Uncomment if available
        "cos_bet1d4_gam53_ss15_q1_r0416_z05_nu1dm11_COR_UHR_2D",
        "cos_bet1d4_gam53_ss20_q1_r0416_z05_nu1dm11_COR_UHR_2D"
    ]
#---------------------------------------------------------------------
    simulations_tempslope = [
        "cos_bet1d4_gam53_ss15_q0_r0615_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0615_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q2_r0615_z05_nu1dm11_COR_LR_2D"
    ]
#---------------------------------------------------------------------
    # Plot vs Viscosity
    sims_visc = [
        'cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm5_COR_HR1502D',
        'cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm6_COR_HR1502D',
        'cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm7_COR_HR1502D',
        'cos_bet1dm4_gam53_ss15_q1_r0516_nu1dm8_COR_HR1502D'
    ]
#---------------------------------------------------------------------
    avg_orbits = 200
    output_dir = "plots_output"

    # --- Create Plots ---
    # Plot 1: vs Beta
    output_file_beta = create_gamma_eff_plot(simulations_2d_beta, simulations_3d_beta, avg_orbits, output_path=output_dir, IDEFIX=IDEFIX)
    if output_file_beta: print(f"\nSuccessfully created plot 1: {output_file_beta}")
    else: print("\nPlot 1 creation failed (vs Beta).")

    if not IDEFIX:
        # ===== NEW: Alternative Torque Plot for Gamma_eff vs Beta =====
        output_file_beta_alt = create_gamma_eff_plot_alternative(simulations_2d_beta, simulations_3d_beta, avg_orbits, output_path=output_dir, IDEFIX=IDEFIX)
        if output_file_beta_alt:
            print(f"\nSuccessfully created alternative torque plot: {output_file_beta_alt}")
        else:
            print("\nAlternative torque plot creation failed (vs Beta).")


        # Plot 2: vs SigmaSlope (Resolution Comparison)
        output_file_sigmaslope = create_gamma_eff_vs_sigmaslope_plot(simulations_sigmaslope_lr, simulations_sigmaslope_hr, avg_orbits, output_path=output_dir, IDEFIX=IDEFIX)
        if output_file_sigmaslope: print(f"\nSuccessfully created plot 2: {output_file_sigmaslope}")
        else: print("\nPlot 2 creation failed (vs SigmaSlope).")

        # Plot 3: vs TempSlope
        output_file_tempslope = create_gamma_eff_vs_tempslope_plot(simulations_tempslope, avg_orbits, output_path=output_dir, IDEFIX=IDEFIX)
        if output_file_tempslope: print(f"\nSuccessfully created plot 3: {output_file_tempslope}")
        else: print("\nPlot 3 creation failed (vs TempSlope).")

        # Plot 4: vs Viscosity
        out_visc = create_gamma_eff_vs_viscosity_plot(sims_visc, avg_orbits, output_path=output_dir, IDEFIX=IDEFIX)
        if out_visc:
            print(f"Created viscosity plot: {out_visc}")
        else:
            print("Failed to create viscosity plot.")

    print("\n=== Script Finished ===")
