import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# --- Assumed to be in PYTHONPATH ---
try:
    from data_storage import determine_base_path, scp_transfer
    from data_reader import read_parameters
    from planet_data import read_alternative_torque, compute_theoretical_torques_PK11, extract_planet_mass_and_migration
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure data_storage.py, data_reader.py, and planet_data.py are in your PYTHONPATH.")
    exit()

def calculate_cumulative_average(data):
    """Calculates the cumulative average of a 1D array."""
    return np.cumsum(data) / np.arange(1, len(data) + 1)

def load_and_process_torque(simulation_name, idefix=False):
    """
    Loads and processes torque data for a given simulation.
    Now calculates GAM0 directly for efficiency and correctness.
    """
    print(f"\n--- Processing simulation: {simulation_name} ---")
    
    # Determine paths and read basic parameters
    base_path = determine_base_path(simulation_name, IDEFIX=idefix)
    summary_file = os.path.join(base_path, "summary0.dat")
    if idefix:
        summary_file = os.path.join(base_path, "idefix.0.log")
        
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found for {simulation_name} at {summary_file}")

    parameters = read_parameters(summary_file, IDEFIX=idefix)
    print("✅ Parameters read.")
    
    if idefix:
        qp = float(parameters.get("planettoprimary", 0.0))
        h = float(parameters.get("h0", 0.05))
        SIGMA0 = float(parameters.get("sigma0", 1.0))
    else:
        qp, _ = extract_planet_mass_and_migration(summary_file)
        h = float(parameters.get("ASPECTRATIO", 0.05))
        SIGMA0 = float(parameters.get("SIGMA0", 1.0))

    # --- BUG FIX: Calculate GAM0 directly instead of making a confusing extra call ---
    # The normalization factor Gamma_0 = (q_p / h)^2 * Sigma_0
    GAM0 = (qp / h)**2 * SIGMA0
    print(f"✅ Calculated GAM0 = {GAM0:.3e} directly.")

    # Read torque data from tqwk0.dat
    tqwk_file = os.path.join(base_path, "tqwk0.dat")
    if not os.path.exists(tqwk_file):
        raise FileNotFoundError(f"Torque file not found: {tqwk_file}")
        
    time_raw, torque_raw, _ = read_alternative_torque(tqwk_file, IDEFIX=idefix)
    print(f"✅ Torque data loaded from tqwk0.dat.")
    
    # Convert time to orbits (FARGO default output is in code units, 2pi = 1 orbit)
    time_orbits = time_raw / (2.0 * np.pi)

    # Normalize the torque from the simulation
    # The torque in tqwk0.dat is the torque ON the planet by the disk.
    # We multiply by qp to match the normalization convention of GAM0.
    torque_normalized = (torque_raw * qp) / GAM0
    
    # Calculate the cumulative average of the normalized torque
    cumulative_avg_torque = calculate_cumulative_average(torque_normalized)

    return {
        "time_orbits": time_orbits,
        "torque_normalized": torque_normalized,
        "cumulative_avg_torque": cumulative_avg_torque,
        "params": parameters,
        "qp": qp,
        "GAM0": GAM0,
        "summary_file": summary_file
    }


def main(sim_3d_name, sim_2d_name):
    """
    Main function to generate the torque comparison plot.
    """
    # --- Load and Process Data ---
    data_3d = load_and_process_torque(sim_3d_name, idefix=False)
    data_2d = load_and_process_torque(sim_2d_name, idefix=False)

    # --- Calculate Theoretical Torques for the 3D Simulation ---
    # This is now the ONLY call to the theory function.
    avg_start_orbit = data_3d['time_orbits'][0]
    avg_end_orbit = data_3d['time_orbits'][-1]
    
    print(f"\nCalculating theoretical torques for {sim_3d_name}...")
    print(f"Averaging turbulent viscosity over orbits: {avg_start_orbit:.1f} to {avg_end_orbit:.1f}")

    total_torque_pk11, lindblad_torque_pk11, _, gamma_eff, nu_used, nu_type = compute_theoretical_torques_PK11(
        data_3d['params'], 
        data_3d['qp'], 
        sim_3d_name, 
        IDEFIX=False,
        summary_file=data_3d['summary_file'],
        avg_start_orbit=avg_start_orbit,
        avg_end_orbit=avg_end_orbit,
        nu_threshold=1e-8
    )
    
    total_torque_theory = total_torque_pk11 * gamma_eff
    lindblad_torque_theory = lindblad_torque_pk11 * gamma_eff

    print(f"\n--- Final Theoretical Values ---")
    print(f"Effective gamma: {gamma_eff:.4f}")
    print(f"Viscosity type used for theory: {nu_type}")
    print(f"Viscosity value used: {nu_used:.2e}")
    print(f"Theoretical Total Torque (scaled): {total_torque_theory:.4f}")
    print(f"Theoretical Lindblad Torque (scaled): {lindblad_torque_theory:.4f}")
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        data_3d['time_orbits'], 
        data_3d['cumulative_avg_torque'] * gamma_eff, 
        label=f'3D Turbulent (COS)',
        color='crimson',
        linewidth=2.5
    )

    ax.plot(
        data_2d['time_orbits'], 
        data_2d['cumulative_avg_torque'] * gamma_eff, 
        label=f'2D Inviscid',
        color='royalblue',
        linewidth=2.5,
        linestyle='-'
    )

    ax.axhline(
        total_torque_theory, 
        color='black', 
        linestyle='--', 
        linewidth=2,
        label=fr'PK11 Total Torque ($\nu_{{turb}}={nu_used:.1e}$)'
    )

    ax.axhline(
        lindblad_torque_theory, 
        color='gray', 
        linestyle=':', 
        linewidth=2,
        label='PK11 Lindblad Torque'
    )
    
    # --- Final Touches ---
    ax.set_xlabel('Time [Orbits]', fontsize=14)
    ax.set_ylabel(r'Cumulative Avg. Torque $\gamma_{\mathrm{eff}} \langle \Gamma \rangle / \Gamma_0$', fontsize=14)
    ax.set_title('Effect of COS Turbulence on Planet Torque', fontsize=16, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    #ax.set_xlim(0, max(data_3d['time_orbits'][-1], data_2d['time_orbits'][-1]))
    ax.set_xlim(0, 1000.)
    ax.set_ylim(-10, 5)
    
    ax.axhline(0, color='black', linewidth=0.75, alpha=0.8)
    plt.tight_layout()
    
    output_filename = 'cos_torque_comparison.pdf'
    plt.savefig(output_filename)
    print(f"\n✅ Plot successfully saved to: {output_filename}")

    # --- SCP Transfer ---
    local_directory = "/Users/mariuslehmann/Downloads/Profiles/"
    print(f"Attempting to transfer {output_filename} to {local_directory} on your laptop...")
    try:
        scp_transfer(output_filename, local_directory, "mariuslehmann")
    except Exception as e:
        print(f"❌ SCP transfer failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a comparison of cumulative average torques from a 3D turbulent and a 2D inviscid simulation."
    )
    parser.add_argument(
        "sim_3d", 
        type=str,
        nargs='?', 
        default=None,
        help="Name of the 3D turbulent simulation."
    )
    parser.add_argument(
        "sim_2d", 
        type=str,
        nargs='?',
        default=None,
        help="Name of the 2D inviscid simulation."
    )
    args = parser.parse_args()

    if not args.sim_3d or not args.sim_2d:
        print("Error: Missing simulation names.\n")
        parser.print_help()
        sys.exit(1)
    
    main(args.sim_3d, args.sim_2d)

