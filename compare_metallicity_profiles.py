import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters, reconstruct_grid

# Simplified version of read_hydro_fields to read a single snapshot and compute metallicity Z
def read_single_snapshot(path, snapshot):
    summary_file = os.path.join(path, "summary0.dat")
    print(f"READING SIMULATION PARAMETERS FROM: {summary_file}")
    
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    # If the specified snapshot does not exist, use the last available one
    dust1dens_file = os.path.join(path, f'dust1dens{snapshot}.dat')
    if not os.path.exists(dust1dens_file):
        snapshot = find_last_available_snapshot(path)
        print(f"Snapshot {snapshot} does not exist. Using last available snapshot: {snapshot}")

    # Read the dust1dens and gasdens arrays for the given snapshot
    dust1dens = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)
    gasdens = read_field_file(path, 'gasdens', snapshot, nx, ny, nz)
    
    return dust1dens, gasdens, xgrid, ygrid, zgrid, parameters

def find_last_available_snapshot(path):
    """Finds the last available snapshot by checking the files in the directory."""
    snapshots = []
    for file in os.listdir(path):
        if file.startswith('dust1dens') and file.endswith('.dat'):
            snapshot_num = int(file[len('dust1dens'):-len('.dat')])
            snapshots.append(snapshot_num)
    if snapshots:
        return max(snapshots)
    else:
        raise FileNotFoundError("No snapshot files found in the directory.")

def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    print(f"Looking for file: {file}")
    
    if os.path.exists(file):
        print(f"Reading file: {file}")
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        print(f"File not found: {file}")
        return np.zeros((ny, nx, nz))

# Compute azimuthally averaged metallicity Z by vertically integrating dust1dens and gasdens
def compute_metallicity(dust1dens, gasdens, zgrid):
    # Integrate dust1dens and gasdens vertically over zgrid
    dz = np.gradient(zgrid)
    dust_column_density = np.trapz(dust1dens, zgrid, axis=2)  # Vertical integration of dust
    gas_column_density = np.trapz(gasdens, zgrid, axis=2)     # Vertical integration of gas

    # Compute metallicity Z
    metallicity_z = dust_column_density / gas_column_density
    
    # Handle potential division by zero
    metallicity_z[np.isnan(metallicity_z)] = 0
    
    # Azimuthal averaging (average over the y-axis)
    metallicity_avg = np.mean(metallicity_z, axis=0)
    
    return metallicity_avg

def extract_stokes_number(simulation_name):
    # Extract the Stokes number from the simulation name
    if "St1dm2" in simulation_name:
        return r'$\tau=0.01$'
    elif "St1dm1" in simulation_name:
        return r'$\tau=0.1$'
    elif "St5dm2" in simulation_name:
        return r'$\tau=0.05$'
    else:
        return r'$\tau=?$'  # Default case, could add more mappings as needed

def extract_metallicity(simulation_name):
    if "Z1dm2" in simulation_name:
        return r'$Z=0.01$'
    elif "Z2dm2" in simulation_name:
        return r'$Z=0.02$'
    elif "Z3dm2" in simulation_name:
        return r'$Z=0.03$'
    elif "Z4dm2" in simulation_name:
        return r'$Z=0.04$'
    elif "Z5dm2" in simulation_name:
        return r'$Z=0.05$'
    elif "Z1dm1" in simulation_name:
        return r'$Z=0.1$'
    else:
        return r'$Z=?$'

def plot_metallicity_profiles(base_paths, sim_pairs, timestep, simulation_dirs, fixed_tau_simulations, one_panel=False, ymin=None, ymax=None):
    # Define fixed radial plot range
    rmin = 0.8
    rmax = 1.2
    # Compute the time corresponding to the snapshot number
    time_orb = 8 * timestep

    if one_panel:
        # Plot a single panel with metallicity profiles for a series of simulations
        fig, ax = plt.subplots(figsize=(8, 6))
        for sim in simulation_dirs:
            sim_base_path = None
            for base_path in base_paths:
                sim_path = os.path.join(base_path, sim)
                if sim_base_path is None and os.path.exists(sim_path):
                    sim_base_path = sim_path
                    print(f"Using base path {sim_base_path} for {sim}")
                    break

            if sim_base_path:
                try:
                    # Read the specified snapshot
                    dust1dens, gasdens, xgrid, _, zgrid, _ = read_single_snapshot(sim_base_path, timestep)
                    metallicity = compute_metallicity(dust1dens, gasdens, zgrid)
                    
                    # Extract label for the plot
                    Z_label = extract_metallicity(sim)
                    
                    # Plot the metallicity profile
                    ax.plot(xgrid, metallicity, label=Z_label)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    continue
        
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        ax.set_xlim([rmin, rmax])  # Set fixed radial plot range
        ax.set_ylabel('Z (Metallicity)')
        ax.set_xlabel('Radial Distance [r]')
        ax.text(0.02, 0.98, f'{time_orb} ORB', transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')  # Add time annotation
        if ymin is not None and ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.legend()
        plt.tight_layout()
        output_filename = 'metallicity_profiles_series.pdf'
    else:
        # Multi-panel plot for simulation pairs
        fig, axs = plt.subplots(4, 1, figsize=(8, 16), sharex=True)
        plt.subplots_adjust(hspace=0.1)

        for idx, (sim1, sim2) in enumerate(sim_pairs):
            Z1_initial, Z1_final = None, None
            Z2_initial, Z2_final = None, None
            r1, r2 = None, None
            sim1_base_path, sim2_base_path = None, None

            for base_path in base_paths:
                sim1_path = os.path.join(base_path, sim1)
                sim2_path = os.path.join(base_path, sim2)

                if sim1_base_path is None and os.path.exists(sim1_path):
                    sim1_base_path = sim1_path
                    print(f"Using base path {sim1_base_path} for {sim1}")

                if sim2_base_path is None and os.path.exists(sim2_path):
                    sim2_base_path = sim2_path
                    print(f"Using base path {sim2_base_path} for {sim2}")

                if sim1_base_path and sim2_base_path:
                    break

            if sim1_base_path and sim2_base_path:
                try:
                    # Read initial snapshot (assuming snapshot=0)
                    dust1dens1_initial, gasdens1_initial, xgrid1, _, zgrid1, _ = read_single_snapshot(sim1_base_path, 0)
                    dust1dens2_initial, gasdens2_initial, xgrid2, _, zgrid2, _ = read_single_snapshot(sim2_base_path, 0)

                    # Read the final snapshot (specified via the single timestep)
                    dust1dens1_final, gasdens1_final, _, _, _, _ = read_single_snapshot(sim1_base_path, timestep)
                    dust1dens2_final, gasdens2_final, _, _, _, _ = read_single_snapshot(sim2_base_path, timestep)

                    # Compute metallicity Z for both simulations (initial and final snapshots)
                    Z1_initial = compute_metallicity(dust1dens1_initial, gasdens1_initial, zgrid1)
                    Z1_final = compute_metallicity(dust1dens1_final, gasdens1_final, zgrid1)
                    Z2_initial = compute_metallicity(dust1dens2_initial, gasdens2_initial, zgrid2)
                    Z2_final = compute_metallicity(dust1dens2_final, gasdens2_final, zgrid2)
                    
                    # Use xgrid from the first simulation
                    r1, r2 = xgrid1, xgrid2
                except FileNotFoundError as e:
                    print(f"Error: {e}")
            else:
                print(f"Error: Could not find data for {sim1} or {sim2}")
                continue

            # Extract the Stokes numbers for the legend
            tau1 = extract_stokes_number(sim1)
            tau2 = extract_stokes_number(sim2)

            # Plot the initial and final metallicity profiles
            axs[idx].plot(r1, Z1_initial, 'k--', label=f'{tau1} Initial')
            axs[idx].plot(r1, Z1_final, 'b-', label=f'{tau1} Final (t={timestep})')
            axs[idx].plot(r2, Z2_initial, 'k--', label=f'{tau2} Initial')
            axs[idx].plot(r2, Z2_final, 'r-', label=f'{tau2} Final (t={timestep})')

            axs[idx].set_yscale('log')  # Set y-axis to logarithmic scale
            axs[idx].set_xlim([rmin, rmax])  # Set fixed radial plot range
            axs[idx].set_ylabel(f'Z (Metallicity) for pair {idx+1}')
            axs[idx].text(0.02, 0.98, f'{time_orb} ORB', transform=axs[idx].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')  # Add time annotation
            axs[idx].legend()

            # Set the vertical plot range if specified
            if ymin is not None and ymax is not None:
                axs[idx].set_ylim([ymin, ymax])

        # Fourth panel: Compare different metallicities at fixed Stokes number (\tau = 0.1)
        for sim in fixed_tau_simulations:
            sim_base_path = None
            for base_path in base_paths:
                sim_path = os.path.join(base_path, sim)
                if sim_base_path is None and os.path.exists(sim_path):
                    sim_base_path = sim_path
                    print(f"Using base path {sim_base_path} for {sim}")
                    break

            if sim_base_path:
                try:
                    dust1dens_initial, gasdens_initial, xgrid, _, zgrid, _ = read_single_snapshot(sim_base_path, 0)
                    dust1dens_final, gasdens_final, _, _, _, _ = read_single_snapshot(sim_base_path, timestep)
                    Z_initial = compute_metallicity(dust1dens_initial, gasdens_initial, zgrid)
                    Z_final = compute_metallicity(dust1dens_final, gasdens_final, zgrid)

                    r = xgrid
                    Z_label = extract_metallicity(sim)

                    axs[3].plot(r, Z_initial, 'k--', label=f'{Z_label} Initial')
                    axs[3].plot(r, Z_final, '-', label=f'{Z_label} Final (t={timestep})')
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    continue

        axs[3].set_yscale('log')  # Set y-axis to logarithmic scale
        axs[3].set_xlim([rmin, rmax])  # Set fixed radial plot range
        axs[3].set_ylabel(f'Z (Metallicity) for fixed $\\tau = 0.1$')
        axs[3].text(0.02, 0.98, f'{time_orb} ORB', transform=axs[3].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')  # Add time annotation
        axs[3].legend()

        # Set the vertical plot range for the fourth panel if specified
        if ymin is not None and ymax is not None:
            axs[3].set_ylim([ymin, ymax])

        axs[-1].set_xlabel('Radial Distance [r]')
        plt.tight_layout()

        output_filename = 'metallicity_profiles_comparison_with_fixed_tau.pdf'

    # Save the plot
    plt.savefig(output_filename)
    plt.close()

    print(f"Plot saved to {output_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Call the scp_transfer function to transfer the plot
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    print(f"Plot transferred to {local_directory}")


def main():
    parser = argparse.ArgumentParser(description="Plot azimuthally averaged metallicity profiles for simulation pairs or a series of simulations.")
    parser.add_argument('--timestep', type=int, required=True, help="Specify the timestep for all simulations.")
    parser.add_argument('--ymin', type=float, help="Specify the minimum y-axis limit for the plot.")
    parser.add_argument('--ymax', type=float, help="Specify the maximum y-axis limit for the plot.")
    parser.add_argument('--one_panel', action='store_true', help="Use this flag to create a single-panel plot for a series of simulations.")
    args = parser.parse_args()

    base_paths = [
        "/tiara/home/mlehmann/data/FARGO3D/outputs",
        "/theory/lts/mlehmann/FARGO3D/outputs"
    ]

    sim_pairs = [
        ("cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150", "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150"),
        ("cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap", "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"),
        ("cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap", "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap")
    ]

    simulation_dirs = [
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack"
    ]

    fixed_tau_simulations = [
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]

    # Call the function with the appropriate arguments
    plot_metallicity_profiles(
        base_paths,
        sim_pairs,
        args.timestep,
        simulation_dirs,
        fixed_tau_simulations,
        one_panel=args.one_panel,
        ymin=args.ymin,
        ymax=args.ymax
    )

if __name__ == "__main__":
    main()
