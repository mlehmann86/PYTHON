import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_reader import read_parameters, reconstruct_grid
from data_storage import scp_transfer
import re

# Function to find the last snapshots up to a specified reference snapshot
def find_last_snapshots(path, reference_snapshot, num_snapshots):
    snapshots = []
    for file in os.listdir(path):
        if file.startswith('dust1dens') and file.endswith('.dat'):
            try:
                snapshot_num = int(file[len('dust1dens'):-len('.dat')])
                if snapshot_num <= reference_snapshot:
                    snapshots.append(snapshot_num)
            except ValueError:
                continue
    snapshots.sort(reverse=True)
    return snapshots[:num_snapshots]

# Helper function to extract the Stokes number from the simulation name
def extract_stokes_number(simulation_name):
    if "St1dm2" in simulation_name:
        return r'$\tau=0.01$'
    elif "St1dm1" in simulation_name:
        return r'$\tau=0.1$'
    elif "St5dm2" in simulation_name:
        return r'$\tau=0.05$'
    elif "St1" in simulation_name:
        return r'$\tau=1.0$'
    else:
        return r'$\tau=?$'  # Default case if Stokes number cannot be identified

# Function to find the last available snapshot, ignoring files with "2d" in their names
def find_last_available_snapshot(path, requested_snapshot):
    snapshots = []
    pattern = re.compile(r'dust1dens(\d+).dat')

    # Extract numeric parts of snapshot files, excluding those with "2d" in the name
    for f in os.listdir(path):
        if '2d' in f:
            continue  # Skip files containing "2d"
        match = pattern.match(f)
        if match:
            snapshot_num = int(match.group(1))
            snapshots.append(snapshot_num)
    
    # Check if the requested snapshot exists
    if requested_snapshot in snapshots:
        return requested_snapshot
    elif snapshots:
        return max(snapshots)
    else:
        return None

# Function to read field files (e.g., dust1dens or gasdens)
def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        print(f"Reading file: {file}")
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    return np.zeros((ny, nx, nz))



# Function to read a single snapshot and extract dust and gas densities
def read_single_snapshot(path, snapshot):
    parameters = read_parameters(os.path.join(path, "summary0.dat"))
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
    
    # Find the last available snapshot if specified one is missing
    snapshot = find_last_available_snapshot(path, snapshot)
    if snapshot is None:
        raise FileNotFoundError("No snapshot files found in the directory.")
    print(f"Using snapshot: {snapshot}")

    # Read the dust and gas densities
    dust1dens = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)
    gasdens = read_field_file(path, 'gasdens', snapshot, nx, ny, nz)
    
    return dust1dens, gasdens, xgrid, ygrid, zgrid



# Helper function to extract metallicity from the simulation name
def extract_metallicity(simulation_name):
    if "Z1dm4" in simulation_name:
        return r'$Z=0.0001$'
    elif "Z1dm3" in simulation_name:
        return r'$Z=0.001$'
    elif "Z1dm2" in simulation_name:
        return r'$Z=0.01$'
    else:
        return r'$Z=?$'

# Function to compute the mass fraction for different epsilon thresholds
def compute_mass_fraction(dust1dens, gasdens, xgrid, ygrid, zgrid, epsilon_values):
    epsilon = dust1dens / gasdens  # Compute epsilon (dust-to-gas ratio)
    
    # Calculate the radial grid spacing (dr), azimuthal spacing (dphi), and vertical spacing (dz)
    dr = np.gradient(xgrid)
    dphi = np.gradient(ygrid)
    dz = np.gradient(zgrid)

    # Create a 3D grid of volume elements in cylindrical coordinates: r * dr * dphi * dz
    r_grid, _, _ = np.meshgrid(ygrid, xgrid, zgrid, indexing='ij')
    volume_elements = r_grid[:, :, :] * dr[None, :, None] * dphi[:, None, None] * dz[None, None, :]

    # Compute the total mass using volume elements (shape should match dust1dens)
    total_mass = np.sum(dust1dens * volume_elements)
    if total_mass <= 0:
        print("Warning: Total mass is zero or negative, which may indicate an issue with the input data.")
        return [0] * len(epsilon_values)

    # Initialize an array to store the fractions for each epsilon threshold
    mass_fractions = []

    for X in epsilon_values:
        # Find the grid points where epsilon > X
        mask = epsilon > X
        
        # Compute the mass above the threshold using the volume elements
        mass_above_threshold = np.sum(dust1dens[mask] * volume_elements[mask])

        # Calculate the fraction of the total mass
        mass_fraction = mass_above_threshold / total_mass
        mass_fractions.append(mass_fraction)

    return mass_fractions

# Function to create the plot and save it as a PDF
def plot_mass_fractions(epsilon_values, mass_fractions_list, metallicities, filename_suffix, fontsize=12):
    plt.figure(figsize=(6, 6))

    # Loop through each set of mass fractions and metallicity labels to plot them
    for mass_fractions, metallicity in zip(mass_fractions_list, metallicities):
        plt.plot(epsilon_values, mass_fractions, label=metallicity)

    # Set logarithmic scales for both axes
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\epsilon$ (dust-to-gas ratio)", fontsize=fontsize)
    plt.ylabel(r"Fraction of $M_d$ with $\epsilon > X$", fontsize=fontsize)

    # Set tick parameters with the specified fontsize
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Fix the vertical plot range from 10^-2 to 1
    plt.ylim(1e-2, 1)

    # Fix the horizontal plot range from 10^-5 to 10^3
    plt.xlim(1e-5, 1e3)

    # Add vertical dashed lines at specified epsilon values
    for x_val in [1e-5, 1e-4, 1e-2, 1, 1e2]:
        plt.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Add horizontal dashed lines at specified fraction values
    for y_val in [1e-2, 1e-1]:
        plt.axhline(y=y_val, color='gray', linestyle='-.', linewidth=0.8, alpha=0.7)

    # Add legend with the specified fontsize
    plt.legend(title="Metallicity", fontsize=fontsize, title_fontsize=fontsize)

    # Apply the same fontsize to the tight layout
    plt.tight_layout()
    
    # Save the plot as a PDF file
    output_filename = f"mass_fraction_{filename_suffix}.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved as {output_filename}")

    # Transfer the file to the local machine using SCP
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    print(f"Plot transferred to {local_directory}")


# Main function to run the script
def main():
    parser = argparse.ArgumentParser(description="Plot dust mass fraction for different epsilon thresholds.")
    parser.add_argument('--raettig', action='store_true', help="Use Raettig simulations.")
    parser.add_argument('--mitigation', action='store_true', help="Use Mitigation simulations.")
    parser.add_argument('snapshot', type=int, help="Reference snapshot number to analyze up to.")
    parser.add_argument('--num_snapshots', type=int, default=1, help="Number of snapshots to average, ending at the reference snapshot.")
    args = parser.parse_args()

    base_paths = [
        "/tiara/home/mlehmann/data/FARGO3D/outputs",
        "/theory/lts/mlehmann/FARGO3D/outputs"
    ]

    # Define the simulation sets
    if args.raettig:
        simulations = [
            "cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150"
        ]
        filename_suffix = "raettig"
    elif args.mitigation:
        simulations = [
            "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
        ]
        filename_suffix = "mitigation"
    else:
        print("Please select one of the simulation sets with --raettig or --mitigation.")
        return

    # Define the epsilon values (logarithmically spaced) to include values down to 10^-5
    epsilon_values = np.logspace(-5, 3, 100)

    # Prepare lists to store results for plotting
    mass_fractions_list = []
    metallicities = []

    # Loop through each simulation, compute mass fractions, and store for plotting
    for sim in simulations:
        sim_path = None
        for base_path in base_paths:
            sim_path = os.path.join(base_path, sim)
            if os.path.exists(sim_path):
                break
        if sim_path is None or not os.path.exists(sim_path):
            print(f"Simulation {sim} not found in any base path.")
            continue

        try:
            # Find the last `num_snapshots` up to the specified reference snapshot
            snapshots = find_last_snapshots(sim_path, args.snapshot, args.num_snapshots)
            if not snapshots:
                print(f"No snapshots found for simulation {sim} up to snapshot {args.snapshot}.")
                continue
            
            # Compute time-averaged mass fractions
            mass_fractions_accum = np.zeros(len(epsilon_values))
            for snapshot in snapshots:
                # Read dust and gas densities for the snapshot
                dust1dens, gasdens, xgrid, ygrid, zgrid = read_single_snapshot(sim_path, snapshot)
                
                # Compute mass fractions for the epsilon values
                mass_fractions = compute_mass_fraction(dust1dens, gasdens, xgrid, ygrid, zgrid, epsilon_values)
                mass_fractions_accum += np.array(mass_fractions)
            
            # Average the accumulated mass fractions
            averaged_mass_fractions = mass_fractions_accum / len(snapshots)
            mass_fractions_list.append(averaged_mass_fractions)
            
            # Extract the metallicity for the legend
            metallicity = extract_metallicity(sim)
            metallicities.append(metallicity)
        
        except FileNotFoundError as e:
            print(f"Error processing {sim}: {e}")
            continue

    # Create the plot for all simulations and save it as a PDF
    plot_mass_fractions(epsilon_values, mass_fractions_list, metallicities, filename_suffix, fontsize=12)

if __name__ == "__main__":
    main()
