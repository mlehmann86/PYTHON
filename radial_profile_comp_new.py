import os
import numpy as np
import re  # Import the re module
import argparse
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from data_reader import read_parameters, determine_nt, reconstruct_grid
from scipy.ndimage import uniform_filter1d


def load_and_compute_radial_profiles(simulations, base_paths, output_path, filename_suffix, N=2, font_size=18, line_thickness=2,
                                     y_range_dP_dr=None, y_range_rho=None, y_range_N2=None):
    """
    Compute and plot radial profiles of various quantities for multiple simulations across different base paths.
    
    Parameters:
    - simulations: List of simulation subdirectory names.
    - base_paths: List of base paths where simulations might be stored.
    - output_path: Directory to save the plot.
    - filename_suffix: Suffix to append to the output filename.
    - N: Number of snapshots to average over.
    - font_size: Font size for the plot.
    - line_thickness: Line thickness for plot lines.
    - y_range_dP_dr: Tuple specifying the y-axis range for dP/dr (e.g., (min, max)).
    - y_range_rho: Tuple specifying the y-axis range for rho (e.g., (min, max)).
    - y_range_N2: Tuple specifying the y-axis range for N² (e.g., (min, max)).
    """

    # Set consistent font sizes for all elements
    label_fontsize = 18
    tick_fontsize = 18
    legend_fontsize = 14

    # Font and axis settings
    plt.rcParams.update({
        'font.size': tick_fontsize,
        'lines.linewidth': line_thickness,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2
    })

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    #fig, axs = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    colors = ['black', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


    smoothing_window = 5
    start_radius = 0.8

    # Flag to track if the initial profile has been plotted
    initial_plotted = False

    # Loop through simulations to compute the radial profiles
    for idx, sim in enumerate(simulations):
        found_file = False
        metallicity = None

        for base_path in base_paths:
            sim_path = os.path.join(base_path, sim)
            summary_file = os.path.join(sim_path, "summary0.dat")
            if os.path.exists(summary_file):
                found_file = True
                parameters = read_parameters(summary_file)
                metallicity = float(parameters.get('METALLICITY', 0.0))
                aspectratio = parameters['ASPECTRATIO']
                print(f"Simulation: {sim}, Extracted Metallicity: Z={metallicity}")
                break

        if not found_file or metallicity is None:
            print(f"Error: summary0.dat not found for {sim} in any base path!")
            continue

        # Load the initial snapshot (snapshot 0) for scaling and plotting
        data_initial = load_simulation_data(sim_path, 0)
        rho_initial = data_initial['gasdens']
        P_initial = data_initial['gasenergy']*(parameters['GAMMA'] - 1)
        xgrid = data_initial['xgrid']

        # Scale quantities
        P_scale = rho_initial[0, rho_initial.shape[1] // 2, rho_initial.shape[2] // 2] * aspectratio**2
        rho_scale = rho_initial[0, rho_initial.shape[1] // 2, rho_initial.shape[2] // 2]
        N2_scale = 1.0

        # Compute initial quantities and apply scaling
        dP_dr_initial_full = np.gradient(P_initial, xgrid, axis=1)
        S_initial = np.log(P_initial / rho_initial**parameters['GAMMA'])
        dS_dr_initial_full = np.gradient(S_initial, xgrid, axis=1)
        N2_initial_full = -(1 / parameters['GAMMA']) * (1 / rho_initial) * dP_dr_initial_full * dS_dr_initial_full

        # Apply scaling
        P_initial_scaled = np.mean(P_initial, axis=(0, 2)) / P_scale
        dP_dr_initial_scaled = np.mean(dP_dr_initial_full, axis=(0, 2)) / P_scale
        rho_initial_scaled = np.mean(rho_initial, axis=(0, 2)) / rho_scale
        N2_initial_scaled = np.mean(N2_initial_full, axis=(0, 2)) / N2_scale

        # Use find_last_snapshots to get the last N snapshots
        nt = determine_nt(sim_path)
        snapshots = find_last_snapshots(sim_path, nt - 1, N)
        print(f"Selected snapshots for {sim}: {snapshots}")  # Debugging output

        if not snapshots:
            print(f"No snapshots found for {sim} in {sim_path}")
            continue

        # Initialize accumulators for averaging
        dP_dr_final_sum = np.zeros_like(dP_dr_initial_scaled)
        rho_final_sum = np.zeros_like(rho_initial_scaled)
        N2_final_sum = np.zeros_like(N2_initial_scaled)

        # Loop over the selected snapshots
        for t in snapshots:
            data = load_simulation_data(sim_path, t)
            rho = data['gasdens']
            P = data['gasenergy']*(parameters['GAMMA'] - 1)
            dP_dr_full = np.gradient(P, xgrid, axis=1)
            S = np.log(P / rho**parameters['GAMMA'])
            dS_dr_full = np.gradient(S, xgrid, axis=1)
            N2_full = -(1 / parameters['GAMMA']) * (1 / rho) * dP_dr_full * dS_dr_full

            # Apply scaling and accumulate
            dP_dr_final_sum += np.mean(dP_dr_full, axis=(0, 2)) / P_scale
            rho_final_sum += np.mean(rho, axis=(0, 2)) / rho_scale
            N2_final_sum += np.mean(N2_full, axis=(0, 2)) / N2_scale

        # Compute the final time-averaged profiles
        dP_dr_final_scaled = dP_dr_final_sum / N
        rho_final_scaled = rho_final_sum / N
        N2_final_scaled = N2_final_sum / N

        if metallicity == 0.01:
            print(f"First 10 entries for (1 / rho) for Z=0.01: {(1 / rho).flatten()[:10]}")
            print(f"First 10 entries for dP_dr_full for Z=0.01: {dP_dr_full.flatten()[:10]}")
            print(f"First 10 entries for dS_dr_full for Z=0.01: {dS_dr_full.flatten()[:10]}")

        # Plot the time-averaged profiles
        label = f"Z={metallicity:.4f}".rstrip('0').rstrip('.')
        axs[0].plot(xgrid, dP_dr_final_scaled, '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)
        axs[1].plot(xgrid, rho_final_scaled, '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)
        axs[2].plot(xgrid, uniform_filter1d(N2_final_scaled, smoothing_window), '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)

        # Overplot the initial profiles once
        if not initial_plotted:
            axs[0].plot(xgrid, dP_dr_initial_scaled, 'k--', label='Initial', linewidth=line_thickness)
            axs[1].plot(xgrid, rho_initial_scaled, 'k--', label='Initial', linewidth=line_thickness)
            axs[2].plot(xgrid, uniform_filter1d(N2_initial_scaled, smoothing_window), 'k--', label='Initial', linewidth=line_thickness)
            initial_plotted = True

    # Set axis labels and limits
    axs[0].set_ylabel(r'$\langle \frac{dP}{dr} \rangle$', fontsize=label_fontsize)
    axs[1].set_ylabel(r'$\langle \rho_g \rangle$', fontsize=label_fontsize)
    axs[2].set_ylabel(r'$\langle N^2 \rangle$', fontsize=label_fontsize)
    axs[1].set_xlabel(r'$r/r_{0}$', fontsize=label_fontsize)

    axs[0].set_xlim([0.8, 1.2])

    # Apply y-axis limits if provided
    if y_range_dP_dr:
        axs[0].set_ylim(y_range_dP_dr)
    if y_range_rho:
        axs[1].set_ylim(y_range_rho)
    if y_range_N2:
        axs[2].set_ylim(y_range_N2)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True)

    axs[0].legend(loc='best', fontsize=legend_fontsize)

    pdf_filename = f"radial_profiles_comparison_{filename_suffix}.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath, bbox_inches="tight")
    plt.close()
    print(f"Radial profile plot saved to {output_filepath}")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")



def find_last_snapshots(path, reference_snapshot, num_snapshots):
    snapshots = []
    pattern = re.compile(r'gasdens(\d+).dat')  # Adjusted to look for gasdens files

    # Check for matches without listing all files
    for file in os.listdir(path):
        match = pattern.match(file)
        if match:
            snapshot_num = int(match.group(1))
            if snapshot_num <= reference_snapshot:
                snapshots.append(snapshot_num)
    
    snapshots.sort(reverse=True)
    return snapshots[:num_snapshots]

# Function to find the last available snapshot, ignoring files with "2d" in their names
def find_last_available_snapshot(path, requested_snapshot):
    snapshots = []
    pattern = re.compile(r'gasdens(\d+).dat')  # Adjusted to look for gasdens files

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

def read_single_snapshot(path, snapshot):
    print(f"Reading parameters from: {os.path.join(path, 'summary0.dat')}")  # Debugging output
    parameters = read_parameters(os.path.join(path, "summary0.dat"))
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
    
    # Find the last available snapshot if the specified one is missing
    print(f"Looking for snapshot files in directory: {path}")  # Debugging output
    snapshot = find_last_available_snapshot(path, snapshot)
    if snapshot is None:
        raise FileNotFoundError("No snapshot files found in the directory.")
    print(f"Using snapshot: {snapshot} in directory: {path}")  # Debugging output
    
    # Read the gas density and gas energy files
    gasdens = read_field_file(path, 'gasdens', snapshot, nx, ny, nz)
    gasenergy = read_field_file(path, 'gasenergy', snapshot, nx, ny, nz)
    
    return gasdens, gasenergy, xgrid, ygrid, zgrid
def load_simulation_data(sim_path, snapshot_index):
    gasdens, gasenergy, xgrid, ygrid, zgrid = read_single_snapshot(sim_path, snapshot_index)
    return {
        'xgrid': xgrid,
        'gasdens': gasdens,
        'gasenergy': gasenergy
    }

def main():
    parser = argparse.ArgumentParser(description="Compute and plot radial profiles for different simulations.")
    parser.add_argument('--raettig', action='store_true', help="Use Raettig simulations.")
    parser.add_argument('--mitigation', action='store_true', help="Use Mitigation simulations.")
    parser.add_argument('--other', action='store_true', help="Use other simulations.")
    parser.add_argument('--N', type=int, default=10, help="Number of snapshots to average over.")

    args = parser.parse_args()

    base_paths = [
        "/tiara/home/mlehmann/data/FARGO3D/outputs",
        "/theory/lts/mlehmann/FARGO3D/outputs"
    ]

    output_path = os.getcwd()

    if args.raettig:
        simulations = [
            "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
            "cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack"
        ]
        filename_suffix = "raettig"
    elif args.mitigation:
        simulations = [
            "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
            "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        ]

       # simulations = [
       #     "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
       #     "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
       # ]
        filename_suffix = "mitigation"
    elif args.other:
        simulations = [
            # Add other simulations here if needed
        ]
        filename_suffix = "other"
    else:
        print("Please select one of the simulation sets with --raettig, --mitigation, or --other.")
        return

    load_and_compute_radial_profiles(simulations, base_paths, output_path, filename_suffix, args.N,
                                 y_range_dP_dr=(-10, 0), y_range_rho=(0.5, 2), y_range_N2=(-0.1, 0.1))

if __name__ == "__main__":
    main()
