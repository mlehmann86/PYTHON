import os
import numpy as np
import re  # Import the re module
import argparse
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from data_reader import read_parameters, determine_nt, reconstruct_grid
from scipy.ndimage import uniform_filter1d



def load_and_compute_radial_profiles(simulations, base_paths, output_path, filename_suffix, font_size=18, line_thickness=2,
                                     y_range_dP_dr=None, y_range_rho=None, y_range_Z=None, horizontal_line=None):
    """
    Compute and plot radial profiles of various quantities for multiple simulations up to the Roche density threshold.
    
    Parameters:
    - simulations: List of simulation subdirectory names.
    - base_paths: List of base paths where simulations might be stored.
    - output_path: Directory to save the plot.
    - filename_suffix: Suffix to append to the output filename.
    - font_size: Font size for the plot.
    - line_thickness: Line thickness for plot lines.
    - y_range_dP_dr: Tuple specifying the y-axis range for dP/dr (e.g., (min, max)).
    - y_range_rho: Tuple specifying the y-axis range for rho (e.g., (min, max)).
    - y_range_Z: Tuple specifying the y-axis range for Z (e.g., (min, max)).
    """
    # Set consistent font sizes for all elements
    label_fontsize = 18
    tick_fontsize = 18
    legend_fontsize = 14

    plt.rcParams.update({
        'font.size': tick_fontsize,
        'lines.linewidth': line_thickness,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2
    })

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    colors = ['black', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    initial_plotted = False

    for idx, sim in enumerate(simulations):
        found_file = False
        metallicity = None
        parameters = None

        # Locate the simulation directory and extract parameters
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

        if not found_file or parameters is None:
            print(f"Error: summary0.dat not found for {sim}. Skipping.")
            continue

        # Locate the .npz file for the simulation
        npz_data = None
        for base_path in base_paths:
            npz_file = os.path.join(base_path, sim, f"{os.path.basename(sim)}_quantities.npz")
            if os.path.exists(npz_file):
                npz_data = np.load(npz_file)
                print(f"Loaded .npz file for {sim}: {npz_file}")
                break

        if npz_data is None:
            print(f"Error: .npz file not found for {sim}. Skipping.")
            continue

        # Load roche_times, time_array, and max_epsilon from the .npz file
        roche_times = npz_data.get('roche_times', None)
        time_array = npz_data.get('time', None)
        max_epsilon = npz_data.get('max_epsilon', None)

        if roche_times is None or time_array is None or max_epsilon is None:
            print(f"Error: Missing 'roche_times', 'time', or 'max_epsilon' in {sim}. Skipping.")
            continue

        # Check if the simulation is dust-free
        is_dust_free = "nodust" in sim

        # Determine the cutoff snapshots for averaging
        if is_dust_free:
            # Dust-free case: average over the last 100 orbits
            cutoff_snapshot = len(time_array) - 1
            snapshots = range(max(0, cutoff_snapshot - 100), cutoff_snapshot + 1)
            print(f"Dust-free simulation detected. Averaging over the last 100 orbits for {sim}: Snapshots {list(snapshots)}")
        else:
            # Standard case: use Roche density or max epsilon criteria
            if roche_times.size > 0:
                first_roche_time = roche_times[0]  # The first time Roche density is exceeded
                cutoff_snapshot = np.searchsorted(time_array, first_roche_time, side="left") - 1
                print(f"Roche density exceeded. Cutoff snapshot for {sim}: {cutoff_snapshot} (time: {time_array[cutoff_snapshot]})")
            else:
                print(f"Roche density never exceeded in {sim}. Selecting cutoff snapshot based on max_epsilon.")
                # Find the snapshot where max_epsilon is the largest
                max_epsilon_idx = np.argmax(max_epsilon)  # Index of the largest value of max_epsilon
                cutoff_snapshot = max_epsilon_idx
                print(f"Max epsilon reached at snapshot {cutoff_snapshot} (time: {time_array[cutoff_snapshot]}).")

            # Select snapshots for averaging
            snapshots = range(max(0, cutoff_snapshot - 15), cutoff_snapshot + 1)
            print(f"Selected snapshots for {sim}: {list(snapshots)}")  # Debugging output



        # Initial snapshot (snapshot 0) for scaling
        sim_path = os.path.join(base_path, sim)
        data_initial = load_simulation_data(sim_path, 0)
        rho_initial = data_initial['gasdens']
        P_initial = data_initial['gasenergy'] * (parameters['GAMMA'] - 1)
        xgrid = data_initial['xgrid']

        # Scaling quantities
        P_scale = rho_initial[0, rho_initial.shape[1] // 2, rho_initial.shape[2] // 2] * parameters['ASPECTRATIO'] ** 2

        # Initialize accumulators for averaging
        dP_dr_final_sum = np.zeros_like(np.mean(P_initial, axis=(0, 2)))
        rho_final_sum = np.zeros_like(np.mean(rho_initial, axis=(0, 2)))

        N=len(snapshots)


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

        # Metallicity (third panel) specific initialization
        dust1dens_initial = data_initial.get('dust1dens', None)
        if dust1dens_initial is not None:
            metallicity_initial_scaled = (
                np.mean(dust1dens_initial, axis=(0, 2)) / np.mean(rho_initial, axis=(0, 2))
            )
        else:
            print("Warning: 'dust1dens' not found. Skipping metallicity computation for initial profiles.")
            metallicity_initial_scaled = np.zeros_like(rho_initial_scaled)



        if not snapshots:
            print(f"No snapshots found for {sim} in {sim_path}")
            continue

        # Initialize accumulators for averaging
        dP_dr_final_sum = np.zeros_like(dP_dr_initial_scaled)
        rho_final_sum = np.zeros_like(rho_initial_scaled)
        metallicity_final_sum = np.zeros_like(metallicity_initial_scaled)

        # Loop over the selected snapshots
        for t in snapshots:
            data = load_simulation_data(sim_path, t)
            rho = data['gasdens']
            P = data['gasenergy']*(parameters['GAMMA'] - 1)
            dust1dens = data.get('dust1dens', None)

            dP_dr_full = np.gradient(P, xgrid, axis=1)
            dP_dr_final_sum += np.mean(dP_dr_full, axis=(0, 2)) / P_scale
            rho_final_sum += np.mean(rho, axis=(0, 2)) / rho_scale

            # Compute metallicity if dust1dens is available
            if dust1dens is not None:
                metallicity_final_sum += np.mean(dust1dens, axis=(0, 2)) / np.mean(rho, axis=(0, 2))
            else:
                print(f"Warning: 'dust1dens' not found for snapshot {t}. Skipping metallicity computation.")

        # Compute the final time-averaged profiles
        dP_dr_final_scaled = dP_dr_final_sum / N
        rho_final_scaled = rho_final_sum / N
        metallicity_final_scaled = metallicity_final_sum / N if metallicity_final_sum.any() else None

        # Plot the time-averaged profiles
        label = f"Z={metallicity:.4f}".rstrip('0').rstrip('.')
        axs[0].plot(xgrid, dP_dr_final_scaled, '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)
        axs[1].plot(xgrid, rho_final_scaled, '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)
        if metallicity_final_scaled is not None:
            axs[2].plot(xgrid, metallicity_final_scaled, '-', color=colors[idx],
                        label=f'{label}', linewidth=line_thickness)

        # Overplot the initial profiles once
        if not initial_plotted:
            axs[0].plot(xgrid, dP_dr_initial_scaled, 'k--', label='Initial', linewidth=line_thickness)
            axs[1].plot(xgrid, rho_initial_scaled, 'k--', label='Initial', linewidth=line_thickness)
            #axs[2].plot(xgrid, metallicity_initial_scaled, 'k--', label='Initial',
            #            linewidth=line_thickness, color=colors[idx])
            initial_plotted = True

        # Set y-axis range for the third panel (Metallicity)
        if y_range_Z:
            axs[2].set_ylim(y_range_Z)
        axs[2].set_yscale('log')  # Log scale for the third panel

        # Overplot the horizontal dotted line for Z_crit and add a single legend
        if horizontal_line is not None:
            axs[2].axhline(horizontal_line, color='black', linestyle=':', linewidth=1.5, label=r"$Z_{\mathrm{crit}}$ (Lim et al. 2024)")

        # Add legends to all panels
        axs[0].legend(loc='best', fontsize=legend_fontsize)
        axs[2].legend(loc='best', fontsize=legend_fontsize, handles=[plt.Line2D([], [], color='black', linestyle=':', linewidth=1.5, label=r"$Z_{\mathrm{crit}}$ (Lim et al. 2024)")])


    # Set axis labels and limits
    axs[0].set_ylabel(r'$\langle \frac{dP}{dr} \rangle$', fontsize=label_fontsize)
    axs[1].set_ylabel(r'$\langle \rho_g \rangle$', fontsize=label_fontsize)
    axs[2].set_ylabel(r'$\langle Z \rangle$', fontsize=label_fontsize)

    # Suppress x-axis label for the second panel
    axs[1].xaxis.set_tick_params(labelbottom=False)  # Turn off x-axis labels for the second panel

    # Activate x-axis label for the third panel
    axs[2].set_xlabel(r'$r/r_{0}$', fontsize=label_fontsize)

    axs[0].set_xlim([0.8, 1.2])
    #axs[2].set_yscale('log')  # Set the y-axis of the third panel to logarithmic
    #axs[2].set_ylim([1e-3, 1])  # Set the y-axis range for the third panel

    # Apply y-axis limits if provided
    if y_range_dP_dr:
        axs[0].set_ylim(y_range_dP_dr)
    if y_range_rho:
        axs[1].set_ylim(y_range_rho)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True)

    # Collect handles and labels
    handles, labels = axs[0].get_legend_handles_labels()

    # Reorder: Place 'Initial' before 'Z=0'
    order = [1, 0] + list(range(2, len(labels)))  # Move 'Initial' (index 1) before 'Z=0' (index 0)

    # Apply the reordered handles and labels to the legend
    axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best', fontsize=legend_fontsize)

    pdf_filename = f"radial_profiles_comparison_Z_{filename_suffix}.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath, bbox_inches="tight")
    plt.close()
    print(f"Radial profile plot saved to {output_filepath}")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


def load_simulation_data(sim_path, snapshot_index):
    gasdens, gasenergy, dust1dens, xgrid, ygrid, zgrid = read_single_snapshot(sim_path, snapshot_index)
    return {
        'xgrid': xgrid,
        'gasdens': gasdens,
        'dust1dens': dust1dens,
        'gasenergy': gasenergy
    }



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
    dust1dens = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)
    
    return gasdens, gasenergy, dust1dens, xgrid, ygrid, zgrid



def main():
    parser = argparse.ArgumentParser(description="Compute and plot radial profiles for different simulations.")
    parser.add_argument('--raettig', action='store_true', help="Use Raettig simulations.")
    parser.add_argument('--mitigation', action='store_true', help="Use Mitigation simulations.")
    parser.add_argument('--other', action='store_true', help="Use other simulations.")

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
        y_range_Z = [2e-5, 2e-1]
        horizontal_line = 0.07
    elif args.mitigation:
        simulations = [
            "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
            "cos_b1d0_us_St1dm1_Z3dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
            "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        ]
        filename_suffix = "mitigation"
        y_range_Z = [1e-3, 5e-1]
        horizontal_line = 0.045
    elif args.other:
        simulations = ["cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
                       "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
                       "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"   
        ]
        filename_suffix = "other"
        y_range_Z = [1e-3, 1]
        horizontal_line = 0.25
    else:
        print("Please select one of the simulation sets with --raettig, --mitigation, or --other.")
        return

    load_and_compute_radial_profiles(
        simulations, base_paths, output_path, filename_suffix,
        y_range_dP_dr=(-10, 0), y_range_rho=(0.5, 2), y_range_Z=y_range_Z,
        horizontal_line=horizontal_line
    )
if __name__ == "__main__":
    main()
