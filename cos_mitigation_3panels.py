import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from data_storage import scp_transfer
from plot_simulations_2D_tau import apply_smoothing

def load_simulation_data(simulation_dirs):
    # Function to load data from the specified directories
    data = {}
    for sim_dir in simulation_dirs:
        print(f"Processing directory: {sim_dir}")  # Debug print statement
        base_paths = [
            "/tiara/home/mlehmann/data/FARGO3D/outputs",
            "/theory/lts/mlehmann/FARGO3D/outputs"
        ]
        
        npz_file = None
        for base_path in base_paths:
            potential_npz_file = os.path.join(base_path, sim_dir, f"{os.path.basename(sim_dir)}_quantities.npz")
            if os.path.exists(potential_npz_file):
                npz_file = potential_npz_file
                break
        
        if npz_file is not None:
            loaded_data = np.load(npz_file, allow_pickle=True)  # Enable loading object arrays
            print(f"Loaded data from: {npz_file}")  # Debug print statement

            # Print all keys and their shapes
            print("Keys in the .npz file:", loaded_data.files)
            for key in loaded_data.files:
                print(f"{key}: {loaded_data[key].shape}")

            # Check and print min/max of xgrid_masked
            if 'xgrid_masked' in loaded_data:
                xgrid_masked = loaded_data['xgrid_masked']
                print(f"xgrid_masked min: {xgrid_masked.min()}, max: {xgrid_masked.max()}")

            # Check and print the smallest Roche time
            if 'roche_times' in loaded_data and len(loaded_data['roche_times']) > 0:
                roche_times = loaded_data['roche_times']
                first_roche_time = np.min(roche_times)
                print(f"First Roche time (smallest): {first_roche_time}")
            else:
                print("No Roche times found or empty Roche times array.")

            data[sim_dir] = {
                'time': loaded_data['time'],
                'alpha_r': loaded_data['alpha_r'],
                'rms_vz': loaded_data['rms_vz'],
                'max_vz': loaded_data['max_vz'],
                'min_vz': loaded_data['min_vz'],
                'max_epsilon': loaded_data['max_epsilon'],
                'H_d': loaded_data['H_d'],
                'roche_times': loaded_data['roche_times'] if 'roche_times' in loaded_data else None,
                'xgrid_masked': loaded_data['xgrid_masked'] if 'xgrid_masked' in loaded_data else None,
            }
        else:
            print(f"WARNING: File not found in both base paths for {sim_dir}!")
    
    return data

def plot_results(simulation_dirs_with_labels):
    # Split the list into simulation directories and labels
    simulation_dirs = simulation_dirs_with_labels[:7]  # First five elements are directories
    labels = simulation_dirs_with_labels[7:]  # Last five elements are labels

    # Adjust figure size for vertically stacked panels
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))  # Four vertically stacked panels

    smoothing_window = 10
    start_time = 80

    # Set font sizes for axes, ticks, and labels
    axis_label_fontsize = 20  # Axis label font size
    tick_label_fontsize = 20  # Tick label font size (numbers)
    legend_fontsize = 16      # Legend font size

    plt.rcParams.update({
        'font.size': tick_label_fontsize,  # Tick labels (numbers) font size
        'lines.linewidth': 3,  
        'axes.linewidth': 2,  
        'xtick.major.width': 2,  
        'ytick.major.width': 2,
        'axes.labelsize': axis_label_fontsize  # Adjust axis label size
    })

    # Updated color scheme: Z=0 is black, Z=0.1 is light orange
    colors = ['black', '#4c72b0', '#55a868', '#e17a36', '#ffa07a', '#8a2be2']  # Adding a medium purple # Z=0 is black, Z=0.1 is light orange
    colors = ['#ffa07a', '#e17a36', '#55a868', '#4c72b0', '#8a2be2', '#2e2d88', 'black']

    print("simulation_dirs:", simulation_dirs)  
    print("labels:", labels)  
    
    data = load_simulation_data(simulation_dirs)

    for sim_dir, label, color in zip(simulation_dirs, labels, colors):
        print(f"Processing directory: {sim_dir}")
        if sim_dir in data:
            time = data[sim_dir]['time']
            roche_times = data[sim_dir].get('roche_times', None)

            # Apply smoothing with a start time
            alpha_r = apply_smoothing(data[sim_dir]['alpha_r'], time, smoothing_window, start_time)
            max_epsilon = data[sim_dir]['max_epsilon']
            rms_vz = data[sim_dir]['rms_vz']
            H_d = data[sim_dir]['H_d']  # H_d is extracted from data

            aspectratio = 0.1

            # Plot in the vertically stacked panels
            axes[0].plot(time, alpha_r, linestyle='solid', color=color, label=label)
            axes[1].plot(time, max_epsilon, linestyle='solid', color=color, label=label)
            axes[2].plot(time, rms_vz / aspectratio, linestyle='solid', color=color, label=label)
            axes[3].plot(time, H_d / aspectratio, linestyle='solid', color=color, label=label)

            # Plot Roche exceedance times as scatter points with matching color
            if roche_times is not None:
                for roche_time in roche_times:
                    epsilon_value_at_time = max_epsilon[time == roche_time]
                    if len(epsilon_value_at_time) > 0:
                        axes[1].scatter(roche_time, epsilon_value_at_time, color=color, marker='o', s=100, zorder=5)

        else:
            print(f"ERROR: Data could not be loaded for {sim_dir}")
    
    # Set y-scales and axis labels
    axes[0].set_yscale('log')
    axes[0].set_ylim(5e-6, 1e-2)
    axes[0].set_ylabel(r"$\alpha_r$", fontsize=axis_label_fontsize)
    
    axes[1].set_yscale('log')
    axes[1].set_ylim(5e-2, 1e4)
    axes[1].set_ylabel(r"$\epsilon_{\max}$", fontsize=axis_label_fontsize)
    
    axes[2].set_yscale('linear')
    axes[2].set_ylim(4e-3, 4e-2)
    axes[2].set_ylabel(r"$RMS(v_z)/c_0$", fontsize=axis_label_fontsize)
    
    axes[3].set_yscale('linear')
    axes[3].set_ylim(0.02, 0.09)  # Adjust as needed for your data range
    axes[3].set_ylabel(r"$\langle H_d / H_g \rangle$", fontsize=axis_label_fontsize)
    axes[3].set_xlabel("Time (Orbits)", fontsize=axis_label_fontsize)

    # Adjust legend and tick marks
    axes[0].legend(loc="lower right", fontsize=legend_fontsize)

    for ax in axes:  # Apply larger tick labels for all axes
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.set_xlim(0, time[-1])  # Set time axis to start at 0 and end at the last time step
    
    plt.tight_layout()

    output_filename = f"cos_mitigation_tau01_4panels_vertical.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")
    
    # Call the scp_transfer function
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


if __name__ == "__main__":
    # Move Z=0 to the front of the list
    simulation_dirs_with_labels = [
        "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",  # Z=0
        "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "Z=0",  # Corresponds to first directory
        "Z=0.002",
        "Z=0.004",
        "Z=0.01",
        "Z=0.03",
        "Z=0.04",
        "Z=0.05"
    ]

    plot_results(simulation_dirs_with_labels)
