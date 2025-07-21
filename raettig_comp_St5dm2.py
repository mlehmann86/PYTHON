import os
import numpy as np
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from plot_simulations_2D_tau import apply_smoothing
from data_storage import determine_base_path


def load_simulation_data(simulation_dirs):
    data = {}

    for sim_dir in simulation_dirs:
        print(f"subdirectory: {sim_dir}")  # Debug print statement
        subdir_path = determine_base_path(sim_dir)
        
        # Load the .npz file containing the required data
        npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
        
        # Open the .npz file and load the contents
        loaded_data = np.load(npz_file)
        
        # Extract metallicity Z from the directory name (assuming Z is in the format Zxxxx)
        metallicity_str = [part for part in sim_dir.split('_') if part.startswith('Z')][0]
        if 'dm' in metallicity_str:
            metallicity = metallicity_str.replace('dm', '*10^{-') + '}'
            metallicity = metallicity.replace("Z", "")  # Remove 'Z' prefix
        else:
            metallicity = metallicity_str

        # Simplify the metallicity label by removing '1*'
        metallicity = metallicity.replace('1*', '')

        # Store the loaded data in the dictionary using metallicity as the key
        data[metallicity] = {
            'time': loaded_data['time'],
            'max_epsilon': loaded_data['max_epsilon'],
            'avg_metallicity': loaded_data['Z_avg'],  # Updated to Z_avg
            'rms_vz': loaded_data['rms_vz'],
            'roche_times': loaded_data.get('roche_times', None)  # Get roche_times if available
        }
    
    return data


def plot_results(simulations_3D, simulations_2D, y_ranges=None, x_range=None, plot_upper_only=False):
    if plot_upper_only:
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        # Set consistent font sizes for all elements
        label_fontsize = 13
        tick_fontsize = 13
        legend_fontsize = 13
        axes = [axes]  # Convert to list to keep consistent with the three-panel approach
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))  # Smaller horizontal size
        # Set consistent font sizes for all elements
        label_fontsize = 26
        tick_fontsize = 26
        legend_fontsize = 26

    smoothing_window = 10
    start_time = 40



    # Font and axis settings
    plt.rcParams.update({
        'font.size': tick_fontsize,  # Global tick label size
        'lines.linewidth': 2,  # Line width
        'axes.linewidth': 2,  # Axis line width
        'xtick.major.width': 2,  # X tick width
        'ytick.major.width': 2   # Y tick width
    })

    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors based on initial metallicity

    for sim_3D_list, sim_2D_list in zip(simulations_3D, simulations_2D):
        # Load data for 3D and 2D simulations
        data_3D = load_simulation_data(sim_3D_list)
        data_2D = load_simulation_data(sim_2D_list)
        
        color_idx = 0  # Reset color index for each set

        for metallicity, data in data_3D.items():
            time = data['time']
            max_epsilon = data['max_epsilon']
            avg_metallicity = data['avg_metallicity']
            rms_vz = data['rms_vz']
            roche_times = data.get('roche_times', None)  # Add roche_times

            # Apply smoothing
            max_epsilon = apply_smoothing(max_epsilon, time, smoothing_window, start_time)
            avg_metallicity = apply_smoothing(avg_metallicity, time, smoothing_window, start_time)
            rms_vz = apply_smoothing(rms_vz, time, smoothing_window, start_time)

            color = consistent_colors[color_idx]
            color_idx += 1
            
            # Plot the 3D simulations
            line, = axes[0].plot(time, max_epsilon, label=f"$Z={metallicity}$", color=color, linestyle='-')
            if not plot_upper_only:
                axes[1].plot(time, avg_metallicity, color=color, linestyle='-')
                axes[2].plot(time, rms_vz, color=color, linestyle='-')

            # Highlight Roche density exceedance in epsilon_max
            if roche_times is not None:
                for roche_time in roche_times:
                    epsilon_value_at_time = max_epsilon[time == roche_time]
                    if len(epsilon_value_at_time) > 0:
                        axes[0].scatter(roche_time, epsilon_value_at_time, color=line.get_color(), marker='o', s=100)

        color_idx = 0  # Reset color index for 2D plots to match 3D simulation colors

        for metallicity, data in data_2D.items():
            time = data['time']
            max_epsilon = data['max_epsilon']
            avg_metallicity = data['avg_metallicity']
            rms_vz = data['rms_vz']
            roche_times = data.get('roche_times', None)  # Add roche_times for 2D

            # Apply smoothing
            max_epsilon = apply_smoothing(max_epsilon, time, smoothing_window, start_time)
            avg_metallicity = apply_smoothing(avg_metallicity, time, smoothing_window, start_time)
            rms_vz = apply_smoothing(rms_vz, time, smoothing_window, start_time)

            color = consistent_colors[color_idx]
            color_idx += 1
            
            # Plot the 2D simulations
            line, = axes[0].plot(time, max_epsilon, color=color, linestyle='--')
            if not plot_upper_only:
                axes[1].plot(time, avg_metallicity, color=color, linestyle='--')
                axes[2].plot(time, rms_vz, color=color, linestyle='--')

            # Highlight Roche density exceedance in epsilon_max for 2D
            if roche_times is not None:
                for roche_time in roche_times:
                    epsilon_value_at_time = max_epsilon[time == roche_time]
                    if len(epsilon_value_at_time) > 0:
                        axes[0].scatter(roche_time, epsilon_value_at_time, color=line.get_color(), marker='o', s=100)

    # Set y-axis labels and log scale for all panels
    axes[0].set_ylabel(r"$\epsilon_{\max}$", fontsize=label_fontsize)
    axes[0].set_yscale('log')  # Logarithmic scale for the first panel

    if not plot_upper_only:
        axes[1].set_ylabel(r"$\langle Z \rangle$", fontsize=label_fontsize)  # Updated label for average metallicity
        axes[1].set_yscale('log')  # Logarithmic scale for the second panel

        axes[2].set_ylabel(r"RMS($v_z$)", fontsize=label_fontsize)
        axes[2].set_yscale('log')  # Logarithmic scale for the third panel

    # Set the x-axis label "Time [Orbits]" for the bottom-most panel
    axes[-1].set_xlabel('Time [Orbits]', fontsize=label_fontsize)

    # Apply user-defined axis ranges if provided
    if y_ranges is not None:
        if 'epsilon_max' in y_ranges:
            axes[0].set_ylim(y_ranges['epsilon_max'])
        if not plot_upper_only:
            if 'Z_avg' in y_ranges:
                axes[1].set_ylim(y_ranges['Z_avg'])
            if 'rms_vz' in y_ranges:
                axes[2].set_ylim(y_ranges['rms_vz'])
    
    if x_range is not None:
        for ax in axes:
            ax.set_xlim(x_range)

    # Add legend only to the top panel
    #axes[0].legend(loc="upper left", fontsize=legend_fontsize)

    # Adjust labels and tick marks for all axes
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    if not plot_upper_only:
        # Remove tick marks from the upper panels' horizontal axes
        axes[0].set_xticklabels([])  # Remove x-tick labels in the first panel
        axes[1].set_xticklabels([])  # Remove x-tick labels in the second panel

    plt.tight_layout()

    # Save the plot
    output_filename = "comparison_max_eps_metallicity_rms_vz_roche.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Call the scp_transfer function to transfer the plot
    scp_transfer(output_filename, local_directory, "mariuslehmann")


if __name__ == "__main__":
    # List of simulations for 3D and 2D cases
    simulations_3D = [
        ["cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
         "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
         "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150"]
    ]
    
    simulations_2D = [
        ["cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_stnew_LR150",
         "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_stnew_LR150",
         "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150"]
    ]

    y_ranges = {
        'epsilon_max': [1e-3, 1e3],  # Set y-axis range for epsilon_max
        'Z_avg': [5e-5, 1e-1],       # Set y-axis range for Z_avg
        'rms_vz': [1e-3, 1.1e-2]     # Set y-axis range for rms_vz
    }

    x_range = [0, 1000]  # Set the x-axis range

    # Call the plot_results function with the option to plot only the upper panel
    plot_results(simulations_3D, simulations_2D, y_ranges=y_ranges, x_range=x_range, plot_upper_only=False)
