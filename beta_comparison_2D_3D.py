import os
import numpy as np
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from data_reader import reconstruct_grid, read_parameters
from plot_simulations_2D_tau import apply_smoothing
from data_storage import determine_base_path
import matplotlib.cm as cm

def load_simulation_data(simulation_dirs):
    data = {}

    for sim_dir in simulation_dirs:
        print(f"subdirectory: {sim_dir}")  # Debug print statement
        subdir_path = determine_base_path(sim_dir)

        # Load the .npz file containing the required data
        npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
        summary_file = os.path.join(subdir_path, "summary0.dat")
        
        try:
            loaded_data = np.load(npz_file)
            parameters = read_parameters(summary_file)
            
            # Reconstruct the grid (in case we need zgrid)
            xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
            
            # Retrieve the aspect ratio from the parameters file
            h0 = float(parameters['ASPECTRATIO'])  # Aspect ratio H_g
            
            # Extract metallicity Z from the directory name (assuming Z is in the format Zxxxx)
            metallicity_str = [part for part in sim_dir.split('_') if part.startswith('Z')][0]
            if 'dm' in metallicity_str:
                metallicity = metallicity_str.replace('dm', '*10^{-') + '}'
                metallicity = metallicity.replace("Z", "")  # Remove 'Z' prefix
            else:
                metallicity = metallicity_str
            
            # Attempt to retrieve the rms_vz data
            rms_vz = loaded_data.get('rms_vz', None)
            if rms_vz is None:
                print(f"Warning: 'rms_vz' not found in {sim_dir}. Skipping.")

            # Attempt to retrieve the rms_vz_profile data (dust)
            rms_vz_profile = loaded_data.get('rms_vz_profile', None)
            if rms_vz_profile is None:
                print(f"Warning: 'rms_vz_profile' (dust) not found in {sim_dir}. Skipping.")

            # Attempt to retrieve the rms_vz_profile_gas data (gas)
            rms_vz_profile_gas = loaded_data.get('rms_vz_profile_gas', None)
            if rms_vz_profile_gas is None:
                print(f"Warning: 'rms_vz_profile_gas' (gas) not found in {sim_dir}. Skipping.")

            # Store the loaded data in the dictionary using metallicity as the key
            data[metallicity] = {
                'time': loaded_data['time'],
                'max_epsilon': loaded_data['max_epsilon'],
                'H_d': loaded_data['H_d'],
                'rms_vz': rms_vz,  # We need to scale this with h0 if it exists
                'rms_vz_profile': rms_vz_profile / h0 if rms_vz_profile is not None else None,  # Normalize dust velocity profile
                'rms_vz_profile_gas': rms_vz_profile_gas / h0 if rms_vz_profile_gas is not None else None,  # Normalize gas velocity profile
                'roche_times': loaded_data.get('roche_times', None),
                'h0': h0,  # Store aspect ratio for scaling later
                'zgrid': zgrid  # Store zgrid for rms_vz_profile plots
            }
        
        except FileNotFoundError:
            print(f"File not found: {npz_file}")
        except KeyError as e:
            print(f"KeyError in {sim_dir}: {e}")
            raise e  # Raise the error again after printing the simulation directory

    return data


def plot_results(simulations_beta1_3D, simulations_beta1_2D,
                 simulations_beta001_3D, simulations_beta001_2D):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))  # 3x2 grid for comparisons
    
    smoothing_window = 8
    smoothing_window_Hd = 1
    start_time = 40

    label_fontsize = 22
    title_fontsize = 24
    tick_fontsize = 22
    legend_fontsize = 18

    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Note: We will swap usage of ylims_Hd and ylims_eps
    ylims_eps = [[1e-2, 1e2], [1e-2, 1e2]]
    ylims_Hd = [[0.03, 0.12], [0.03, 0.12]]
    ylims_rms_vz = [[4e-3, 2e-2], [4e-3, 2e-2]]

    labels = [1, 0.001]

    metallicity_map = {
        "1*10^{-2}": "0.01",
        "5*10^{-2}": "0.05",
        "1*10^{-1}": "0.1"
    }

    marker_size = 8  # used for 2D simulation markers

    # --- MAIN LOOP: Now top row is H_d/H_g, middle row is epsilon_max ---
    for i, (sim_3D_list, sim_2D_list) in enumerate(zip([simulations_beta1_3D, simulations_beta001_3D],
                                                       [simulations_beta1_2D, simulations_beta001_2D])):

        col_idx = i
        data_3D = load_simulation_data(sim_3D_list)
        data_2D = load_simulation_data(sim_2D_list)
        
        color_idx = 0

        # --- Plot 3D simulation data ---
        for metallicity, data in data_3D.items():
            time = data['time']
            max_epsilon = apply_smoothing(data['max_epsilon'], time, 1, start_time)
            H_d_H_g = apply_smoothing(data['H_d'] / data['h0'], time, smoothing_window, start_time)
            rms_vz_c0 = (apply_smoothing(data['rms_vz'] / data['h0'], time, smoothing_window, start_time)
                         if data['rms_vz'] is not None else None)

            color = consistent_colors[color_idx % len(consistent_colors)]
            color_idx += 1

            # Use metallicity label only on the top-row plots (so we can build a metallicity legend)
            metallicity_label = f"$Z={metallicity_map.get(metallicity, metallicity)}$"

            # TOP row -> H_d/H_g
            axes[0, col_idx].plot(time, H_d_H_g, label=metallicity_label,
                                  color=color, linestyle='-', linewidth=3, alpha=0.8)
            # MIDDLE row -> epsilon_max
            axes[1, col_idx].plot(time, max_epsilon, color=color, linestyle='-', linewidth=3, alpha=0.8)
            # BOTTOM row -> RMS(v_z)/c0
            if rms_vz_c0 is not None:
                axes[2, col_idx].plot(time, rms_vz_c0, color=color, linestyle='-', linewidth=3, alpha=0.8)

        # --- Plot 2D simulation data ---
        color_idx = 0
        for metallicity, data in data_2D.items():
            time = data['time']
            max_epsilon = apply_smoothing(data['max_epsilon'], time, 1, start_time)
            H_d_H_g = apply_smoothing(data['H_d'] / data['h0'], time, smoothing_window, start_time)
            rms_vz_c0 = (apply_smoothing(data['rms_vz'] / data['h0'], time, smoothing_window, start_time)
                         if data['rms_vz'] is not None else None)

            color = consistent_colors[color_idx % len(consistent_colors)]
            color_idx += 1

            marker_interval = 20
            marker_indices = np.arange(0, len(time), marker_interval)

            # TOP row -> H_d/H_g
            axes[0, col_idx].plot(time, H_d_H_g, color=color, linestyle='--', linewidth=2,
                                  marker='o', markevery=marker_indices, alpha=1.0, markersize=marker_size)
            # MIDDLE row -> epsilon_max
            axes[1, col_idx].plot(time, max_epsilon, color=color, linestyle='--', linewidth=2,
                                  marker='o', markevery=marker_indices, alpha=1.0, markersize=marker_size)
            # BOTTOM row -> RMS(v_z)/c0
            if rms_vz_c0 is not None:
                axes[2, col_idx].plot(time, rms_vz_c0, color=color, linestyle='--', linewidth=2,
                                      marker='o', markevery=marker_indices, alpha=1.0, markersize=marker_size)

        # --- Set axis limits, scales, titles, and labels ---
        # TOP row -> H_d/H_g
        axes[0, col_idx].set_ylim(ylims_Hd[col_idx])
        axes[0, col_idx].set_ylabel(r"$\langle H_d / H_g \rangle$", fontsize=label_fontsize)
        axes[0, col_idx].set_title(r"$\beta = " + str(labels[i]) + "$", fontsize=title_fontsize)

        # MIDDLE row -> epsilon_max (log scale)
        axes[1, col_idx].set_ylim(ylims_eps[col_idx])
        axes[1, col_idx].set_yscale('log')
        axes[1, col_idx].set_ylabel(r"$\epsilon_{\max}$", fontsize=label_fontsize)

        # BOTTOM row -> RMS(v_z)/c0
        axes[2, col_idx].set_ylim(ylims_rms_vz[col_idx])
        axes[2, col_idx].set_ylabel(r"RMS$(v_z)/c_0$", fontsize=label_fontsize)

        # Remove y-ticks for the right column
        if col_idx == 1:
            for ax in axes[:, col_idx]:
                ax.set_ylabel('')
                ax.set_yticklabels([])

        # Make ticks consistent
        for ax in axes[:, col_idx]:
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.5)
            ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize, width=1.0)

    # Remove x tick labels from the top and middle rows
    for ax in axes[0:2, :].flatten():
        ax.set_xticklabels([])

    # Set x label only on the bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (Orbits)', fontsize=label_fontsize)

    # ----------------------------------------------------------------
    # MOVE BOTH LEGENDS INTO THE TOP-RIGHT PANEL (axes[0,1]):
    # 1) Metallicities (color-coded) in the top-left corner
    # 2) 3D vs. 2D simulation (line style) in the top-right corner
    # ----------------------------------------------------------------

    # 1) Metallicities legend: gather from the top-left panel (axes[0,0])
    metal_handles, metal_labels = axes[0, 0].get_legend_handles_labels()
    # Remove any existing legend from that panel
    axes[0, 0].legend_ = None

    # Place metallicities legend in top-right panel, top-left corner
    leg_metal = axes[0, 1].legend(metal_handles, metal_labels, loc='upper left',
                                  fontsize=legend_fontsize, frameon=False)
    axes[0, 1].add_artist(leg_metal)

    # 2) 3D vs 2D legend (line styles) in the top-right corner of top-right panel
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='3D Simulation'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, marker='o',
               markersize=marker_size, label='2D Simulation')
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right',
                      fontsize=legend_fontsize, frameon=False)

    plt.tight_layout()
    output_filename = "comparison_epsilon_Hd_rms_vz_beta1_beta001.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Call the scp_transfer function to transfer the plot
    scp_transfer(output_filename, local_directory, "mariuslehmann")



def plot_rms_vz_profile(simulations_beta1_3D, simulations_beta1_2D, simulations_beta001_3D, simulations_beta001_2D):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Two side-by-side panels

    label_fontsize = 22
    title_fontsize = 24
    tick_fontsize = 22
    legend_fontsize = 18

    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    labels = [1, 0.001]

    metallicity_map = {
        "1*10^{-2}": "0.01",
        "5*10^{-2}": "0.05",
        "1*10^{-1}": "0.1"
    }

    for i, (sim_3D_list, sim_2D_list) in enumerate(zip([simulations_beta1_3D, simulations_beta001_3D],
                                                       [simulations_beta1_2D, simulations_beta001_2D])):
        col_idx = i
        data_3D = load_simulation_data(sim_3D_list)
        data_2D = load_simulation_data(sim_2D_list)

        color_idx = 0

        for metallicity, data in data_3D.items():
            zgrid = data['zgrid']
            rms_vz_profile = data['rms_vz_profile']

            if rms_vz_profile is None:
                continue  # Skip if rms_vz_profile is missing

            color = consistent_colors[color_idx % len(consistent_colors)]
            color_idx += 1

            metallicity_label = f"$Z={metallicity_map.get(metallicity, metallicity)}$"

            # Plot 3D simulations (solid lines)
            axes[col_idx].plot(zgrid, rms_vz_profile, label=metallicity_label, color=color, linestyle='-', linewidth=3, alpha=0.8)

        color_idx = 0  # Reset color index for 2D simulations

        from matplotlib.lines import Line2D

        for metallicity, data in data_2D.items():
            zgrid = data['zgrid']
            rms_vz_profile = data['rms_vz_profile']

            if rms_vz_profile is None:
                continue  # Skip if rms_vz_profile is missing

            color = consistent_colors[color_idx % len(consistent_colors)]
            color_idx += 1

            marker_interval = 10
            marker_indices = np.arange(0, len(zgrid), marker_interval)
            marker_size = 6

            # Plot 2D simulations (dashed lines with markers)
            axes[col_idx].plot(zgrid, rms_vz_profile, color=color, linestyle='--', linewidth=2, marker='o',
                               markevery=marker_indices, alpha=1.0, markersize=marker_size)

        # Axis labels and formatting
        axes[col_idx].set_xlabel(r"$z/H_g$", fontsize=label_fontsize)  # Swapped to x-axis
        axes[col_idx].set_ylabel(r"RMS$(v_z)/c_0$", fontsize=label_fontsize)  # Swapped to y-axis
        axes[col_idx].set_title(r"$\beta = " + str(labels[i]) + "$", fontsize=title_fontsize)
        axes[col_idx].tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Add legend to the right panel
        if col_idx == 1:
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='-', linewidth=3, label='3D Simulation'),
                Line2D([0], [0], color='black', linestyle='--', linewidth=2, marker='o', markersize=6, label='2D Simulation')
            ]
            axes[col_idx].legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize, frameon=False)

    plt.tight_layout()

    # Save the figure
    output_filename = "rms_vz_profile_2D_vs_3D.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Transfer via SCP
    scp_transfer(output_filename, local_directory, "mariuslehmann")



def plot_rms_vz_dust_vs_gas(simulations_beta1_2D, simulations_beta001_2D):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Two side-by-side panels

    label_fontsize = 22
    title_fontsize = 24
    tick_fontsize = 22
    legend_fontsize = 18

    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    labels = [1, 0.001]

    metallicity_map = {
        "1*10^{-2}": "0.01",
        "5*10^{-2}": "0.05",
        "1*10^{-1}": "0.1"
    }

    for i, sim_2D_list in enumerate([simulations_beta1_2D, simulations_beta001_2D]):
        col_idx = i
        data_2D = load_simulation_data(sim_2D_list)

        color_idx = 0

        for metallicity, data in data_2D.items():
            zgrid = data['zgrid']
            rms_vz_profile_dust = data.get('rms_vz_profile', None)  # Dust velocity
            rms_vz_profile_gas = data.get('rms_vz_profile_gas', None)  # Gas velocity

            if rms_vz_profile_dust is None or rms_vz_profile_gas is None:
                continue  # Skip if either quantity is missing

            color = consistent_colors[color_idx % len(consistent_colors)]
            color_idx += 1

            metallicity_label = f"$Z={metallicity_map.get(metallicity, metallicity)}$"

            # Plot dust velocity (solid line)
            axes[col_idx].plot(zgrid, rms_vz_profile_dust, label=f"{metallicity_label} (Dust)", 
                               color=color, linestyle='-', linewidth=3, alpha=0.8)

            # Plot gas velocity (dashed line)
            axes[col_idx].plot(zgrid, rms_vz_profile_gas, label=f"{metallicity_label} (Gas)", 
                               color=color, linestyle='--', linewidth=2, alpha=0.8)

        # Axis labels and formatting
        axes[col_idx].set_xlabel(r"$z/H_g$", fontsize=label_fontsize)
        axes[col_idx].set_ylabel(r"RMS$(v_z)/c_0$", fontsize=label_fontsize)
        axes[col_idx].set_title(r"$\beta = " + str(labels[i]) + "$", fontsize=title_fontsize)
        axes[col_idx].tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Add legend to the right panel
        axes[col_idx].legend(loc='upper right', fontsize=legend_fontsize, frameon=False)

    plt.tight_layout()

    # Save the figure
    output_filename = "rms_vz_dust_vs_gas_2D.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Transfer via SCP
    scp_transfer(output_filename, local_directory, "mariuslehmann")
if __name__ == "__main__":
    # List of simulations with beta=1
    simulations_beta1_3D = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]
    
    simulations_beta1_2D = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
    ]

    # List of simulations with beta=0.001
    simulations_beta001_3D = [
        "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]



    simulations_beta001_2D = [
        "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
        "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
    ]

    
    plot_results(
        simulations_beta1_3D, simulations_beta1_2D,
        simulations_beta001_3D, simulations_beta001_2D
    )

    plot_rms_vz_profile(
        simulations_beta1_3D, simulations_beta1_2D,
        simulations_beta001_3D, simulations_beta001_2D
    )

    # Call function to compare dust and gas RMS v_z profiles in 2D simulations
    plot_rms_vz_dust_vs_gas(
        simulations_beta1_2D, simulations_beta001_2D
    )
