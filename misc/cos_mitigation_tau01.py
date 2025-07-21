import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
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
            loaded_data = np.load(npz_file)
            print(f"Loaded data from: {npz_file}")  # Debug print statement

            # Print all keys and their shapes
            print("Keys in the .npz file:", loaded_data.files)
            for key in loaded_data.files:
                print(f"{key}: {loaded_data[key].shape}")

            data[sim_dir] = {
                'time': loaded_data['time'],
                'alpha_r': loaded_data['alpha_r'],
                'rms_vz': loaded_data['rms_vz'],
                #'max_vz': loaded_data['max_vz'],
                #'min_vz': loaded_data['min_vz'],
                'max_epsilon': loaded_data['max_epsilon'],
                'H_d_array': loaded_data['H_d'],
                'roche_times': loaded_data['roche_times'] if 'roche_times' in loaded_data else None,  # Handle older .npz files without roche_times
            }
        else:
            print(f"WARNING: File not found in both base paths for {sim_dir}!")
    
    return data



def plot_results(simulation_dirs_with_labels):
    # Split the list into simulation directories and labels
    simulation_dirs = simulation_dirs_with_labels[:4]  # First four elements are directories
    labels = simulation_dirs_with_labels[4:]  # Last four elements are labels

    #default_linewidth = 2.5  # Adjust this value to control the thickness

    # Set the global font size
    plt.rcParams.update({'font.size': 20})  # Adjust the number to increase or decrease font size

    smoothing_window = 10
    start_time = 80

    # Set consistent font sizes for all elements
    label_fontsize = 24
    tick_fontsize = 24
    legend_fontsize = 24

    # Font and axis settings
    plt.rcParams.update({
        'font.size': tick_fontsize,  # Global tick label size
        'lines.linewidth': 3,  # Line width
        'axes.linewidth': 2,  # Axis line width
        'xtick.major.width': 2,  # X tick width
        'ytick.major.width': 2   # Y tick width
    })


    # Colors for different simulations
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    print("simulation_dirs:", simulation_dirs)  # Debug print statement
    print("labels:", labels)  # Debug print statement
    
    data = load_simulation_data(simulation_dirs)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    for sim_dir, label, color in zip(simulation_dirs, labels, colors):
        print(f"Processing directory: {sim_dir}")
        if sim_dir in data:
            time = data[sim_dir]['time']
            roche_times = data[sim_dir].get('roche_times', None)

            # Apply smoothing with a start time
            alpha_r = apply_smoothing(data[sim_dir]['alpha_r'], time, smoothing_window, start_time)
            #max_epsilon = apply_smoothing(data[sim_dir]['max_epsilon'], time, smoothing_window, start_time)
            #rms_vz = apply_smoothing(data[sim_dir]['rms_vz'], time, smoothing_window, start_time)
            max_epsilon = data[sim_dir]['max_epsilon']
            rms_vz = data[sim_dir]['rms_vz']
            #max_vz = data[sim_dir]['max_vz']
            #min_vz = data[sim_dir]['min_vz']

            aspectratio = 0.1

         
            axes[0].plot(time, alpha_r, linestyle='solid', color=color, label=label)
            axes[1].plot(time, max_epsilon, linestyle='solid', color=color, label=label)
            #axes[2].plot(time, (max_vz - min_vz) / aspectratio, linestyle='solid', color=color, label=label)
            axes[2].plot(time, rms_vz / aspectratio, linestyle='solid', color=color, label=label)

         

            # Plot Roche exceedance times as scatter points with matching color
            if roche_times is not None:
                for roche_time in roche_times:
                    epsilon_value_at_time = max_epsilon[time == roche_time]
                    if len(epsilon_value_at_time) > 0:
                        axes[1].scatter(roche_time, epsilon_value_at_time, color=color, marker='o', s=100, zorder=5)

        else:
            print(f"ERROR: Data could not be loaded for {sim_dir}")
    
    # Add a subtle vertical black dashed line at 728 orbits to all panels
    #for ax in axes:
    #    ax.axvline(x=728, color='black', linestyle='dashed', linewidth=1)
    #    ax.axvline(x=400, color='black', linestyle='dashed', linewidth=1)

    # Set the vertical axes to logarithmic scale and adjust the ranges
    axes[0].set_yscale('log')
    axes[0].set_ylim(5e-6, 1e-2)
    
    axes[1].set_yscale('log')
    axes[1].set_ylim(5e-2, 1e4)
    
    axes[2].set_yscale('linear')
    #axes[2].set_ylim(3e-2, 1e0)
    axes[2].set_ylim(1e-2, 5e-2)
    
    axes[0].set_ylabel(r"$\alpha_r$")
    axes[1].set_ylabel(r"$\epsilon_{\max}$")
    #axes[2].set_ylabel(r"$[max(v_z)-min(v_z)]/c_0$")
    axes[2].set_ylabel(r"$RMS(v_z)/c_0$")
    axes[2].set_xlabel("Time (Orbits)")
    
    # Adjust legend and tick marks
    axes[1].legend(loc="lower right",fontsize=17)  # Single legend in bottom panel

    for ax in axes[:2]:  # Remove tick marks on the horizontal axes in the upper and middle panels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"cos_mitigation_tau01.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")
    
    # Call the scp_transfer function
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")



if __name__ == "__main__":
    simulation_dirs_with_labels = [
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "Z=0.01",
        "Z=0.02",
        "Z=0.05",
        "Z=0.1"
    ]

    plot_results(simulation_dirs_with_labels)



