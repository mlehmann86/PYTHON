import os
import numpy as np
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from plot_simulations_2D_tau import apply_smoothing



def determine_base_path(subdirectory):

    print(f"subdirectory: {subdirectory}")  # Debugging print statement

    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]
    
    subdir_path = None
    for base_path in base_paths:
        potential_path = os.path.join(base_path, subdirectory)
        if os.path.exists(potential_path):
            subdir_path = potential_path
            break

    if subdir_path is None:
        raise FileNotFoundError(f"Subdirectory {subdirectory} not found in any base path.")
    
    return subdir_path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def load_simulation_data(simulation_dirs):
    data = {}

    for sim_dir in simulation_dirs:
        print(f"subdirectory: {sim_dir}")  # Debug print statement
        subdir_path = determine_base_path(sim_dir)
        
        # Load the .npz file containing the required data
        npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
        loaded_data = np.load(npz_file)
        
        # Extract metallicity Z from the directory name (assuming Z is in the format Zxxxx)
        metallicity_str = [part for part in sim_dir.split('_') if part.startswith('Z')][0]
        if 'dm' in metallicity_str:
            metallicity = metallicity_str.replace('dm', '*10^{-') + '}'
            metallicity = metallicity.replace("Z", "")  # Remove 'Z' prefix
        else:
            metallicity = metallicity_str
        
        # Store the loaded data in the dictionary using metallicity as the key
        data[metallicity] = {
            'time': loaded_data['time'],
            'H_d': loaded_data['H_d'],
        }
    
    return data

def plot_results(simulations_3D, simulations_2D, labels):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    smoothing_window=10
    start_time=80

    # Before plotting, directly apply the font size settings:
    plt.rcParams.update({
    'font.size': 20,  # Global font size
    'lines.linewidth': 2,  # Line width
    'axes.linewidth': 2,  # Axis line width
    'xtick.major.width': 2,  # X tick width
    'ytick.major.width': 2   # Y tick width
    })



    colors = {
        'Z1*10^{-4}': 'blue',
        'Z5*10^{-2}': 'green',
        'Z1*10^{-1}': 'red',
        'Z2*10^{-3}': 'cyan',
        'Z5*10^{-3}': 'magenta',
        # Add more predefined colors as needed
    }

    # Generate a colormap if metallicity doesn't have a predefined color
    colormap = cm.get_cmap('tab10')  # Use any suitable colormap
    color_idx = 0

    # Y-axis limits for each panel
    ylims = [
        [2e-3, 2e-2],  # First panel
        [1e-3, 1e0],  # Second panel
        [1e-3, 1e-2]   # Third panel
    ]

 

    for i in range(len(simulations_3D)):
        sim_3D_list = simulations_3D[i]
        sim_2D_list = simulations_2D[i]
        label = labels[i]
        
        # Load data for each individual 3D and 2D simulation
        data_3D = load_simulation_data(sim_3D_list)
        data_2D = load_simulation_data(sim_2D_list)
        
        for metallicity, data in data_3D.items():
            time = data['time']
            H_d = data['H_d']
            H_d = apply_smoothing(H_d, time, smoothing_window, start_time)
            # Dynamically assign color if not predefined
            if metallicity not in colors:
                colors[metallicity] = colormap(color_idx)
                color_idx += 1
            # Plot the 3D simulations with a label
            line, = axes[i].plot(time, H_d, label=f"$Z={metallicity}$", color=colors[metallicity], linestyle='-')
            
           

        for metallicity, data in data_2D.items():
            time = data['time']
            H_d = data['H_d']
            H_d = apply_smoothing(H_d, time, smoothing_window, start_time)

            # Use the same color as for the corresponding 3D simulation
            if metallicity not in colors:
                colors[metallicity] = colormap(color_idx)
                color_idx += 1
            # Plot the 2D simulations without a label
            axes[i].plot(time, H_d, color=colors[metallicity], linestyle='--')
            
          
        axes[i].set_ylim(ylims[i])
        axes[i].set_yscale('log')
        axes[i].set_ylabel(r"$H_{d}/H_{g}$")
        axes[i].set_title(label)
        axes[i].legend(loc="upper left")


    for ax in axes:
        ax.set_xlabel('Time (Orbits)', fontsize=20)
        ax.set_ylabel(r"$H_{d}/H_{g}$", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)

    # Adjust legend font size explicitly
    for legend in axes:
        legend.legend(fontsize=18)

    # Remove tick marks and axis labels on all horizontal axes, apart from the bottom panel
    for ax in axes[:-1]:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    axes[-1].set_xlabel("Time (Orbits)")
    
    plt.tight_layout()

    # Save the plot
    output_filename = "HdHg_comparison.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Call the scp_transfer function to transfer the plot
    scp_transfer(output_filename, local_directory, "mariuslehmann")



if __name__ == "__main__":
    # List of simulations
    simulations_tau_0_01_3D = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]
    
    simulations_tau_0_01_2D = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
    ]
    
    simulations_tau_0_05_3D = [
        "cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150"
    ]

    simulations_tau_0_05_2D = [
        "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150"
    ]
    
    simulations_tau_0_1_3D = [
        "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]

    simulations_tau_0_1_2D = [
        "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm3_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
    ]

    simulations_tau_0_1_3D = [
        "cos_b1d0_us_St1dm1_Z5dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]

    simulations_tau_0_1_2D = [
        "cos_b1d0_us_St1dm1_Z5dm3_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap"
    ]
    
    labels = [
        r"$\tau = 0.01$",
        r"$\tau = 0.05$",
        r"$\tau = 0.1$"
    ]
    
    plot_results(
        [simulations_tau_0_01_3D, simulations_tau_0_05_3D, simulations_tau_0_1_3D],
        [simulations_tau_0_01_2D, simulations_tau_0_05_2D, simulations_tau_0_1_2D],
        labels
    )
