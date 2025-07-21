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
from plot_simulations_2D_tau import apply_smoothing




def load_simulation_data(simulation_dirs):
    # Function to load data from the specified directories
    data = {}
    for sim_dir in simulation_dirs:
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

            # Print all keys and their shapes
            print("Keys in the .npz file:", loaded_data.files)
            for key in loaded_data.files:
                print(f"{key}: {loaded_data[key].shape}")

            data[sim_dir] = {
                'time': loaded_data['time'],
                'alpha_r': loaded_data['alpha_r'],
                'rms_vz': loaded_data['rms_vz'],
                'max_epsilon': loaded_data['max_epsilon'],
                'H_d_array': loaded_data['H_d'],
                'roche_times': loaded_data['roche_times'] if 'roche_times' in loaded_data else None,  # Handle older .npz files without roche_times
            }
        else:
            print(f"WARNING: File not found in both base paths for {sim_dir}!")
    
    return data



def plot_results(simulation_dirs):

    default_linewidth = 2  # Adjust this value to control the thickness

    # Set the global font size
    plt.rcParams.update({'font.size': 14.5})  # Adjust the number to increase or decrease font size


    data = load_simulation_data(simulation_dirs)
    
    # Split the simulations into LR and MR categories
    LR_runs = [d for d in simulation_dirs if "LR" in d]
    MR_runs = [d for d in simulation_dirs if "MR" in d]

    smoothing_window=10
    start_time=80
    
    # Colors and linestyles for different metallicity values
    colors = {
    'Z1dm3': '#1f77b4',  # Blue
    'Z1dm2': '#ff7f0e',  # Orange
    'Z5dm2': '#2ca02c',  # Green
    'Z1dm1': '#d62728'   # Red
    }

    linestyles = {
        'bet1d0': 'dashed',
        'bet1dm3': 'solid'
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    labels = {
        "LR": r"$res = 200/H$",
        "MR": r"$res = 400/H$"
    }
  

    # Determine vertical range limits for each row
    ranges = {
        "H_d_H_g": [np.inf, -np.inf],
        "epsilon_max": [np.inf, -np.inf],
        "rms_vz": [np.inf, -np.inf]
    }
    
    # First pass to calculate the consistent y-axis ranges
    aspectratio=0.1
    for runs in [LR_runs, MR_runs]:
        for sim_dir in runs:
            time = data[sim_dir]['time']
            # Define the smoothed arrays
            H_d_H_g = data[sim_dir]['H_d_array'] / aspectratio
            epsilon_max = data[sim_dir]['max_epsilon']
            rms_vz = data[sim_dir]['rms_vz']

            # Update range limits
            ranges["H_d_H_g"][0] = min(ranges["H_d_H_g"][0], np.min(H_d_H_g))
            ranges["H_d_H_g"][1] = max(ranges["H_d_H_g"][1], np.max(H_d_H_g))
            ranges["epsilon_max"][0] = min(ranges["epsilon_max"][0], np.min(epsilon_max))
            ranges["epsilon_max"][1] = max(ranges["epsilon_max"][1], np.max(epsilon_max))
            ranges["rms_vz"][0] = min(ranges["rms_vz"][0], np.min(rms_vz))
            ranges["rms_vz"][1] = max(ranges["rms_vz"][1], np.max(rms_vz))
    
    for idx, (runs, title) in enumerate([(LR_runs, labels["LR"]), (MR_runs, labels["MR"])]):
        for sim_dir in runs:
            metallicity = None
            for key in colors.keys():
                if key in sim_dir:
                    metallicity = key
            
            # Assign linestyle based on beta value in the directory name
            if any(b in sim_dir for b in ["bet1d0", "b1d0"]):
                linestyle = linestyles['bet1d0']
            elif any(b in sim_dir for b in ["bet1dm3", "b1dm3"]):
                linestyle = linestyles['bet1dm3']
            else:
                print(f"Warning: No beta value found in {sim_dir}. Defaulting to solid line.")
                linestyle = 'solid'  # Fallback to solid if neither is found
    
            label = f"{metallicity.replace('Z', 'Z=')}" if any(b in sim_dir for b in ["bet1d0", "b1d0"]) else None  # Avoid redundant labels
            
            time = data[sim_dir]['time']
            # Apply smoothing with a start time
            H_d_H_g = apply_smoothing(data[sim_dir]['H_d_array'] / aspectratio, time, smoothing_window, start_time)
            epsilon_max = apply_smoothing(data[sim_dir]['max_epsilon'], time, smoothing_window, start_time)
            rms_vz = apply_smoothing(data[sim_dir]['rms_vz'], time, smoothing_window, start_time)

            # Plot H_d / H_g (First row)
            axes[0, idx].plot(time, H_d_H_g, color=colors[metallicity], linestyle=linestyle, label=label, linewidth=default_linewidth)
            axes[0, idx].set_ylim(ranges["H_d_H_g"])
            
            # Plot max epsilon (Second row)
            axes[1, idx].plot(time, epsilon_max, color=colors[metallicity], linestyle=linestyle, label=label, linewidth=default_linewidth)
            axes[1, idx].set_ylim(ranges["epsilon_max"])

            # Use the same color for the scatter points as the curve
            curve_color = colors[metallicity]

            # Plot the Roche exceed times with the corresponding color
            if 'roche_times' in data[sim_dir]:
                for roche_time in data[sim_dir]['roche_times']:
                    # Assuming you want to plot the marker at the max epsilon value for that time
                    epsilon_value_at_time = epsilon_max[time == roche_time]  # Find epsilon value at roche_time
                    if len(epsilon_value_at_time) > 0:
                        axes[1, idx].scatter(roche_time, epsilon_value_at_time, color=curve_color, marker='o', s=100, label='Roche Density Reached')
            
            # Plot RMS(vz) (Third row)
            axes[2, idx].plot(time, rms_vz, color=colors[metallicity], linestyle=linestyle, label=label, linewidth=default_linewidth)
            axes[2, idx].set_ylim(ranges["rms_vz"])

        # Set titles and labels
        axes[0, idx].set_title(title)
        axes[0, idx].set_ylabel(r"$H_d / H_g$")
        axes[1, idx].set_ylabel(r"$\epsilon_{\mathrm{max}}$")
        axes[2, idx].set_ylabel(r"RMS($v_z$)")
        axes[2, idx].set_xlabel("Orbits")
        axes[0, idx].set_yscale('log')
        axes[1, idx].set_yscale('log')
        axes[2, idx].set_yscale('log')
        
        for ax in axes[0, :]:  # This targets the first row (H_d/H_g plots)
            ax.set_ylim([5e-3, 5e-1])


        for ax in axes[:, idx]:
            ax.set_xlim([0, 1000])
            ax.set_yscale('log')
    
            # Show grid lines for major ticks only
            ax.grid(True, which='major', linestyle='--', linewidth=0.5)  # Keep major grid lines
    
            # Keep minor ticks but turn off the minor grid lines
            ax.grid(False, which='minor')  # Disable grid lines for minor ticks
        
        # Remove vertical axis labels for right columns
        if idx == 1:
            for ax in axes[:, idx]:
                ax.set_ylabel("")
        
        # Remove x-axis ticks except for the bottom row
        for ax in axes[:-1, idx]:
            ax.set_xticklabels([])

    # Add legends
    for ax in axes[0, :]:
        ax.legend(loc="upper left", fontsize='small')

    # Modify the linestyles of the upper right legend to be solid
    legend = axes[0, 1].get_legend()
    for line in legend.get_lines():
        line.set_linestyle('solid')

    # Create custom lines for the legend
    solid_line = mlines.Line2D([], [], color='black', linestyle='solid', label=r'$\beta = 10^{-3}$', linewidth=default_linewidth)
    dashed_line = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'$\beta = 1$', linewidth=default_linewidth)
    
    # Add the custom legend in the upper left panel
    axes[0, 0].legend(handles=[solid_line, dashed_line], loc="upper left", fontsize='small')

    # Save the plots as a PDF
    pdf_filename = "combined_simulations_plots_res.pdf"
    with PdfPages(pdf_filename) as pdf:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Plots saved as {pdf_filename}")

if __name__ == "__main__":
    simulation_dirs = [
    "cos_b1d0_us_St1dm1_Z1dm3_r6H_z08H_stnew_MR400",
    "cos_b1dm3_us_St1dm1_Z1dm3_r6H_z08H_stnew_MR400",
    "cos_b1dm3_us_St1dm1_Z1dm1_r6H_z08H_stnew_MR400_tap",
    "cos_b1dm3_us_St1dm1_Z1dm2_r6H_z08H_stnew_MR400",
    "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_stnew_MR400_tap",
    "cos_b1dm3_us_St1dm1_Z5dm2_r6H_z08H_stnew_MR400_tap",
    "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_stnew_MR400",
    "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_stnew_MR400_tap",
    "cos_bet1dm3_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1dm3_St1dm1_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1dm3_St1dm1_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1d0_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1d0_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1d0_St1dm1_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1d0_St1dm1_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap"
        # Add the remaining directories here
    ]
    plot_results(simulation_dirs)


