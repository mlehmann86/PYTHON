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

def compute_eddy_time(rms_vz, H_d_array, H_g, tau):
    """
    Compute the eddy time t_e using the formula:
    t_e/OmK = H_g^2 / <v_z^2> * [ (H_d / H_g)^(-2) - 1 ]^(-1) * tau
    """
    factor = (H_d_array / H_g) ** (-2) - 1
    # To avoid division by zero, handle cases where factor is very small
    factor[factor <= 0] = np.inf
    t_e = (H_g / rms_vz)**2 * (1 / factor) * tau
    return t_e

def apply_smoothing(data, time_array, smoothing_window, start_time):
    # Determine the index corresponding to the start_time
    start_index = np.searchsorted(time_array, start_time)
    
    # Smooth only the data after the start_time
    smoothed_data = np.copy(data)  # Copy the original data
    smoothed_data[start_index:] = uniform_filter1d(data[start_index:], size=smoothing_window)
    
    return smoothed_data


def load_simulation_data(simulation_dirs):
    # List of possible base paths
    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]

    data = {}
    for sim_dir in simulation_dirs:
        # Try each base path
        npz_file = None
        for base_path in base_paths:
            potential_path = os.path.join(base_path, sim_dir, f"{os.path.basename(sim_dir)}_quantities.npz")
            if os.path.exists(potential_path):
                npz_file = potential_path
                break
        
        if npz_file:
            print(f"Loading: {npz_file}")
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
                'roche_times': loaded_data.get('roche_times', None)  # Handle older .npz files without roche_times
            }
        else:
            print(f"WARNING: File not found for {sim_dir} in any base path!")
            continue  # Skip to the next directory if the file is not found

    return data



def plot_results(simulation_dirs):

    default_linewidth = 2  # Adjust this value to control the thickness

    # Set the global font size
    plt.rcParams.update({'font.size': 14.5})  # Adjust the number to increase or decrease font size


    data = load_simulation_data(simulation_dirs)
    
    # Split the simulations into St1dm3 and St1dm1 categories
    st1dm3_runs = [d for d in simulation_dirs if "St1dm3" in d]
    st1dm1_runs = [d for d in simulation_dirs if "St1dm1" in d]

    smoothing_window=1
    start_time=80
    
    colors = {
    'Z1dm3': '#1f77b4',  # Blue
    'Z1dm2': '#ff7f0e',  # Orange
    'Z3dm2': '#9467bd',  # Purple
    'Z5dm2': '#2ca02c',  # Green
    'Z7dm2': '#8c564b',  # Brown
    'Z1dm1': '#d62728',  # Red
    'Z5dm3': '#17becf'   # Cyan (New entry for Z5dm3)
    } 

    linestyles = {
        'bet1d0': 'dashed',
        'bet1dm3': 'solid'
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    labels = {
        "St1dm3": r"$\tau = 10^{-3}$",
        "St1dm1": r"$\tau = 10^{-1}$"
    }

    tau_values = {"St1dm3": 0.001, "St1dm1": 0.1}
  

    # Determine vertical range limits for each row
    ranges = {
        "H_d_H_g": [np.inf, -np.inf],
        "epsilon_max": [np.inf, -np.inf],
        "rms_vz": [1e-4,2e-1]
    }
    
    # First pass to calculate the consistent y-axis ranges
    aspectratio=0.1
    for runs in [st1dm3_runs, st1dm1_runs]:
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
    
    for idx, (runs, title) in enumerate([(st1dm3_runs, labels["St1dm3"]),  (st1dm1_runs, labels["St1dm1"])]):
        for sim_dir in runs:
            metallicity = None
            for key in colors.keys():
                if key in sim_dir:
                    metallicity = key
                    print(f"Recognized metallicity: {metallicity} in {sim_dir}")  # Debug print
                    if metallicity == 'Z1dm1':
                        print(f"Processing data for Z = 0.1 in: {sim_dir}")
            
            bet = 'bet1d0' if 'bet1d0' in sim_dir else 'bet1dm3'
            linestyle = linestyles[bet]
            label = f"{metallicity.replace('Z', 'Z=')}" if bet == "bet1d0" else None  # Avoid redundant labels
            
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
            axes[2, idx].plot(time, rms_vz / aspectratio, color=colors[metallicity], linestyle=linestyle, label=label, linewidth=default_linewidth)
            axes[2, idx].set_ylim(ranges["rms_vz"])

        # Set titles and labels
        axes[0, idx].set_title(title)
        axes[0, idx].set_ylabel(r"$\langle H_d / H_g \rangle$")
        axes[1, idx].set_ylabel(r"$\epsilon_{\mathrm{max}}$")
        axes[2, idx].set_ylabel(r"RMS($v_z$)/$c_0$")
        axes[2, idx].set_xlabel("Orbits")
        axes[0, idx].set_yscale('log')
        axes[1, idx].set_yscale('log')
        axes[2, idx].set_yscale('log')

       
        for ax in axes[0, :]:  # This targets the first row (H_d/H_g plots)
            ax.set_ylim([5e-3, 5e-1])

        for ax in axes[2, :]:  # This targets the first row (RMS(v_z) plots)
            ax.set_ylim([1e-4, 2e-1])


        for ax in axes[:, idx]:
            ax.set_xlim([0, 1000])
            ax.set_yscale('log')
    
        import matplotlib.ticker as ticker

        # Enable grid lines for major ticks only and add minor ticks without grid lines
        for row in axes:
            for ax in row:
                #ax.grid(True, which='major', linestyle='--', linewidth=0.5)  # Major grid lines only
                ax.minorticks_on()  # Enable minor ticks

                # Set minor ticks specifically for logarithmic scale
                if ax.get_yscale() == 'log':
                    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
        
        # Remove vertical axis labels for right columns but keep the ticks
        if idx == 1:
            for ax in axes[:, idx]:
                ax.tick_params(axis='y', which='both', labelleft=False)  # Disable only the labels, keep ticks and grid lines

        
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
    pdf_filename = "combined_simulations_plots_tau.pdf"
    with PdfPages(pdf_filename) as pdf:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Plots saved as {pdf_filename}")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Profiles"

    # Call the scp_transfer function to transfer the plot
    scp_transfer(pdf_filename, local_directory, "mariuslehmann")





def plot_eddy_time(simulation_dirs):
    data = load_simulation_data(simulation_dirs)

    # Set a uniform font size for the entire plot
    font_size = 16
    plt.rcParams.update({'font.size': font_size})

    # Split the simulations into St1dm3 and St1dm1 categories
    st1dm3_runs = [d for d in simulation_dirs if "St1dm3" in d]
    st1dm2_runs = [d for d in simulation_dirs if "St1dm2" in d]
    st1dm1_runs = [d for d in simulation_dirs if "St1dm1" in d]

    tau_values = {"St1dm3": 0.001, "St1dm2": 0.01, "St1dm1": 0.1}
    aspectratio = 0.1  # H_g

    # Initialize the metallicities and eddy_times dictionaries
    metallicities = {'St1dm3': [], 'St1dm2': [], 'St1dm1': []}
    eddy_times = {'St1dm3': [], 'St1dm2': [], 'St1dm1': []}

    # First plot: Eddy time evolution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = {"St1dm3": r"$\tau = 10^{-3}$", "St1dm2": r"$\tau = 10^{-2}$", "St1dm1": r"$\tau = 10^{-1}$"}

    colors = {
        'Z1dm3': '#1f77b4',  # Blue
        'Z1dm2': '#ff7f0e',  # Orange
        'Z3dm2': '#9467bd',  # Purple
        'Z5dm2': '#2ca02c',  # Green
        'Z7dm2': '#8c564b',  # Brown
        'Z1dm1': '#d62728',  # Red
        'Z5dm3': '#17becf'   # Cyan
    }

    linestyles = {'bet1d0': 'dashed', 'bet1dm3': 'solid'}

    for idx, (runs, tau_label) in enumerate([(st1dm3_runs, "St1dm3"), (st1dm2_runs, "St1dm2"), (st1dm1_runs, "St1dm1")]):
        tau = tau_values[tau_label]
        for sim_dir in runs:
            metallicity = None
            for key in colors.keys():
                if key in sim_dir:
                    metallicity = key
                    print(f"Recognized metallicity: {metallicity} in {sim_dir}")
                    if metallicity not in metallicities[tau_label]:
                        metallicities[tau_label].append(metallicity)

            if metallicity is None:
                print(f"ERROR: Metallicities not recognized in {sim_dir}")
                continue

            bet = 'bet1d0' if 'bet1d0' in sim_dir else 'bet1dm3'
            linestyle = linestyles[bet]
            label = f"{metallicity.replace('Z', 'Z=')}"

            time = data[sim_dir]['time']
            rms_vz = data[sim_dir]['rms_vz']
            H_d_array = data[sim_dir]['H_d_array']

            # Compute eddy time t_e
            t_e = compute_eddy_time(rms_vz, H_d_array, aspectratio, tau)

            # Apply smoothing
            t_e_smoothed = apply_smoothing(t_e, time, smoothing_window=10, start_time=80)
            axes[idx].plot(time, t_e_smoothed, color=colors[metallicity], linestyle=linestyle, label=label, linewidth=2)
            axes[idx].set_yscale('log')

            # Set x-range based on Stokes number
            if tau_label == "St1dm3":  # For Stokes number 0.001
                axes[idx].set_xlim([0, 4000])
            elif tau_label == "St1dm2":  # For Stokes number 0.01
                axes[idx].set_xlim([0, 2000])
            elif tau_label == "St1dm1":  # For Stokes number 0.1
                axes[idx].set_xlim([0, 1000])
            
            axes[idx].set_ylim([1e-3, 1e1])
            axes[idx].set_title(labels[tau_label], fontsize=font_size)
            axes[idx].set_ylabel(r"$t_e$ (orbits)", fontsize=font_size)
            axes[idx].set_xlabel("Orbits", fontsize=font_size)

            # Compute the time average
            if tau_label == "St1dm3":  # For \tau = 0.001, use the last 200 orbits
                last_200_indices = time >= (time[-1] - 500)
                if np.sum(last_200_indices) == 0:
                    print(f"WARNING: No data points found for averaging in {sim_dir}")
                    continue
                t_e_avg = np.mean(t_e[last_200_indices])
            else:  # For \tau = 0.1, use the 200-600 orbit range
                range_indices = (time >= 200) & (time <= 800)
                if np.sum(range_indices) == 0:
                    print(f"WARNING: No data points found for averaging in {sim_dir}")
                    continue
                t_e_avg = np.mean(t_e[range_indices])

            eddy_times[tau_label].append(t_e_avg)

        axes[idx].legend(loc="upper left", fontsize=font_size - 2)  # Slightly smaller font for the legend

    # Save the first plot
    pdf_filename = "eddy_time_plots_tau.pdf"
    with PdfPages(pdf_filename) as pdf:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Eddy time plots saved as {pdf_filename}")
    scp_transfer(pdf_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

    # Second plot: Time-averaged eddy times vs. metallicity
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each tau series separately without requiring matching metallicities
    for tau_label, color, marker in [("St1dm3", 'blue', 'o'), ("St1dm2", 'green', 'o'), ("St1dm1", 'red', 's')]:
        if metallicities[tau_label] and eddy_times[tau_label]:
            # Convert metallicities to numerical values
            metallicity_values = [float(m.replace('Z', '').replace('dm', 'e-')) for m in metallicities[tau_label]]
            ax.plot(metallicity_values, eddy_times[tau_label], marker + '-', label=labels[tau_label], color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Metallicity $Z$", fontsize=font_size)
    ax.set_ylabel("Time-averaged $t_e$ (orbits)", fontsize=font_size)
    ax.legend(fontsize=font_size - 2)  # Slightly smaller font for the legend

    # Save the second plot
    pdf_filename_avg = "time_averaged_eddy_times_vs_metallicity.pdf"
    with PdfPages(pdf_filename_avg) as pdf:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Time-averaged eddy time plot saved as {pdf_filename_avg}")
    scp_transfer(pdf_filename_avg, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

if __name__ == "__main__":

    simulation_dirs = [
    "cos_bet1dm3_St1dm3_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1dm3_St1dm3_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1dm3_St1dm3_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1dm3_St1dm3_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1d0_St1dm3_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1d0_St1dm3_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1d0_St1dm3_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1d0_St1dm3_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap",
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


    simulation_dirs_2 = [
    "cos_bet1dm3_St1dm3_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm3_Z5dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm3_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm3_Z3dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm3_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm3_Z7dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm3_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
    "cos_bet1dm3_St1dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew", #OK
    "cos_bet1dm3_St1dm2_Z3dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", #OK
    "cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", #OK
    "cos_bet1dm3_St1dm2_Z3dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", #OK
    "cos_bet1dm3_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
    "cos_bet1dm3_St1dm1_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap",
    "cos_bet1dm3_St1dm1_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap"
]
    
    plot_results(simulation_dirs)

    plot_eddy_time(simulation_dirs_2)


