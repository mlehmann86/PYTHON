import numpy as np
import os
import matplotlib.pyplot as plt
from plot_fargo import determine_base_path
from scipy.ndimage import uniform_filter1d  # For smoothing

# Helper function to extract simulation metadata
def extract_simulation_metadata(simulation_name):
    if "Z1dm4" in simulation_name:
        metallicity = r'$Z=0.0001$'
    elif "Z1dm3" in simulation_name:
        metallicity = r'$Z=0.001$'
    elif "Z1dm2" in simulation_name:
        metallicity = r'$Z=0.01$'
    elif "Z2dm2" in simulation_name:
        metallicity = r'$Z=0.02$'
    elif "Z5dm2" in simulation_name:
        metallicity = r'$Z=0.05$'
    elif "Z3dm2" in simulation_name:
        metallicity = r'$Z=0.03$'
    elif "Z1dm1" in simulation_name:
        metallicity = r'$Z=0.1$'
    else:
        metallicity = r'$Z=0$'

    stokes_dict = {
        "St1dm2": r"$\tau=0.01$",
        "St5dm2": r"$\tau=0.05$",
        "St1dm1": r"$\tau=0.1$"
    }
    stokes_number = next((v for k, v in stokes_dict.items() if k in simulation_name), r"$\tau=0$")

    beta_dict = {
        "b1d0": r"$\beta=1$",
        "b1dm3": r"$\beta=0.001$",
    }
    cooling_time = next((v for k, v in beta_dict.items() if k in simulation_name), r"$\beta=\mathrm{unknown}$")

    return metallicity, stokes_number, cooling_time

# Function to load simulation data
def load_simulation_data(sim_dir):
    subdir_path = determine_base_path(sim_dir)
    if isinstance(subdir_path, tuple):
        subdir_path = subdir_path[0]
    npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
    print(f"Loading file: {npz_file}")

    try:
        loaded_data = np.load(npz_file)
        time = loaded_data['time']
        vort_min = loaded_data['vort_min']
        return time, vort_min
    except Exception as e:
        print(f"Error loading file {npz_file}: {e}")
        return None, None

# Function to smooth data
def smooth_data(data, window_size=10):
    return uniform_filter1d(data, size=window_size)

# Function to plot vorticity data
def plot_vorticity(simulations, output_file="vorticity_time_plot.pdf"):
    plt.figure(figsize=(10, 6))

    # Extract unique values for metadata
    unique_metallicities = list(set(extract_simulation_metadata(sim)[0] for sim in simulations))
    unique_stokes_numbers = list(set(extract_simulation_metadata(sim)[1] for sim in simulations))
    unique_cooling_times = list(set(extract_simulation_metadata(sim)[2] for sim in simulations))

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_metallicities)))  # Color palette
    markers = ['o', 's'][:len(unique_stokes_numbers)]  # Limit markers to number of unique Stokes numbers
    linestyles = ['-', '--'][:len(unique_cooling_times)]  # Limit linestyles to number of unique cooling times

    # Set font sizes
    plt.rcParams.update({'font.size': 14})

    # Plot each simulation
    for sim_name in simulations:
        time, vort_min = load_simulation_data(sim_name)

        if time is None or vort_min is None:
            continue
        metallicity, stokes_number, cooling_time = extract_simulation_metadata(sim_name)

        # Determine indices for metadata
        color_idx = unique_metallicities.index(metallicity)
        marker_idx = unique_stokes_numbers.index(stokes_number)
        linestyle_idx = unique_cooling_times.index(cooling_time)

        # Smooth data
        vort_min_smoothed = smooth_data(vort_min)

        # Plot data with metadata
        label = f"{metallicity}, {stokes_number}, {cooling_time}"
        plt.plot(time, vort_min_smoothed, label=label, linestyle=linestyles[linestyle_idx],
                 marker=markers[marker_idx], markevery=50, color=colors[color_idx])

    # Add labels, title, and legend
    plt.xlabel("Time", fontsize=16)
    plt.ylabel(r"$\min (\langle \omega_z \rangle_z)$", fontsize=16)
    plt.title("Vorticity Evolution Over Time", fontsize=18)
    plt.legend(fontsize=12, loc="best", frameon=True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    simulations = [
        "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1dm3_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
    ]

    simulations = [
        "cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack"
    ]

    plot_vorticity(simulations)
       
