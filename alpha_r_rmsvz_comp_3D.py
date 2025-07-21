import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from data_storage import scp_transfer
from plot_simulations_2D_tau import apply_smoothing

def determine_base_path(subdirectory):
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

def load_simulation_data(sim_dir):
    data = {}
    subdir_path = determine_base_path(sim_dir)
    file_name = os.path.join(subdir_path, f"{os.path.basename(subdir_path)}_quantities.npz")
    
    try:
        loaded_data = np.load(file_name)
        data = {
            'time': loaded_data['time'],
            'alpha_r': loaded_data['alpha_r'],
            'rms_vz': loaded_data.get('rms_vz', None)  # Handle missing rms_vz
        }
    except KeyError as e:
        print(f"KeyError: {e} in file {file_name}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    
    return data

def extract_metallicity(sim_dir):
    """Extract metallicity from the simulation directory name."""
    metallicity_map = {
        "Z1dm2": "1*10^{-2}",
        "Z5dm2": "5*10^{-2}",
        "Z1dm1": "1*10^{-1}"
    }
    for key in metallicity_map:
        if key in sim_dir:
            return metallicity_map[key]
    return "Unknown"

def plot_results(simulations_tau_0_01_3D, simulations_tau_0_1_3D, labels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    smoothing_window = 5
    start_time = 80

    plt.rcParams.update({
        'font.size': 16,
        'lines.linewidth': 2,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2
    })

    # Define colors for metallicities
    colors = {
        '1*10^{-2}': 'blue',
        '5*10^{-2}': 'green',
        '1*10^{-1}': 'red'
    }

    # Process each simulation set for tau = 0.01 and tau = 0.1 separately
    for i, sim_3D_list in enumerate([simulations_tau_0_01_3D, simulations_tau_0_1_3D]):
        for sim_dir in sim_3D_list:
            # Load data for each individual 3D simulation
            data = load_simulation_data(sim_dir)
            time = data['time']
            alpha_r = apply_smoothing(data['alpha_r'], time, smoothing_window, start_time)
            rms_vz = apply_smoothing(data['rms_vz'], time, smoothing_window, start_time)
            metallicity = extract_metallicity(sim_dir)

            # Plot alpha_r for tau = 0.01 (Top row)
            axes[0, i].plot(time, alpha_r, label=f"Z={metallicity}", color=colors.get(metallicity, 'black'))
            axes[0, i].set_yscale('log')
            axes[0, i].set_ylabel(r"$\alpha_r$")
            axes[0, i].legend()

            # Plot rms_vz for tau = 0.01 (Bottom row)
            if rms_vz is not None:  # Check if rms_vz exists
                axes[1, i].plot(time, rms_vz, label=f"Z={metallicity}", color=colors.get(metallicity, 'black'))
            axes[1, i].set_yscale('log')
            axes[1, i].set_ylabel(r"RMS($v_z$)")
            axes[1, i].legend()

    # Set x labels and adjust layout
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (Orbits)')
    
    plt.tight_layout()
    output_filename = "alpha_r_rmsvz_comparison_3D.pdf"
    plt.savefig(output_filename)
    plt.close()

    # Transfer plot via SCP
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


# ============================== #
#       SCRIPT ENTRY POINT       #
# ============================== #

if __name__ == "__main__":
    simulations_tau_0_01_3D = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]
    simulations_tau_0_1_3D = [
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]

    labels = [r"$\tau = 0.01$", r"$\tau = 0.1$"]

    plot_results(simulations_tau_0_01_3D, simulations_tau_0_1_3D, labels)
