import os
import numpy as np
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from data_storage import determine_base_path
from plot_simulations_2D_tau import apply_smoothing
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, NullFormatter


def format_stokes_label(stokes_label):
    stokes_value = stokes_label[2:].replace('dm', 'e-')
    return f"$\\tau={float(stokes_value):.0e}$"

def format_metallicity_label(metallicity_label):
    metallicity_value = metallicity_label[1:].replace('dm', 'e-')
    return f"$Z={float(metallicity_value):.0e}$"

def load_simulation_data(simulation_dirs, set_type):
    data = {}
    for sim_dir in simulation_dirs:
        subdir_path = determine_base_path(sim_dir)

        if set_type == 'set1':
            stokes_str = [part for part in sim_dir.split('_') if part.startswith('St')][0]
            tau_label = format_stokes_label(stokes_str)

            npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
            loaded_data = np.load(npz_file)

            # Store data as before
            data[tau_label] = {
                'time': loaded_data['time'],
                'max_epsilon': loaded_data['max_epsilon'],
                'vort_min': loaded_data['vort_min'],
                'roche_times': loaded_data.get('roche_times', np.array([]))  # Default to empty array if roche_times is not available
            }

            # Check if "xgrid_masked" exists in the file and print its min and max values
            if 'xgrid_masked' in loaded_data:
                xgrid_masked = loaded_data['xgrid_masked']
                min_xgrid = np.min(xgrid_masked)
                max_xgrid = np.max(xgrid_masked)
                print(f"Minimum value of xgrid_masked: {min_xgrid}")
                print(f"Maximum value of xgrid_masked: {max_xgrid}")
            else:
                print("xgrid_masked not found in the .npz file.")

        elif set_type == 'set2':
            metallicity_str = [part for part in sim_dir.split('_') if part.startswith('Z')][0]
            Z_label = format_metallicity_label(metallicity_str)

            npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
            loaded_data = np.load(npz_file)

            data[Z_label] = {
                'time': loaded_data['time'],
                'max_epsilon': loaded_data['max_epsilon'],
                'vort_min': loaded_data['vort_min'],
                'roche_times': loaded_data.get('roche_times', np.array([]))  # Default to empty array if roche_times is not available
            }

            # Check if "xgrid_masked" exists in the file and print its min and max values
            if 'xgrid_masked' in loaded_data:
                xgrid_masked = loaded_data['xgrid_masked']
                min_xgrid = np.min(xgrid_masked)
                max_xgrid = np.max(xgrid_masked)
                print(f"Minimum value of xgrid_masked: {min_xgrid}")
                print(f"Maximum value of xgrid_masked: {max_xgrid}")
            else:
                print("xgrid_masked not found in the .npz file.")
    return data


def plot_results(simulations_set1, simulations_set2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Adjusted figure size

    # Simulation data
    data_set1 = load_simulation_data(simulations_set1, set_type='set1')
    data_set2 = load_simulation_data(simulations_set2, set_type='set2')

    max_eps_set1, roche_time_set1 = [], []
    max_eps_set2, roche_time_set2 = [], []
    tau_set1, Z_set2 = [], []

    # Color normalization for Roche exceeding time
    norm = Normalize(vmin=200, vmax=1000)
    cmap = plt.cm.viridis

    # Plot Set 1 (circles) associated with tau (top axis)
    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim([1.0e-2, 1.0e-1])  # Fixed range for tau
    ax_top.set_xlabel(r"$\tau$", fontsize=18)
    ax_top.tick_params(axis='x', which='both', labelsize=18)

    # Use LogLocator to control tick positions and avoid overlaps
    log_locator = LogLocator(base=10.0, subs=(1.0, 5.0), numticks=10)  # Only 1, 2, and 5 ticks
    ax_top.xaxis.set_major_locator(log_locator)
    ax_top.xaxis.set_minor_formatter(NullFormatter())  # Hide minor tick labels

    # Set tick label font size
    ax_top.tick_params(axis='x', which='both', labelsize=18)

    for sim_label, data in data_set1.items():
        max_epsilon = data['max_epsilon']
        roche_times = data.get('roche_times', None)
        tau_value = float(sim_label.split('=')[1].replace('$', '').strip())

        max_eps_set1.append(np.max(max_epsilon))
        tau_set1.append(tau_value)

        if roche_times is not None and roche_times.size > 0:
            first_roche_time = roche_times.min()
            roche_time_set1.append(first_roche_time)
            color = cmap(norm(first_roche_time))
            marker_style = 'o'
            fill_style = color
        else:
            roche_time_set1.append(None)
            color = 'black'
            marker_style = 'o'
            fill_style = 'none'

        ax_top.scatter(tau_value, np.max(max_epsilon), color=color, marker=marker_style, edgecolor='black', facecolors=fill_style, s=200, zorder=3)

    # Connect points for Set 1 (tau)
    ax_top.plot(tau_set1, max_eps_set1, linestyle='--', color='gray', linewidth=1)

    # Plot Set 2 (squares) associated with Z (bottom axis)
    for sim_label, data in data_set2.items():
        max_epsilon = data['max_epsilon']
        roche_times = data.get('roche_times', None)
        Z_value = float(sim_label.split('=')[1].replace('$', '').strip())

        max_eps_set2.append(np.max(max_epsilon))
        Z_set2.append(Z_value)

        if roche_times is not None and roche_times.size > 0:
            first_roche_time = roche_times.min()
            roche_time_set2.append(first_roche_time)
            color = cmap(norm(first_roche_time))
            marker_style = 's'
            fill_style = color
        else:
            roche_time_set2.append(None)
            color = 'black'
            marker_style = 's'
            fill_style = 'none'

        ax.scatter(Z_value, np.max(max_epsilon), color=color, marker=marker_style, edgecolor='black', facecolors=fill_style, s=200, zorder=3)

    # Connect points for Set 2 (Z)
    ax.plot(Z_set2, max_eps_set2, linestyle=':', color='gray', linewidth=1)

    # Configure the main (bottom) axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$Z$", fontsize=18)
    ax.set_ylabel(r"$\epsilon_{\max}$", fontsize=18)
    ax.tick_params(axis='both', which='both', labelsize=18)
    ax.set_xlim([1.0e-3, 1.0e-1])  # Set the range for the bottom Z-axis

    # Add colorbar and adjust its position
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, 
        ax=ax, 
        orientation='vertical', 
        pad=0.02, 
        shrink=1.0, 
        anchor=(1.8, 0.2)  # Adjust for bottom-right placement
    )
    cbar.set_label(r"Clumping time", fontsize=18)  # Explicit label font size
    cbar.ax.tick_params(labelsize=18)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Reduced top margin for better fit
    plt.savefig("final_results_tau_and_z_corrected.pdf", bbox_inches="tight")
    plt.close()

    # Optional: SCP transfer
    scp_transfer("final_results_tau_and_z_corrected.pdf", "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


if __name__ == "__main__":
    simulations_set1 = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St2dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St3dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St4dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack"
    ]



    simulations_set2 = [
        "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z3dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z5dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]

#python3 plot_fargo.py cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap



#isothermal simulations = [
#"cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
#"cos_b1dm3_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
#"cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
#"cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
#]



    plot_results(simulations_set1, simulations_set2)
