import os
import numpy as np
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from data_storage import determine_base_path
from plot_simulations_2D_tau import apply_smoothing

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

            data[tau_label] = {
                'time': loaded_data['time'],
                'max_epsilon': loaded_data['max_epsilon'],
                'vort_min': loaded_data['vort_min'],
                'roche_times': loaded_data.get('roche_times', np.array([]))  # Default to empty array if roche_times is not available
            }

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
    return data

def plot_results(simulations_set1, simulations_set2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    smoothing_window = 10
    start_time = 40

    label_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 14

    plt.rcParams.update({
        'font.size': tick_fontsize,
        'lines.linewidth': 2,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2
    })

    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    data_set1 = load_simulation_data(simulations_set1, set_type='set1')
    data_set2 = load_simulation_data(simulations_set2, set_type='set2')

    max_eps_set1, vort_min_set1, tau_set1 = [], [], []  # Initialize arrays for Set 1
    max_eps_set2, vort_min_set2, Z_set2 = [], [], []    # Initialize arrays for Set 2
    roche_exceeded_set1, roche_exceeded_set2 = [], []   # Initialize arrays for Roche checks

    # Panel 1 and 2: Plotting each set separately
    for idx, (sim_label, data) in enumerate(data_set1.items()):
        time = data['time']
        max_epsilon = apply_smoothing(data['max_epsilon'], time, smoothing_window, start_time)
        vort_min = apply_smoothing(data['vort_min'], time, smoothing_window, start_time)
        roche_times = data.get('roche_times', None)

        color = consistent_colors[idx % len(consistent_colors)]
        axes[0].plot(time, max_epsilon, color=color)  # Plot on both panels
        axes[1].plot(time, vort_min, color=color)

        #max_eps_set1.append(np.max(max_epsilon))   # Store max epsilon
        #vort_min_set1.append(np.min(vort_min))     # Store min vorticity
        max_eps_set1.append(np.max(max_epsilon))   # Store max epsilon
        vort_min_set1.append(np.mean(vort_min))     # Store mean vorticity


        tau_value = sim_label.split('=')[1].replace('$', '').strip()
        tau_set1.append(float(tau_value))          # Store tau values

        # Print debugging values
        print(f"Set 1, {sim_label}: max_eps = {np.max(max_epsilon)}, vort_min = {np.min(vort_min)}")

        # Check for Roche density exceedance
        roche_exceeded_set1.append(roche_times is not None and roche_times.size > 0)

    for idx, (sim_label, data) in enumerate(data_set2.items()):
        time = data['time']
        max_epsilon = apply_smoothing(data['max_epsilon'], time, smoothing_window, start_time)
        vort_min = apply_smoothing(data['vort_min'], time, smoothing_window, start_time)
        roche_times = data.get('roche_times', None)

        color = consistent_colors[idx % len(consistent_colors)]
        axes[0].plot(time, max_epsilon, color=color, linestyle='--')  # Plot on both panels
        axes[1].plot(time, vort_min, color=color, linestyle='--')

        #max_eps_set2.append(np.max(max_epsilon))   # Store max epsilon
        #vort_min_set2.append(np.min(vort_min))     # Store min vorticity
        max_eps_set2.append(np.max(max_epsilon))   # Store max epsilon
        vort_min_set2.append(np.mean(vort_min))     # Store mean  vorticity


        Z_value = sim_label.split('=')[1].replace('$', '').strip()
        Z_set2.append(float(Z_value))              # Store Z values

        # Print debugging values
        print(f"Set 2, {sim_label}: max_eps = {np.max(max_epsilon)}, vort_min = {np.min(vort_min)}")

        # Check for Roche density exceedance
        roche_exceeded_set2.append(roche_times is not None and roche_times.size > 0)

    axes[0].set_yscale('log')
    axes[0].set_ylabel(r"$\epsilon_{\max}$ (log scale)")
    axes[1].set_ylabel(r"Vorticity (min)")
    axes[0].set_xlabel('Time (Orbits)')
    axes[1].set_xlabel('Time (Orbits)')

    # Manually create legends for Set 1 and Set 2
    legend_1 = [plt.Line2D([0], [0], color=consistent_colors[i % len(consistent_colors)], lw=2) for i in range(len(data_set1))]
    legend_2 = [plt.Line2D([0], [0], color=consistent_colors[i % len(consistent_colors)], lw=2, linestyle='--') for i in range(len(data_set2))]

    axes[0].legend(legend_1, list(data_set1.keys()), fontsize=legend_fontsize, title="Set 1: $\\tau$", loc="upper left")
    axes[1].legend(legend_2, list(data_set2.keys()), fontsize=legend_fontsize, title="Set 2: Z", loc="upper left")

    from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter

    # Third panel: Plot max_eps and vort_min for each set

    # Bottom axis (for Z) and right vertical axis for set 2
    ax_bottom = axes[2]
    ax_bottom_right = ax_bottom.twinx()

    # Top axis (for tau) and right vertical axis for set 1
    ax_top = ax_bottom.twiny()  # Link this to the bottom axis but handle separately
    ax_top_right = ax_top.twinx()  # New right y-axis for set 1

    # Set specific limits for both axes
    tau_xlim = [0.01, 0.1]  # Tau range
    Z_xlim = [1e-3, 1e-1]   # Z range

    # Plot for Set 1 (using tau for the top axis)
    for idx, (tau, max_eps, vort_min, roche_exceeded) in enumerate(zip(tau_set1, max_eps_set1, vort_min_set1, roche_exceeded_set1)):
        color = consistent_colors[idx % len(consistent_colors)]
        marker_style = 'o'
        fill_style = color if roche_exceeded else 'none'  # Use color for filled, 'none' for open

        # Plot max_eps against tau (top axis)
        ax_top.scatter(tau, max_eps, color=color, marker=marker_style, facecolors=fill_style, s=200, zorder=3)
        ax_top.plot(tau_set1, max_eps_set1, color=color, linestyle='--', zorder=2)  # Connect points with a dashed line

        
        # Plot vort_min against the new right axis for tau (top axis)
        ax_top_right.scatter(tau, vort_min, color=color, marker='s', facecolors=fill_style, s=200, zorder=3)
        ax_top_right.plot(tau_set1, vort_min_set1, color=color, linestyle='--', zorder=2)  # Connect points with a dashed line


    # Plot for Set 2 (using Z for the bottom axis)
    for idx, (Z, max_eps, vort_min, roche_exceeded) in enumerate(zip(Z_set2, max_eps_set2, vort_min_set2, roche_exceeded_set2)):
        color = consistent_colors[idx % len(consistent_colors)]
        marker_style = 'o'
        fill_style = color if roche_exceeded else 'none'  # Use color for filled, 'none' for open

        # Plot max_eps against Z (bottom axis)
        ax_bottom.scatter(Z, max_eps, color=color, marker=marker_style, facecolors=fill_style, s=100, zorder=2)
        ax_bottom.plot(Z_set2, max_eps_set2, color=color, linestyle=':', zorder=1)  # Connect points with a dotted line


        
        # Plot vort_min against the right axis for Z (bottom axis)
        ax_bottom_right.scatter(Z, vort_min, color=color, marker='s', facecolors=fill_style, s=100, zorder=2)
        ax_bottom_right.plot(Z_set2, vort_min_set2, color=color, linestyle=':', zorder=1)  # Connect points with a dotted line

    from matplotlib.ticker import LogLocator, NullFormatter, LogFormatter

    # Set the xlim for the tau axis (top axis)
    ax_top.set_xlim(tau_xlim)
    ax_top.set_xscale('log')  # Render tau-axis as logarithmic

    # Set major and minor ticks for tau axis
    ax_top.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))  # Major ticks at 10^-2, 10^-1
    ax_top.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2.0, 3.0, 5.0, 7.0], numticks=10))  # Minor ticks at 2, 3, 5, 7
    ax_top.xaxis.set_minor_formatter(NullFormatter())  # Hide minor tick labels
    ax_top.xaxis.set_major_formatter(LogFormatter(base=10.0))  # Format major ticks as powers of 10
    ax_top.set_xlabel(r"$\tau$", fontsize=label_fontsize)

    # Force the minor ticks to display
    ax_top.minorticks_on()  # Ensure minor ticks are enabled
    # Increase tick length for both major and minor ticks
    ax_top.tick_params(axis='x', which='major', length=10)  # Major ticks length
    ax_top.tick_params(axis='x', which='minor', length=6)   # Minor ticks length

    # Set axis labels and limits for the third panel
    ax_bottom.set_xlabel("Z", fontsize=label_fontsize)
    ax_bottom.set_ylabel(r"$\epsilon_{\max}$", fontsize=label_fontsize)
    ax_bottom.set_xscale('log')  # Render Z-axis as logarithmic

    # Right y-axis for vort_min for set 2
    ax_bottom_right.set_ylabel(r"min$(<\omega_z>_z - <\omega_z>_{z\varphi})$", fontsize=label_fontsize)

    # Right y-axis for vort_min for set 1
    ax_top_right.set_ylabel(r"min$(<\omega_z>_z - <\omega_z>_{z\varphi})$", fontsize=label_fontsize)

    # Set epsilon_max to be logarithmic on the bottom axis
    ax_bottom.set_yscale('log')

    # Optional y-axis limit control
    ax_bottom.set_ylim([1e-1, 1e4])
    ax_bottom_right.set_ylim([-1., 0])
    ax_top_right.set_ylim([-1., 0])  # Match the limits for set 1

    plt.tight_layout()
    plt.savefig("comparison_max_eps_vort_min.pdf")
    plt.close()

    # Reintroduce SCP transfer
    scp_transfer("comparison_max_eps_vort_min.pdf", "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

if __name__ == "__main__":
    simulations_set1 = [
        "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St2dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St3dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St4dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150"
    ]

    simulations_set2 = [
        "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z3dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
        "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        "cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
    ]

    plot_results(simulations_set1, simulations_set2)
