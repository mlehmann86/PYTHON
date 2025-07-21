import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from data_storage import scp_transfer
from data_reader import read_parameters  # Importing read_parameters from data_reader
from scipy.ndimage import uniform_filter1d
from plot_simulations_2D_tau import apply_smoothing

def load_and_compute_radial_profiles(simulations, base_paths, output_path, filename_suffix, N=2, font_size=18, line_thickness=2):
    """
    Compute and plot radial profiles of various quantities for multiple simulations across different base paths.
    
    Parameters:
    - simulations: List of simulation subdirectory names.
    - base_paths: List of base paths where simulations might be stored.
    - output_path: Directory to save the plot.
    - filename_suffix: Suffix to append to the output filename.
    - N: Number of snapshots to average over.
    - font_size: Font size for the plot.
    - line_thickness: Line thickness for plot lines.
    """

    # Set consistent font sizes for all elements
    label_fontsize = 18
    tick_fontsize = 18
    legend_fontsize = 14

    # Font and axis settings
    plt.rcParams.update({
        'font.size': tick_fontsize,
        'lines.linewidth': line_thickness,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2
    })

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    smoothing_window = 5
    start_radius = 0.8

    for idx, sim in enumerate(simulations):
        found_file = False
        metallicity = None

        for base_path in base_paths:
            sim_path = os.path.join(base_path, sim)
            summary_file = os.path.join(sim_path, "summary0.dat")
            if os.path.exists(summary_file):
                found_file = True
                parameters = read_parameters(summary_file)
                metallicity = float(parameters.get('METALLICITY', 0.0))
                aspectratio = parameters['ASPECTRATIO']
                break

        if not found_file or metallicity is None:
            print(f"Error: summary0.dat not found for {sim} in any base path!")
            continue

        # Load the initial snapshot (snapshot 0) for scaling
        data_initial = load_simulation_data(sim_path, 0)
        rho_initial = data_initial['gasdens']
        P_initial = data_initial['gasenergy']

        # Scale quantities
        P_scale = rho_initial[0, rho_initial.shape[1] // 2, rho_initial.shape[2] // 2] * aspectratio**2
        rho_scale = rho_initial[0, rho_initial.shape[1] // 2, rho_initial.shape[2] // 2]
        N2_scale = 1.0

        # Compute initial quantities and apply scaling
        dP_dr_initial_full = np.gradient(P_initial, data_initial['xgrid'], axis=1)
        S_initial = np.log(P_initial / rho_initial**parameters['GAMMA'])
        dS_dr_initial_full = np.gradient(S_initial, data_initial['xgrid'], axis=1)
        N2_initial_full = -(1 / parameters['GAMMA']) * (1 / rho_initial) * dP_dr_initial_full * dS_dr_initial_full

        dP_dr_initial_avg = np.mean(dP_dr_initial_full, axis=(0, 2)) / P_scale
        rho_initial_avg = np.mean(rho_initial, axis=(0, 2)) / rho_scale
        N2_initial_avg = np.mean(N2_initial_full, axis=(0, 2)) / N2_scale

        # Initialize accumulators for time-averaged quantities
        dP_dr_final_sum = np.zeros_like(dP_dr_initial_avg)
        rho_final_sum = np.zeros_like(rho_initial_avg)
        N2_final_sum = np.zeros_like(N2_initial_avg)

        # Find the last N snapshots
        nt = determine_nt(sim_path)
        snapshots = find_last_snapshots(sim_path, nt - 1, N)
        if not snapshots:
            print(f"No snapshots found for {sim} in {sim_path}")
            continue

        # Loop over the selected snapshots
        for t in snapshots:
            data = load_simulation_data(sim_path, t)
            rho = data['gasdens']
            P = data['gasenergy']
            dP_dr_full = np.gradient(P, data['xgrid'], axis=1)
            S = np.log(P / rho**parameters['GAMMA'])
            dS_dr_full = np.gradient(S, data['xgrid'], axis=1)
            N2_full = -(1 / parameters['GAMMA']) * (1 / rho) * dP_dr_full * dS_dr_full

            # Compute and accumulate scaled quantities
            dP_dr_final_sum += np.mean(dP_dr_full, axis=(0, 2)) / P_scale
            rho_final_sum += np.mean(rho, axis=(0, 2)) / rho_scale
            N2_final_sum += np.mean(N2_full, axis=(0, 2)) / N2_scale

        # Compute the final time-averaged profiles
        dP_dr_final = dP_dr_final_sum / N
        rho_final = rho_final_sum / N
        N2_final = N2_final_sum / N

        # Plot the time-averaged profiles
        label = f"Z={metallicity:.4f}".rstrip('0').rstrip('.')
        axs[0].plot(data_initial['xgrid'], dP_dr_final, '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)
        axs[1].plot(data_initial['xgrid'], rho_final, '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)
        axs[2].plot(data_initial['xgrid'], uniform_filter1d(N2_final, smoothing_window), '-', color=colors[idx], label=f'{label}', linewidth=line_thickness)

        # Overplot the initial profiles
        axs[0].plot(data_initial['xgrid'], dP_dr_initial_avg, 'k--', label='Initial', linewidth=line_thickness)
        axs[1].plot(data_initial['xgrid'], rho_initial_avg, 'k--', label='Initial', linewidth=line_thickness)
        axs[2].plot(data_initial['xgrid'], uniform_filter1d(N2_initial_avg, smoothing_window), 'k--', label='Initial', linewidth=line_thickness)

    # Set axis labels and limits
    axs[0].set_ylabel(r'$\langle \frac{dP}{dr} \rangle$', fontsize=label_fontsize)
    axs[1].set_ylabel(r'$\langle \rho_g \rangle$', fontsize=label_fontsize)
    axs[2].set_ylabel(r'$\langle N^2 \rangle$', fontsize=label_fontsize)
    axs[2].set_xlabel(r'$r$', fontsize=label_fontsize)

    axs[0].set_xlim([0.8, 1.2])

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True)

    axs[0].legend(loc='best', fontsize=legend_fontsize)

    pdf_filename = f"radial_profiles_comparison_{filename_suffix}.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath, bbox_inches="tight")
    plt.close()
    print(f"Radial profile plot saved to {output_filepath}")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

def main():
    # Use argparse to allow the user to specify which set of simulations to run
    parser = argparse.ArgumentParser(description="Load and plot radial profiles for different simulations.")
    parser.add_argument('--raettig', action='store_true', help="Use Raettig simulations.")
    parser.add_argument('--mitigation', action='store_true', help="Use Mitigation simulations.")
    parser.add_argument('--other', action='store_true', help="Use other simulations.")

    args = parser.parse_args()

    base_paths = [
        "/tiara/home/mlehmann/data/FARGO3D/outputs",
        "/theory/lts/mlehmann/FARGO3D/outputs"
    ]

    output_path = os.getcwd()  # Save to current directory

    # Define the simulation sets
    if args.raettig:
        simulations = [
            "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
            "cos_b1d0_us_St5dm2_Z1dm4_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St5dm2_Z1dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150"
        ]
        filename_suffix = "raettig"
    elif args.mitigation:
       # simulations = [
       #     "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
       #     "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
       #     "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
       #     "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
       #     "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
       #     "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
       # ]

        simulations = [
            "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
            "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
        ]


        filename_suffix = "mitigation"
    elif args.other:
        simulations = [
            # Add a different set of simulations here if needed
        ]
        filename_suffix = "other"
    else:
        print("Please select one of the simulation sets with --raettig, --mitigation, or --other.")
        return

    # Load and plot the selected simulation set
    load_and_plot_radial_profiles(simulations, base_paths, output_path, filename_suffix)

if __name__ == "__main__":
    main()
