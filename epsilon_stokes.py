import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from plot_fargo import determine_base_path

# Helper function to extract simulation metadata and correlation times
correlation_times = {
    "cos_bet1d0_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew": 0.2,
    "cos_bet1d0_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.15,
    "cos_bet1d0_St3dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.15,
    "cos_bet1d0_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.15,
    "cos_bet1d0_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.15,
    "cos_bet1d0_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.15,
    "cos_bet1d0_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew": 0.1,
    "cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew": 0.40,
    "cos_bet1dm3_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.30,
    "cos_bet1dm3_St3dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.25,
    "cos_bet1dm3_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.20,
    "cos_bet1dm3_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.20,
    "cos_bet1dm3_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.20,
    "cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew": 0.15,
##########NEED TO BE REDONE################
    "cos_bet1dm3_St1dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.45,
    "cos_bet1dm3_St2dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.35,
    "cos_bet1dm3_St4dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.30,
    "cos_bet1dm3_St6dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.25,
    "cos_bet1dm3_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack": 0.20
}

# Updated extract_simulation_metadata function
def extract_simulation_metadata(simulation_name):
    # Extract Stokes number (\tau)
    if "St1dm2" in simulation_name:
        tau = 0.01
    elif "St2dm2" in simulation_name:
        tau = 0.02
    elif "St3dm2" in simulation_name:
        tau = 0.03
    elif "St4dm2" in simulation_name:
        tau = 0.04
    elif "St6dm2" in simulation_name:
        tau = 0.06
    elif "St8dm2" in simulation_name:
        tau = 0.08
    elif "St1dm1" in simulation_name:
        tau = 0.1
    else:
        tau = None

    # Extract metallicity (Z)
    if "Z1dm3" in simulation_name:
        Z = 0.001
    elif "Z1dm2" in simulation_name:
        Z = 0.01
    else:
        Z = None

    # Extract cooling time (\beta)
    beta = "0.001" if "bet1dm3" in simulation_name else "1"
    return tau, beta, Z

def load_mean_epsilon_max_corrected(sim_dir):
    subdir_path = determine_base_path(sim_dir)

    # Ensure subdir_path is a single path (extract the first element if it's a tuple)
    if isinstance(subdir_path, tuple):
        subdir_path = subdir_path[0]

    npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
    print(f"Loading: {npz_file}")
    loaded_data = np.load(npz_file)
    print("Keys in the .npz file:", loaded_data.files)

    try:
        loaded_data = np.load(npz_file)
        #max_epsilon = loaded_data['max_epsilon']
        max_epsilon = loaded_data['max_epsilon_avg']
        rms_vz = loaded_data['rms_vz']
        avg_metallicity = loaded_data['Z_avg']
        time = loaded_data['time']

        # Select the last 500 orbits; subsequent outputs are 8 orbits apart
        last_500_indices = time >= (time[-1] - 500)

        # Compute the time average of avg_metallicity over the last 500 orbits
        avg_metallicity_last_500 = np.mean(avg_metallicity[last_500_indices])

        # Compute the correction factor
        correction_factor = avg_metallicity[0] / avg_metallicity_last_500
        print(f"Simulation: {sim_dir}, Correction Factor: {correction_factor:.3f}")

        # Correct max_epsilon
        mean_max_epsilon_corrected = np.mean(max_epsilon[last_500_indices]) * correction_factor

        H_g=0.1
        mean_rms_vz = np.mean(rms_vz[last_500_indices])/H_g

        return mean_max_epsilon_corrected, mean_rms_vz
    except FileNotFoundError:
        print(f"File not found: {npz_file}")
        return None, None


def plot_2x2_panel(simulations, output_file="2x2_panel_figure_with_labels.pdf"):
    # Initialize data series
    tau_values_beta_001_z_001, tau_values_beta_001_z_01, tau_values_beta_1 = [], [], []
    max_eps_values_beta_001_z_001, max_eps_values_beta_001_z_01, max_eps_values_beta_1 = [], [], []
    rms_vz_values_beta_001_z_001, rms_vz_values_beta_001_z_01, rms_vz_values_beta_1 = [], [], []
    t_eddy_values_beta_001_z_001, t_eddy_values_beta_001_z_01, t_eddy_values_beta_1 = [], [], []
    dz_values_beta_001_z_001, dz_values_beta_001_z_01, dz_values_beta_1 = [], [], []

    # Process simulations
    for sim_name in simulations:
        tau, beta, Z = extract_simulation_metadata(sim_name)
        mean_max_epsilon_corrected, mean_rms_vz = load_mean_epsilon_max_corrected(sim_name)

        if tau is not None and mean_max_epsilon_corrected is not None and mean_rms_vz is not None:
            t_eddy = correlation_times.get(sim_name, None)
            if beta == "0.001" and Z == 0.001:
                tau_values_beta_001_z_001.append(tau)
                max_eps_values_beta_001_z_001.append(mean_max_epsilon_corrected)
                rms_vz_values_beta_001_z_001.append(mean_rms_vz)
                if t_eddy is not None:
                    t_eddy_values_beta_001_z_001.append(t_eddy)
                    dz_values_beta_001_z_001.append(mean_rms_vz**2 * t_eddy)
            elif beta == "0.001" and Z == 0.01:
                tau_values_beta_001_z_01.append(tau)
                max_eps_values_beta_001_z_01.append(mean_max_epsilon_corrected)
                rms_vz_values_beta_001_z_01.append(mean_rms_vz)
                if t_eddy is not None:
                    t_eddy_values_beta_001_z_01.append(t_eddy)
                    dz_values_beta_001_z_01.append(mean_rms_vz**2 * t_eddy)
            elif beta == "1":
                tau_values_beta_1.append(tau)
                max_eps_values_beta_1.append(mean_max_epsilon_corrected)
                rms_vz_values_beta_1.append(mean_rms_vz)
                if t_eddy is not None:
                    t_eddy_values_beta_1.append(t_eddy)
                    dz_values_beta_1.append(mean_rms_vz**2 * t_eddy)

    def perform_fit_and_plot(ax, x_data, y_data, label, color, marker):
        if len(x_data) > 1:
            try:
                popt, _ = curve_fit(lambda x, a, b: a * x**b, x_data, y_data)
                fit_x = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)
                fit_y = popt[0] * fit_x**popt[1]
                ax.plot(fit_x, fit_y, linestyle="-", color=color, label=f"Fit {label}: $\\propto \\tau^{{{popt[1]:.2f}}}$")
            except Exception as e:
                print(f"Fit failed for {label}: {e}")
        ax.scatter(x_data, y_data, color=color, marker=marker, label=f"Simulation Data {label}")

    # Plotting logic
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))


    # Adjust font sizes
    fontsize = 14
    plt.rcParams.update({'font.size': fontsize})

    # Apply consistent font size to tick labels in all subplots
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    # Panel 1: max_epsilon vs tau
    if len(tau_values_beta_001_z_001) > 0:
        constant_beta_001_z_001 = max_eps_values_beta_001_z_001[0] / np.sqrt(tau_values_beta_001_z_001[0] / dz_values_beta_001_z_001[0])
        #constant_beta_001_z_001 = 0.01*0.8/np.sqrt(2*np.pi)
        eps_fit_001_z_001 = [
            constant_beta_001_z_001 * np.sqrt(tau / dz) 
            for tau, dz in zip(tau_values_beta_001_z_001, dz_values_beta_001_z_001)
        ]
    #    axs[0, 0].plot(tau_values_beta_001_z_001, eps_fit_001_z_001, linestyle="--", color="blue")

    if len(tau_values_beta_001_z_01) > 0:
        constant_beta_001_z_01 = max_eps_values_beta_001_z_01[0] / np.sqrt(tau_values_beta_001_z_01[0] / dz_values_beta_001_z_01[0])
        #constant_beta_001_z_01 = 0.01*0.8/np.sqrt(2*np.pi)
        eps_fit_001_z_01 = [
            constant_beta_001_z_01 * np.sqrt(tau / dz) 
            for tau, dz in zip(tau_values_beta_001_z_01, dz_values_beta_001_z_01)
        ]
        axs[0, 0].plot(tau_values_beta_001_z_01, eps_fit_001_z_01, linestyle="--", color="cyan")

    if len(tau_values_beta_1) > 0:
        constant_beta_1 = max_eps_values_beta_1[0] / np.sqrt(tau_values_beta_1[0] / dz_values_beta_1[0])
        eps_fit_1 = [
            constant_beta_1 * np.sqrt(tau / dz) 
            for tau, dz in zip(tau_values_beta_1, dz_values_beta_1)
        ]
        axs[0, 0].plot(tau_values_beta_1, eps_fit_1, linestyle="--", color="orange")

    # Plot Simulation Data
    #perform_fit_and_plot(axs[0, 0], tau_values_beta_001_z_001, max_eps_values_beta_001_z_001, r"$\beta=0.001$", "blue", "o")
    perform_fit_and_plot(axs[0, 0], tau_values_beta_001_z_01, max_eps_values_beta_001_z_01, r"$\beta=0.001$", "cyan", "s")
    perform_fit_and_plot(axs[0, 0], tau_values_beta_1, max_eps_values_beta_1, r"$\beta=1$", "orange", "d")

    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylabel(r"$\langle \mathrm{max}(\epsilon) \rangle_t$", fontsize=fontsize)
    axs[0, 0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Custom Legend
    axs[0, 0].legend(
        handles=[
            plt.Line2D([], [], color="cyan", marker="s", linestyle="", label=r"$\beta=0.001 + \propto \tau^{0.22}$"),
            plt.Line2D([], [], color="orange", marker="d", linestyle="", label=r"$\beta=1 + \propto \tau^{0.25}$"),
            plt.Line2D([], [], color="black", linestyle="--", label=r"$\propto \sqrt{\tau / D_z}$")
        ],
        loc="upper left",
        fontsize=fontsize
    )

    # Upper Right Panel: rms_vz vs tau
    #perform_fit_and_plot(axs[0, 1], tau_values_beta_001_z_001, rms_vz_values_beta_001_z_001, r"$\beta=0.001, Z=0.001$", "blue", "o")
    perform_fit_and_plot(axs[0, 1], tau_values_beta_001_z_01, rms_vz_values_beta_001_z_01, r"$\beta=0.001, Z=0.01$", "cyan", "s")
    perform_fit_and_plot(axs[0, 1], tau_values_beta_1, rms_vz_values_beta_1, r"$\beta=1$", "orange", "d")
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_ylabel(r"$\langle \mathrm{RMS}(v_z)/c_{0} \rangle_t$", fontsize=fontsize)
    axs[0, 1].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Lower Left Panel: t_eddy vs tau
    #perform_fit_and_plot(axs[1, 0], tau_values_beta_001_z_001, t_eddy_values_beta_001_z_001, r"$\beta=0.001, Z=0.001$", "blue", "o")
    perform_fit_and_plot(axs[1, 0], tau_values_beta_001_z_01, t_eddy_values_beta_001_z_01, r"$\beta=0.001, Z=0.01$", "cyan", "s")
    perform_fit_and_plot(axs[1, 0], tau_values_beta_1, t_eddy_values_beta_1, r"$\beta=1$", "orange", "d")
    axs[1, 0].set_xscale("log")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel(r"$\tau$", fontsize=fontsize)
    axs[1, 0].set_ylabel(r"$t_{\mathrm{eddy}} \, \Omega_0$", fontsize=fontsize)
    axs[1, 0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Lower Right Panel: D_z vs tau
    #perform_fit_and_plot(axs[1, 1], tau_values_beta_001_z_001, dz_values_beta_001_z_001, r"$\beta=0.001, Z=0.001$", "blue", "o")
    perform_fit_and_plot(axs[1, 1], tau_values_beta_001_z_01, dz_values_beta_001_z_01, r"$\beta=0.001, Z=0.01$", "cyan", "s")
    perform_fit_and_plot(axs[1, 1], tau_values_beta_1, dz_values_beta_1, r"$\beta=1$", "orange", "d")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel(r"$\tau$", fontsize=fontsize)
    axs[1, 1].set_ylabel(r"$D_z / (H_{g0}^2 \, \Omega_0)$", fontsize=fontsize)
    axs[1, 1].grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"2x2 panel figure saved to {output_file}")

if __name__ == "__main__":
    simulations = [
        "cos_bet1d0_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1d0_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1d0_St3dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1d0_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1d0_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1d0_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1d0_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1dm3_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St3dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1dm3_St1dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St2dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St4dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St6dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack",
        "cos_bet1dm3_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack"
    ]
    plot_2x2_panel(simulations)


