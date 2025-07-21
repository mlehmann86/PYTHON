import os
import numpy as np
import matplotlib.pyplot as plt
from data_reader import read_parameters
from data_reader import read_single_snapshot
from data_storage import scp_transfer

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

def plot_vertical_buoyancy(simulations, snapshot, beta_case, is_3D=True, yrange_N2=None, yrange_dustdens=None, xrange=None):

    """
    Plot the vertical buoyancy (N_z^2) and dust density profiles for multiple 2D and 3D simulations.
    Now also plots vertical shear (r d \Omega / d z)^2.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

    # Set consistent font sizes for all elements
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

    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for each case

    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]

    def get_metallicity(sim):
        # Extract metallicity from summary0.dat using read_parameters
        for base_path in base_paths:
            summary_file = os.path.join(base_path, sim, "summary0.dat")
            if os.path.exists(summary_file):
                parameters = read_parameters(summary_file)
                return float(parameters.get('METALLICITY', 0.0))  # Default to 0.0 if not found
        return None

    def compute_vertical_shear(data, xgrid, zgrid):
        """Compute vertical shear (r dΩ/dz)^2 for a given snapshot."""

        # Check if gasdens exists in the data dictionary
        if 'gasdens' not in data:
            raise KeyError("'gasdens' not found in data dictionary")

        gasdens = data['gasdens']
        dustdens = data['dust1dens']
        v_g_phi = data['gasvx']
        v_d_phi = data['dust1vx']

        # Calculate the center of mass azimuthal velocity
        v_cms_phi = (gasdens * v_g_phi + dustdens * v_d_phi) / (gasdens + dustdens)

        # Calculate omega = v_cms_phi / r
        omega = v_cms_phi / xgrid[np.newaxis, :, np.newaxis]

        # Compute the vertical gradient dΩ/dz
        d_omega_dz = np.gradient(omega, zgrid, axis=2)

        # Compute vertical shear (r dΩ/dz)^2
        vertical_shear = (xgrid[np.newaxis, :, np.newaxis] * d_omega_dz) ** 2

        return vertical_shear

    # Loop through simulations and compute buoyancy and shear
    for idx, sim in enumerate(simulations):
        print(f"Processing simulation: {sim}")
        subdir_path = determine_base_path(sim)
        data, xgrid, ygrid, zgrid, params = read_single_snapshot(subdir_path, snapshot, 
                                                                 read_dust1dens=True, read_gasdens=True, 
                                                                 read_gasvx=True, read_dust1vx=True)
        
        dustdens = data['dust1dens']  # Dust density from simulation
        dz = np.gradient(zgrid)  # Calculate gradient with respect to z

        # Define the radial mask for the region [0.8, 1.2]
        radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
        
        # Compute the vertical gradient of dust density along the z-axis (axis=2 is the vertical axis)
        grad_dustdens = np.gradient(dustdens, zgrid, axis=2)
        dustdens_masked = dustdens[:, radial_mask, :]
        grad_dustdens_masked = grad_dustdens[:, radial_mask, :]
        grad_dustdens_avg = np.mean(grad_dustdens_masked, axis=(0, 1))  # Averaging over azimuth and radius
        dustdens_avg = np.mean(dustdens_masked, axis=(0, 1))  # Averaging over azimuth and radius

        # Compute N_z^2 using Omega_K = 1 (in code units)
        N2_z = -1 * (zgrid * grad_dustdens_avg)

        label = f"{'3D' if is_3D else '2D'} Z={params['METALLICITY']:.4f}".rstrip('0').rstrip('.')

        # Panel 1: Plot the buoyancy profile (N_z^2)
        axs[0].plot(zgrid, N2_z, label=f'Buoyancy {label}', color=consistent_colors[idx % len(consistent_colors)], linestyle='-', linewidth=2)

        # Panel 2: Plot the averaged dust density vertical profile
        axs[1].plot(zgrid, dustdens_avg, label=label, color=consistent_colors[idx % len(consistent_colors)], linestyle='-', linewidth=2)

        # Compute and plot vertical shear for both 2D and 3D simulations
        vertical_shear = compute_vertical_shear(data, xgrid, zgrid)
        vertical_shear_masked = vertical_shear[:, radial_mask, :]
        vertical_shear_avg = np.mean(vertical_shear_masked, axis=(0, 1))
        # Print vertical shear values (mean for simplicity)
        print(f"Vertical shear for {label}: Mean value = {np.mean(vertical_shear_avg)}")

        axs[0].plot(zgrid, vertical_shear_avg, label=f'Shear {label}', color=consistent_colors[idx % len(consistent_colors)], linestyle='--', linewidth=2)

    # Set labels and titles
    axs[0].set_ylabel(r'$\langle N_z^2 \rangle$ and Shear', fontsize=label_fontsize)
    axs[0].set_title(rf'$\beta={beta_case}$', fontsize=label_fontsize)
    axs[0].grid(True)
    axs[0].legend(loc='best', fontsize=legend_fontsize)

    axs[1].set_ylabel(r'$\langle \rho_d \rangle$', fontsize=label_fontsize)
    axs[1].set_xlabel('z', fontsize=label_fontsize)
    axs[1].grid(True)

    if yrange_N2:
        axs[0].set_ylim(yrange_N2)
    if yrange_dustdens:
        axs[1].set_ylim(yrange_dustdens)
    if xrange:
        axs[0].set_xlim(xrange)
        axs[1].set_xlim(xrange)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()
    plot_type = "3D" if is_3D else "2D"
    output_filename = f"N2_dust_density_shear_profile_{plot_type}_beta{beta_case}.pdf"
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

    local_directory = "/Users/mariuslehmann/Downloads/Profiles"
    scp_transfer(output_filename, local_directory, "mariuslehmann")

    print(f"Plot saved as {output_filename} and transferred to {local_directory}")

# Define lists for simulations (for beta=1 and beta=0.001)
simulations_beta1_3D = [
    "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
    "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
    "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
]

simulations_beta1_2D = [
    "cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
    "cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
    "cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
]

simulations_beta001_3D = [
    "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
    "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
    "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
]

simulations_beta001_2D = [
    "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
    "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
    "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
]


if __name__ == "__main__":
    snapshot = 112  # Example snapshot number

    # Call the plotting function for 3D simulations
    plot_vertical_buoyancy(simulations_beta1_3D, snapshot, beta_case=1, is_3D=True, yrange_N2=[-0.0005, 0.01], yrange_dustdens=[0, 0.001], xrange=[-0.04, 0.04])

    # Call the plotting function for 2D simulations
    plot_vertical_buoyancy(simulations_beta1_2D, snapshot, beta_case=1, is_3D=False, yrange_N2=[-0.0005, 0.01], yrange_dustdens=[0, 0.001], xrange=[-0.04, 0.04])

    # For beta = 0.001
    plot_vertical_buoyancy(simulations_beta001_3D, snapshot, beta_case=0.001, is_3D=True, yrange_N2=[0.000, 0.014], yrange_dustdens=[0, 0.0022], xrange=[-0.03, 0.03])

    plot_vertical_buoyancy(simulations_beta001_2D, snapshot, beta_case=0.001, is_3D=False, yrange_N2=[0.000, 0.014], yrange_dustdens=[0, 0.0022], xrange=[-0.03, 0.03])
