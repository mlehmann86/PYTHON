import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage import gaussian_filter
from data_reader import read_single_snapshot, read_initial_snapshot, determine_nt, read_parameters, reconstruct_grid
from data_storage import scp_transfer  # Import the SCP transfer function
from plot_fargo import determine_base_path

plt.rcParams['text.usetex'] = True

# List of simulations for the side-by-side omega_z plots

#simulations = [
#    "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
#    "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
#    "cos_b1d0_us_St1dm1_Z3dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
#    "cos_b1d0_us_St1dm1_Z4dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack"
#]

#3D NONISO
simulations = [
    "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
    "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
    "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack",
    "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap_hack"
]
'''
#2D NONISO
simulations = [
    "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150",
    "cos_b1d0_us_St1dm1_Z2dm3_r6H_z08H_fim053_ss203_stnew_LR150_tap_hack",
    "cos_b1d0_us_St1dm1_Z4dm3_r6H_z08H_fim053_ss203_stnew_LR150_tap_hack",
    "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap_hack"
]
'''
##########################################################################

#3D ISO
simulations = [
"cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
"cos_b1dm3_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
"cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
"cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
]


'''
#3D NONISO
simulations = [
"cos_b1d0_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
"cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
"cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
"cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
]
'''



#3D NONISO
simulations = [
"cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
"cos_b1d0_us_St1dm1_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
"cos_b1d0_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
"cos_b1d0_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
]






# Helper function to extract simulation parameters from the simulation name
def extract_simulation_metadata(simulation_name):
    # Extract metallicity (Z)
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
    elif "Z2dm3" in simulation_name:
        metallicity = r'$Z=0.002$'
    elif "Z4dm3" in simulation_name:
        metallicity = r'$Z=0.004$'
    elif "Z4dm2" in simulation_name:
        metallicity = r'$Z=0.04$'
    elif "Z1dm1" in simulation_name:
        metallicity = r'$Z=0.1$'
    else:
        metallicity = r'$Z=0$'

    # Extract Stokes number (\tau)
    stokes_dict = {
        "St1dm2": r"$\tau=0.01$",
        "St2dm2": r"$\tau=0.02$",
        "St3dm2": r"$\tau=0.03$",
        "St4dm2": r"$\tau=0.04$",
        "St5dm2": r"$\tau=0.05$",
        "St1dm1": r"$\tau=0.1$"
    }
    tau = next((stokes_dict[key] for key in stokes_dict if key in simulation_name), r"$\tau=\mathrm{N/A}$")

    # Extract beta (\beta)
    beta_dict = {
        "b1d0": r"$\beta=1$",
        "b1dm3": r"$\beta=0.001$"
    }
    beta = next((beta_dict[key] for key in beta_dict if key in simulation_name), r"$\beta=\mathrm{N/A}$")

    # Combine the metadata into the title
    title = f"{metallicity}, {tau}, {beta}"
    return title

# Function to load simulation data
def load_simulation_data(sim_dir):
    # Ensure determine_base_path returns a single path
    subdir_path = determine_base_path(sim_dir)
    
    # If subdir_path is a tuple, extract the first element
    if isinstance(subdir_path, tuple):
        subdir_path = subdir_path[0]
    
    npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
    
    try:
        loaded_data = np.load(npz_file)
        time = loaded_data['time']
        vortr_avg = loaded_data['vortr_avg']
        vortz_avg = loaded_data['vortz_avg']
        return time, vortr_avg, vortz_avg
    except FileNotFoundError:
        print(f"File not found: {npz_file}")
        return None, None, None

def compute_vorticity(data_arrays, initial_arrays, xgrid, ygrid, zgrid, sigma=1):
    r = xgrid[np.newaxis, :, np.newaxis]  # Expand r to match the dimensions (1, nx, 1)
    ny = data_arrays['gasvy'].shape[0]  # Check if it's a 2D or 3D simulation

    # Apply Gaussian smoothing to the velocity components
    smoothed_vr = gaussian_filter(data_arrays['gasvy'], sigma=sigma)  # Radial velocity
    smoothed_vphi = gaussian_filter(data_arrays['gasvx'], sigma=sigma)  # Azimuthal velocity
    smoothed_vz = gaussian_filter(data_arrays.get('gasvz', np.zeros_like(smoothed_vr)), sigma=sigma)  # Vertical velocity

    # Initialize vorticity components
    d_vz_dr = np.gradient(smoothed_vz, xgrid, axis=1)
    d_vr_dz = np.gradient(smoothed_vr, zgrid, axis=2)
    omega_phi = d_vz_dr - d_vr_dz

    if ny > 1:  # 3D case
        d_vphi_dz = np.gradient(smoothed_vphi, zgrid, axis=2)
        d_vz_dphi = np.gradient(smoothed_vz, ygrid, axis=0) / r
        omega_r = d_vphi_dz - d_vz_dphi

        d_vr_dphi = np.gradient(smoothed_vr, ygrid, axis=0) / r
        d_vphi_dr = np.gradient(r * smoothed_vphi, xgrid, axis=1) / r
        omega_z = d_vphi_dr - d_vr_dphi
    else:  # 2D case
        # No azimuthal derivatives in 2D simulations
        omega_r = np.gradient(smoothed_vphi, zgrid, axis=2)
        omega_z = np.gradient(r * smoothed_vphi, xgrid, axis=1) / r

    # Compute the initial vorticity using the initial velocity fields
    initial_vr = gaussian_filter(initial_arrays.get('gasvy', np.zeros_like(smoothed_vr)), sigma=sigma)
    initial_vphi = gaussian_filter(initial_arrays.get('gasvx', np.zeros_like(smoothed_vphi)), sigma=sigma)
    initial_vz = gaussian_filter(initial_arrays.get('gasvz', np.zeros_like(smoothed_vz)), sigma=sigma)

    initial_d_vz_dr = np.gradient(initial_vz, xgrid, axis=1)
    initial_d_vr_dz = np.gradient(initial_vr, zgrid, axis=2)
    initial_omega_phi = initial_d_vz_dr - initial_d_vr_dz

    if ny > 1:  # 3D case
        initial_d_vphi_dz = np.gradient(initial_vphi, zgrid, axis=2)
        initial_d_vz_dphi = np.gradient(initial_vz, ygrid, axis=0) / r
        initial_omega_r = initial_d_vphi_dz - initial_d_vz_dphi

        initial_d_vr_dphi = np.gradient(initial_vr, ygrid, axis=0) / r
        initial_d_vphi_dr = np.gradient(r * initial_vphi, xgrid, axis=1) / r
        initial_omega_z = initial_d_vphi_dr - initial_d_vr_dphi
    else:  # 2D case
        # No azimuthal derivatives in 2D simulations
        initial_omega_r = np.gradient(initial_vphi, zgrid, axis=2)
        initial_omega_z = np.gradient(r * initial_vphi, xgrid, axis=1) / r

    # Subtract the initial vorticity from the current vorticity
    delta_omega_r = omega_r - initial_omega_r
    delta_omega_phi = omega_phi - initial_omega_phi
    delta_omega_z = omega_z - initial_omega_z

    # Apply azimuthal averaging for 2D simulations (ny=1)
    if ny == 1:
        delta_omega_r = np.mean(delta_omega_r, axis=0)
        delta_omega_phi = np.mean(delta_omega_phi, axis=0)
        delta_omega_z = np.mean(delta_omega_z, axis=0)
    else:
        # Apply the specified averaging for 3D simulations
        delta_omega_r = np.mean(delta_omega_r, axis=0)
        delta_omega_phi = np.mean(delta_omega_phi, axis=0)
        delta_omega_z = np.mean(delta_omega_z, axis=2)

    return delta_omega_r, delta_omega_phi, delta_omega_z



def compute_kappa_squared(data_arrays, xgrid, ygrid, zgrid):
    """
    Compute the epicyclic frequency squared, kappa^2, using the angular velocity (Omega)
    in the inertial frame.
    """
    # Transform gasvx into the inertial frame
    r = xgrid[np.newaxis, :, np.newaxis]  # Expand r to match array dimensions
    gasvx_inertial = data_arrays['gasvx']# + r  # Transform gasvx to the inertial frame

    # Compute Omega (angular velocity) in the inertial frame
    Omega_inertial = gasvx_inertial / r

    # Compute the radial derivatives for kappa^2
    term1 = r**4 * Omega_inertial**2
    d_term1_dr = np.gradient(term1, xgrid, axis=1)  # Second derivative

    # Compute kappa^2
    kappa_squared = d_term1_dr / r**3

    # Perform vertical (z) averaging
    kappa_squared_z_avg = np.mean(kappa_squared, axis=2)  # Average over z

    return kappa_squared_z_avg  # Return as a 2D array (phi, r)


def plot_vorticity_components_or_omega_z(omega_r_avg, omega_phi_avg, omega_z_avg, xgrid, ygrid, zgrid, data_arrays, output_path, snapshot, plot_omega_z_only=False, y_range=None):
    if plot_omega_z_only:
        plt.figure(figsize=(24, 6))
        for i, sim in enumerate(simulations):
            # Load data for each simulation
            sim_subdir_path = determine_base_path(sim)
            
            # Ensure sim_subdir_path is a string
            if isinstance(sim_subdir_path, tuple):
                sim_subdir_path = sim_subdir_path[0]
            
            data_arrays, _, _, _, _ = read_single_snapshot(sim_subdir_path, snapshot=snapshot, read_gasvx=True, read_gasvy=True, read_gasvz=True)
            initial_arrays = read_initial_snapshot(sim_subdir_path, read_gasvx=True, read_gasvy=True, read_gasvz=True)
            data_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]  # Transform to the inertial frame
            initial_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]  # Transform initial to the inertial frame

            # Transform gasvx into the inertial frame
            r = xgrid[np.newaxis, :, np.newaxis]  # Expand r for proper broadcasting
            gasvx_inertial = data_arrays['gasvx']# + r  # Transform gasvx to the inertial frame

            # Compute Keplerian frequency squared in code units
            Omega_K_squared = xgrid**(-3)  # Keplerian frequency squared
            half_Omega_K_squared = 0.5 * Omega_K_squared
            radial_mask = (xgrid >= 0.71) & (xgrid <= 1.3)
            xgrid_masked = xgrid[radial_mask]
            Omega_K_squared_masked = Omega_K_squared[radial_mask]
            half_Omega_K_squared_masked = half_Omega_K_squared[radial_mask]

            # Debugging: Compute kappa_squared as Omega^2 = (v_phi / r)^2
            v_phi_inertial = gasvx_inertial / r  # Compute v_phi in the inertial frame
            Omega_squared = (v_phi_inertial)**2  # Omega^2 = (v_phi / r)^2

            # Perform z- and phi-averaging at the last step
            kappa_squared_z_avg = np.mean(Omega_squared, axis=2)  # z-averaged
            kappa_squared_avg = np.mean(kappa_squared_z_avg, axis=0)[radial_mask]  # Phi-averaged and apply radial mask

            # Extract specific azimuth value (debugging for kappa_squared_at_min_diff)
            diff = np.abs(kappa_squared_z_avg[:, radial_mask] - half_Omega_K_squared_masked[np.newaxis, :])
            azimuth_index = np.unravel_index(np.argmin(diff), diff.shape)[0]
            kappa_squared_at_min_diff = kappa_squared_z_avg[azimuth_index, radial_mask]

            # Plot omega_z_avg for each simulation
            ax1 = plt.subplot(1, 4, i + 1)
            img = ax1.imshow(omega_z_avg, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
                             aspect='auto', origin='lower', cmap='hot')
            plt.colorbar(img, label=r'$\langle \Delta \omega_z \rangle_z$')
            plt.xlabel(r'$r/r_0$')
            plt.ylabel(r'$\phi$')
            plt.title(extract_simulation_metadata(sim))  # Use the metallicity label

            # Plot kappa^2 and Omega_K^2
            ax2 = ax1.twinx()
            ax2.plot(xgrid, kappa_squared_at_min_diff, 'b--', label=r'$\kappa^2$ (specific azimuth)', linewidth=1.5)
            ax2.plot(xgrid, kappa_squared_avg, 'y--', label=r'$\langle \kappa^2 \rangle$', linewidth=1.5)
            ax2.plot(xgrid, Omega_K_squared, 'g-', label=r'$\Omega_K^2$', linewidth=1.5)
            ax2.plot(xgrid, half_Omega_K_squared, 'm-.', label=r'$0.5 \Omega_K^2$', linewidth=1.5)
            specific_phi = ygrid[azimuth_index]
            ax1.axhline(y=specific_phi, color='b', linestyle='--', linewidth=1.0)

            ax2.tick_params(axis='y', which='both', left=False, right=True)
            ax2.legend(loc='upper right')

            if y_range:
                ax2.set_ylim(y_range)

        plt.tight_layout()
        pdf_filename = os.path.join("contours", f"omega_z_comparison_snapshot{snapshot}.pdf")
        plt.savefig(pdf_filename)
        plt.close()
        print(f"Omega_z comparison plot saved to {pdf_filename}")
        scp_transfer(pdf_filename, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")

      
       

def plot_vorticity_contours(omega_r_avg, omega_phi_avg, omega_z_avg, xgrid, ygrid, zgrid, data_arrays, initial_arrays, output_path, snapshot, y_range=None):
    # Check if gasdens exists in data_arrays
    if 'gasdens' not in data_arrays:
        raise KeyError("The 'gasdens' field is missing from the data_arrays. Ensure it is loaded correctly.")

    ny = data_arrays['gasvy'].shape[0]  # Check dimensionality

    # Transform gasvx into the inertial frame
    r = xgrid[np.newaxis, :, np.newaxis]  # Expand r for proper broadcasting
    gasvx_inertial = data_arrays['gasvx']# + r  # Transform gasvx to the inertial frame

    # Compute Keplerian frequency squared in code units
    Omega_K_squared = xgrid**(-3)  # Keplerian frequency squared
    half_Omega_K_squared = 0.5 * Omega_K_squared
    radial_mask = (xgrid >= 0.71) & (xgrid <= 1.3)
    xgrid_masked = xgrid[radial_mask]
    Omega_K_squared_masked = Omega_K_squared[radial_mask]
    half_Omega_K_squared_masked = half_Omega_K_squared[radial_mask]

    # Debugging: Compute kappa_squared as Omega^2 = (v_phi / r)^2
    v_phi_inertial = gasvx_inertial / r  # Compute v_phi in the inertial frame
    Omega_squared = (v_phi_inertial)**2  # Omega^2 = (v_phi / r)^2

    # Perform z- and phi-averaging at the last step
    kappa_squared_z_avg = np.mean(Omega_squared, axis=2)  # z-averaged
    kappa_squared_avg = np.mean(kappa_squared_z_avg, axis=0)[radial_mask]  # Phi-averaged and apply radial mask

    # Compute kappa^2 using the provided function
    kappa_squared_z_avg = compute_kappa_squared(data_arrays, xgrid, ygrid, zgrid)  # z-averaged
    # Perform phi-averaging and apply radial mask
    kappa_squared_avg = np.mean(kappa_squared_z_avg, axis=0)[radial_mask]  # Phi-averaged and apply radial mask

    # Extract specific azimuth value (kappa^2 at the azimuth of minimal difference with 0.5 Omega_K^2)
    diff = np.abs(kappa_squared_z_avg[:, radial_mask] - half_Omega_K_squared_masked[np.newaxis, :])
    azimuth_index = np.unravel_index(np.argmin(diff), diff.shape)[0]
    kappa_squared_at_min_diff = kappa_squared_z_avg[azimuth_index, radial_mask]

    # Extract specific azimuth value (debugging for kappa_squared_at_min_diff)
    diff = np.abs(kappa_squared_z_avg[:, radial_mask] - half_Omega_K_squared_masked[np.newaxis, :])
    azimuth_index = np.unravel_index(np.argmin(diff), diff.shape)[0]
    kappa_squared_at_min_diff = kappa_squared_z_avg[azimuth_index, radial_mask]

    # Mask omega_phi_avg for the radial range
    omega_phi_avg_masked = omega_phi_avg[radial_mask, :] if ny > 1 else omega_phi_avg[radial_mask, :]

    # Prepare the figure with adjusted spacing
    plt.figure(figsize=(21, 6))  # Increase figure width to spread panels

    # Compute RMS azimuthal velocity deviations
    gasvx_diff = data_arrays['gasvx'] - initial_arrays['gasvx']  # Difference: current vs. initial gasvx
    rms_vphi_deviation = np.sqrt(np.mean(gasvx_diff**2, axis=(0, 2)))  # RMS over azimuthal and vertical axes
    rms_vphi_deviation_masked = rms_vphi_deviation[radial_mask]  # Mask for radial range



  
    plt.subplot(1, 3, 1)
    plt.imshow(omega_r_avg.T, extent=[0.71, 1.3, zgrid.min(), zgrid.max()],
                   aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label=r'$\langle \Delta \omega_r \rangle_\phi$')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel(r'$z/r_0$')
    plt.title(r'Radial Vorticity Component (2D)')

    # Add radial RMS azimuthal velocity deviations
    ax1_twin = plt.gca().twinx()
    ax1_twin.plot(xgrid_masked, rms_vphi_deviation_masked, 'g-', label=r'$\mathrm{RMS}(\Delta v_\phi)$', linewidth=1.5)
    ax1_twin.set_ylabel(r'$\mathrm{RMS}(\Delta v_\phi)$', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1_twin.legend(loc='upper right')

    # Frame 2: omega_phi in the r-z plane
    plt.subplot(1, 3, 2)
    img = plt.imshow(omega_phi_avg_masked.T if ny > 1 else omega_phi_avg.T,
                     extent=[0.71, 1.3, zgrid.min(), zgrid.max()],
                     aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(img, label=r'$\langle \Delta \omega_\phi \rangle_\phi$')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel(r'$z/r_0$')
    plt.title(r'Azimuthal Vorticity Component')


    # Add radial gas density profile
    gasdens_avg = np.mean(data_arrays['gasdens'], axis=0)[radial_mask]
    ax2 = plt.gca().twinx()
    ax2.plot(xgrid_masked, gasdens_avg, 'b-', label=r'$\langle \rho_{\mathrm{gas}} \rangle_{\phi, z}$', linewidth=1.5)
    ax2.set_ylabel(r'$\langle \rho_{\mathrm{gas}} \rangle_{\phi, z}$')

    # Ensure only the first handle and label are used for the legend
    handles, labels = ax2.get_legend_handles_labels()
    if handles and labels:  # Check if there are any handles and labels
        ax2.legend([handles[0]], [labels[0]], loc='upper right')


    # Frame 3: omega_z in the r-phi plane (3D) or r-z plane (2D)
    plt.subplot(1, 3, 3)
    if ny > 1:  # 3D case
        img = plt.imshow(omega_z_avg, extent=[0.71, 1.3, ygrid.min(), ygrid.max()],
                         aspect='auto', origin='lower', cmap='hot')
        plt.colorbar(img, label=r'$\langle \Delta \omega_z \rangle_z$')
        plt.xlabel(r'$r/r_0$')
        plt.ylabel(r'$\phi$')
        plt.title(r'Vertical Vorticity Component')
    else:  # 2D case
        img = plt.imshow(omega_z_avg.T, extent=[0.71, 1.3, zgrid.min(), zgrid.max()],
                         aspect='auto', origin='lower', cmap='hot')
        plt.colorbar(img, label=r'$\langle \Delta \omega_z \rangle_\phi$')
        plt.xlabel(r'$r/r_0$')
        plt.ylabel(r'$z/r_0$')
        plt.title(r'Vertical Vorticity Component (2D)')

    # Add kappa^2 and Omega_K^2 curves
    ax2 = plt.gca().twinx()
    # Average kappa^2
    ax2.plot(xgrid_masked, kappa_squared_avg, 'b--', label=r'$\langle \kappa^2 \rangle$', linewidth=1.5)
    # Specific azimuth kappa^2
    ax2.plot(xgrid_masked, kappa_squared_at_min_diff, 'c-', label=r'$\kappa^2$ (specific azimuth)', linewidth=1.5)
    # Omega_K^2
    ax2.plot(xgrid_masked, Omega_K_squared_masked, 'g-', label=r'$\Omega_K^2$', linewidth=1.5)
    # 0.5 Omega_K^2
    ax2.plot(xgrid_masked, half_Omega_K_squared_masked, 'm-.', label=r'$0.5 \Omega_K^2$', linewidth=1.5)
    # Adjust the y-axis range
    ax2.set_ylim([-5, 5])

    # Legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:4], labels[:4], loc='upper right')

    # Add legend
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    ax2.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    plt.tight_layout()
    contours_dir = "contours"
    os.makedirs(contours_dir, exist_ok=True)
    pdf_filename = os.path.join(contours_dir, f"vorticity_contours_snapshot{snapshot}_{os.path.basename(output_path)}.pdf")
    plt.savefig(pdf_filename)
    plt.close()
    print(f"Vorticity contour plots saved to {pdf_filename}")

    # Transfer the file using SCP
    local_directory = "/Users/mariuslehmann/Downloads/Contours"
    scp_transfer(pdf_filename, local_directory, "mariuslehmann")

def compute_angular_momentum_flux(data_arrays, initial_arrays, xgrid, ygrid, zgrid):
    # Compute deviations from the initial values
    delta_v_gr = data_arrays['gasvy'] - initial_arrays['gasvy']
    delta_v_gphi = data_arrays['gasvx'] - initial_arrays['gasvx']
    delta_v_dr = data_arrays.get('dust1vy', np.zeros_like(data_arrays['gasvy'])) - initial_arrays.get('dust1vy', np.zeros_like(data_arrays['gasvy']))
    delta_v_dphi = data_arrays.get('dust1vx', np.zeros_like(data_arrays['gasvx'])) - initial_arrays.get('dust1vx', np.zeros_like(data_arrays['gasvx']))
    
    # Compute epsilon
    gasdens = data_arrays['gasdens']
    dust1dens = data_arrays.get('dust1dens', np.zeros_like(gasdens))
    epsilon = np.where(gasdens > 0, dust1dens / gasdens, 0)  # Avoid division by zero
    
    # Compute the angular momentum flux F
    F = (
        delta_v_gr * (delta_v_gphi + epsilon * delta_v_dphi) +
        epsilon * delta_v_dr * (delta_v_gphi + epsilon * delta_v_dphi)
    )

    #F = (
    #    delta_v_gr * delta_v_gphi + epsilon * delta_v_dr * delta_v_dphi
    #)
    
    # Z-average F
    F_z_avg = np.mean(F, axis=2)
    
    return F_z_avg

def plot_angular_momentum_flux(simulations, xgrid, ygrid, zgrid, snapshot):
    plt.figure(figsize=(24, 6))
    
    all_min_f = []
    all_max_f = []
    twin_axes = []

    # First pass to compute global min/max for the color scale
    for sim in simulations:
        sim_subdir_path = determine_base_path(sim)
        if isinstance(sim_subdir_path, tuple):
            sim_subdir_path = sim_subdir_path[0]
        
        data_arrays, _, _, _, _ = read_single_snapshot(
            sim_subdir_path, snapshot=snapshot, 
            read_gasvx=True, read_gasvy=True, read_gasdens=True,
            read_dust1dens=True, read_dust1vx=True, read_dust1vy=True
        )
        initial_arrays = read_initial_snapshot(sim_subdir_path, read_gasvx=True, read_gasvy=True, read_dust1vx=True, read_dust1vy=True)

        data_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]
        initial_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]

        # Compute epsilon
        epsilon = data_arrays['dust1dens'] / data_arrays['gasdens']
        
        # Deviation velocities
        delta_gas_vr = data_arrays['gasvy'] - initial_arrays['gasvy']
        delta_gas_vphi = data_arrays['gasvx'] - initial_arrays['gasvx']
        delta_dust_vr = data_arrays['dust1vy'] - initial_arrays['dust1vy']
        delta_dust_vphi = data_arrays['dust1vx'] - initial_arrays['dust1vx']

        # Compute F (angular momentum flux)
        F = (
            delta_gas_vr * (delta_gas_vphi + epsilon * delta_dust_vphi) +
            epsilon * delta_dust_vr * (delta_gas_vphi + epsilon * delta_dust_vphi)
        )
        F_z_avg = np.mean(F, axis=2)  # Z-average
        all_min_f.append(np.min(F_z_avg))
        all_max_f.append(np.max(F_z_avg))

    # Determine global color scale range
    global_min_f = min(all_min_f)
    global_max_f = max(all_max_f)

    # Second pass to plot data
    for i, sim in enumerate(simulations):
        sim_subdir_path = determine_base_path(sim)
        if isinstance(sim_subdir_path, tuple):
            sim_subdir_path = sim_subdir_path[0]
        
        data_arrays, _, _, _, _ = read_single_snapshot(
            sim_subdir_path, snapshot=snapshot, 
            read_gasvx=True, read_gasvy=True, read_gasdens=True,
            read_dust1dens=True, read_dust1vx=True, read_dust1vy=True
        )
        initial_arrays = read_initial_snapshot(sim_subdir_path, read_gasvx=True, read_gasvy=True, read_dust1vx=True, read_dust1vy=True)

        data_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]
        initial_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]

        # Compute epsilon
        epsilon = data_arrays['dust1dens'] / data_arrays['gasdens']
        
        # Deviation velocities
        delta_gas_vr = data_arrays['gasvy'] - initial_arrays['gasvy']
        delta_gas_vphi = data_arrays['gasvx'] - initial_arrays['gasvx']
        delta_dust_vr = data_arrays['dust1vy'] - initial_arrays['dust1vy']
        delta_dust_vphi = data_arrays['dust1vx'] - initial_arrays['dust1vx']

        # Compute F (angular momentum flux)
        F = (
            delta_gas_vr * (delta_gas_vphi + epsilon * delta_dust_vphi) +
            epsilon * delta_dust_vr * (delta_gas_vphi + epsilon * delta_dust_vphi)
        )
        F_z_avg = np.mean(F, axis=2)  # Z-average
        F_r = np.mean(F_z_avg, axis=0)  # Azimuthally average

        # Compute pure gas F (epsilon -> 0)
        F_gas = delta_gas_vr * delta_gas_vphi
        F_gas_z_avg = np.mean(F_gas, axis=2)  # Z-average
        F_gas_r = np.mean(F_gas_z_avg, axis=0)  # Azimuthally average

        # Plot contour
        ax1 = plt.subplot(1, 4, i + 1)
        img = ax1.imshow(F_z_avg, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
                         aspect='auto', origin='lower', cmap='hot', vmin=global_min_f, vmax=global_max_f)
        plt.colorbar(img, label=r'$\langle F \rangle_z$')
        plt.xlabel(r'$r/r_0$')
        plt.ylabel(r'$\phi$')
        plt.title(extract_simulation_metadata(sim))

        # Plot radial profiles on twin axis
        ax2 = ax1.twinx()
        ax2.plot(xgrid, F_r, 'b-', label=r'$F(r)$', linewidth=1.5)
        ax2.plot(xgrid, F_gas_r, 'k--', label=r'$F_{\mathrm{gas}}(r)$', linewidth=1.5)
        twin_axes.append(ax2)
        ax2.legend(loc='upper right')

    # Set consistent y-limits for all twin axes
    global_min_profile = min(ax.get_ylim()[0] for ax in twin_axes)
    global_max_profile = max(ax.get_ylim()[1] for ax in twin_axes)
    for ax in twin_axes:
        ax.set_ylim(global_min_profile, global_max_profile)

    plt.tight_layout()
    pdf_filename = os.path.join("contours", f"angular_momentum_flux_snapshot{snapshot}.pdf")
    plt.savefig(pdf_filename)
    plt.close()
    print(f"Angular momentum flux comparison plot saved to {pdf_filename}")
    scp_transfer(pdf_filename, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")

def plot_epsilon_contours(simulations, xgrid, ygrid, zgrid, snapshot, twin_y_range=None):
    plt.figure(figsize=(24, 6))
    
    # Loop through simulations
    for i, sim in enumerate(simulations):
        # Load data for each simulation
        sim_subdir_path = determine_base_path(sim)
        if isinstance(sim_subdir_path, tuple):
            sim_subdir_path = sim_subdir_path[0]
        
        data_arrays, _, _, _, _ = read_single_snapshot(sim_subdir_path, snapshot=snapshot, 
                                                       read_gasdens=True, read_dust1dens=True)
        
        # Compute epsilon = dust1dens / gasdens
        epsilon = data_arrays['dust1dens'] / data_arrays['gasdens']
        log_epsilon = np.log10(epsilon)
        log_epsilon_z_avg = np.mean(log_epsilon, axis=2)  # Z-average of log(epsilon)
        epsilon_z_avg = np.mean(epsilon, axis=2)  # Z-average

        midplane_index = zgrid.shape[0] // 2  # Midplane index
        epsilon_midplane = data_arrays['dust1dens'][:, :, midplane_index] / np.maximum(data_arrays['gasdens'][:, :, midplane_index], 1e-12)
        log_epsilon_midplane = np.log10(np.maximum(epsilon_midplane, 1e-12))

        # Compute radial profile (azimuthally averaged)
        epsilon_radial_profile = np.mean(epsilon_z_avg, axis=0)

        # Plot epsilon contours
        ax1 = plt.subplot(1, 4, i + 1)
        img = ax1.imshow(log_epsilon_midplane, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
                 aspect='auto', origin='lower', cmap='coolwarm', vmin=-5, vmax=0)
        plt.colorbar(img, label=r'$\log_{10}(\epsilon)$')
        plt.xlabel(r'$r/r_0$')
        plt.ylabel(r'$\phi$')
        plt.title(extract_simulation_metadata(sim))

        # Over-plot radial profile on a twin axis
        ax2 = ax1.twinx()
        ax2.plot(xgrid, epsilon_radial_profile, color='blue', linestyle='-', label=r'$\langle \epsilon \rangle_\phi$', linewidth=1.5)
        ax2.tick_params(axis='y', which='both', left=False, right=True)
        ax2.legend(loc='upper right')

        # Adjust twin y-axis range if specified
        if twin_y_range:
            ax2.set_ylim(twin_y_range)

    plt.tight_layout()
    contours_dir = "epsilon_contours"
    os.makedirs(contours_dir, exist_ok=True)
    pdf_filename = os.path.join(contours_dir, f"epsilon_contours_snapshot{snapshot}.pdf")
    plt.savefig(pdf_filename)
    plt.close()
    print(f"Epsilon contour plots saved to {pdf_filename}")
    scp_transfer(pdf_filename, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate vorticity contour plots or omega_z comparison plot.')
    parser.add_argument('--dir', type=str, required=True, help='Directory of the simulation')
    parser.add_argument('--snapshot', type=int, required=True, help='Snapshot number')
    parser.add_argument('--series', action='store_true', help='Plot only omega_z for different simulations')

    args = parser.parse_args()

    subdir_path = determine_base_path(args.dir)
    if isinstance(subdir_path, tuple):
        subdir_path = subdir_path[0]

    summary_file = os.path.join(subdir_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    data_arrays, _, _, _, _ = read_single_snapshot(subdir_path, snapshot=args.snapshot, read_gasvx=True, read_gasvy=True, read_gasvz=True, read_gasdens=True)
    # Transform to inertial frame
    data_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]
    initial_arrays = read_initial_snapshot(subdir_path, read_gasvx=True, read_gasvy=True, read_gasvz=True)
    initial_arrays['gasvx'] += xgrid[np.newaxis, :, np.newaxis]

    omega_r_avg, omega_phi_avg, omega_z_avg = compute_vorticity(data_arrays, initial_arrays, xgrid, ygrid, zgrid)


    if args.series:
        # Plot omega_z and angular momentum flux for different simulations
        plot_vorticity_components_or_omega_z(omega_r_avg, omega_phi_avg, omega_z_avg, xgrid, ygrid, zgrid, data_arrays, subdir_path, args.snapshot, plot_omega_z_only=True, y_range=(-5, 5))
        #plot_angular_momentum_flux(simulations, xgrid, ygrid, zgrid, snapshot=args.snapshot)
        plot_epsilon_contours(simulations, xgrid, ygrid, zgrid, snapshot=args.snapshot, twin_y_range=(0, 0.1))
    else:
        # Ensure that gasdens is loaded
        if 'gasdens' not in data_arrays:
            print("Loading 'gasdens' field into data_arrays...")
            if 'gasdens' not in data_arrays:
                raise KeyError("The 'gasdens' field is missing even after attempting to load it.")

        # Plot original three-panel vorticity components
        plot_vorticity_contours(
            omega_r_avg, omega_phi_avg, omega_z_avg, 
            xgrid, ygrid, zgrid, 
            data_arrays, initial_arrays,subdir_path, 
            args.snapshot, y_range=(-5, 5)
    )

        # Load time evolution data and plot
        #time, vortr_avg, vortz_avg = load_simulation_data(args.dir)
        #if time is not None:
        #    plot_time_evolution(time, vortr_avg, vortz_avg, subdir_path)



#DIRECTIONS:
#	To plot the original three vorticity components:
#python script_name.py --dir your_simulation_directory --snapshot your_snapshot_number
#	•	To plot \omega_z for four simulations:
#python script_name.py --dir your_simulation_directory --snapshot your_snapshot_number --series
