import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from plot_fargo import determine_base_path
from data_reader import reconstruct_grid, read_parameters
from data_storage import scp_transfer
from joblib import Parallel, delayed



def extract_parameters_from_name(sim_name):
    if "Z1dm3" in sim_name:
        Z = 0.001
    elif "Z3dm3" in sim_name:
        Z = 0.003
    elif "Z1dm2" in sim_name:
        Z = 0.01
    elif "Z3dm2" in sim_name:
        Z = 0.03
    elif "Z1dm1" in sim_name:
        Z = 0.1
    else:
        Z = "unknown"

    if "St1dm3" in sim_name:
        tau = 0.001
    elif "St1dm2" in sim_name:
        tau = 0.01
    elif "St1dm1" in sim_name:
        tau = 0.1
    else:
        tau = "unknown"

    return Z, tau

def gaussian(z, A, z0, sigma):
    return A * np.exp(-0.5 * ((z - z0) / sigma) ** 2)

def compute_ri_t_z_sampled(velocities, avg_velocity, t_values, dt, xgrid, zgrid, n_sample_x, n_sample_z):
    sampled_x_indices = np.linspace(0, len(xgrid) - 1, n_sample_x, dtype=int)
    sampled_z_indices = np.linspace(0, len(zgrid) - 1, n_sample_z, dtype=int)
    nt = len(t_values)

    ri_t_z = np.zeros((len(sampled_z_indices), nt))

    for z_idx, z_index in enumerate(tqdm(sampled_z_indices, desc="Computing R_i(t, z)")):
        ri_t_accum = []
        for x_index in sampled_x_indices:
            velocity_series = velocities[0, x_index, z_index, :].flatten()  # Time series
            avg_velocity_value = avg_velocity[0, x_index, z_index]  # Single value

            ri_t = []
            for t in t_values:
                integrand = []
                for tau in range(nt - t):
                    delta_v_tau = velocity_series[tau] - avg_velocity_value
                    delta_v_tau_t = velocity_series[tau + t] - avg_velocity_value
                    product = delta_v_tau * delta_v_tau_t
                    integrand.append(product)
                ri_t.append(trapezoid(integrand, dx=dt))
            ri_t_accum.append(ri_t)

        # Compute mean and normalize
        ri_t_mean = np.mean(ri_t_accum, axis=0)
        ri_t_mean /= ri_t_mean[0] if ri_t_mean[0] != 0 else 1
        ri_t_z[z_idx, :] = ri_t_mean

    return ri_t_z

from joblib import Parallel, delayed

def compute_ri_t_z_sampled_parallel(velocities, avg_velocity, t_values, dt, xgrid, zgrid, n_sample_x, n_sample_z, n_jobs=16):
    sampled_x_indices = np.linspace(0, len(xgrid) - 1, n_sample_x, dtype=int)
    sampled_z_indices = np.linspace(0, len(zgrid) - 1, n_sample_z, dtype=int)
    nt = len(t_values)

    # Helper function for parallel processing
    def compute_ri_t_for_z(z_index):
        ri_t_accum = []
        for x_index in sampled_x_indices:
            velocity_series = velocities[0, x_index, z_index, :].flatten()  # Time series
            avg_velocity_value = avg_velocity[0, x_index, z_index]  # Single value

            ri_t = []
            for t in t_values:
                integrand = [
                    (velocity_series[tau] - avg_velocity_value) *
                    (velocity_series[tau + t] - avg_velocity_value)
                    for tau in range(nt - t)
                ]
                ri_t.append(trapezoid(integrand, dx=dt))
            ri_t_accum.append(ri_t)

        # Compute mean and normalize
        ri_t_mean = np.mean(ri_t_accum, axis=0)
        ri_t_mean /= ri_t_mean[0] if ri_t_mean[0] != 0 else 1
        return ri_t_mean

    # Parallel computation
    print(f"Parallelizing computation across {n_jobs} processes...")
    ri_t_z = Parallel(n_jobs=n_jobs)(
        delayed(compute_ri_t_for_z)(z_index) for z_index in tqdm(sampled_z_indices, desc="Computing R_i(t, z)")
    )

    # Convert to numpy array
    return np.array(ri_t_z)


def find_eddy_times_z(t_values, ri_t_z, dt):
    print("Starting computation of eddy times t_eddy(z)...")
    t_eddy_z = []
    for z_idx in range(ri_t_z.shape[0]):  # Iterate over the z-dimension
        ri_t = ri_t_z[z_idx, :]  # Extract time series for this z-index
        times = t_values * dt
        indices = np.where(ri_t <= 0.5)[0]
        if len(indices) > 0:
            t_eddy_z.append(times[indices[0]])
        else:
            t_eddy_z.append(float('nan'))
    print("Completed computation of eddy times t_eddy(z).")
    return np.array(t_eddy_z)

def integrate_density_profile_symmetric(zgrid, t_eddy_z, rms_vz_profile, omega, tau_s):
    print("Starting numerical integration of density profile symmetrically...")
    dz = np.diff(zgrid)
    rho_d = np.zeros_like(zgrid)
    
    # Initialize midplane density
    midplane_index = len(zgrid) // 2
    rho_d[midplane_index] = 1.0  # Normalized at the midplane

    # Interpolate rms_vz_profile to match t_eddy_z length
    f_rms_vz = interp1d(zgrid, rms_vz_profile, kind='linear', fill_value='extrapolate')
    rms_vz_sampled = f_rms_vz(zgrid[:len(t_eddy_z)])  # Match sampled zgrid

    D = rms_vz_sampled ** 2 * t_eddy_z

    # Integrate upward from the midplane
    for i in range(midplane_index + 1, len(zgrid)):
        rho_d[i] = rho_d[i - 1] * np.exp(-omega ** 2 * tau_s * zgrid[i - 1] * dz[i - 1] / D[i - 1])

    # Integrate downward from the midplane
    for i in range(midplane_index - 1, -1, -1):
        rho_d[i] = rho_d[i + 1] * np.exp(omega ** 2 * tau_s * zgrid[i] * dz[i] / D[i])

    print("Completed numerical integration of density profile symmetrically.")
    return rho_d

# Function to read velocity data from a file
def read_velocity_data(file_path, nx, ny, nz):
    if os.path.exists(file_path):
        return np.fromfile(file_path, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        raise FileNotFoundError(f"File {file_path} not found.")

# Function to compute the average velocity over the full time range
def compute_average_velocity(velocities):
    return np.mean(velocities, axis=3)

def compute_center_of_mass_and_corrugation_velocity(dust1dens, dust1vz, zgrid):
    """
    Computes the center of mass (z_CMS) and vertical corrugation velocity (vz_corr).
    
    Parameters:
        dust1dens (ndarray): Dust density field (ny, nx, nz).
        dust1vz (ndarray): Dust vertical velocity field (ny, nx, nz).
        zgrid (ndarray): Vertical grid array.
    
    Returns:
        z_cms (ndarray): Center of mass (CMS) along the radial direction (nx,).
        vz_corr (ndarray): Corrugation velocity along the radial direction (nx,).
    """
    dz = np.diff(zgrid, append=zgrid[-1])  # Approximate vertical integration weights
    z_cms = np.sum(dust1dens * zgrid[None, None, :] * dz[None, None, :], axis=2) / np.sum(dust1dens * dz[None, None, :], axis=2)
    vz_corr = np.sum(dust1dens * dust1vz * dz[None, None, :], axis=2) / np.sum(dust1dens * dz[None, None, :], axis=2)
    return z_cms[0], vz_corr[0]  # Remove the redundant ny dimension

def main():
    print("Starting main script...")

    simulations = [
        ("cos_bet1dm3_St1dm3_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400),
        ("cos_bet1dm3_St1dm3_Z3dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400),
        ("cos_bet1dm3_St1dm3_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400),
        ("cos_bet1dm3_St1dm3_Z3dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400),
        ("cos_bet1dm3_St1dm3_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400),
        ("cos_bet1dm3_St1dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew", 251, 200), #OK
        ("cos_bet1dm3_St1dm2_Z3dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 251, 200), #OK
        ("cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 251, 200), #OK
        ("cos_bet1dm3_St1dm2_Z3dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 251, 200), #OK
        ("cos_bet1dm3_St1dm2_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap", 251, 200), #OK
        ("cos_bet1dm3_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
        ("cos_bet1dm3_St1dm1_Z3dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
        ("cos_bet1dm3_St1dm1_Z3dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
        ("cos_bet1dm3_St1dm1_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap", 125, 200),
    ]



    results = []
    Z_values_tau = {0.001: [], 0.01: [], 0.1: []}
    H_d_H_g_measured_tau = {0.001: [], 0.01: [], 0.1: []}
    H_d_H_g_predicted_tau = {0.001: [], 0.01: [], 0.1: []}

    for idx, (sim_name, file_offset, num_files) in enumerate(simulations):
        print(f"Processing simulation: {sim_name}")
        Z, tau_s = extract_parameters_from_name(sim_name)
        sim_subdir_path = determine_base_path(sim_name)[0]
        summary_file = os.path.join(sim_subdir_path, "summary0.dat")
        parameters = read_parameters(summary_file)
        xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
        dt = 1 / 20
        H_g=0.1

        # Modified code for loading velocity and density data
        velocities = []

        z_cms_list = []  # Store z_CMS for each time step
        for i in tqdm(range(num_files), desc="Loading velocity and density data"):
            file_index = file_offset + i
            
            # Load dust1vz
            vz_file_name = f"dust1vz{file_index}.dat"
            vz_file_path = os.path.join(sim_subdir_path, vz_file_name)
            try:
                dust1vz_data = read_velocity_data(vz_file_path, nx, ny, nz)
            except ValueError as e:
                print(f"ERROR: {e}. File: {vz_file_path}, Shape: {nx}, {ny}, {nz}")
                continue  # Skip to the next file if there is a mismatch

            # Load dust1dens only for CMS computation
            dens_file_name = f"dust1dens{file_index}.dat"
            dens_file_path = os.path.join(sim_subdir_path, dens_file_name)
            try:
                dust1dens_data = read_velocity_data(dens_file_path, nx, ny, nz)
            except ValueError as e:
                print(f"ERROR: {e}. File: {dens_file_path}, Shape: {nx}, {ny}, {nz}")
                continue  # Skip to the next file if there is a mismatch

            # Compute z_CMS and vz_corr
            z_cms, vz_corr = compute_center_of_mass_and_corrugation_velocity(dust1dens_data, dust1vz_data, zgrid)
            z_cms_list.append(z_cms)  # Append CMS for this time step

            # Correct the vertical velocity by subtracting vz_corr along the z-column
            corrected_dust1vz = dust1vz_data - vz_corr[None, :, None]

            # Perform a vertical shift to align the center of mass with the midplane (z=0)
            for radius_idx, z_cms_value in enumerate(z_cms):
                z_cms_index = np.argmin(np.abs(zgrid - z_cms_value))  # Find the closest index to z_cms
                shift = z_cms_index - nz // 2  # Compute the shift to bring z_cms to the midplane

                # Apply the shift for this radius
                corrected_dust1vz[0, radius_idx, :] = np.roll(corrected_dust1vz[0, radius_idx, :], -shift, axis=-1)          

            # Store corrected velocities
            velocities.append(corrected_dust1vz)

        # If no valid data was loaded, skip this simulation
        if not velocities:
            print(f"No valid data found for simulation {sim_name}. Skipping.")
            continue

        velocities = np.stack(velocities, axis=3)
        print("All velocity data loaded successfully.") 

        # Compute the average velocity
        avg_velocity = compute_average_velocity(velocities)
        print("Average velocity computed.")

        t_values = np.arange(0, num_files, 1).astype(int)

        n_sample_x = 40
        n_sample_z = 120
        sampled_z_indices = np.linspace(0, len(zgrid) - 1, n_sample_z, dtype=int)
        sampled_zgrid = zgrid[sampled_z_indices]

        ri_t_z = compute_ri_t_z_sampled_parallel(velocities, avg_velocity, t_values, dt, xgrid, zgrid, n_sample_x, n_sample_z, n_jobs=16)
        t_eddy_z = find_eddy_times_z(t_values, ri_t_z, dt)

        npz_file = os.path.join(sim_subdir_path, f"{os.path.basename(sim_name)}_quantities.npz")
        loaded_data = np.load(npz_file)
        rms_vz_profile = loaded_data['rms_vz_profile'][sampled_z_indices]
        H_d_array = loaded_data['H_d_corr']
        time = loaded_data['time']

        rho_d = integrate_density_profile_symmetric(sampled_zgrid, t_eddy_z, rms_vz_profile, omega=1.0, tau_s=tau_s)

        popt, _ = curve_fit(gaussian, sampled_zgrid, rho_d, p0=[1.0, 0.0, 0.1])
        H_d_H_g_predicted = np.abs(popt[2] / 0.1)

        last_200_orbits_index = np.where(time >= (time[-1] - 200))[0]
        H_d_H_g_measured = (
            np.mean(H_d_array[last_200_orbits_index] / 0.1)
            if len(last_200_orbits_index) > 0
            else float("nan")
        )

        results.append((Z, tau_s, H_d_H_g_predicted, H_d_H_g_measured))

        Z_values_tau[tau_s].append(Z)
        H_d_H_g_measured_tau[tau_s].append(H_d_H_g_measured)
        H_d_H_g_predicted_tau[tau_s].append(H_d_H_g_predicted)

        if idx == 0:
            # Plotting results
            plt.figure(figsize=(15, 5))

            # RMS vz^2 vs sampled z
            plt.subplot(1, 3, 1)
            plt.plot(sampled_zgrid, rms_vz_profile, label=r"$\mathrm{RMS} \, v_z^2$")
            plt.xlabel("z")
            plt.ylabel(r"$\mathrm{RMS} \, v_z^2$")
            plt.title("RMS vz^2 vs z")
            plt.legend()

            # Eddy Times vs sampled z
            plt.subplot(1, 3, 2)
            plt.plot(sampled_zgrid, t_eddy_z, label=r"$t_\mathrm{eddy}(z)$")
            plt.xlabel("z")
            plt.ylabel(r"$t_\mathrm{eddy}(z)$")
            plt.title("Eddy Times vs z")
            plt.legend()

             # Plot the density profile with the predicted and measured Gaussian fits
            plt.subplot(1, 3, 3)
            plt.plot(sampled_zgrid, rho_d, label=r"$\rho_d(z)$ (Predicted)")
            plt.plot(
                sampled_zgrid,
                gaussian(sampled_zgrid, 1.0, 0.0, H_d_H_g_predicted * H_g),
                linestyle="--",
                label=r"Gaussian (Predicted $H_d$)",
            )
            plt.plot(
                sampled_zgrid,
                gaussian(sampled_zgrid, 1.0, 0.0, H_d_H_g_measured * H_g),
                linestyle="--",
                label=r"Gaussian (Measured $H_d$)",
            )
            plt.xlabel("z")
            plt.ylabel(r"$\rho_d(z)$")
            plt.title("Density Profile vs z")
            plt.ylim(0, 1)  # Limit y-axis to [0, 1] for better visibility
            plt.legend()

            plt.tight_layout()
            plt.savefig("density_profile_vs_z_with_gaussian.pdf")
            print("Plot saved as density_profile_vs_z_with_gaussian.pdf")

    print("Results:")
    for Z, tau, predicted, measured in results:
        print(f"Z: {Z}, tau: {tau}, Predicted: {predicted}, Measured: {measured}")

    from matplotlib import rc

    # Set global font sizes
    rc('font', size=16)  # Default text size
    rc('axes', titlesize=16)  # Title font size
    rc('axes', labelsize=16)  # Axes labels font size
    rc('xtick', labelsize=16)  # X-axis tick labels font size
    rc('ytick', labelsize=16)  # Y-axis tick labels font size
    rc('legend', fontsize=15)  # Legend font size


    # Plotting results
    plt.figure(figsize=(10, 6))

    colors = {0.001: 'blue', 0.01: 'green', 0.1: 'orange'}
    markers = {0.001: 'o', 0.01: '^', 0.1: 's'}
    size=80

    for tau, color in colors.items():
        plt.scatter(Z_values_tau[tau], H_d_H_g_measured_tau[tau], color=color, marker=markers[tau],
                    label=f"Measured (τ={tau})", s=size)
        plt.scatter(Z_values_tau[tau], H_d_H_g_predicted_tau[tau], color=color, marker=markers[tau],
                    facecolors='none', label=f"Predicted (τ={tau})", s=size)
        for z, measured, predicted in zip(Z_values_tau[tau], H_d_H_g_measured_tau[tau], H_d_H_g_predicted_tau[tau]):
            plt.plot([z, z], [measured, predicted], color=color, linestyle=':', linewidth=0.8)

    plt.xscale('log')
    plt.xlabel("Metallicity Z")
    plt.ylabel(r"$H_d / H_g$")
    plt.title("Comparison of Predicted and Measured Dust Scale Heights")
    plt.ylim(0.01, 0.08)  # Limit y-axis to [0, 1] for better visibility
    plt.legend()
    plt.tight_layout()

    output_filename = "dust_scale_height_vs_metallicity_zdep.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Transfer the plot
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")


if __name__ == "__main__":
    main()
