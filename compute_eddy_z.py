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


def main():
    print("Starting main script...")

    simulations = [
        ("cos_bet1dm3_St1dm3_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400)
    ]
   # simulations = [
   #     ("cos_bet1dm3_St1dm2_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap", 251, 200)
   #]




    sim_name, file_offset, num_files = simulations[0]
    Z, tau_s = extract_parameters_from_name(sim_name)

    sim_subdir_path = determine_base_path(sim_name)[0]
    summary_file = os.path.join(sim_subdir_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
    dt = 1 / 20

    velocities = []
    for i in tqdm(range(num_files), desc="Loading velocity data"):
        file_index = file_offset + i
        file_name = f"gasvz{file_index}.dat"
        file_path = os.path.join(        sim_subdir_path, file_name)
        try:
            velocity_data = np.fromfile(file_path, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping.")
            continue
        velocities.append(velocity_data)

    velocities = np.stack(velocities, axis=3)
    avg_velocity = np.mean(velocities, axis=3)

    t_values = np.arange(0, num_files, 1).astype(int)

    print("Computing R_i(t, z)...")
    n_sample_x = 40
    n_sample_z = 120
    sampled_z_indices = np.linspace(0, len(zgrid) - 1, n_sample_z, dtype=int)
    sampled_zgrid = zgrid[sampled_z_indices]

    #ri_t_z = compute_ri_t_z_sampled(velocities, avg_velocity, t_values, dt, xgrid, zgrid, n_sample_x, n_sample_z)
    ri_t_z = compute_ri_t_z_sampled_parallel(velocities, avg_velocity, t_values, dt, xgrid, zgrid, n_sample_x, n_sample_z, n_jobs=16)

    print("Finished computing R_i(t, z).")

    t_eddy_z = find_eddy_times_z(t_values, ri_t_z, dt)

    npz_file = os.path.join(sim_subdir_path, f"{os.path.basename(sim_name)}_quantities.npz")
    loaded_data = np.load(npz_file)
    rms_vz_profile = loaded_data['rms_vz_profile'][sampled_z_indices]
    H_d_array = loaded_data['H_d']
    time = loaded_data['time']

    omega = 1.0
    H_g = 0.1
    rho_d = integrate_density_profile_symmetric(sampled_zgrid, t_eddy_z, rms_vz_profile, omega, tau_s)

    popt, _ = curve_fit(gaussian, sampled_zgrid, rho_d, p0=[1.0, 0.0, 0.1])
    H_d_H_g_predicted = np.abs(popt[2]/H_g)

    print(f"Predicted H_d/H_g: {H_d_H_g_predicted}")

    # Compute measured H_d/H_g
    last_200_orbits_index = np.where(time >= (time[-1] - 200))[0]
    H_d_H_g_measured = (
        np.mean(H_d_array[last_200_orbits_index] / H_g)
        if len(last_200_orbits_index) > 0
        else float("nan")
    )

    print(f"Measured H_d/H_g: {H_d_H_g_measured}")

    # Generate Gaussian using measured H_d
    H_d_measured = H_d_H_g_measured * H_g
    rho_d_measured_gaussian = gaussian(sampled_zgrid, 1.0, 0.0, H_d_measured)

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

    # Output results
    print(f"Predicted H_d/H_g: {H_d_H_g_predicted}")
    print(f"Measured H_d/H_g (time-averaged over last 200 orbits): {H_d_H_g_measured}")

if __name__ == "__main__":
    main()

