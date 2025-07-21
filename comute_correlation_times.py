import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from plot_fargo import determine_base_path
from data_reader import reconstruct_grid, read_parameters
from data_storage import scp_transfer
from matplotlib import rc
from matplotlib import rcParams



# Corrected load_simulation_data function
def load_simulation_data(simulation_dirs):
    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]

    data = {}
    for sim_tuple in simulation_dirs:
        sim_name, _, _ = sim_tuple  # Unpack the tuple
        npz_file = None
        for base_path in base_paths:
            potential_path = os.path.join(base_path, sim_name, f"{os.path.basename(sim_name)}_quantities.npz")
            if os.path.exists(potential_path):
                npz_file = potential_path
                break
        
        if npz_file:
            print(f"Loading: {npz_file}")
            loaded_data = np.load(npz_file)
            print("Keys in the .npz file:", loaded_data.files)

            data[sim_name] = {
                'time': loaded_data['time'],
                'alpha_r': loaded_data['alpha_r'],
                'rms_vz': loaded_data['rms_vz'],
                'max_epsilon': loaded_data['max_epsilon'],
                'H_d_array': loaded_data['H_d'],
                'H_d_corr_array': loaded_data['H_d_corr'],
                'roche_times': loaded_data.get('roche_times', None)
            }
        else:
            print(f"WARNING: File not found for {sim_name} in any base path!")
            continue

    return data

# Function to compute time-averaged RMS velocity squared over the last 500 orbits
def compute_time_avg_rms_vz_squared(time, rms_vz):
    last_500_orbits_index = np.where(time >= (time[-1] - 500))[0]
    if len(last_500_orbits_index) == 0:
        raise ValueError("Not enough data points for the last 500 orbits.")
    rms_vz_squared_avg = np.mean(rms_vz[last_500_orbits_index]**2)
    return rms_vz_squared_avg

# Function to predict H_d/H_g using the given formula
def predict_dust_scale_height(eddy_time, H_g, tau, rms_vz_squared_avg):
    H_d_H_g_predicted = 1/np.sqrt(H_g**2 * tau / (rms_vz_squared_avg * eddy_time) + 1)
    # Compute the vertical dust diffusion coefficient
    D_z = rms_vz_squared_avg * eddy_time / (H_g**2)
    print(f"Vertical dust diffusion coefficient (D_z) = {D_z}")
    return H_d_H_g_predicted

# Function to read velocity data from a file
def read_velocity_data(file_path, nx, ny, nz):
    if os.path.exists(file_path):
        return np.fromfile(file_path, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        raise FileNotFoundError(f"File {file_path} not found.")

# Function to compute the average velocity over the full time range
def compute_average_velocity(velocities):
    return np.mean(velocities, axis=3)

# Function to compute R_i(t) using scipy's trapezoidal integration
def compute_ri_t(velocity_series, avg_velocity, t_values, dt):
    nt = len(velocity_series)
    ri_t = np.zeros(len(t_values))

    for t_index, t in enumerate(t_values):
        integrand = []
        for tau in range(nt - t):
            delta_v_tau = velocity_series[tau] - avg_velocity
            delta_v_tau_t = velocity_series[tau + t] - avg_velocity
            product = delta_v_tau * delta_v_tau_t
            integrand.append(product)
        ri_t[t_index] = trapezoid(integrand, dx=dt)

    return ri_t

# Function to extract Z and tau from the simulation name
def extract_parameters_from_name(sim_name):
    if "Z1dm3" in sim_name:
        Z = "0.001"
    elif "Z3dm3" in sim_name:
        Z = "0.003"
    elif "Z1dm2" in sim_name:
        Z = "0.01"
    elif "Z3dm2" in sim_name:
        Z = "0.03"
    elif "Z1dm1" in sim_name:
        Z = "0.1"
    else:
        Z = "unknown"

    if "St1dm3" in sim_name:
        tau = "0.001"
    elif "St1dm2" in sim_name:
        tau = "0.01"
    elif "St2dm2" in sim_name:
        tau = "0.02"
    elif "St4dm2" in sim_name:
        tau = "0.04"
    elif "St6dm2" in sim_name:
        tau = "0.06"
    elif "St8dm2" in sim_name:
        tau = "0.08"
    elif "St1dm1" in sim_name:
        tau = "0.1"
    else:
        tau = "unknown"

    return Z, tau

# Function to find the correlation time
def find_correlation_time(t_values, ri_t_average, dt):
    # Multiply t_values by dt to get the time in orbits
    times = t_values * dt

    # Find the indices where R_i(t) crosses 0.5
    indices = np.where(ri_t_average <= 0.5)[0]
    if len(indices) > 0:
        # Get the smallest time where R_i(t) crosses 0.5
        correlation_time = times[indices[0]]
    else:
        # If R_i(t) never crosses 0.5, return NaN
        correlation_time = float('nan')
    
    return correlation_time


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




# Main function
def main():

    #Comparison of predicted and measured H_d/H_g
    # Simulation directories and parameters
    simulations = [
	#("cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150", 125, 200),
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
        ("cos_bet1dm3_St1dm1_Z1dm1_r6H_z08H_LR_PRESETx10_stnew_tap", 125, 200)
    ]

    #Comparison of predicted and measured epsilon_max
    simulations = [
        ("cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150", 125, 200),
        ("cos_bet1d0_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
        ("cos_bet1d0_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1d0_St3dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1d0_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1d0_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1d0_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1d0_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
        ("cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
        ("cos_bet1dm3_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1dm3_St3dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1dm3_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1dm3_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1dm3_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
        ("cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200)
    ]

    #simulations = [
    #    ("cos_bet1dm3_St1dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
    #    ("cos_bet1dm3_St2dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
    #    ("cos_bet1dm3_St4dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
    #    ("cos_bet1dm3_St6dm2_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200),
    #    ("cos_bet1dm3_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 125, 200)
    #]

    simulations = [
        ("cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150", 125, 200),
        ("cos_bet1dm3_St1dm3_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 502, 400),
        ("cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", 251, 200), #OK
        ("cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", 125, 200),
    ]


    # Load parameters and reconstruct grid using the first simulation
    sim_subdir_path = determine_base_path(simulations[0][0])[0]
    summary_file = os.path.join(sim_subdir_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
    dt = 1 / 20

    # Define radial and vertical locations
    radial_indices = np.linspace(0.8, 1.2, 20)
    radial_indices = [np.argmin(np.abs(xgrid - r)) for r in radial_indices]
    #vertical_indices = range(nz // 2 - 5, nz // 2 + 5)

    # Determine the largest time span
    max_num_files = max(sim[2] for sim in simulations)
    total_time = max_num_files * dt

    # Create a list to store all t_values
    t_values_list = []
    sampling_interval = 1  # Adjust the sampling interval as needed

    # Generate t_values for each simulation
    for _, _, num_files in simulations:
        t_values = np.arange(0, num_files, sampling_interval).astype(int)
        t_values_list.append(t_values)


    # Initialize the plot for the simplified case
    plt.figure()

    # Define a list of simulations to include in the simplified plot
    selected_simulations = [
	("cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150", "Z=0, τ=0, β=1"),
        ("cos_bet1dm3_St1dm3_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", "Z=0.01, τ=0.001, β=0.001"),
        ("cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", "Z=0.01, τ=0.01, β=0.001"),
        ("cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", "Z=0.01, τ=0.1, β=0.001")
    ]




    #selected_simulations = [
    #    ("cos_bet1dm3_St1dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", "Z=0.01, τ=0.01, \beta=0.001"),
    #    ("cos_bet1dm3_St2dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", "Z=0.01, τ=0.02, \beta=0.001"),
    #    ("cos_bet1dm3_St4dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", "Z=0.01, τ=0.04, \beta=0.001"),
    #    ("cos_bet1dm3_St6dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", "Z=0.01, τ=0.06, \beta=0.001"),
    #    ("cos_bet1dm3_St8dm2_Z1dm2_r6H_z08H_LR_PRESETx10_stnew_tap_hack", "Z=0.01, τ=0.08, \beta=0.001"),
    #    ("cos_bet1dm3_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew", "Z=0.01, τ=0.1, \beta=0.001")
    #]
    


    # Loop over all simulations for full computation
    correlation_times = []
    eddy_times = {}
    alternative_H_d_H_g = {}  # Store alternative H_d/H_g predictions
   
    # Initialize dictionary to store R_i(t) averages for selected simulations
    selected_simulations_ri_t = {}


    # Modify the main function to dynamically set grid dimensions
    for sim_index, (sim_name, file_offset, num_files) in enumerate(simulations):
        print(f"\nProcessing simulation: {sim_name}")


        # Skip the "NOFB" simulations for velocity data processing
        if "NOFB" in sim_name:
            print(f"Skipping velocity data and R_i(t) computation for {sim_name} (NOFB simulation).")
            continue  # Skip the rest of the loop for these simulations


        sim_subdir_path = determine_base_path(sim_name)[0]

        # Load parameters and grid dimensions for this simulation
        summary_file = os.path.join(sim_subdir_path, "summary0.dat")
        parameters = read_parameters(summary_file)
        xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
        dt = 1 / 20  # Time step (update if required)
     

        # Modified code for loading velocity and density data
        velocities = []

        z_cms_list = []  # Store z_CMS for each time step
        for i in tqdm(range(num_files), desc="Loading velocity and density data"):
            file_index = file_offset + i

            # Check if the simulation is the "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150"
            if sim_name == "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150":
                # Read gasvz instead of dust1vz
                vz_file_name = f"gasvz{file_index}.dat"
                vz_file_path = os.path.join(sim_subdir_path, vz_file_name)
                try:
                    dust1vz_data = read_velocity_data(vz_file_path, nx, ny, nz)
                except ValueError as e:
                    print(f"ERROR: {e}. File: {vz_file_path}, Shape: {nx}, {ny}, {nz}")
                    continue  # Skip to the next file if there is a mismatch

                # Create a null array for dust1dens (filled with very small numbers)
                dust1dens_data = np.full((nx, ny, nz), 1e-10, dtype=np.float64)
            else:
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

            # After this point, the code can proceed as usual with the original vertical_indices:
            vertical_indices = range(nz // 2 - 4, nz // 2 + 4)

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

        # Compute R_i(t) for each radial and vertical location with progress tracking
        t_values = t_values_list[sim_index]
        total_iterations = len(radial_indices) * len(vertical_indices)
        ri_t_values = []

        with tqdm(total=total_iterations, desc="Computing R_i(t) over all locations") as pbar:
            for r_index in radial_indices:
                for z_index in vertical_indices:
                    velocity_time_series = velocities[0, r_index, z_index, :]
                    avg_velocity_value = avg_velocity[0, r_index, z_index]
                    ri_t = compute_ri_t(velocity_time_series, avg_velocity_value, t_values, dt)
                    ri_t_values.append(ri_t)
                    pbar.update(1)

        print("All R_i(t) values computed for this simulation.")

        # Average R_i(t) over all sampled locations
        ri_t_average = np.mean(ri_t_values, axis=0)

        # Normalize R_i(t) by its initial value
        ri_t_average /= ri_t_average[0] if ri_t_average[0] != 0 else 1

        # Store R_i(t) for selected simulations
        if sim_name in [s[0] for s in selected_simulations]:
            selected_simulations_ri_t[sim_name] = (t_values * dt, ri_t_average)

        # Find the correlation time
        correlation_time = find_correlation_time(t_values, ri_t_average, dt)
        correlation_times.append(correlation_time)
        eddy_times[sim_name] = correlation_time
        print(f"Correlation time for {sim_name}: {correlation_time:.2f} orbits")


        # Compute the alternative predicted H_d/H_g
        Z, tau = extract_parameters_from_name(sim_name)
        if Z == "unknown" or tau == "unknown":
            print(f"WARNING: Z or tau value not found for {sim_name}")
            continue



    # Set the global font size
    font_size = 14  # Adjust this value as needed
    rcParams.update({'font.size': font_size})

    # Simplified plot for selected simulations
    plt.figure()
    for sim_name, label in selected_simulations:
        if sim_name in selected_simulations_ri_t:
            t_values_plot, ri_t_average_plot = selected_simulations_ri_t[sim_name]
            plt.plot(t_values_plot, ri_t_average_plot, label=label)

    # Finalize the simplified plot
    plt.xlabel("Time (orbits)")
    plt.ylabel(r"$R_z(t)$")
    plt.legend()
    plt.title("Turbulent Correlation Times")
    plt.xscale("log")
    plt.xlim([0.05, total_time])
    plt.ylim([0, 1])
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8)
    output_filename = f"turbulent_correlation_time_log.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Call the scp_transfer function
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

###############################################################################



    # Load the data
    data = load_simulation_data(simulations)
    tau_values = {"St1dm3": 0.001, "St1dm2": 0.01, "St1dm1": 0.1}

    # Reference correlation time (from the dust-free simulation)
    reference_simulation = "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_LR150"
    reference_correlation_time = eddy_times.get(reference_simulation, None)
    if reference_correlation_time is None:
        print(f"WARNING: Correlation time not found for {reference_simulation}.")
        reference_correlation_time = 0.1  # Default fallback value

  
    # Prepare lists to store data for plotting
    Z_values_tau_0_001 = []
    H_d_H_g_measured_tau_0_001 = []
    H_d_H_g_predicted_tau_0_001 = []

    Z_values_tau_0_01 = []
    H_d_H_g_measured_tau_0_01 = []
    H_d_H_g_predicted_tau_0_01 = []

    Z_values_tau_0_1 = []
    H_d_H_g_measured_tau_0_1 = []
    H_d_H_g_predicted_tau_0_1 = []

    # Print all H_d values
    print("\n--- Predicted and Measured H_d / H_g Values ---")

    for sim_tuple in simulations:
        sim_name, _, _ = sim_tuple  # Unpack the tuple
        sim_data = data.get(sim_name)
        
        if sim_data is None:
            print(f"WARNING: Data not found for {sim_name}")
            continue

        # Check if H_d_corr key exists
        if 'H_d_corr_array' not in sim_data:
            print(f"ALERT: 'H_d_corr' not found in sim_data for simulation {sim_name}.")
            continue

        try:
            # Extract data for the simulation
            time = sim_data['time']
            rms_vz = sim_data['rms_vz']
            H_d_array = sim_data['H_d_corr_array']  # Use 'H_d_array' instead of 'H_d'
        except KeyError as e:
            print(f"ERROR: Unexpected missing key {e} in sim_data for simulation {sim_name}.")
            continue

        # Compute the time-averaged RMS velocity squared
        rms_vz_squared_avg = compute_time_avg_rms_vz_squared(time, rms_vz)
        print(f"Time-averaged <v_z^2> for {sim_name}: {rms_vz_squared_avg}")


         # Extract Z and tau from the simulation name
        Z, tau = extract_parameters_from_name(sim_name)
        print(f"Simulation: {sim_name}, Z: {Z}, tau: {tau}")

        if Z == "unknown" or tau == "unknown":
            print(f"WARNING: Z or tau value not found for {sim_name}")
            continue

        Z = float(Z)
        tau = float(tau)
        print(f"Final Z: {Z}, Final tau: {tau}")

        # Use the known eddy time from the previous computation
        eddy_time = eddy_times.get(sim_name, None)
        if eddy_time is None:
            print(f"WARNING: Eddy time not found for {sim_name}")
            continue

        
        H_g=0.1
        compute_time_avg_rms_vz_squared(time, rms_vz)
        H_d_H_g_predicted = predict_dust_scale_height(eddy_time, H_g, tau, rms_vz_squared_avg)

        # Compute the measured H_d/H_g
        last_500_orbits_index = np.where(time >= (time[-1] - 500))[0]
        H_d_H_g_measured = (
            np.mean(H_d_array[last_500_orbits_index] / H_g)
            if len(last_500_orbits_index) > 0
            else float('nan')
        )

        # Append to the correct lists based on tau
        if tau == 0.001:
            Z_values_tau_0_001.append(Z)
            H_d_H_g_measured_tau_0_001.append(H_d_H_g_measured)
            H_d_H_g_predicted_tau_0_001.append(H_d_H_g_predicted)
            print(f"Appending to tau=0.001 lists: Z={Z}, Measured={H_d_H_g_measured}, Predicted={H_d_H_g_predicted}")
        elif tau == 0.01:
            Z_values_tau_0_01.append(Z)
            H_d_H_g_measured_tau_0_01.append(H_d_H_g_measured)
            H_d_H_g_predicted_tau_0_01.append(H_d_H_g_predicted)
            print(f"Appending to tau=0.01 lists: Z={Z}, Measured={H_d_H_g_measured}, Predicted={H_d_H_g_predicted}")
        elif tau == 0.1:
            Z_values_tau_0_1.append(Z)
            H_d_H_g_measured_tau_0_1.append(H_d_H_g_measured)
            H_d_H_g_predicted_tau_0_1.append(H_d_H_g_predicted)
            print(f"Appending to tau=0.1 lists: Z={Z}, Measured={H_d_H_g_measured}, Predicted={H_d_H_g_predicted}")

    # Print the contents of the lists
    print("\n--- Lists of H_d/H_g Values for Plotting ---")
    print("Z_values_tau_0_001:", Z_values_tau_0_001)
    print("H_d_H_g_measured_tau_0_001:", H_d_H_g_measured_tau_0_001)
    print("H_d_H_g_predicted_tau_0_001:", H_d_H_g_predicted_tau_0_001)

    print("Z_values_tau_0_01:", Z_values_tau_0_01)
    print("H_d_H_g_measured_tau_0_01:", H_d_H_g_measured_tau_0_01)
    print("H_d_H_g_predicted_tau_0_01:", H_d_H_g_predicted_tau_0_01)

    print("Z_values_tau_0_1:", Z_values_tau_0_1)
    print("H_d_H_g_measured_tau_0_1:", H_d_H_g_measured_tau_0_1)
    print("H_d_H_g_predicted_tau_0_1:", H_d_H_g_predicted_tau_0_1)


    # Set font sizes globally
    rc('font', size=14)  # Default text size
    rc('axes', titlesize=16)  # Title font size
    rc('axes', labelsize=14)  # Axes labels font size
    rc('xtick', labelsize=12)  # X-axis tick labels font size
    rc('ytick', labelsize=12)  # Y-axis tick labels font size
    rc('legend', fontsize=12)  # Legend font size

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot for tau = 0.1
    plt.scatter(Z_values_tau_0_1, H_d_H_g_measured_tau_0_1, color='orange', marker='s', label='Measured (τ=0.1)', s=50)
    plt.scatter(Z_values_tau_0_1, H_d_H_g_predicted_tau_0_1, color='orange', marker='s', facecolors='none', label='Predicted (τ=0.1)', s=80)
    for z, measured, predicted in zip(Z_values_tau_0_1, H_d_H_g_measured_tau_0_1, H_d_H_g_predicted_tau_0_1):
        plt.plot([z, z], [measured, predicted], color='orange', linestyle=':', linewidth=0.8)

    # Plot for tau = 0.01
    plt.scatter(Z_values_tau_0_01, H_d_H_g_measured_tau_0_01, color='green', marker='^', label='Measured (τ=0.01)', s=50)
    plt.scatter(Z_values_tau_0_01, H_d_H_g_predicted_tau_0_01, color='green', marker='^', facecolors='none', label='Predicted (τ=0.01)', s=80)
    for z, measured, predicted in zip(Z_values_tau_0_01, H_d_H_g_measured_tau_0_01, H_d_H_g_predicted_tau_0_01):
        plt.plot([z, z], [measured, predicted], color='green', linestyle=':', linewidth=0.8)

    # Plot for tau = 0.001
    plt.scatter(Z_values_tau_0_001, H_d_H_g_measured_tau_0_001, color='blue', marker='o', label='Measured (τ=0.001)', s=50)
    plt.scatter(Z_values_tau_0_001, H_d_H_g_predicted_tau_0_001, color='blue', marker='o', facecolors='none', label='Predicted (τ=0.001)', s=80)
    for z, measured, predicted in zip(Z_values_tau_0_001, H_d_H_g_measured_tau_0_001, H_d_H_g_predicted_tau_0_001):
        plt.plot([z, z], [measured, predicted], color='blue', linestyle=':', linewidth=0.8)

    # Finalize the plot
    plt.xscale('log')
    plt.xlabel("Metallicity Z")
    plt.ylabel(r"$H_d/H_g$")
    plt.title("Comparison of Predicted and Measured Dust Scale Heights vs. Metallicity")
    plt.legend()

    output_filename = "dust_scale_height_vs_metallicity.pdf"
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

    # Call the scp_transfer function
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

if __name__ == "__main__":
    main()
