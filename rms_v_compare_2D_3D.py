import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from data_storage import determine_base_path, scp_transfer
from data_reader import reconstruct_grid, read_parameters

# Define the local directory for file transfer
local_directory = "/Users/mariuslehmann/Downloads/Profiles"

# Function to load simulation data
def load_simulation_data(simulation_dirs):
    data = {}
    
    for sim_dir in simulation_dirs:
        print(f"Loading data for: {sim_dir}")
        subdir_path = determine_base_path(sim_dir)
        npz_file = os.path.join(subdir_path, f"{os.path.basename(sim_dir)}_quantities.npz")
        summary_file = os.path.join(subdir_path, "summary0.dat")
        
        try:
            loaded_data = np.load(npz_file)
            parameters = read_parameters(summary_file)
            # Reconstruct the grid to obtain zgrid
            xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
            h0 = float(parameters['ASPECTRATIO'])  # Aspect ratio used for scaling
            
            available_keys = loaded_data.keys()
            
            # Check if rms_vr, rms_vphi, rms_vz, profiles, and time exist, and store them if available
            rms_vr_present = 'rms_vr' in available_keys and 'rms_vr_profile' in available_keys
            rms_vphi_present = 'rms_vphi' in available_keys and 'rms_vphi_profile' in available_keys
            rms_vz_present = 'rms_vz' in available_keys and 'rms_vz_profile' in available_keys
            time_present = 'time' in available_keys
            
            if rms_vr_present and rms_vphi_present and rms_vz_present and time_present:
                # Store scaled RMS velocities, time, profiles, and zgrid
                data[sim_dir] = {
                    'rms_vr': loaded_data['rms_vr'] / h0,
                    'rms_vphi': loaded_data['rms_vphi'] / h0,
                    'rms_vz': loaded_data['rms_vz'] / h0,
                    'rms_vr_profile': loaded_data['rms_vr_profile'] / h0,
                    'rms_vphi_profile': loaded_data['rms_vphi_profile'] / h0,
                    'rms_vz_profile': loaded_data['rms_vz_profile'] / h0,
                    'time': loaded_data['time'],
                    'zgrid': zgrid  # Use reconstructed zgrid
                }
            else:
                print(f"Skipping {sim_dir} - missing required data.")
        
        except FileNotFoundError:
            print(f"File not found: {npz_file}")
    
    return data

# Function to plot both time evolution and vertical profiles of the RMS velocities
def plot_rms_velocities(simulations_3D, simulations_2D, y_ranges=None, z_ranges=None, labels=None):
    # Plot time evolution of RMS velocities
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 vertical panels for the 3 velocities
    
    # Load data for the simulations
    data_3D = load_simulation_data(simulations_3D)
    data_2D = load_simulation_data(simulations_2D)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors for different metallicities

    # Plot RMS radial, azimuthal, and vertical velocities (time evolution)
    for i, (ax, velocity_key, ylabel) in enumerate(zip(axes, 
                                                      ['rms_vr', 'rms_vphi', 'rms_vz'], 
                                                      [r'RMS$(v_r)/c_0$', r'RMS$(v_\varphi)/c_0$', r'RMS$(v_z)/c_0$'])):
        # Plot 3D simulations with solid lines
        for idx, sim_dir in enumerate(simulations_3D):
            if sim_dir in data_3D:
                data = data_3D[sim_dir]
                time = data['time']  # Get the time array
                ax.plot(time, data[velocity_key], label=f"3D {labels[idx]}", color=colors[idx], linestyle='-')
        
        # Plot 2D simulations with dashed lines
        for idx, sim_dir in enumerate(simulations_2D):
            if sim_dir in data_2D:
                data = data_2D[sim_dir]
                time = data['time']  # Get the time array
                ax.plot(time, data[velocity_key], label=f"2D {labels[idx]}", color=colors[idx], linestyle='--')
        
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        # Set y-axis range if provided
        if y_ranges and i < len(y_ranges):
            ax.set_ylim(y_ranges[i])

    # Set common labels and title for time evolution plot
    axes[-1].set_xlabel('Time (Orbits)', fontsize=14)
    plt.tight_layout()
    
    # Save the time evolution plot to a PDF file
    output_filename = f"rms_velocities_time_evolution_beta_{beta}.pdf"
    plt.savefig(output_filename)
    plt.close()  # Close the plot to avoid opening the window

    # Transfer the file
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    print(f"Time evolution plot transferred to {local_directory}")
    
    # Plot vertical profiles of time-averaged RMS velocities
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 vertical panels for the 3 profiles

    # Plot RMS radial, azimuthal, and vertical velocities (vertical profiles)
    for i, (ax, profile_key, ylabel) in enumerate(zip(axes, 
                                                     ['rms_vr_profile', 'rms_vphi_profile', 'rms_vz_profile'], 
                                                     [r'RMS$(v_r)/c_0$', r'RMS$(v_\varphi)/c_0$', r'RMS$(v_z)/c_0$'])):
        # Plot 3D simulations with solid lines
        for idx, sim_dir in enumerate(simulations_3D):
            if sim_dir in data_3D:
                data = data_3D[sim_dir]
                zgrid = data['zgrid']  # Use reconstructed zgrid
                ax.plot(zgrid, data[profile_key], label=f"3D {labels[idx]}", color=colors[idx], linestyle='-')
        
        # Plot 2D simulations with dashed lines
        for idx, sim_dir in enumerate(simulations_2D):
            if sim_dir in data_2D:
                data = data_2D[sim_dir]
                zgrid = data['zgrid']  # Use reconstructed zgrid
                ax.plot(zgrid, data[profile_key], label=f"2D {labels[idx]}", color=colors[idx], linestyle='--')
        
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        # Set z-axis range if provided
        if z_ranges and i < len(z_ranges):
            ax.set_ylim(z_ranges[i])

    # Set common labels and title for vertical profile plot
    axes[-1].set_xlabel('z', fontsize=14)
    plt.tight_layout()
    
    # Save the vertical profile plot to a PDF file
    output_filename = f"rms_velocities_vertical_profiles_beta_{beta}.pdf"
    plt.savefig(output_filename)
    plt.close()  # Close the plot to avoid opening the window

    # Transfer the file
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    print(f"Vertical profile plot transferred to {local_directory}")

# Function to plot the Z=0 simulations for time evolution in 1 panel with logarithmic y-axis
def plot_rms_velocities_z0(simulation_3D_z0, simulation_2D_z0, y_ranges=None):
    # Load data for the Z=0 simulations
    data_3D_z0 = load_simulation_data([simulation_3D_z0])
    data_2D_z0 = load_simulation_data([simulation_2D_z0])
    
    # Create a single panel plot for time evolution of RMS velocities with distinct colors for different components
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_z0 = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each velocity component

    for idx, (velocity_key, ylabel) in enumerate(zip(['rms_vr', 'rms_vphi', 'rms_vz'], 
                                                     [r'RMS$(v_r)$', r'RMS$(v_\varphi)$', r'RMS$(v_z)$'])):
        # Plot 3D Z=0 simulation with solid lines
        if simulation_3D_z0 in data_3D_z0:
            data = data_3D_z0[simulation_3D_z0]
            time = data['time']
            ax.plot(time, data[velocity_key], label=f"3D {ylabel}", color=colors_z0[idx], linestyle='-')
        
        # Plot 2D Z=0 simulation with dashed lines
        if simulation_2D_z0 in data_2D_z0:
            data = data_2D_z0[simulation_2D_z0]
            time = data['time']
            ax.plot(time, data[velocity_key], label=f"2D {ylabel}", color=colors_z0[idx], linestyle='--')

    ax.set_yscale('log')  # Set y-axis to logarithmic
    ax.set_ylabel("RMS velocities (log scale)")
    ax.set_xlabel("Time (Orbits)")
    ax.legend(fontsize=12)
    ax.grid(True)

    # Save the single panel time evolution plot for Z=0 to a PDF file
    output_filename_time_z0 = f"rms_velocities_time_evolution_Z0_beta_{beta}.pdf"
    plt.tight_layout()
    plt.savefig(output_filename_time_z0)
    plt.close()

    # Transfer the file
    scp_transfer(output_filename_time_z0, local_directory, "mariuslehmann")
    print(f"Z=0 time evolution plot transferred to {local_directory}")

if __name__ == "__main__":
    # Parse beta argument from command line
    beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001  # Default to beta=0.001 if not provided

    # Define simulation lists for beta=1 and beta=0.001
    if beta == 1:
        simulations_3D = [
            "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150",
            "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
            "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
        ]
        simulations_2D = [
            "cos_nu1dm9_bet1d0_us_nodust_zpm0025_fim053_ss203",
            "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
            "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
           "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
        ]
        # Also plot Z=0 cases
        simulation_3D_z0 = "cos_b1d0_us_nodust_r6H_z08H_fim053_ss203_3D_2PI_LR150"  # Z=0 3D
        simulation_2D_z0 = "cos_nu1dm9_bet1d0_us_nodust_zpm0025_fim053_ss203"  # Z=0 2D
        labels = [r"$Z=0$", r"$Z=0.01$", r"$Z=0.05$", r"$Z=0.1$"]
    else:  # beta = 0.001
        simulations_3D = [
            "cos_b1dm3_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
            "cos_b1dm3_us_St5dm2_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap",
            "cos_b1dm3_us_St5dm2_Z1dm1_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
        ]
        simulations_2D = [
            "cos_b1dm3_us_St1dm2_Z1dm2_r6H_z08H_fim053_ss203_stnew_LR150",
            "cos_b1dm3_us_St1dm2_Z5dm2_r6H_z08H_fim053_ss203_stnew_LR150_tap",
           "cos_b1dm3_us_St1dm2_Z1dm1_r6H_z08H_fim053_ss203_stnew_LR150_tap"
        ]
        labels = [r"$Z=0.01$", r"$Z=0.05$", r"$Z=0.1$"]
    
    # Adjust the y-range and z-range for each panel (optional)
    y_ranges = [(0, 0.125), (0, 0.3), (0, 0.02)]  # Example ranges for radial, azimuthal, and vertical RMS velocities (time evolution)
    z_ranges = [(0, 0.125), (0, 0.3), (0, 0.02)]  # Example ranges for vertical profiles
    
    # Plot the original time evolution and vertical profiles for the selected beta case
    plot_rms_velocities(simulations_3D, simulations_2D, y_ranges=y_ranges, z_ranges=z_ranges, labels=labels)

    # Plot Z=0 simulations only if beta=1
    if beta == 1:
        plot_rms_velocities_z0(simulation_3D_z0, simulation_2D_z0, y_ranges=y_ranges)
