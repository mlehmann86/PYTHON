import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from data_storage import determine_base_path  # Importing determine_base_path as requested
from data_reader import read_parameters, reconstruct_grid  # Importing from data_reader

# Simplified version of read_hydro_fields to read a single snapshot
def read_single_snapshot(path, snapshot, read_dust1dens=False):
    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)  # Using the imported function
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)  # Using reconstruct_grid from data_reader

    data_arrays = {}
    
    if read_dust1dens:
        data_arrays['dust1dens'] = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)

    return data_arrays, xgrid, ygrid, zgrid, parameters

# Corrected version for reading and reshaping the file
def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        print(f"Reading file: {file}")
        # Ensure correct reshaping and transposing of the data
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        print(f"File not found: {file}")
        return np.zeros((ny, nx, nz))

# Main function to plot radial, vertical, and azimuthal profiles of dust1dens
def plot_profiles_with_non_averaged(simulation_name, snapshot):
    # Define the base path for the simulation data
    base_path = determine_base_path(simulation_name)
    
    # Load the simulation data
    data_arrays, xgrid, ygrid, zgrid, parameters = read_single_snapshot(base_path, snapshot, read_dust1dens=True)
    dust1dens = data_arrays['dust1dens']
    
    # Print the shape of dust1dens
    print(f"Shape of dust1dens: {dust1dens.shape}")
    
    # Compute the azimuthally averaged dust1dens
    dust1dens_avg = np.mean(dust1dens, axis=0)  # Azimuthal averaging over y-axis (axis=0)

    # Extract radial profile at the midplane (z = nz/2)
    midplane_idx = len(zgrid) // 2
    radial_profile_avg = dust1dens_avg[:, midplane_idx]
    radial_profile_non_avg = dust1dens[0, :, midplane_idx]  # y = 0

    # Extract vertical profile at the center radius (x = nx/2)
    center_radius_idx = len(xgrid) // 2
    vertical_profile_avg = dust1dens_avg[center_radius_idx, :]
    vertical_profile_non_avg = dust1dens[0, center_radius_idx, :]  # y = 0

    # Extract azimuthal profile at (x = nx/2, z = nz/2)
    azimuthal_profile = dust1dens[:, center_radius_idx, midplane_idx]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the radial profile at the midplane (z = nz/2) in the first subplot
    ax1.plot(xgrid, radial_profile_avg, label=f'Azimuthally Averaged (z={zgrid[midplane_idx]:.2f})', marker='o')
    ax1.plot(xgrid, radial_profile_non_avg, label=f'Non-Averaged (y=0, z={zgrid[midplane_idx]:.2f})', marker='x')
    ax1.set_xlabel('Radial Coordinate (x)')
    ax1.set_ylabel('Dust Density')
    ax1.set_title('Radial Profile at Midplane')
    ax1.legend()

    # Plot the vertical profile at the center radius (x = nx/2) in the second subplot
    ax2.plot(zgrid, vertical_profile_avg, label=f'Azimuthally Averaged (x={xgrid[center_radius_idx]:.2f})', marker='o')
    ax2.plot(zgrid, vertical_profile_non_avg, label=f'Non-Averaged (y=0, x={xgrid[center_radius_idx]:.2f})', marker='x')
    ax2.set_xlabel('Vertical Coordinate (z)')
    ax2.set_ylabel('Dust Density')
    ax2.set_title('Vertical Profile at Center Radius')
    ax2.legend()

    # Plot the azimuthal profile along the y-axis in the third subplot
    ax3.plot(ygrid, azimuthal_profile, label=f'Azimuthal Profile (x={xgrid[center_radius_idx]:.2f}, z={zgrid[midplane_idx]:.2f})', marker='o')
    ax3.set_xlabel('Azimuthal Coordinate (y)')
    ax3.set_ylabel('Dust Density')
    ax3.set_title('Azimuthal Profile (Center Radius & Midplane)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('profiles_azimuthally_averaged_and_non_averaged_with_azimuthal.pdf')
    plt.show()

    # Halt execution after plotting
    print("Halting execution after plotting radial, vertical, and azimuthal profiles.")
    return

if __name__ == "__main__":
    # Command-line argument handling
    parser = argparse.ArgumentParser(description="Plot azimuthally averaged and non-averaged dust1dens profiles.")
    parser.add_argument('simulation_name', type=str, help="Name of the simulation.")
    parser.add_argument('snapshot', type=int, help="Snapshot number.")
    
    args = parser.parse_args()
    
    # Call the function to plot the profiles
    plot_profiles_with_non_averaged(args.simulation_name, args.snapshot)
