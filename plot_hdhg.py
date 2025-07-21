import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.optimize import curve_fit
import warnings
from plotting_functions import gaussian  # Assuming the gaussian function is defined in this module

# Function to read parameters from the summary file
def read_parameters(summary_file):
    parameters = {}
    with open(summary_file, 'r') as f:
        for line in f:
            key, value = line.split('=')
            parameters[key.strip()] = float(value)
    return parameters

# Function to reconstruct grid based on parameters
def reconstruct_grid(parameters):
    nx, ny, nz = int(parameters['NX']), int(parameters['NY']), int(parameters['NZ'])
    xgrid = np.linspace(parameters['XMIN'], parameters['XMAX'], nx)
    ygrid = np.linspace(parameters['YMIN'], parameters['YMAX'], ny)
    zgrid = np.linspace(parameters['ZMIN'], parameters['ZMAX'], nz)
    return xgrid, ygrid, zgrid, ny, nx, nz

# Simplified version of read_hydro_fields to read a single snapshot
def read_single_snapshot(path, snapshot, read_dust1dens=False):
    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    data_arrays = {}
    
    if read_dust1dens:
        data_arrays['dust1dens'] = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)

    return data_arrays, xgrid, ygrid, zgrid, parameters

def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        print(f"Reading file: {file}")
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        print(f"File not found: {file}")
        return np.zeros((ny, nx, nz))

# Function to compute dust scale height for each (x, y) in the grid
def compute_dust_scale_height(dust1dens, zgrid):
    H_d = np.zeros(dust1dens.shape[1:3])  # Assuming shape is (z, y, x)
    
    for i in range(H_d.shape[1]):  # Loop over x-grid
        for j in range(H_d.shape[0]):  # Loop over y-grid
            dust_profile = np.mean(dust1dens[:, j, i], axis=0)
            p0 = [np.max(dust_profile), 0, np.std(zgrid), np.min(dust_profile)]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(gaussian, zgrid, dust_profile, p0=p0, maxfev=1000)
                H_d[j, i] = abs(popt[2])  # Store the dust scale height (sigma)
            except RuntimeError:
                H_d[j, i] = 1e-15  # Default value when fitting fails
    
    return H_d

# Main plotting function
def plot_dust_scale_height(simulation_name, snapshot, azimuth_angle):
    # Define the base path for the simulation data
    base_path = determine_base_path(simulation_name)
    
    # Load the simulation data
    data_arrays, xgrid, ygrid, zgrid, parameters = read_single_snapshot(base_path, snapshot, read_dust1dens=True)
    dust1dens = data_arrays['dust1dens']
    
    # Compute the dust scale height
    H_d = compute_dust_scale_height(dust1dens, zgrid)
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the 2D contour of the dust scale height in the left panel
    cax = axes[0].imshow(H_d, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
                         origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=axes[0])
    axes[0].set_title('Dust Scale Height (H_d)')
    axes[0].set_xlabel('Radial (x)')
    axes[0].set_ylabel('Azimuthal (y)')
    
    # Plot the radial profile for a specified azimuth in the right panel
    azimuth_idx = (np.abs(ygrid - azimuth_angle)).argmin()
    radial_profile = H_d[azimuth_idx, :]
    
    axes[1].plot(xgrid, radial_profile, label=f'Azimuth = {azimuth_angle}')
    axes[1].set_title('Radial Profile of Dust Scale Height')
    axes[1].set_xlabel('Radial (x)')
    axes[1].set_ylabel('H_d')
    axes[1].legend()

    # Save the plot as a PDF
    plt.tight_layout()
    output_filename = f"{simulation_name}_snapshot_{snapshot}_dust_scale_height.pdf"
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Command-line argument handling
    parser = argparse.ArgumentParser(description="Plot 2D contour and radial profile of dust scale height (H_d).")
    parser.add_argument('simulation_name', type=str, help="Name of the simulation.")
    parser.add_argument('snapshot', type=int, help="Snapshot number.")
    parser.add_argument('--azimuth', type=float, default=0, help="Azimuth angle for the radial profile (default=0).")
    
    args = parser.parse_args()
    
    # Call the plotting routine
    plot_dust_scale_height(args.simulation_name, args.snapshot, args.azimuth)
