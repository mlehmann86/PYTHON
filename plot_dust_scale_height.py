import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters, reconstruct_grid

# Simplified version of read_hydro_fields to read a single snapshot
def read_single_snapshot(path, snapshot, read_dust1dens=False, read_gasenergy=False):
    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    data_arrays = {}

    if read_dust1dens:
        data_arrays['dust1dens'] = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)
    if read_gasenergy:
        data_arrays['gasenergy'] = read_field_file(path, 'gasenergy', snapshot, nx, ny, nz)

    return data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, parameters

def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        print(f"Reading file: {file}")
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        print(f"File not found: {file}")
        return np.zeros((ny, nx, nz))

# Helper function to extract metallicity from the simulation name
def extract_metallicity(simulation_name):
    if "Z1dm4" in simulation_name:
        return r'$Z=0.0001$'
    elif "Z1dm3" in simulation_name:
        return r'$Z=0.001$'
    elif "Z1dm2" in simulation_name:
        return r'$Z=0.01$'
    elif "Z2dm2" in simulation_name:
        return r'$Z=0.02$'
    elif "Z5dm2" in simulation_name:
        return r'$Z=0.05$'
    else:
        return r'$Z=?$'

# Main plotting function
def plot_epsilon_with_pressure_gradient(simulations, snapshots, rmin=0.7, rmax=1.3, fontsize=14):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.3)
    
    all_epsilon_values = []
    all_pressure_gradients = []
    initial_pressure_gradient = None
    midplane_pressure_value = None

    # Load the initial gasenergy and calculate the initial pressure gradient once
    base_path = determine_base_path(simulations[0])  # Use the first simulation's base path
    data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, parameters = read_single_snapshot(
        base_path, snapshots[0], read_dust1dens=False, read_gasenergy=True
    )
    initial_gasenergy = load_initial_gasenergy(base_path, nx, ny, nz)
    midplane_pressure_value = initial_gasenergy[ny // 2, nx // 2, nz // 2]  # Midplane pressure at r=1, time=0
    initial_pressure_gradient = np.gradient(np.mean(initial_gasenergy, axis=(0, 2)), xgrid)

    for i, (sim_name, snapshot) in enumerate(zip(simulations, snapshots)):
        # Determine the base path for the simulation data
        base_path = determine_base_path(sim_name)
        
        # Read the snapshot data
        data_arrays, xgrid, ygrid, zgrid, ny, nx, nz, parameters = read_single_snapshot(
            base_path, snapshot, read_dust1dens=True, read_gasenergy=True
        )
        dust1dens = data_arrays['dust1dens']
        gasenergy = data_arrays['gasenergy']

        # Radial masking
        radial_mask = (xgrid >= rmin) & (xgrid <= rmax)
        xgrid_masked = xgrid[radial_mask]
        dust1dens_masked = dust1dens[:, radial_mask, :]
        gasenergy_masked = gasenergy[:, radial_mask, :]

        # Calculate epsilon (midplane value or vertical average)
        epsilon = dust1dens_masked / (np.ones_like(dust1dens_masked) + 1e-10)  # Assuming gasdens=1 as placeholder
        epsilon_avg = np.mean(epsilon, axis=2)  # Vertical average

        all_epsilon_values.append(epsilon_avg)

        # Compute radial pressure gradient
        pressure_gradient = np.gradient(np.mean(gasenergy_masked, axis=(0, 2)), xgrid_masked)
        all_pressure_gradients.append(pressure_gradient)

        # Extract and display the metallicity
        metallicity_label = extract_metallicity(sim_name)
        axes[i].text(0.05, 0.95, metallicity_label, transform=axes[i].transAxes,
                     fontsize=fontsize, color='white', ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))

    # Determine the global min and max values for the colorbar
    vmin = min(np.min(values) for values in all_epsilon_values)
    vmax = max(np.max(values) for values in all_epsilon_values)

    # Plot each panel with the global colorbar range and pressure gradients
    for i, (epsilon_avg, pressure_gradient) in enumerate(zip(all_epsilon_values, all_pressure_gradients)):
        # Plot epsilon
        im = axes[i].imshow(np.log10(epsilon_avg), extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid.min(), ygrid.max()],
                            aspect='auto', origin='lower', cmap='inferno', vmin=np.log10(vmin), vmax=np.log10(vmax))
        axes[i].set_xlabel(r'$r/r_0$', fontsize=fontsize)
        axes[i].set_ylabel(r'$\varphi$', fontsize=fontsize)
        axes[i].tick_params(axis='both', labelsize=fontsize)

        # Create a secondary y-axis for the pressure gradient
        ax2 = axes[i].twinx()
        scaling_factor = midplane_pressure_value  # Use the midplane pressure value for scaling
        ax2.plot(xgrid_masked, pressure_gradient * scaling_factor, color='red', label='Current Pressure Gradient')
        ax2.plot(xgrid_masked, initial_pressure_gradient * scaling_factor, color='blue', linestyle='--', label='Initial Pressure Gradient')
        ax2.tick_params(axis='y', labelsize=fontsize, colors='red')

    # Add a single legend to the first panel
    axes[0].legend(fontsize=fontsize, loc='upper left')

    # Add a single colorbar for all panels
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label(r'$\log_{10}(\epsilon)$', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Save the plot without opening a window
    output_filename = "epsilon_with_pressure_gradient.pdf"
    plt.savefig(output_filename)
    
    # Transfer the plot file using SCP
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")

# List of simulations and individual snapshot numbers
simulations = [
    "cos_b1d0_us_St1dm1_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
    "cos_b1d0_us_St1dm1_Z2dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150",
    "cos_b1d0_us_St1dm1_Z5dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_tap"
]
snapshots = [36, 36, 121]  # Example snapshot numbers for each simulation

# Run the plotting function
plot_epsilon_with_pressure_gradient(simulations, snapshots)
