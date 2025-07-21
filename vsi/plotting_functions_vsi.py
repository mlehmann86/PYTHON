# Standard library imports
import os
import sys
import warnings
import subprocess
import math

# Third-party library imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed

from data_storage_vsi import save_simulation_quantities  # Import data storage functions

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import the module from the parent directory
from data_storage import  scp_transfer
from data_reader import determine_nt, read_parameters, reconstruct_grid

# Optionally, remove the parent directory from sys.path after import to clean up
sys.path.pop(0)

def plot_initial_profiles(data_arrays, xgrid, ygrid, zgrid, time_array, output_path):
    """
    Plot debugging profiles in a spherical coordinate frame. 
    Includes gasdens, gasvx, gasvy, gasvz, plotted against height z.
    """
    # Get the index for the initial time step: time=0
    gasdens = data_arrays['gasdens']
    gasvx = data_arrays['gasvx']
    gasvy = data_arrays['gasvy']
    gasvz = data_arrays['gasvz']

    nx = gasdens.shape[1]
    ny = gasdens.shape[0]
    nz = gasdens.shape[2]

    # Set the initial time index
    time_index = 0

    # Convert zgrid (theta) to actual height z
    theta_grid = zgrid  # Rename for clarity
    r_grid = xgrid
    z_values = r_grid[nx // 2] * np.cos(theta_grid)  # Compute z for mid-radius (r = xgrid[nx/2])

    # Print the minimum and maximum values of the computed z
    print(f"Minimum height (z): {np.min(z_values)}")
    print(f"Maximum height (z): {np.max(z_values)}")

    def plot_variable(variable, var_name, y_label, subdir_name):
        """Helper function to plot a given variable."""
        plt.figure(figsize=(12, 6))

        # Plot r (xgrid) versus variable(0, :, nz/2, time_index)
        plt.subplot(1, 3, 1)
        plt.plot(r_grid, variable[0, :, nz//2, time_index], label=f'Time = {time_array[time_index]}')
        plt.xlabel('Radial Location (r)')
        plt.ylabel(f'{y_label} (y=0, z=mid-plane)')
        plt.title(f'Raw Profile: r vs {var_name}(y=0, :, z=mid-plane)')
        plt.legend()
        plt.grid(True)

        # Plot ygrid versus variable(:, nx/2, nz/2, time_index) if ny > 1
        if ny > 1:
            plt.subplot(1, 3, 2)
            plt.plot(ygrid, variable[:, nx//2, nz//2, time_index], label=f'Time = {time_array[time_index]}')
            plt.xlabel('Azimuthal Location (y)')
            plt.ylabel(f'{y_label} (:, x=mid-radial, z=mid-plane)')
            plt.title(f'Raw Profile: y vs {var_name}(:, x=mid-radial, z=mid-plane)')
            plt.legend()
            plt.grid(True)

        # Plot z (height) versus variable(0, nx/2, :, time_index)
        plt.subplot(1, 3, 3)
        plt.plot(z_values, variable[0, nx//2, :, time_index], label=f'Time = {time_array[time_index]}')
        plt.xlabel('Height (z)')
        plt.ylabel(f'{y_label} (y=0, x=mid-radial)')
        plt.title(f'Raw Profile: z vs {var_name}(y=0, x=mid-radial, :)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save the plot with the subdirectory name in the file name
        pdf_filename = f"{subdir_name}_debugging_profiles_{var_name}.pdf"
        output_filepath = os.path.join(output_path, pdf_filename)
        plt.savefig(output_filepath)
        plt.close()
        print(f"#######################################")
        print(f"Debug plot for {var_name} saved to {output_filepath}")
        print(f"#######################################")

        # Call the scp_transfer function
        scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

    # Extract the subdirectory name
    subdir_name = os.path.basename(output_path)

    # Plot each variable
    plot_variable(gasdens, 'gasdens', 'Gas Density', subdir_name)
    plot_variable(gasvx, 'gasvx', 'Gas Azimuthal Velocity', subdir_name)
    plot_variable(gasvy, 'gasvy', 'Gas Radial Velocity', subdir_name)
    plot_variable(gasvz, 'gasvz', 'Gas Vertical Velocity', subdir_name)

###########################################################################################################

def create_simulation_movie_axi(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, dust=False):
    # Extract necessary data for the plots
    gasdens = data_arrays['gasdens']
    if dust:
        dust1dens = data_arrays['dust1dens']
        dust1vz = data_arrays['dust1vz']
    else:
        gasvx = data_arrays['gasvx']
        gasvz = data_arrays['gasvz']
    gasenergy = data_arrays['gasenergy']


    if not dust:
        # Initial azimuthal velocity at time=0 (precompute)
        gasvx0 = gasvx[:, :, :, 0]
        # Calculate delta v_phi (deviation from initial azimuthal velocity)
        delta_v_phi = gasvx - gasvx0[:, :, :, np.newaxis]

    # Define initial values for scaling
    initial_rho_value = gasdens[0, len(xgrid) // 2, len(zgrid) // 2, 0]
    initial_pg_value = gasenergy[0, len(xgrid) // 2, len(zgrid) // 2, 0]

    # Initial gas density profile for reference
    initial_gas_density = np.mean(np.mean(gasdens[:, :, :, 0], axis=0), axis=1)  # Azimuthal and vertical average

    # Calculate the initial pressure gradient
    initial_pressure_gradient = np.mean(np.gradient(np.mean(gasenergy[:, :, :, 0], axis=0), xgrid, axis=0), axis=1)

    # Scale the initial pressure gradient
    initial_scaled_pg = initial_pressure_gradient / initial_pg_value

    # Load saved quantities
    quantities_file = os.path.join(output_path, f"{os.path.basename(output_path)}_quantities.npz")
    loaded_data = np.load(quantities_file)
    alpha_r = loaded_data['alpha_r']
    alpha_z = loaded_data['alpha_z']
    rms_vz = loaded_data['rms_vz']
    if dust:
        max_epsilon = loaded_data['max_epsilon']
        H_d_array = loaded_data['H_d']

    # Read disk aspect ratio (precomputed once)
    summary_file = os.path.join(output_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    aspectratio = parameters['ASPECTRATIO']
    H_g = float(aspectratio)  # Disk aspect ratio
   
    if dust:
        # Roche density calculation, reshaped to match gasdens and dust1dens
        roche_density = (9 / (4 * np.pi * np.power(xgrid, 3)))[np.newaxis, :, np.newaxis]


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # Adjust horizontal and vertical spacing between plots
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.25)

    # Create the secondary y-axis for the radial plot
    ax3_right = ax3.twinx()

    # Set fixed positions for colorbars
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("top", size="5%", pad=0.3)  # Adjusted for better placement
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("top", size="5%", pad=0.3)  # Adjusted for better placement

    cbar1, cbar2 = None, None
    time_text = fig.text(0.51, 0.95, '', ha='center', va='center', fontsize=12)  # Add time annotation

    if dust:
        # Initialize a list to store indices where Roche density is exceeded
        roche_exceeded_times = []

    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(time_array), desc="Creating Movie", ncols=100)

    def animate(i):
        nonlocal cbar1, cbar2

        # Apply radial mask
        radial_mask = (xgrid >= 0.7) & (xgrid <= 1.3)
        xgrid_masked = xgrid[radial_mask]

        # Calculate epsilon averages without masking first
        if dust:
            plotar_az_avg = np.mean(dust1dens[:, :, :, i] / gasdens[:, :, :, i], axis=0)  # Shape: (radial, z)
            dustvz_az_avg = np.mean(dust1vz[:, :, :, i], axis=0)  # Shape: (radial, z)
        else:
            plotar_az_avg = np.mean(delta_v_phi[:, :, :, i] , axis=0)  # Shape: (radial, z)
            gasvz_az_avg = np.mean(gasvz[:, :, :, i], axis=0)  # Shape: (radial, z)

        # Apply radial mask AFTER averaging
        plotar_az_avg_masked = plotar_az_avg[radial_mask, :]  # Now mask the radial axis
        if dust:
            dustvz_az_avg_masked = dustvz_az_avg[radial_mask, :]  # Now mask the radial axis
        else:
            gasvz_az_avg_masked = gasvz_az_avg[radial_mask, :]  # Now mask the radial axis

        # Adjust theta_grid (zgrid) to ensure symmetric alignment
        theta_grid = zgrid - np.mean(zgrid) + np.pi / 2  # Shift zgrid to center around midplane (pi/2)

        # Compute proper grid edges for pcolormesh
        theta_edges = np.linspace(theta_grid.min(), theta_grid.max(), len(theta_grid) + 1)  # Edges for theta
        r_edges = np.linspace(xgrid_masked.min(), xgrid_masked.max(), len(xgrid_masked) + 1)  # Edges for radius

        # Create 2D grids for the edges
        r_edge_grid, theta_edge_grid = np.meshgrid(r_edges, theta_edges, indexing='ij')  # Grid of edges

        # Convert edges to Cartesian coordinates
        X_edges = r_edge_grid * np.sin(theta_edge_grid)  # X-coordinate for edges
        Z_edges = r_edge_grid * np.cos(theta_edge_grid)  # Z-coordinate for edges


        # Use pcolormesh with properly computed edges
        if dust:
            im1 = ax1.pcolormesh(X_edges, Z_edges, np.sqrt(plotar_az_avg_masked), cmap='hot', shading='auto')
            im2 = ax2.pcolormesh(X_edges, Z_edges, dustvz_az_avg_masked, cmap='hot', shading='auto')
        else:
            im1 = ax1.pcolormesh(X_edges, Z_edges, plotar_az_avg_masked, cmap='hot', shading='auto')
            im2 = ax2.pcolormesh(X_edges, Z_edges, gasvz_az_avg_masked, cmap='hot', shading='auto')

        # Adjust aspect ratio for flaring disc geometry
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')


        # Add labels to the upper panels with the desired LaTeX formatting
        if dust:
            ax1.text(0.025, 0.95, r"$\epsilon$", transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        else:
            ax1.text(0.025, 0.95, r"$\delta v_{g\varphi}$", transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
        ax2.text(0.025, 0.95, r"$v_{gz}$", transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


        if cbar1 is None and cbar2 is None:
            cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
        else:
            for c in cbar1.ax.collections:
                c.remove()
            for c in cbar2.ax.collections:
                c.remove()
            cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
            cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')

        # Update tick marks and labels
        if dust:
            cbar1.set_ticks(np.linspace(np.sqrt(np.min(plotar_az_avg_masked)), np.sqrt(np.max(plotar_az_avg_masked)), 5))
            cbar2.set_ticks(np.linspace(np.min(dustvz_az_avg_masked), np.max(dustvz_az_avg_masked), 5))
        else:
            cbar1.set_ticks(np.linspace(np.min(plotar_az_avg_masked), np.max(plotar_az_avg_masked), 5))
            cbar2.set_ticks(np.linspace(np.min(gasvz_az_avg_masked), np.max(gasvz_az_avg_masked), 5))
        # Update the plots with relevant data for this time step
        ax3.clear()
        ax3_right.clear()

        # Compute the relevant quantities
        radial_gas_density = np.mean(np.mean(gasdens[:, :, :, i], axis=0), axis=1)
        pressure_gradient = np.mean(np.gradient(np.mean(gasenergy[:, :, :, i], axis=0), xgrid, axis=0), axis=1)
        scaled_pg = pressure_gradient / initial_pg_value

        # Plot gas density on the primary y-axis
        line1, = ax3.plot(xgrid, radial_gas_density / initial_rho_value, color='red', lw=2, label=r"Current $\rho_{g}$")
        line2, = ax3.plot(xgrid, initial_gas_density / initial_rho_value, color='black', lw=2, linestyle='--', label=r"Initial $\rho_{g}$")

        # Plot pressure gradients on the secondary y-axis
        line3, = ax3_right.plot(xgrid, scaled_pg, color='blue', lw=2, label=r"$\frac{dP}{dr}$")
        line4, = ax3_right.plot(xgrid, initial_scaled_pg, color='black', lw=2, linestyle='--', label=r"Initial $\frac{dP}{dr}$")

        # Set axis labels and limits
        ax3.set_xlabel(r"$r/r_{0}$")
        ax3.set_xlim([0.8, 1.2])
        ax3.set_ylim([0, 3])  # Fixed y-axis range for the left axis
        ax3_right.set_ylim([-40, 5])  # Fixed y-axis range for the right axis

        # Use a combined legend for both axes
        lines = [line1, line2, line3, line4]
        labels = [line.get_label() for line in lines]
        ax3.legend(lines, labels, loc='center right', fontsize='small')

        if dust:
            # Check if the Roche density is exceeded and store the time index
            radial_mask = (xgrid > 0.8) & (xgrid < 1.2)
            if np.any((gasdens[:, :, :, i] + dust1dens[:, :, :, i] >= roche_density) & radial_mask[np.newaxis, :, np.newaxis]):
                roche_exceeded_times.append(i)
    
        ax4.clear()
        ax4.plot(time_array[:i + 1], alpha_r[:i + 1], label=r'$\alpha_r(t)$')
        ax4.plot(time_array[:i + 1], alpha_z[:i + 1], label=r'$\alpha_z(t)$')
        ax4.plot(time_array[:i + 1], rms_vz[:i + 1], label=r'RMS(Vertical Velocity)')
        if dust:
            ax4.plot(time_array[:i + 1], max_epsilon[:i + 1], label=r'Max $\epsilon$', color='green', lw=2)
    
            # Highlight the points where Roche density was exceeded
            ax4.plot(np.array(time_array)[roche_exceeded_times], np.array(max_epsilon)[roche_exceeded_times], 
                 label=None, color='green', marker='o', linestyle='None')

            ax4.plot(time_array[:i + 1], H_d_array[:i + 1] / H_g, label=r'Dust Scale Height $H_d/H_g$')
        ax4.set_xlim([0, np.max(time_array)])  # Fixed x-axis range
        ax4.set_ylim([1e-5, 1e3])  # Fixed y-axis range
        ax4.set_yscale('log')
        ax4.set_xlabel("Time [Orbits]")
        ax4.set_ylabel(" ")  # Empty vertical axis label
        ax4.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax4.legend(loc='upper left', fontsize='small')  # Fixed legend position

        # Add labels to the upper panels
        ax1.set_xlabel(r'$r/r_0$')
        ax1.set_ylabel(r'$z/r_0$')
        ax2.set_xlabel(r'$r/r_0$')
        ax2.set_ylabel(r'$z/r_0$')

        # Set axis limits for the flaring disc
        # Set axis limits to enforce symmetry
        ax1.set_xlim([0.8, 1.2])  # Explicitly set y-axis range
        ax1.set_ylim([-0.5, 0.5])  # Explicitly set y-axis range
        ax2.set_xlim([0.8, 1.2])  # Explicitly set y-axis range
        ax2.set_ylim([-0.5, 0.5])  # Explicitly set y-axis range

        # Update the time annotation
        time_text.set_text(f"{time_array[i]:.2f} Orbits")

        # Update the progress bar
        progress_bar.update(1)

    # Create the movie
    ani = animation.FuncAnimation(fig, animate, frames=len(time_array), repeat=False)
    subdir_name = os.path.basename(output_path)  # Extract the subdirectory name from the output path
    movie_filename = f"movie_{subdir_name}.mp4"
    output_filepath = os.path.join(output_path, movie_filename)  # Full path to the movie file
    ani.save(output_filepath, writer='ffmpeg', dpi=300, fps=15)

    # Close the progress bar
    progress_bar.close()

    plt.close(fig)
    print(f"#######################################")
    print(f"simulation movie saved to {output_filepath}")
    print(f"#######################################")

    # Call the scp_transfer function to transfer the movie to your laptop
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Movies", "mariuslehmann")


#############################################################################################################


def plot_alpha_r(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, nsteps, dust=False):
    """
    Calculate and plot the turbulent alpha parameter (alpha_r), RMS of vertical dust velocity, 
    maximum value of epsilon, the dust scale height (H_d), total gas mass (m_gas), 
    total dust mass (m_dust) scaled by initial values, average metallicity <Z>,
    and highlight times when the Roche density is achieved during the simulation.

    Parameters:
    - data_arrays: Dictionary containing the simulation data.
    - xgrid: Radial grid.
    - zgrid: Vertical grid.
    - time_array: Array of time steps.
    - output_path: Directory to save the plot.
    """

    # Apply radial masks
    radial_mask = (xgrid >= 0.9) & (xgrid <= 1.2)
    radial_mask_2 = (xgrid >= 0.9) & (xgrid <= 1.2)

    nz = len(zgrid)
    ny = len(ygrid)

    if dust:
        # Calculate the Roche density at each radial location
        roche_density_masked = 9 / (4 * np.pi * xgrid[radial_mask]**3)

    # Initialize arrays to hold the computed values
    vort_min = np.zeros(len(time_array))  # New 
    vortz_avg = np.zeros(len(time_array))  # New 
    vortr_avg = np.zeros(len(time_array))  # New 
    alpha_r = np.zeros(len(time_array))
    alpha_z = np.zeros(len(time_array))
    rms_vr = np.zeros(len(time_array))
    rms_vphi = np.zeros(len(time_array))
    rms_vz = np.zeros(len(time_array))
    max_vz = np.zeros(len(time_array))
    min_vz = np.zeros(len(time_array))
    max_epsilon = np.zeros(len(time_array))
    H_d_array = np.zeros(len(time_array))
    m_gas = np.zeros(len(time_array))
    m_dust = np.zeros(len(time_array)) if dust else None
    avg_metallicity = np.zeros(len(time_array))  # New array for <Z>


    # New arrays for vertical profiles
    rms_vr_profile = np.zeros(nz)
    rms_vphi_profile = np.zeros(nz)
    rms_vz_profile = np.zeros(nz)


    roche_times = []  # To store the time steps where the Roche density is reached

    # Read disk aspect ratio (precomputed once)
    summary_file = os.path.join(output_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    aspectratio = parameters['ASPECTRATIO']

    # Disk aspect ratio H_g
    H_g = float(aspectratio)

    # Precompute the std of zgrid if it's constant
    zgrid_std = np.std(zgrid)

    r = xgrid[np.newaxis, :, np.newaxis]  # Radial grid expanded to shape (1, nx, 1)
    #Smoothing in (azimuthal, radial) directions
    nsx=100 #def=100
    nsy=10 #def=10

    # Compute the initial gas and dust mass at t_idx=0 for scaling
    gasdens_initial = data_arrays['gasdens'][:, radial_mask, :, 0]
    m_gas_initial = np.sum(gasdens_initial)
    m_dust_initial = None
    if dust:
        dust1dens_initial = data_arrays['dust1dens'][:, radial_mask, :, 0]
        m_dust_initial = np.sum(dust1dens_initial)

    print('COMPUTING TIME EVOLUTION OF VARIOUS QUANTITIES')

    def process_time_step(t_idx):
        """Process a single time step."""
        # Extract data for this time step and apply radial masks dynamically
        gasdens_t = data_arrays['gasdens'][:, radial_mask, :, t_idx]
        dust1dens_t = data_arrays['dust1dens'][:, radial_mask, :, t_idx] if dust else None

        # Compute total gas mass m_gas
        m_gas_val = np.sum(gasdens_t) / m_gas_initial  # Scale by initial gas mass

        if dust:
            # Compute total dust mass m_dust and scale by initial dust mass
            m_dust_val = np.sum(dust1dens_t) / m_dust_initial
            total_density = gasdens_t + dust1dens_t

            # Create 2D grids for r and theta
            r, theta = np.meshgrid(xgrid, zgrid, indexing='ij')  # r has shape (nx, nz), theta has shape (nx, nz)

            # Compute cos(theta) for the height factors
            sin_theta = np.sin(theta)  # cos(theta) has shape (nx, nz)

            # Compute height factors: r * cos(theta)
            height_factors = r * sin_theta  # Shape (nx, nz)

            # Define the angular grid step size (dtheta)
            dtheta = np.abs(zgrid[1] - zgrid[0])  # Assuming zgrid is evenly spaced

            # Compute vertically integrated surface densities using explicit integration
            sigma_gas = np.sum(gasdens_t * height_factors[None, :, :] * dtheta, axis=2)  # Shape (ny, nx)
            sigma_dust = np.sum(dust1dens_t * height_factors[None, :, :] * dtheta, axis=2)  # Shape (ny, nx)

            # Compute metallicity (Z = sigma_dust / sigma_gas)
            metallicity = sigma_dust / sigma_gas  # Shape (ny, nx)
            avg_metallicity_val = np.mean(metallicity)  # Average over radial direction

            # Check if Roche density is exceeded
            roche_exceed_mask = total_density >= roche_density_masked[:, np.newaxis, np.newaxis]
            roche_reached = np.any(roche_exceed_mask)
            
            if roche_reached:
                roche_times.append(time_array[t_idx])

            # Compute max_epsilon
            epsilon = dust1dens_t / gasdens_t
            max_epsilon_val = np.max(epsilon)

            # Step 1: Identify columns (r, phi values) where any epsilon > 10
            column_mask = np.all(epsilon <= 10, axis=2)  # Shape (ny, nx), True if all epsilon values in the column are <= 10

            # Step 2: Filter out entire columns where epsilon > 10 in any vertical slice
            dust1dens_t_filtered = dust1dens_t * column_mask[:, :, np.newaxis]  # Retain only columns that pass the mask

            # Step 3: Compute the radial and azimuthal average to get the dust profile as a function of theta
            dust_profile = np.mean(dust1dens_t_filtered, axis=(0, 1))  # Average over radial and azimuthal directions

            # Step 4: Compute the vertical coordinate (height) as r * cos(theta)
            r_mid = np.mean(xgrid)  # Use the mean radial position as a representative r
            z_values = r_mid * np.cos(zgrid)  # Compute height z for each theta

            # Step 5: Fit a Gaussian to the dust profile
            p0 = [np.max(dust_profile), 0, np.std(z_values), np.min(dust_profile)]  # Initial guess for the Gaussian fit

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    popt, _ = curve_fit(gaussian, z_values, dust_profile, p0=p0, maxfev=1000)
                H_d_val = abs(popt[2])  # Use the absolute value of sigma as the dust scale height
            except RuntimeError:
                print(f"Fitting failed at time step {t_idx}. Setting H_d/H_g = 1.")
                H_d_val = aspectratio  # Default value when fitting fails
     
        else:
            m_dust_val = None
            max_epsilon_val = 1e-15
            H_d_val = 1e-15
            avg_metallicity_val = 0.0  # No metallicity if dust is not present

        # Extract the polar angle theta grid
        theta = zgrid  # zgrid represents theta in spherical coordinates
        sin_theta = np.sin(theta)  # Compute sin(theta)
        cos_theta = np.cos(theta)  # Compute cos(theta)

        # Apply the correction to project it to the disc mid-plane
        gasvy_corrected = data_arrays['gasvy'][:, radial_mask, :, t_idx] * sin_theta[np.newaxis, np.newaxis, :] + data_arrays['gasvz'][:, radial_mask, :, t_idx] * cos_theta[np.newaxis, np.newaxis, :]
        gasvz_corrected = data_arrays['gasvy'][:, radial_mask, :, t_idx] * cos_theta[np.newaxis, np.newaxis, :] - data_arrays['gasvz'][:, radial_mask, :, t_idx] * sin_theta[np.newaxis, np.newaxis, :]
        gasvx_corrected = data_arrays['gasvx'][:, radial_mask, :, t_idx] - data_arrays['gasvx'][:, radial_mask, :, 0]

        # Perform the calculations for turbulent alpha
        numerator = np.mean(gasdens_t * gasvy_corrected * gasvx_corrected)
        denominator = np.mean(data_arrays['gasenergy'][:, radial_mask, :, t_idx])
        alpha_r_val = numerator / denominator

        # Create mask for above and below the midplane
        above_midplane_mask = zgrid > np.pi / 2  # Theta values above the midplane
     
        # Apply the masks and compute the numerator above the midplane
        numerator_above = np.mean(
            gasdens_t[:, :, above_midplane_mask] *
            gasvz_corrected[:, :, above_midplane_mask] * gasvx_corrected[:, :, above_midplane_mask]
        )

        # Apply the masks and compute the numerator below the midplane
        numerator_below = -np.mean(
            gasdens_t[:, :, ~above_midplane_mask] *
            gasvz_corrected[:, :, ~above_midplane_mask] * gasvx_corrected[:, :, above_midplane_mask]
        )

        # Combine the contributions from above and below the midplane
        numerator = numerator_above + numerator_below

        # Compute the denominator (same for the entire domain)
        denominator = np.mean(data_arrays['gasenergy'][:, radial_mask, :, t_idx])

        # Compute alpha_z
        alpha_z_val = numerator / denominator

        dum = sin_theta[np.newaxis, np.newaxis, :]
        if dust:
            rms_vr_val = np.sqrt(np.mean((data_arrays['dust1vy'][:, radial_mask, (nz // 2 - 4):(nz // 2 + 4), t_idx]*dum)**2))
            rms_vphi_val = np.sqrt(np.mean((data_arrays['dust1vx'][:, radial_mask, (nz // 2 - 4):(nz // 2 + 4), t_idx]-data_arrays['dust1vx'][:, radial_mask, (nz // 2 - 4):(nz // 2 + 4), 0])**2))
            rms_vz_val = np.sqrt(np.mean(data_arrays['dust1vz'][:, radial_mask, (nz // 2 - 4):(nz // 2 + 4), t_idx]**2))

            max_vz_val = np.max(data_arrays['dust1vz'][:, radial_mask_2, :, t_idx])
            min_vz_val = np.min(data_arrays['dust1vz'][:, radial_mask_2, :, t_idx])


            # Compute vertical profiles of RMS velocities averaged over radius and azimuth
            rms_vr_z = np.sqrt(np.mean(np.mean((data_arrays['dust1vy'][:, radial_mask, :, t_idx]*dum)**2, axis=0), axis=0))
            rms_vphi_z = np.sqrt(np.mean(np.mean((data_arrays['dust1vx'][:, radial_mask, :, t_idx]-data_arrays['dust1vx'][:, radial_mask, :, 0])**2, axis=0), axis=0))
            rms_vz_z = np.sqrt(np.mean(np.mean(data_arrays['dust1vz'][:, radial_mask, :, t_idx]**2, axis=0), axis=0))
        else:
            rms_vr_val = np.sqrt(np.mean((data_arrays['gasvy'][:, radial_mask, :, t_idx]*dum)**2))
            rms_vphi_val = np.sqrt(np.mean((data_arrays['gasvx'][:, radial_mask, :, t_idx]-data_arrays['gasvx'][:, radial_mask, :, 0])**2))
            #rms_vz_val = np.sqrt(np.mean(data_arrays['gasvz'][:, radial_mask, :, t_idx]**2))
            rms_vz_val = np.sqrt(np.mean(data_arrays['gasvz'][:, radial_mask, (nz // 2 - 4):(nz // 2 + 4), t_idx]**2))
            max_vz_val = np.max(data_arrays['gasvz'][:, radial_mask_2, :, t_idx])
            min_vz_val = np.min(data_arrays['gasvz'][:, radial_mask_2, :, t_idx])
            # Compute vertical profiles of RMS velocities averaged over radius and azimuth
            rms_vr_z = np.sqrt(np.mean(np.mean((data_arrays['gasvy'][:, radial_mask, :, t_idx]*dum)**2, axis=0), axis=0))
            rms_vphi_z = np.sqrt(np.mean(np.mean((data_arrays['gasvx'][:, radial_mask, :, t_idx]-data_arrays['gasvx'][:, radial_mask, :, 0])**2, axis=0), axis=0))
            rms_vz_z = np.sqrt(np.mean(np.mean(data_arrays['gasvz'][:, radial_mask, :, t_idx]**2, axis=0), axis=0))
        

        # Pre-compute sin(theta) for use in the calculations
        sin_theta = np.sin(zgrid)  # zgrid represents theta in spherical coordinates
        r_grid, theta_grid = np.meshgrid(xgrid, zgrid, indexing='ij')  # r and theta grids

        # Compute 1/r and 1/(r sin(theta)) to avoid repetitive calculations
        inv_r = 1.0 / r_grid
        inv_r_sin_theta = 1.0 / (r_grid * sin_theta)

        # Compute the vertical (theta) component of vorticity: omega_theta
        d_vphi_dr = np.gradient(r_grid * (data_arrays['gasvx'][:, :, :, t_idx] - data_arrays['gasvx'][:, :, :, 0]), xgrid, axis=1) / r_grid  # 1/r * d(r v_phi)/dr
        if ny > 1:
            d_vr_dphi = np.gradient(data_arrays['gasvy'][:, :, :, t_idx], ygrid, axis=0) * inv_r  # d(v_r)/dphi
        else:
            d_vr_dphi = 0  # If ny == 1, set d(v_r)/dphi to 0

        omega_theta = d_vphi_dr - d_vr_dphi  # Vertical vorticity component

        # Compute the radial (r) component of vorticity: omega_r
        d_vphi_dtheta = np.gradient(data_arrays['gasvx'][:, :, :, t_idx] * sin_theta[np.newaxis, np.newaxis, :], zgrid, axis=2) * inv_r_sin_theta  # 1/(r sin(theta)) * d(v_phi sin(theta))/dtheta
        if ny > 1:
            d_vtheta_dphi = np.gradient(data_arrays['gasvz'][:, :, :, t_idx], ygrid, axis=0) * inv_r_sin_theta  # 1/(r sin(theta)) * d(v_theta)/dphi
        else:
            d_vtheta_dphi = 0  # If ny == 1, set d(v_theta)/dphi to 0

        omega_r = d_vphi_dtheta - d_vtheta_dphi  # Radial vorticity component

        # Compute the azimuthal (phi) component of vorticity: omega_phi
        #d_vr_dtheta = np.gradient(data_arrays['gasvy'][:, :, :, t_idx], zgrid, axis=2) * inv_r  # 1/r * d(v_r)/dtheta
        #d_vtheta_dr = np.gradient(data_arrays['gasvz'][:, :, :, t_idx], xgrid, axis=1)  # d(v_theta)/dr

        #omega_phi = d_vr_dtheta - d_vtheta_dr  # Azimuthal vorticity component

        # Calculate vertical averages for omega_theta (vertical component of vorticity)
        omega_theta_avg_z = np.mean(omega_theta, axis=2)  # Average over vertical (theta) direction

        # Calculate vertical averages for omega_r (radial component of vorticity)
        omega_r_avg_z = np.mean(omega_r, axis=2)  # Average over vertical (theta) direction

        # Optional: Compute global averages over all dimensions if needed
        vortz_avg_val = np.mean(omega_theta, axis=(0, 1, 2))  # Global average of omega_theta
        vortr_avg_val = np.mean(omega_r, axis=(0, 1, 2))  # Global average of omega_r

       
        omega_z_avg_z = omega_theta_avg_z


        # Apply boxcar smoothing in the azimuthal (y) and radial (x) directions
        omega_z_smoothed = uniform_filter(omega_z_avg_z, size=(nsy, nsx))  # Apply filter to (ny, nx) dimensions

        # Calculate <omega_z>_zphi (vertical and azimuthal average)
        omega_z_avg_zphi = np.mean(omega_z_smoothed, axis=0)  # Shape (nx)

        # Calculate the difference <omega_z>_z - <omega_z>_zphi
        vorticity_diff = omega_z_smoothed - omega_z_avg_zphi[np.newaxis, :]  # Shape (ny, nx)

        # Compute the minimum across the azimuthal direction
        vorticity_diff_min = np.min(vorticity_diff, axis=0)  # Shape (nx)

        # Apply radial mask
        vort_min_val=np.min(vorticity_diff_min[radial_mask]) # single value
        #vort_min_val=np.mean(vorticity_diff_min[radial_mask]) # single value

        # Return calculated values, including avg_metallicity
        return alpha_r_val, alpha_z_val, rms_vr_val, rms_vphi_val, rms_vz_val, max_vz_val, min_vz_val, max_epsilon_val, H_d_val,  m_gas_val, m_dust_val, avg_metallicity_val, vort_min_val, rms_vr_z, rms_vphi_z, rms_vz_z, vortr_avg_val, vortz_avg_val

    # Process the time steps in parallel
    results = Parallel(n_jobs=8, backend='threading')(delayed(process_time_step)(t_idx) for t_idx in tqdm(range(len(time_array)), desc="Processing time steps"))

    # Unpack results
    for t_idx, (alpha_r_val, alpha_z_val, rms_vr_val, rms_vphi_val, rms_vz_val, max_vz_val, min_vz_val, max_epsilon_val, H_d_val, m_gas_val, m_dust_val, avg_metallicity_val, vort_min_val, rms_vr_z, rms_vphi_z, rms_vz_z, vortr_avg_val, vortz_avg_val) in enumerate(results):
        alpha_r[t_idx] = alpha_r_val
        alpha_z[t_idx] = alpha_z_val
        rms_vr[t_idx] = rms_vr_val
        rms_vphi[t_idx] = rms_vphi_val
        rms_vz[t_idx] = rms_vz_val
        max_vz[t_idx] = max_vz_val
        min_vz[t_idx] = min_vz_val
        max_epsilon[t_idx] = max_epsilon_val
        H_d_array[t_idx] = H_d_val
        m_gas[t_idx] = m_gas_val
        vort_min[t_idx]=vort_min_val
        vortr_avg[t_idx]=vortr_avg_val
        vortz_avg[t_idx]=vortz_avg_val
        if dust:
            m_dust[t_idx] = m_dust_val
            avg_metallicity[t_idx] = avg_metallicity_val


        # Accumulate vertical profiles
        rms_vr_profile += rms_vr_z
        rms_vphi_profile += rms_vphi_z
        rms_vz_profile += rms_vz_z

    # Normalize the profiles by the number of time steps
    rms_vr_profile /= len(time_array)
    rms_vphi_profile /= len(time_array)
    rms_vz_profile /= len(time_array)

    # Save the computed quantities, including avg_metallicity
    if nsteps == 1:
        xgrid_masked=xgrid[radial_mask]
        save_simulation_quantities(output_path, time_array, alpha_r, alpha_z, rms_vr, rms_vphi, rms_vz, max_vz, min_vz, max_epsilon, H_d_array, roche_times, m_gas, m_dust, avg_metallicity, vort_min, xgrid_masked, rms_vr_profile, rms_vphi_profile, rms_vz_profile, vortr_avg, vortz_avg)



    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, alpha_r, label=r'$\alpha_r(t)$')
    plt.plot(time_array, alpha_z, label=r'$\alpha_z(t)$')
    plt.plot(time_array, rms_vz, label='RMS(Vertical Velocity)')
    plt.plot(time_array, m_gas, label='Total Gas Mass (scaled)')
    if dust:
        plt.plot(time_array, m_dust, label='Total Dust Mass (scaled)')
        plt.plot(time_array, max_epsilon, label=r'Max $\epsilon$')
        plt.plot(time_array, H_d_array / H_g, label=r'Dust Scale Height $H_d/H_g$')
        plt.plot(time_array, avg_metallicity, label=r'$\langle Z \rangle$ (Averaged Metallicity)')

        # Plot Roche density exceed markers without repeated legend entry
        roche_label_added = False
        for roche_time in roche_times:
            roche_idx = np.where(time_array == roche_time)[0]
            if len(roche_idx) > 0:
                roche_idx = roche_idx[0]
                # Add label only for the first occurrence
                if not roche_label_added:
                    plt.scatter(roche_time, max_epsilon[roche_idx], color='red', marker='o', s=100, label='Roche Density Reached')
                    roche_label_added = True
                else:
                    plt.scatter(roche_time, max_epsilon[roche_idx], color='red', marker='o', s=100)

    # Set the y-axis limits based on whether dust is present
    if dust:
        plt.ylim(1e-6, 1e3)  # Y-axis limits when dust is present
    else:
        plt.ylim(1e-6, 1)  # Y-axis limits when no dust is present

    plt.xlabel('Time [Orbits]')
    plt.ylabel('Values (scaled)')
    plt.yscale('log')
    plt.title(r'$\alpha_r$, $\alpha_r$, $RMS(v_z)$, Max $\epsilon$, $H_d/H_g$, Total Masses (scaled), $\langle Z \rangle$')

    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend()

    # Save the plot as a PDF
    pdf_filename = f"{os.path.basename(output_path)}_alpha_r_scaled_masses.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()

    print(f"Alpha_r, RMS dust velocity, max epsilon, scale height, and scaled masses plot saved to {output_filepath}")
    
    # Call the scp_transfer function if needed
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")



###########################################################################################################################


def plot_azimuthal_velocity_deviation(data_arrays, xgrid, time_array, output_path):
    """
    Plot the space-time contour of the azimuthal velocity deviation from its initial value, scaled by the disk aspect ratio.

    Parameters:
    - data_arrays (dict): Dictionary containing data arrays including 'gasvx' (azimuthal velocity).
    - xgrid (array): The grid array for the radial coordinate.
    - time_array (array): The array of time steps.
    - output_path (str): Path to save the plot.
    """

    # Set global font size and styling
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 24
    })

    # Read disk aspect ratio (precomputed once)
    summary_file = os.path.join(output_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    aspectratio = parameters['ASPECTRATIO']

    # Retrieve azimuthal velocity (gasvx) data and calculate initial azimuthal velocity
    azimuthal_velocity = data_arrays['gasvx']  # Shape (ny, nx, nz, nts)
    initial_azimuthal_velocity = azimuthal_velocity[..., 0]  # Initial azimuthal velocity snapshot (ny, nx, nz)

    # Calculate deviation of azimuthal velocity from initial value, scaled by aspect ratio
    azimuthal_velocity_deviation = (azimuthal_velocity - initial_azimuthal_velocity[..., np.newaxis]) / aspectratio  # Shape (ny, nx, nz, nts)

    # Integrate over the azimuthal and vertical directions (y and z axes)
    azimuthal_velocity_deviation_avg = np.mean(azimuthal_velocity_deviation, axis=(0, 2))  # Shape (nx, nts)

    # Apply radial mask
    radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
    xgrid_masked = xgrid[radial_mask]
    azimuthal_vel_dev_masked = np.cbrt(azimuthal_velocity_deviation_avg[radial_mask, :])  # Shape (masked nx, nts)
    #azimuthal_vel_dev_masked = azimuthal_velocity_deviation_avg[radial_mask, :]  # Shape (masked nx, nts)

    # Create the contour plot
    plt.figure(figsize=(10, 6))
    cp = plt.imshow(azimuthal_vel_dev_masked.T, extent=[xgrid_masked.min(), xgrid_masked.max(), time_array.min(), time_array.max()],
                    aspect='auto', origin='lower', cmap='seismic', vmin=np.min(azimuthal_vel_dev_masked),
                    vmax=np.max(azimuthal_vel_dev_masked))
    plt.colorbar(cp, label=r'$(\Delta v_{\phi})^{1/3}$')
    #plt.colorbar(cp, label=r'$\Delta v_{\phi}$')
    plt.xlabel(r'$r/r_0$')
    plt.ylabel('Orbits')
    #plt.title(r'Space-Time Plot of Scaled Azimuthal Velocity Deviation')

    # Extract the subdirectory name for saving the plot and npz file
    subdir_name = os.path.basename(output_path)
    
    # Save the plot
    pdf_filename = f"{subdir_name}_space_time_plot_azimuthal_velocity_deviation.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"#######################################")
    print(f"Azimuthal velocity deviation space-time plot saved to {output_filepath}")
    print(f"#######################################")

    # Save the required data in an npz file
    npz_filename = f"{subdir_name}_azimuthal_velocity_deviation_data.npz"
    npz_filepath = os.path.join(output_path, npz_filename)
    np.savez(npz_filepath, azimuthal_vel_dev=azimuthal_vel_dev_masked.T, time_array=time_array, xgrid=xgrid_masked)
    print(f"#######################################")
    print(f"Azimuthal velocity deviation data saved to {npz_filepath}")
    print(f"#######################################")

    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")



###########################################################################################################################

def plot_metallicity(data_arrays, xgrid, zgrid, time_array, output_path):
    """
    Plot the space-time contour of the metallicity Z across the entire azimuth at the disk midplane in spherical coordinates.

    Parameters:
    - xgrid (array): The grid array for the radial coordinate.
    - zgrid (array): The grid array for the polar angle (\(\theta\)).
    - time_array (array): The array of time steps.
    - data_arrays (dict): A dictionary containing gas and dust density arrays.
    - output_path (str): Path to save the plot.
    """

    # Set global font size and other styling properties using rcParams
    plt.rcParams.update({
        'font.size': 16,        # Set default font size for all text
        'axes.titlesize': 20,   # Set font size for the title of each subplot
        'axes.labelsize': 18,   # Set font size for x and y labels
        'xtick.labelsize': 16,  # Set font size for x-axis tick labels
        'ytick.labelsize': 16,  # Set font size for y-axis tick labels
        'legend.fontsize': 14,  # Set font size for the legend
        'figure.titlesize': 24  # Set font size for the overall figure title
    })

    # Pre-compute geometric factors
    r, theta = np.meshgrid(xgrid, zgrid, indexing='ij')
    sin_theta = np.sin(theta)
    dtheta = np.abs(zgrid[1] - zgrid[0])  # Angular step size (assuming uniform spacing)

    # Compute the height factor for integration (r * sin(theta) * dtheta)
    height_factors = r * sin_theta * dtheta  # Shape (nx, nz)

    # Step 1: Compute surface mass densities by integrating over the vertical direction (\(\theta\))
    sigma_gas = np.sum(data_arrays['gasdens'] * height_factors[np.newaxis, :, :], axis=2)  # Shape (ny, nx, nts)
    sigma_dust = np.sum(data_arrays['dust1dens'] * height_factors[np.newaxis, :, :], axis=2)  # Shape (ny, nx, nts)

    # Step 2: Calculate the metallicity Z = sigma_dust / sigma_gas
    Z = sigma_dust / sigma_gas  # Shape (ny, nx, nts)

    # Step 3: Take the maximum value across the azimuthal direction (y-axis)
    Z_max = np.max(Z, axis=0)  # Shape (nx, nts)

    # Step 4: Apply radial mask
    radial_mask = (xgrid >= 0.9) & (xgrid <= 1.2)
    xgrid_masked = xgrid[radial_mask]
    Z_max_masked = Z_max[radial_mask, :]  # Shape (masked nx, nts)

    # Step 5: Create the contour plot
    plt.figure(figsize=(10, 6))
    cp = plt.imshow(np.sqrt(Z_max_masked.T), extent=[xgrid_masked.min(), xgrid_masked.max(), time_array.min(), time_array.max()],
                    aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(cp, label=r'$Z^{1/2}_{\max}(\varphi)$')
    plt.xlabel(r'$r/r_{0}$')
    plt.ylabel('Orbits')
    plt.title(r' ')

    # Extract the subdirectory name
    subdir_name = os.path.basename(output_path)

    # Highlight specific feedback times for certain simulations
    if subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_fboff":
        feedback_off_time = 726
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5, label='Feedback Off')
        plt.legend()

    elif subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150":
        feedback_off_time = 0
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5)
        plt.legend([plt.Line2D([0], [0], linestyle='None', marker='None', label='Feedback')], ['Feedback'])

    elif subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_nfb":
        feedback_off_time = 0
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5)
        plt.legend([plt.Line2D([0], [0], linestyle='None', marker='None', label='No Feedback')], ['No Feedback'])

    elif subdir_name == "cos_b1d0_us_St5dm2_Z1dm2_r6H_z08H_fim053_ss203_3D_2PI_stnew_LR150_nfb_fbon":
        feedback_off_time = 400
        plt.axhline(y=feedback_off_time, color='k', linestyle='--', linewidth=1.5, label='Feedback On')
        plt.legend()

    # Step 6: Save the plot with the subdirectory name in the file name
    pdf_filename = f"{subdir_name}_space_time_plot_metallicity_Z.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()
    print(f"#######################################")
    print(f"Metallicity space-time plot saved to {output_filepath}")
    print(f"#######################################")

    # Step 7: Save the required data in an npz file
    npz_filename = f"{subdir_name}_metallicity_data.npz"
    npz_filepath = os.path.join(output_path, npz_filename)
    np.savez(npz_filepath, metallicity=Z_max_masked.T, time_array=time_array, xgrid=xgrid_masked)
    print(f"#######################################")
    print(f"Metallicity data saved to {npz_filepath}")
    print(f"#######################################")

    # Transfer the plot to the specified local directory
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")


