import numpy as np
import matplotlib.pyplot as plt
import os
from data_reader import read_single_snapshot, read_parameters, reconstruct_grid
from data_storage import scp_transfer

plt.rcParams['text.usetex'] = True

# Define a font size parameter
font_size = 18

colormap_vort = "hot"
colormap_eps = "viridis"

def plot_snapshots(simulation_subdir, snapshots, x_range=[0.9, 1.2]):
    # Base path setup
    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]

    subdir_path = None
    for base_path in base_paths:
        potential_path = os.path.join(base_path, simulation_subdir)
        if os.path.exists(potential_path):
            subdir_path = potential_path
            break

    if subdir_path is None:
        raise FileNotFoundError(f"Subdirectory {simulation_subdir} not found in any base path.")

    # Read parameters and set up grids once
    parameters = read_parameters(os.path.join(subdir_path, "summary0.dat"))
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)
    
    # Apply radial mask for specified x_range
    radial_mask = (xgrid >= x_range[0]) & (xgrid <= x_range[1])
    xgrid_masked = xgrid[radial_mask]

    # Set color bar ranges manually
    vorticity_min, vorticity_max = -0.8, 1.  # Adjust as needed
    epsilon_min, epsilon_max = -2, 2  # Adjust as needed

    # Compute the initial vorticity for subtraction
    initial_data_arrays, _, _, _, _ = read_single_snapshot(
        subdir_path, 
        snapshot=0,
        read_gasvx=True, 
        read_gasvy=True
    )
    initial_vorticity_z = compute_initial_vorticity(initial_data_arrays, xgrid, ygrid)

    # Print dimensions before applying the radial mask
    print("Dimensions of initial_vorticity_z before masking:", initial_vorticity_z.shape)
    print("Dimensions of radial_mask:", radial_mask.shape)

    # Apply radial mask along the middle dimension (nx) of initial_vorticity_z
    initial_vorticity_z = initial_vorticity_z[:, radial_mask, :]

    # Adjust layout to add space between columns and set explicit colorbar size
    fig, axes = plt.subplots(len(snapshots), 2, figsize=(14, 9), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3, 'hspace': 0.05})
    plt.rcParams.update({'font.size': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size})

    # Plot each snapshot
    for idx, snapshot in enumerate(snapshots):
        # Load data for each snapshot
        data_arrays, _, _, _, _ = read_single_snapshot(
            subdir_path, 
            snapshot=snapshot,
            read_gasvx=True, 
            read_gasvy=True, 
            read_gasdens=True, 
            read_dust1dens=True
        )

        # Calculate quantities with initial vorticity subtracted, and apply radial mask to the middle index
        vorticity_z, epsilon = calculate_quantities(data_arrays, xgrid, ygrid, nz, initial_vorticity_z, radial_mask)

        # Plot vorticity
        ax_vorticity = axes[idx, 0]
        im_vorticity = ax_vorticity.imshow(vorticity_z, vmin=vorticity_min, vmax=vorticity_max, origin="lower", extent=[x_range[0], x_range[1], ygrid.min(), ygrid.max()], aspect='auto', cmap=colormap_vort)

        # Set tick parameters for vorticity axis
        ax_vorticity.tick_params(labelsize=font_size)
        
        # Time label
        ax_vorticity.text(0.77, 0.9, f"{snapshot * 8} ORB", transform=ax_vorticity.transAxes, color="white", fontsize=font_size, bbox=dict(facecolor="black", alpha=0.5))
        
        # Titles and labels
        if idx == 0:
            ax_vorticity.set_title(r"$\langle \omega_z \rangle_z$", fontsize=font_size)
        if idx == len(snapshots) - 1:
            ax_vorticity.set_xlabel(r"$r/r_0$", fontsize=font_size)
            # Define x-axis ticks based on x_range
            tick_start, tick_end = x_range[0], x_range[1]
            tick_step = 0.1  # Define your preferred tick spacing
            ax_vorticity.set_xticks(np.arange(tick_start, tick_end + tick_step, tick_step))
        else:
            ax_vorticity.set_xticks([])  # Remove x-axis ticks for upper rows
        ax_vorticity.set_ylabel(r"$\varphi$", fontsize=font_size)

        # Plot epsilon
        ax_epsilon = axes[idx, 1]
        im_epsilon = ax_epsilon.imshow(np.log10(epsilon), vmin=epsilon_min, vmax=epsilon_max, origin="lower", extent=[x_range[0], x_range[1], ygrid.min(), ygrid.max()], aspect='auto', cmap=colormap_eps)

        # Set tick parameters for epsilon axis
        ax_epsilon.tick_params(labelsize=font_size)
        
        # Time label for epsilon
        ax_epsilon.text(0.77, 0.9, f"{snapshot * 8} ORB", transform=ax_epsilon.transAxes, color="white", fontsize=font_size, bbox=dict(facecolor="black", alpha=0.5))
        
        if idx == 0:
            ax_epsilon.set_title(r"$\log_{10}(\epsilon)$", fontsize=font_size)
        if idx == len(snapshots) - 1:
            ax_epsilon.set_xlabel(r"$r/r_0$", fontsize=font_size)
            tick_start, tick_end = x_range[0], x_range[1]
            tick_step = 0.1  # Define your preferred tick spacing
            ax_epsilon.set_xticks(np.arange(tick_start, tick_end + tick_step, tick_step))
        else:
            ax_epsilon.set_xticks([])  # Remove x-axis ticks for upper rows

    # Add shared colorbars for each column, explicitly setting height and width
    cbar_vorticity = fig.colorbar(im_vorticity, ax=axes[:, 0], orientation="vertical", fraction=0.05, pad=0.02)
    cbar_vorticity.ax.tick_params(labelsize=font_size)
    cbar_epsilon = fig.colorbar(im_epsilon, ax=axes[:, 1], orientation="vertical", fraction=0.05, pad=0.02)
    cbar_epsilon.ax.tick_params(labelsize=font_size)

    # Save the plot
    pdf_filename = "vorticity_epsilon_sequence.pdf"
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved as {pdf_filename}")

    # SCP transfer to local directory
    local_directory = "/Users/mariuslehmann/Downloads/Contours"
    scp_transfer(pdf_filename, local_directory, "mariuslehmann")


def calculate_quantities(data_arrays, xgrid, ygrid, nz, initial_vorticity_z, radial_mask):
    from scipy.ndimage import gaussian_filter

    # Apply the radial mask to xgrid and calculate the masked vorticity
    masked_xgrid = xgrid[radial_mask]
    
    # Smoothed velocity components
    smoothed_vx = gaussian_filter(data_arrays['gasvx'][:, radial_mask, :], sigma=1)
    smoothed_vy = gaussian_filter(data_arrays['gasvy'][:, radial_mask, :], sigma=1)
    
    r = masked_xgrid[np.newaxis, :, np.newaxis]
    d_vphi_dr = np.gradient(r * smoothed_vx, masked_xgrid, axis=1) / r
    d_vr_dphi = np.gradient(smoothed_vy, ygrid, axis=0)
    vorticity_z = d_vphi_dr - d_vr_dphi

    # Subtract initial vorticity
    vorticity_z -= initial_vorticity_z

    # Calculate epsilon and apply radial mask to it as well
    epsilon = data_arrays['dust1dens'][:, radial_mask, :] / data_arrays['gasdens'][:, radial_mask, :]
    
    return np.mean(vorticity_z, axis=2), epsilon[:, :, nz // 2]

def compute_initial_vorticity(initial_data_arrays, xgrid, ygrid):
    from scipy.ndimage import gaussian_filter

    # Smoothed velocity components
    smoothed_vx = gaussian_filter(initial_data_arrays['gasvx'], sigma=1)
    smoothed_vy = gaussian_filter(initial_data_arrays['gasvy'], sigma=1)
    
    r = xgrid[np.newaxis, :, np.newaxis]
    d_vphi_dr = np.gradient(r * smoothed_vx, xgrid, axis=1) / r
    d_vr_dphi = np.gradient(smoothed_vy, ygrid, axis=0)
    initial_vorticity_z = d_vphi_dr - d_vr_dphi

    return initial_vorticity_z

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot vorticity and epsilon for multiple snapshots.")
    parser.add_argument("simulation_subdir", type=str, help="Subdirectory of the simulation")
    parser.add_argument("--snapshots", nargs="+", type=int, help="Snapshot numbers to plot", required=True)
    args = parser.parse_args()
    
    plot_snapshots(args.simulation_subdir, args.snapshots)
