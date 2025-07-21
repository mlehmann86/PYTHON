import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.optimize import curve_fit
import warnings
from scipy.ndimage import gaussian_filter, uniform_filter
from joblib import Parallel, delayed
from tqdm import tqdm
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters, reconstruct_grid
from plotting_functions import gaussian

# Compute RMS(dust1vz)
def compute_rms_vertical_velocity(dust1vz, zgrid, method="z-average"):
    """
    Compute the RMS vertical dust velocity.

    Parameters:
    - dust1vz: 3D numpy array of the vertical dust velocity (shape: [ny, nx, nz]).
    - zgrid: 1D numpy array of the vertical grid points.
    - method: str, "z-average" to compute the vertically averaged RMS velocity, 
              or "midplane" to use the value at the midplane.

    Returns:
    - rms_velocity: 2D numpy array of the RMS vertical dust velocity (shape: [ny, nx]).
    """
    if method == "z-average":
        # Compute the RMS velocity as a z-average
        #midplane_index = np.argmin(np.abs(zgrid))
        #rms_velocity = np.sqrt(np.mean(dust1vz[:, :,(midplane_index - 4):(midplane_index + 4) ]**2, axis=2))
        rms_velocity = np.max(dust1vz, axis=2) - np.min(dust1vz, axis=2)
    elif method == "midplane":
        # Find the index closest to the midplane (z = 0)
        midplane_index = np.argmin(np.abs(zgrid))
        # Use the value at the midplane
        rms_velocity = np.abs(dust1vz[:, :, midplane_index])
    else:
        raise ValueError("Invalid method. Choose 'z-average' or 'midplane'.")

    return rms_velocity

# Simplified version of read_hydro_fields to read a single snapshot
def read_single_snapshot(path, snapshot, read_dust1dens=False, read_gasdens=False, read_dust1vz=False):
    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    data_arrays = {}

    if read_dust1dens:
        data_arrays['dust1dens'] = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)
    if read_gasdens:
        data_arrays['gasdens'] = read_field_file(path, 'gasdens', snapshot, nx, ny, nz)
    if read_dust1vz:
        data_arrays['dust1vz'] = read_field_file(path, 'dust1vz', snapshot, nx, ny, nz)

    return data_arrays, xgrid, ygrid, zgrid, parameters
def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if os.path.exists(file):
        print(f"Reading file: {file}")
        return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
    else:
        print(f"File not found: {file}")
        return np.zeros((ny, nx, nz))

# Compute dust scale height
def compute_single_scale_height(dust_profile, zgrid):
    dust_profile_smooth = gaussian_filter(dust_profile, sigma=1)
    
    p0 = [np.max(dust_profile_smooth), 0, np.std(zgrid), np.min(dust_profile_smooth)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(gaussian, zgrid, dust_profile_smooth, p0=p0, maxfev=1000)
        return abs(popt[2]), popt  # Return scale height and fit parameters
    except RuntimeError:
        return np.nan, None

# Parallel computation for H_d
def compute_H_d(y_idx, x_idx, dust1dens_smooth, zgrid):
    return compute_single_scale_height(dust1dens_smooth[y_idx, x_idx, :], zgrid)

def filter_outliers(H_d_scaled, threshold):
    """
    Replace outlier values (greater than threshold) and NaNs in H_d with the 
    average of their neighboring non-outlier, non-NaN values.

    Parameters:
    - H_d_scaled: 2D array of dust scale heights (already scaled by H_g)
    - threshold: The threshold value above which H_d is considered an outlier
    
    Returns:
    - H_d_filtered: The filtered array with outliers and NaNs replaced by local averages
    """
    H_d_filtered = H_d_scaled.copy()
    
    #print(f"Max value of H_d before filtering: {np.nanmax(H_d_filtered)}")
    
    # Mask for outliers (values exceeding the threshold)
    outlier_mask = H_d_filtered > threshold
    
    # Mask for NaN values
    nan_mask = np.isnan(H_d_filtered)

    # Combined mask for NaNs and outliers
    combined_mask = outlier_mask | nan_mask

    # Process NaNs and outliers
    if np.any(combined_mask):
        #print(f"Values greater than threshold ({threshold}): {H_d_filtered[outlier_mask]}")
        
        # Loop over the positions where we have NaNs or outliers
        for y_idx, x_idx in zip(*np.where(combined_mask)):
            # Define a neighboring region around the problematic value
            y_min = max(0, y_idx - 1)
            y_max = min(H_d_filtered.shape[0], y_idx + 2)  # +2 because slice is exclusive
            x_min = max(0, x_idx - 1)
            x_max = min(H_d_filtered.shape[1], x_idx + 2)

            # Extract the neighboring region and ignore NaNs and outliers
            neighbors = H_d_filtered[y_min:y_max, x_min:x_max]
            neighbors = neighbors[(neighbors <= threshold) & ~np.isnan(neighbors)]

            # If we have valid neighbors, replace the outlier/NaN with their mean
            if neighbors.size > 0:
                H_d_filtered[y_idx, x_idx] = np.mean(neighbors)
            else:
                # If no valid neighbors, leave as NaN (or set to zero depending on preference)
                H_d_filtered[y_idx, x_idx] = np.nan
    
    #print(f"Max value of H_d after filtering: {np.nanmax(H_d_filtered)}")
    
    return H_d_filtered

def compute_H_d_avg(x_idx, dust1dens_avg, zgrid):
    """
    Compute H_d from azimuthally averaged dust density profile for each radius (x_idx).
    """
    return compute_single_scale_height(dust1dens_avg[x_idx, :], zgrid)


def plot_dust_scale_height(simulation_name, snapshot, azimuth_angle, rmin, rmax, phimin, phimax, vmin=None, vmax=None, right_panel="radial"):
    thresh = 3
    fontsize = 20
    left = 0.05
    right = 0.95
    top = 0.9
    bottom = 0.15
    azimuthal_shift = -np.pi *0.5

    plt.rcParams.update({'font.size': fontsize})

    base_path = determine_base_path(simulation_name)
    data_arrays, xgrid, ygrid, zgrid, parameters = read_single_snapshot(
        base_path, snapshot, read_dust1dens=True, read_gasdens=(right_panel == "epsilon"), read_dust1vz=True
    )
    dust1dens = data_arrays['dust1dens']
    gasdens = data_arrays.get('gasdens', None)
    dust1vz = data_arrays.get('dust1vz', None)

    # Compute the radial and azimuthal masks before using them
    radial_mask = (xgrid >= rmin) & (xgrid <= rmax)
    azimuth_mask = (ygrid >= phimin) & (ygrid <= phimax)
    xgrid_masked = xgrid[radial_mask]
    ygrid_masked = ygrid[azimuth_mask]

    sigma = 2
    dust1dens_smooth = uniform_filter(dust1dens, size=(sigma, sigma, 1), mode='nearest')

    H_d = np.zeros((ygrid.shape[0], xgrid.shape[0]))
    results = Parallel(n_jobs=32)(delayed(compute_single_scale_height)(dust1dens_smooth[y_idx, x_idx, :], zgrid)
                                  for y_idx in range(H_d.shape[0]) for x_idx in range(H_d.shape[1]))
    for idx, (H_d_value, _) in tqdm(enumerate(results), total=len(results)):
        y_idx = idx // H_d.shape[1]
        x_idx = idx % H_d.shape[1]
        H_d[y_idx, x_idx] = H_d_value

    summary_file = os.path.join(base_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    aspectratio = parameters['ASPECTRATIO']
    H_d_scaled = H_d / aspectratio

    if thresh is not None:
        H_d_scaled = filter_outliers(H_d_scaled, thresh)

    epsilon_avg = np.mean(dust1dens / (gasdens), axis=2) if gasdens is not None else None
    H_d_avg = np.nanmean(H_d_scaled, axis=0)
    shift_index = int(azimuthal_shift / (2 * np.pi) * ygrid.shape[0])
    H_d_scaled_shifted = np.roll(H_d_scaled, shift_index, axis=0)
    epsilon_avg_shifted = np.roll(epsilon_avg, shift_index, axis=0) if epsilon_avg is not None else None

    if dust1vz is not None:
        rms_dust1vz = compute_rms_vertical_velocity(dust1vz, zgrid, method="z-average")
        # or
        #rms_dust1vz = compute_rms_vertical_velocity(dust1vz, zgrid, method="midplane")
        rms_dust1vz_avg = np.mean(rms_dust1vz, axis=0)
        rms_dust1vz_shifted = np.roll(rms_dust1vz, shift_index, axis=0)  # Apply azimuthal shift
        rms_dust1vz_masked = rms_dust1vz_shifted[azimuth_mask, :][:, radial_mask]

    H_d_scaled_masked = H_d_scaled_shifted[azimuth_mask, :][:, radial_mask]
    epsilon_masked = epsilon_avg_shifted[azimuth_mask, :][:, radial_mask] if epsilon_avg_shifted is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.4})
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    ax1 = axes[0]
    im1 = ax1.imshow(np.log10(H_d_scaled_masked), extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid_masked.min(), ygrid_masked.max()],
                     aspect='auto', origin='lower', cmap='inferno', vmin=-2, vmax=np.max(np.log10(H_d_scaled_masked)))
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label(r'$\log_{10}(H_d / H_g)$', fontsize=fontsize)
    ax1.set_xlabel(r'$r/r_0$', fontsize=fontsize)
    ax1.set_ylabel(r'$\varphi$', fontsize=fontsize)
    #ax1.set_title(r'Dust Scale Height $\log_{10}(H_d / H_g)$', fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2 = axes[1]
    if right_panel == "radial":
        H_d_avg_masked = H_d_avg[radial_mask]
        ax2.plot(xgrid_masked, H_d_avg_masked, label='Azimuthally Averaged', color='blue')
        azimuth_idx = np.abs(ygrid - azimuth_angle).argmin()
        ax2.plot(xgrid_masked, H_d_scaled_masked[azimuth_idx, :], linestyle='--', color='orange', label=f'Azimuth = {azimuth_angle:.2f}')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$r/r_0$', fontsize=fontsize)
        ax2.set_ylabel(r'$H_d / H_g$', fontsize=fontsize)
        ax2.legend(fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)
    elif right_panel == "epsilon" and epsilon_masked is not None:
        im2 = ax2.imshow(np.log10(epsilon_masked), extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid_masked.min(), ygrid_masked.max()],
                         aspect='auto', origin='lower', cmap='viridis', vmin=-3, vmax=np.max(np.log10(epsilon_masked)))
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label(r'$\log_{10}(\langle \epsilon \rangle_{z})$', fontsize=fontsize)
        ax2.set_xlabel(r'$r/r_0$', fontsize=fontsize)
        ax2.set_ylabel(r'$\varphi$', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)
        cbar2.ax.tick_params(labelsize=fontsize)

    Hg=0.1
    ax3 = axes[2]
    im3 = ax3.imshow(rms_dust1vz_masked/Hg, extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid_masked.min(), ygrid_masked.max()],
                     aspect='auto', origin='lower', cmap='plasma')
    cbar3 = fig.colorbar(im3, ax=ax3)
    #cbar3.set_label(r'$\text{RMS}(\text{dust1vz})/c_0$', fontsize=fontsize)
    cbar3.set_label(r'$[max(v_{dz}-min(v_{dz})]/c_0$', fontsize=fontsize)
    ax3.set_xlabel(r'$r/r_0$', fontsize=fontsize)
    ax3.set_ylabel(r'$\varphi$', fontsize=fontsize)
    #ax3.set_title(r'RMS Vertical Dust Velocity', fontsize=fontsize)
    #ax3.set_title(r'$[max(v_{dz}-min(v_{dz})]/c_0$', fontsize=fontsize)
    ax3.tick_params(axis='both', labelsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)

    output_filename = f"{simulation_name}_snapshot_{snapshot}_dust_scale_height_rms_dustvz.pdf"
    plt.savefig(output_filename)
    scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Contours", "mariuslehmann")

# Function to handle fraction or integer input
def parse_snapshot(value):
    try:
        # Try evaluating the input as a mathematical expression (e.g., 64/8)
        return int(eval(value))
    except (SyntaxError, NameError, ValueError):
        raise argparse.ArgumentTypeError(f"Invalid snapshot value: {value}. Must be an integer or a fraction like 64/8.")


# Command-line argument handling
parser = argparse.ArgumentParser(description="Plot 2D contour and radial profile of dust scale height (H_d), and RMS dust vertical velocity.")
parser.add_argument('simulation_name', type=str, help="Name of the simulation.")
parser.add_argument('snapshot', type=parse_snapshot, help="Snapshot number (integer or fraction like 64/8).")
parser.add_argument('--azimuth', type=float, default=0, help="Azimuth angle for the radial profile (default=0).")
parser.add_argument('--rmin', type=float, default=0.9, help="Minimum radial coordinate.")
parser.add_argument('--rmax', type=float, default=1.3, help="Maximum radial coordinate.")
parser.add_argument('--phimin', type=float, default=0, help="Minimum azimuthal coordinate.")
parser.add_argument('--phimax', type=float, default=2 * np.pi, help="Maximum azimuthal coordinate.")
parser.add_argument('--vmin', type=float, default=None, help="Minimum vertical value for right panel.")
parser.add_argument('--vmax', type=float, default=None, help="Maximum vertical value for right panel.")
parser.add_argument('--right_panel', type=str, choices=['radial', 'epsilon'], default='radial', help="Choose right panel plot: 'radial' or 'epsilon'")

args = parser.parse_args()

# Call the plotting routine
plot_dust_scale_height(args.simulation_name, args.snapshot, args.azimuth, args.rmin, args.rmax, args.phimin, args.phimax, args.vmin, args.vmax, args.right_panel)
