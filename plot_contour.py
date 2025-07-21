import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from data_reader import read_single_snapshot, read_initial_snapshot, determine_nt, read_parameters, reconstruct_grid
from data_storage import scp_transfer
import subprocess


plt.rcParams['text.usetex'] = True


def plot_contours(data_arrays, initial_arrays, xgrid, ygrid, zgrid, quantity, output_path, azimuth, snapshot, arrows=False):
    h0 = float(parameters['ASPECTRATIO'])

    # Set the global font size
    plt.rcParams.update({'font.size': 24})  # Adjust the number to increase or decrease font size

    quantity_map = {}
    
    if 'gasvx' in data_arrays:
        quantity_map['gasvx'] = (data_arrays['gasvx'] - initial_arrays.get('gasvx', np.zeros_like(data_arrays['gasvx']))) / h0
    
    if 'gasvy' in data_arrays:
        quantity_map['gasvy'] = data_arrays['gasvy'] / h0
    
    if 'gasvz' in data_arrays:
        quantity_map['gasvz'] = data_arrays['gasvz'] / h0
    
    if 'dust1vx' in data_arrays:
        quantity_map['dust1vx'] = (data_arrays.get('dust1vx', np.zeros_like(data_arrays['dust1vx'])) - initial_arrays.get('dust1vx', np.zeros_like(data_arrays['dust1vx']))) / h0
    
    if 'dust1vy' in data_arrays:
        quantity_map['dust1vy'] = data_arrays.get('dust1vy', np.zeros_like(data_arrays['dust1vy'])) / h0
    
    if 'dust1vz' in data_arrays:
        quantity_map['dust1vz'] = data_arrays.get('dust1vz', np.zeros_like(data_arrays['dust1vz'])) / h0
    
    if 'gasenergy' in data_arrays:
        quantity_map['pressure'] = (data_arrays['gasenergy'] - initial_arrays.get('gasenergy', np.ones_like(data_arrays['gasenergy']))) / initial_arrays.get('gasenergy', np.ones_like(data_arrays['gasenergy']))
    
    if 'gasvy' in data_arrays and 'gasvx' in data_arrays:
        # Radial grid expanded to match the shape of the velocity arrays
        r = xgrid[np.newaxis, :, np.newaxis]  # Shape (1, nx, 1)

        from scipy.ndimage import gaussian_filter

        sigma = 1 #2
        # Apply Gaussian smoothing with a chosen sigma (e.g., sigma=1)
        smoothed_vx = gaussian_filter(data_arrays['gasvx'], sigma=sigma)
        smoothed_vy = gaussian_filter(data_arrays['gasvy'], sigma=sigma)

        # Now compute the gradients using the smoothed data
        d_vphi_dr = np.gradient(r * smoothed_vx, xgrid, axis=1) / r
        d_vr_dphi = np.gradient(smoothed_vy, ygrid, axis=0) if len(ygrid) > 1 else 0
        omega_z = d_vphi_dr - d_vr_dphi

        # Subtract the initial vorticity as before
        initial_d_vphi_dr = np.gradient(r * initial_arrays.get('gasvx', np.zeros_like(data_arrays['gasvx'])), xgrid, axis=1) / r
        initial_d_vr_dphi = np.gradient(initial_arrays.get('gasvy', np.zeros_like(data_arrays['gasvy'])), ygrid, axis=0) if len(ygrid) > 1 else 0
        initial_omega_z = initial_d_vphi_dr - initial_d_vr_dphi

        delta_omega_z = omega_z - initial_omega_z

        # Subtract the initial vorticity from the current vorticity
        quantity_map['vort'] = delta_omega_z

    if 'dust1dens' in data_arrays and 'gasdens' in data_arrays:
        data_arrays['dust1dens'][:, 0, :] = 0
        quantity_map['eps'] = data_arrays['dust1dens'] / data_arrays['gasdens']
    
    label_map = {
    'gasvx': r'\delta v_{g\varphi}/c_{0}',
    'gasvy': r'v_{gr}/c_{0}',
    'gasvz': r'v_{gz}/c_{0}',
    'dust1vx': r'\delta v_{d\varphi}/c_{0}',
    'dust1vy': r'v_{dr}/c_{0}',
    'dust1vz': r'v_{dz}/c_{0}',
    'pressure': r'P/(\rho_{g0} c_{0}^2)',
    'vort': r'\delta \omega_{z}',
    'eps': r'\epsilon^{1/2}'
}



    print(f"Available quantities in data_arrays: {list(data_arrays.keys())}")
    
    if quantity not in quantity_map:
        raise ValueError(f"Invalid or unsupported quantity '{quantity}'.")

    if quantity == 'eps':
        quantity_data = np.sqrt(quantity_map[quantity])
    else:
        quantity_data = quantity_map[quantity]
    
    print(f"creating plot of: {quantity}")
    # Print the dimensions of quantity_data before applying the mask
    print(f"Dimensions of quantity_data: {quantity_data.shape}")
    print(f"Dimensions of xgrid: {xgrid.shape}")


    # Apply radial/azimuthal mask
    radial_mask = (xgrid >= 0.8) & (xgrid <= 1.2)
    xgrid_masked = xgrid[radial_mask]
    
    azimuthal_mask = (ygrid >= 0.*np.pi) & (ygrid <= 2.*np.pi)
    #azimuthal_mask = (ygrid >= 1.5) & (ygrid <= 3.5)
    ygrid_masked = ygrid[azimuthal_mask]
    
    # Apply masks to the quantity_data array
    # Masking the first and second dimensions respectively
    quantity_data_masked = quantity_data[azimuthal_mask, :, :][:, radial_mask, :]

    if ny>1:
        # Define the shift angle in radians
        phi_shift = np.pi / 2 *0  # example shift of 45 degrees

        # Calculate the corresponding shift in grid points
        delta_phi = ygrid[1] - ygrid[0]  # assuming uniform spacing in ygrid
        shift_steps = int(phi_shift / delta_phi)

        # Shift the quantity_data_masked array in the azimuthal direction (first dimension)
        quantity_data_masked = np.roll(quantity_data_masked, shift_steps, axis=0)

        # Shift the ygrid_masked similarly
        ygrid_masked = np.roll(ygrid_masked, shift_steps)

        # If necessary, adjust ygrid_shifted to wrap around the 2*pi boundary
        ygrid_masked = (ygrid_masked + phi_shift) % (2 * np.pi)

        # Determine the azimuth index
        azimuth_index = np.abs(ygrid_masked - (azimuth+phi_shift)).argmin()


    print(f"Dimensions of quantity_data_masked: {quantity_data_masked.shape}")
    print(f"Dimensions of xgrid_masked: {xgrid_masked.shape}")
    
    Full = True
    if Full == False:
        print(f"*******************************************************************************************")
        print(f"WARNING: plotting only meriodonal contour of 3D simulation. Set Full=True to get all plots!")
        print(f"*******************************************************************************************")
    cmap = 'inferno'

    labels = False
    titles = False
    labelfont=24

    # Extracting beta and tau as floats, with default values if not present
    beta = float(parameters.get('BETA', 1.0))
    tau = float(parameters.get('STOKES1', 0.001))
    metal = float(parameters.get('METALLICITY', 0.01))

    labelstring = rf"$\beta={beta}, \tau={tau}$"
    #labelstring = rf"$\beta={beta}$"
    #if ny > 1:
    #    labelstring = "3D"
    #    labelstring = rf"$Z={metal}$"
    #else:
    #    labelstring = "2D"

    # XZ plot (Azimuthal average)
    if ny > 1 and Full == True:
        plt.figure(figsize=(21, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(np.mean(quantity_data_masked, axis=0).T, extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
           aspect='auto', origin='lower', cmap=cmap)
    else:
        plt.figure(figsize=(6, 5))
        plt.subplot(1, 1, 1)      
        plt.imshow(quantity_data_masked[:,:].T, extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
           aspect='auto', origin='lower', cmap=cmap)
    
    plt.colorbar()
    plt.xlabel('$r/r_{0}$')
    plt.ylabel('$z/r_{0}$')
    # Title for Azimuthal Average
    if titles:
        if ny > 1 and Full:
            plt.title(rf'$\langle {label_map[quantity]} \rangle_\varphi$')
        else:
            plt.title(rf'${label_map[quantity]}$')


    if labels:
        # Add text to the upper left corner of the first plot
        ax1 = plt.gca()
        ax1.text(0.025, 0.97, labelstring, transform=ax1.transAxes, fontsize=labelfont, 
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    if ny > 1 and Full == True:
        # XY plot (Midplane)
        plt.subplot(1, 3, 2)
        if quantity == 'vort' or 'gasvz':
            plt.imshow(np.mean(quantity_data_masked, axis=2), extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid_masked.min(), ygrid_masked.max()],
           aspect='auto', origin='lower', cmap=cmap)
        else:
            plt.imshow(quantity_data_masked[:, :, len(zgrid) // 2], extent=[xgrid_masked.min(), xgrid_masked.max(), ygrid_masked.min(), ygrid_masked.max()],  aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar()
        plt.xlabel('$r/r_{0}$')
        plt.ylabel(r'$\varphi$')
        if quantity == 'vort' or 'gasvz':
            if titles:
                # Title using \mathrm for plain text
                plt.title(rf'$\langle {label_map[quantity]} \rangle_z$')
            else:
                #plt.title(rf'${label_map[quantity]}_{{z=0}}$') 
                plt.title(rf'${label_map[quantity]} \, (z=0)$')

        if labels:
            # Add text to the upper left corner of the first plot
            ax2 = plt.gca()
            ax2.text(0.625, 0.97, labelstring, transform=ax2.transAxes, fontsize=labelfont, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


        # Plot the white solid line first
        plt.axhline(y=ygrid_masked[azimuth_index], color='white', linestyle='-', linewidth=2.0)
        # Plot the black dashed line on top
        plt.axhline(y=ygrid_masked[azimuth_index], color='k', linestyle='--', linewidth=1.0, dashes=(5, 10))

        # XZ plot at a specific azimuth
        plt.subplot(1, 3, 3)
        im = plt.imshow(quantity_data_masked[azimuth_index, :, :].T, 
                extent=[xgrid_masked.min(), xgrid_masked.max(), zgrid.min(), zgrid.max()],
                aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar(cmap=cmap)

        if arrows:
            # Ensure that gasvy and gasvz are loaded correctly from the quantity_map
            gasvy = quantity_map['gasvy']
            gasvz = quantity_map['gasvz']

            # Thin out the grid points for the arrows
            npz = 5  # Change this value to control the density of arrows
            npx = 25

            # Thin out the velocity components in the same way
            gasvz = gasvz[azimuth_index, radial_mask, :].T
            gasvy = gasvy[azimuth_index, radial_mask, :].T

            gasvz_thin = gasvz[::npz, ::npx]
            gasvy_thin = gasvy[::npz, ::npx]

            # Create 2D grid arrays for xgrid and zgrid with matching shapes to the velocity arrays
            xgrid_2d, zgrid_2d = np.meshgrid(xgrid_masked, zgrid)

            # Now thin out the grid arrays using the same index step 'n'
            xgrid_thin = xgrid_2d[::npz, ::npx]
            zgrid_thin = zgrid_2d[::npz, ::npx]

            # Calculate the scaling factors based on the aspect ratio and grid extents
            x_length = xgrid_masked.max() - xgrid_masked.min()
            z_length = zgrid.max() - zgrid.min()

            # Get the aspect ratio of the specific panel
            ax = plt.gca()
            bbox = ax.get_position()  # Get the bounding box of the subplot
            panel_width = bbox.width
            panel_height = bbox.height
            aspect_ratio_panel = panel_height / panel_width  # Aspect ratio of the individual panel

            # Calculate the scaling factors for quiver
            scalx = (z_length / x_length) / aspect_ratio_panel
            scalz = 2  # Keeping this as 1 for relative scaling, can adjust for overall scale

            # Overplot the arrows on the third panel, swapping gasvy_thin and gasvz_thin
            plt.quiver(xgrid_thin, zgrid_thin, gasvy_thin * scalx, gasvz_thin * scalz, color='white', scale=1, width=0.005)

        #plt.colorbar(cmap=cmap)
        plt.xlabel('$r/r_{0}$')
        plt.ylabel('$z/r_{0}$')
        if titles:
            plt.title(rf'${label_map[quantity]}$')

        if labels:
            # Add text to the upper left corner of the first plot
            ax3 = plt.gca()
            if labels:
                ax3.text(0.625, 0.97, labelstring, transform=ax3.transAxes, fontsize=labelfont, 
                 verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Adjust the layout to increase space between plots and reduce left margin
    if ny > 1 and Full == True:
        plt.subplots_adjust(left=0.07, right=0.97, wspace=0.45, bottom=0.16, top=0.92)
    else:
        plt.subplots_adjust(left=0.21, right=0.87, wspace=0.45, bottom=0.16, top=0.92)
    
    # Create "contours" directory if it doesn't exist
    contours_dir = "contours"
    os.makedirs(contours_dir, exist_ok=True)

    pdf_filename = os.path.join(contours_dir, f"contour_{os.path.basename(output_path)}_{quantity}_snapshot{snapshot}.pdf")
    plt.savefig(pdf_filename)
    plt.close()
    print(f"Contour plots saved to {pdf_filename}")
    print(f"#####################################")
    print(f"evince {pdf_filename} &")
    print(f"#####################################")

    # Define the local directory on your laptop
    local_directory = "/Users/mariuslehmann/Downloads/Contours"

    # Call the scp_transfer function to transfer the plot
    scp_transfer(pdf_filename, local_directory, "mariuslehmann")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot contour plots of specified quantities.')
    parser.add_argument('subdirectory', type=str, help='Subdirectory of the simulation')
    parser.add_argument('quantity', type=str, help='Quantity to plot')
    parser.add_argument('snapshot', type=int, help='Snapshot number')
    parser.add_argument('azimuth', type=float, help='Azimuth angle in radians')
    parser.add_argument('--arrows', action='store_true', help='Overlay arrows to indicate velocity field in the rz-plane')

    args = parser.parse_args()

    # Base paths
    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]

    # Find the correct subdirectory path
    subdir_path = None
    for base_path in base_paths:
        potential_path = os.path.join(base_path, args.subdirectory)
        if os.path.exists(potential_path):
            subdir_path = potential_path
            break

    if subdir_path is None:
        raise FileNotFoundError(f"Subdirectory {args.subdirectory} not found in any base path.")

    # Determine the number of available snapshots
    nt = determine_nt(subdir_path)

    # Build the list of required files based on the quantity and arrows flag
    required_files = []
    if args.quantity in ['gasvx', 'vort'] or args.arrows:
        required_files.append(f"gasvx{args.snapshot}.dat")
    if args.quantity in ['gasvy', 'vort'] or args.arrows:
        required_files.append(f"gasvy{args.snapshot}.dat")
    if args.quantity == 'gasvz' or args.arrows:
        required_files.append(f"gasvz{args.snapshot}.dat")
    if args.quantity in ['eps', 'pressure']:
        required_files.append(f"gasdens{args.snapshot}.dat")
    if args.quantity == 'pressure':
        required_files.append(f"gasenergy{args.snapshot}.dat")
    if args.quantity == 'dust1vx':
        required_files.append(f"dust1vx{args.snapshot}.dat")
    if args.quantity == 'dust1vy':
        required_files.append(f"dust1vy{args.snapshot}.dat")
    if args.quantity == 'dust1vz':
        required_files.append(f"dust1vz{args.snapshot}.dat")
    if args.quantity == 'eps':
        required_files.append(f"dust1dens{args.snapshot}.dat")

    # Debug print statement to show which files are needed
    print("Required files:", required_files)

    # Check if all required files exist
    all_files_exist = all(os.path.isfile(os.path.join(subdir_path, file)) for file in required_files)

    if not all_files_exist:
        print(f"Snapshot {args.snapshot} not found or incomplete. Using the latest available snapshot {nt - 1}.")
        args.snapshot = nt - 1

    # Ensure that the path to the summary0.dat file is correct
    summary_file = os.path.join(subdir_path, "summary0.dat")

    # Read simulation parameters
    print("READING SIMULATION PARAMETERS")
    parameters = read_parameters(summary_file)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    # Determine which fields need to be read based on the quantity and arrows flag
    read_gasvx = args.quantity in ['gasvx', 'vort'] or args.arrows
    read_gasvy = args.quantity in ['gasvy', 'vort'] or args.arrows
    read_gasvz = args.quantity == 'gasvz' or args.arrows
    read_gasdens = args.quantity in ['eps', 'pressure']
    read_gasenergy = args.quantity == 'pressure'
    read_dust1vx = args.quantity == 'dust1vx'
    read_dust1vy = args.quantity == 'dust1vy'
    read_dust1vz = args.quantity == 'dust1vz'
    read_dust1dens = args.quantity == 'eps'

    # Debug print statements
    print(f"read_gasvx: {read_gasvx}")
    print(f"read_gasvy: {read_gasvy}")
    print(f"read_gasvz: {read_gasvz}")
    print(f"read_gasdens: {read_gasdens}")
    print(f"read_gasenergy: {read_gasenergy}")
    print(f"read_dust1vx: {read_dust1vx}")
    print(f"read_dust1vy: {read_dust1vy}")
    print(f"read_dust1vz: {read_dust1vz}")
    print(f"read_dust1dens: {read_dust1dens}")

    # Read the current snapshot with appropriate fields
    print(f"Reading single snapshot")
    data_arrays, xgrid, ygrid, zgrid, parameters = read_single_snapshot(
        subdir_path, 
        snapshot=args.snapshot, 
        read_gasvx=read_gasvx, 
        read_gasvy=read_gasvy, 
        read_gasvz=read_gasvz, 
        read_gasdens=read_gasdens, 
        read_gasenergy=read_gasenergy, 
        read_dust1vx=read_dust1vx, 
        read_dust1vy=read_dust1vy, 
        read_dust1vz=read_dust1vz, 
        read_dust1dens=read_dust1dens
    )

    # Read the initial snapshot if needed
    initial_arrays = {}
    if args.quantity in ['gasvx', 'vort', 'pressure']:
        print(f"Reading initial snapshot")
        initial_arrays = read_initial_snapshot(
            subdir_path,
            read_gasvx=read_gasvx,
            read_gasvy=read_gasvy,
            read_gasenergy=read_gasenergy,
            read_gasdens=read_gasdens,
            read_dust1vx=read_dust1vx,
            read_dust1vy=read_dust1vy,
            read_dust1vz=read_dust1vz,
            read_dust1dens=read_dust1dens,
        )

    # Plot the selected quantity
    print(f"plotting the contours")
    plot_contours(data_arrays, initial_arrays, xgrid, ygrid, zgrid, args.quantity, args.subdirectory, args.azimuth, args.snapshot, arrows=args.arrows)
