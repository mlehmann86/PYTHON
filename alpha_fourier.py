def alpha_fourier(data_arrays, xgrid, ygrid, zgrid, time_array, output_path, nsteps, dust=False):
    """
    Calculate and plot the turbulent alpha parameter (alpha_r).

    Parameters:
    - data_arrays: Dictionary containing the simulation data.
    - xgrid: Radial grid.
    - zgrid: Vertical grid.
    - time_array: Array of time steps.
    - output_path: Directory to save the plot.
    """

    # Apply radial masks
    radial_mask = (xgrid >= 0.9) & (xgrid <= 1.2)

    nz = len(zgrid)
    ny = len(ygrid)


    # Initialize arrays to hold the computed values
    alpha_r = np.zeros(len(time_array))
    

    # Read disk aspect ratio (precomputed once)
    summary_file = os.path.join(output_path, "summary0.dat")
    parameters = read_parameters(summary_file)
    aspectratio = parameters['ASPECTRATIO']

    # Disk aspect ratio H_g
    H_g = float(aspectratio)

    # Precompute the std of zgrid if it's constant
    zgrid_std = np.std(zgrid)

    r = xgrid[np.newaxis, :, np.newaxis]  # Radial grid expanded to shape (1, nx, 1)
  
    # Compute the initial gas and dust mass at t_idx=0 for scaling
    gasdens_initial = data_arrays['gasdens'][:, radial_mask, :, 0]
   
  

    print('COMPUTING TIME EVOLUTION OF VARIOUS QUANTITIES')

    def process_time_step(t_idx):
        """Process a single time step."""
        # Extract data for this time step and apply radial masks dynamically
        gasdens_t = data_arrays['gasdens'][:, radial_mask, :, t_idx]

 

        # Perform the calculations for other quantities
        numerator = np.mean(gasdens_t * data_arrays['gasvy'][:, radial_mask, :, t_idx] * 
                            (data_arrays['gasvx'][:, radial_mask, :, t_idx] - data_arrays['gasvx'][:, radial_mask, :, 0]))
        denominator = np.mean(data_arrays['gasenergy'][:, radial_mask, :, t_idx])
        alpha_r_val = numerator / denominator
        
    

        # Return calculated values, including avg_metallicity
        return alpha_r_val

    # Process the time steps in parallel
    results = Parallel(n_jobs=8, backend='threading')(delayed(process_time_step)(t_idx) for t_idx in tqdm(range(len(time_array)), desc="Processing time steps"))

    # Unpack results
    for t_idx, (alpha_r_val) in enumerate(results):
        alpha_r[t_idx] = alpha_r_val
       



    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, alpha_r, label=r'$\alpha_r(t)$')
   

    plt.ylim(1e-6, 1)  # Y-axis limits when no dust is present

    plt.xlabel('Time [Orbits]')
    plt.ylabel('Values (scaled)')
    plt.yscale('log')
    plt.title(r'$\alpha_r$')

    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend()

    # Save the plot as a PDF
    pdf_filename = f"{os.path.basename(output_path)}_alpha_fourier.pdf"
    output_filepath = os.path.join(output_path, pdf_filename)
    plt.savefig(output_filepath)
    plt.close()

    print(f"Alpha_r, RMS dust velocity, max epsilon, scale height, and scaled masses plot saved to {output_filepath}")
    
    # Call the scp_transfer function if needed
    scp_transfer(output_filepath, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

