#plot_fargo.py (main routine)
gas = True  # or False, depending on whether you want to read gas or dust components
test = False  # Set to 'yes' to read only gasdens for testing, 'no' otherwise
noniso= True
movie=False
sptd=True
planet=True
nsteps=1 # skip intermediate time steps
parallel=True #file reading serial or parallel

itstart=0
itend=500000

import argparse
import numpy as np
import os
import sys
import subprocess
import time  # Import the time module
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================== #
#        MAIN EXECUTION SECTION  #
# ============================== #

from data_reader import (
    read_hydro_fields_serial, 
    read_hydro_fields_parallel_optimized, 
    read_hydro_fields_parallel_filetype, 
    determine_nt, 
    read_parameters, 
    reconstruct_grid,
    check_nfluids,
    check_planet_presence
)

from data_storage import determine_base_path

from plotting_functions import (
    #plot_debugging_profiles,
    plot_radial_profiles_over_time,
    plot_metallicity, 
    plot_pressure_gradient_deviation, 
    #plot_vertically_averaged_gas_density,
    plot_alpha_r, 
    plot_vorticity_difference, 
    #plot_radial_profiles, 
    create_simulation_movie, 
    create_simulation_movie_noz, 
    create_simulation_movie_axi, 
    create_combined_movie 
)


import os



# ============================== #
#       SCRIPT ENTRY POINT       #
# ============================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Fargo simulation results")
    parser.add_argument("subdirectory", type=str, help="Subdirectory containing the simulation results")
    
    # Optional flags
    parser.add_argument("--movie", action="store_true", help="Generate a movie")
    parser.add_argument("--sptd", action="store_true", help="Generate space time plots")
    parser.add_argument("--quick", action="store_true", help="Skip combined movie creation")

    args = parser.parse_args()

    # Determine the subdir path
    subdir_path = determine_base_path(args.subdirectory)

    # Access the flags
    movie = args.movie
    sptd = args.sptd
    quick = args.quick

    # Check number of fluids, planet presence, etc.
    global dust
    dust = True
    summary_file = os.path.join(subdir_path, "summary0.dat")
    dust = check_nfluids(summary_file, dust)

    planet = check_planet_presence(summary_file)
    if planet:
        print("Planet present: Yes")
    else:
        print("Planet present: No")

    # Print the evaluation lines
    print(f"***************************************")
    print(f"Running evaluation with gas={gas}")
    print(f"Running evaluation with dust={dust}")
    print(f"Running evaluation with test={test}")
    print(f"Running evaluation with movie={movie}")
    print(f"Running evaluation with sptd={sptd}")
    print(f"Running evaluation with noniso={noniso}")
    print(f"Running evaluation with planet={planet}")
    print(f"Quick mode is set to {quick}")
    print(f"***************************************")

    # ============================== #
    #        DATA READING SECTION   #
    # ============================== #

    # Read parameters from summary0.dat
    parameters = read_parameters(summary_file)

    # Extract grid information
    gammas = parameters['GAMMA']
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    # Measure time for reading arrays
    start_time = time.time()

    # Parallel or serial data reading
    if parallel:    
        data_arrays, data_types, xgrid, ygrid, zgrid, time_array = read_hydro_fields_parallel_filetype(
            subdir_path, gas=gas, dust=dust, noniso=noniso, itstart=itstart, itend=itend, 
            nsteps=nsteps, test=test, backend='threading'
        )
    else:    
        data_arrays, data_types, xgrid, ygrid, zgrid, time_array = read_hydro_fields_serial(
            subdir_path, gas=gas, dust=dust, noniso=noniso, itstart=0, 
            nsteps=nsteps, test=test
        )

    # Print debug information
    print(f'nx = {nx}, ny = {ny}, nz = {nz}')
    print(f"Number of outputs (time steps): {len(time_array)}")
    print(f"Simulation time = {max(time_array)} ORB")

    # Check if dimensions agree
    if len(time_array) != data_arrays['gasdens'].shape[-1]:
        print("Warning: Number of time steps does not match in time array and gasdens array!")

    if len(xgrid) != data_arrays['gasdens'].shape[1]:
        print("Warning: Number of entries in x grid does not match gasdens array!")

    if len(ygrid) != data_arrays['gasdens'].shape[0]:
        print("Warning: Number of entries in y grid does not match gasdens array!")

    if len(zgrid) != data_arrays['gasdens'].shape[2]:
        print("Warning: Number of entries in z grid does not match gasdens array!")

    # ============================== #
    #        DATA PLOTTING SECTION  #
    # ============================== #

    # Plot debugging profiles
    #plot_debugging_profiles(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path)

    plot_radial_profiles_over_time(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path)
    
    if sptd:
        plot_pressure_gradient_deviation(data_arrays, xgrid, time_array, subdir_path, planet=planet)
        #plot_azimuthal_velocity_deviation(data_arrays, xgrid, time_array, subdir_path, planet=planet)
        #plot_vertically_averaged_gas_density(data_arrays, xgrid, time_array, subdir_path, planet=planet)

    # space time plots
    if dust and sptd:
        plot_metallicity(data_arrays, xgrid, time_array, subdir_path, planet=planet)
        plot_vorticity_difference(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, planet=planet)
    
    if not test:
        if nsteps == 1:
            plot_alpha_r(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, nsteps, dust=dust, planet=planet)
            #alpha_fourier(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, nsteps, dust=dust)
        #plot_radial_profiles(data_arrays, xgrid, subdir_path, nsteps=nsteps, itstart=itstart, itend=itend, planet=planet)

    # Create movies, unless --quick is set to skip the combined movie
    if movie:
        if ny > 1:
            if len(zgrid) > 1:  # Check if there is no vertical dimension
                create_simulation_movie(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, dust=dust, planet=planet)
            create_simulation_movie_noz(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, planet=planet)
            
            # Skip combined movie if --quick is set
            if planet and not quick:
                create_combined_movie(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path)
        else:
            create_simulation_movie_axi(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, dust=dust, planet=planet)


    # ============================== #
    #        EXAMPLE EXECUTION      #
    # ============================== #

    # Example command to run the script:
    #   python3 plot_fargo.py my_subdir --movie --sptd --quick
