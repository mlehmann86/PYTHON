#plot_fargo.py (main routine)
gas = True  # or False, depending on whether you want to read gas or dust components
test = False  # Set to 'yes' to read only gasdens for testing, 'no' otherwise
noniso= True
movie=False
sptd=True
nsteps=1 # skip intermediate time steps
parallel=True #file reading serial or parallel

itstart=0
itend=251


import argparse
import numpy as np
import os
import sys
import subprocess
import time  # Import the time module
import matplotlib.pyplot as plt
from tqdm import tqdm

from plotting_functions_vsi import (
    plot_initial_profiles,
    create_simulation_movie_axi,
    plot_alpha_r,
    plot_azimuthal_velocity_deviation
)

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import the module from the parent directory

from data_reader import (
    read_hydro_fields_serial, 
    read_hydro_fields_parallel_optimized, 
    read_hydro_fields_parallel_filetype, 
    determine_nt, 
    read_parameters, 
    reconstruct_grid,
    check_nfluids
)



from data_storage_vsi import save_simulation_quantities  # Import data storage functions

# Optionally, remove the parent directory from sys.path after import to clean up
sys.path.pop(0)


def determine_base_path(subdirectory):
    """
    Determine the base path for a given simulation subdirectory by checking if the directory exists.

    Args:
        subdirectory (str): The simulation subdirectory name.

    Returns:
        tuple: (subdir_path, base_path), where:
            subdir_path (str): The full path to the simulation subdirectory.
            base_path (str): The base path where the subdirectory was found.

    Raises:
        FileNotFoundError: If the subdirectory is not found in any base path.
    """
    print(f"Searching for subdirectory: {subdirectory}")

    base_paths = [
        "/theory/lts/mlehmann/FARGO3D/outputs",
        "/tiara/home/mlehmann/data/FARGO3D/outputs"
    ]

    for base_path in base_paths:
        potential_path = os.path.join(base_path, subdirectory)
        print(f"Checking path: {potential_path}")
        if os.path.exists(potential_path):
            print(f"Directory found: {potential_path}")
            return potential_path, base_path

    # If no valid base path is found
    print("Simulation directory not found in any base path.")
    raise FileNotFoundError(f"Simulation directory '{subdirectory}' not found in any base path.")


# ============================== #
#       SCRIPT ENTRY POINT       #
# ============================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Fargo simulation results")
    parser.add_argument("subdirectory", type=str, help="Subdirectory containing the simulation results")
    
    # Optional flag for movie/sptd mode
    parser.add_argument("--movie", action="store_true", help="Generate a movie")
    parser.add_argument("--sptd", action="store_true", help="Generate space time plots")
    
    args = parser.parse_args()

    # Determine the correct base path
    subdir_path, base_path = determine_base_path(args.subdirectory)

    # Access the flags
    movie = args.movie
    sptd = args.sptd


    global dust
    dust=True
    summary_file = os.path.join(subdir_path, "summary0.dat")
    dust = check_nfluids(summary_file, dust)

    # Print the evaluation lines
    print(f"***************************************")
    print(f"Running evaluation with gas={gas}")
    print(f"Running evaluation with dust={dust}")
    print(f"Running evaluation with test={test}")
    print(f"Running evaluation with movie={movie}")
    print(f"Running evaluation with sptd={sptd}")
    print(f"Running evaluation with noniso={noniso}")
    print(f"***************************************")

# ============================== #
#        DATA READING SECTION     #
# ============================== #
        

    # Read parameters from summary0.dat
    summary_file = os.path.join(subdir_path, "summary0.dat")
    parameters = read_parameters(summary_file)

    # Extract grid information
    gammas = parameters['GAMMA']
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    # Measure time for reading arrays
    start_time = time.time()

    # Assuming this is within your main function or main script block
    if parallel:    
        data_arrays, data_types, xgrid, ygrid, zgrid, time_array = read_hydro_fields_parallel_filetype(subdir_path, gas=gas, dust=dust, noniso=noniso, itstart=itstart,itend=itend, nsteps=nsteps, test=test, backend='threading')

    else:    
        data_arrays, data_types, xgrid, ygrid, zgrid, time_array = read_hydro_fields_serial(subdir_path, gas=gas, dust=dust, noniso=noniso, itstart=0, nsteps=nsteps, test=test)

    # Print debug information
    print(f'nx = {nx}, ny = {ny}, nz = {nz}')
    print(f"Number of outputs (time steps): {len(time_array)}")
    print(f"Simulation time = {max(time_array)} ORB")

    gasdens = data_arrays['gasdens']
    # Check if dimensions agree
    if len(time_array) != gasdens.shape[-1]:
        print("Warning: Number of time steps does not match in time array and gasdens array!")

    if len(xgrid) != gasdens.shape[1]:
        print("Warning: Number of entries in x grid does not match gasdens array!")

    if len(ygrid) != gasdens.shape[0]:
        print("Warning: Number of entries in y grid does not match gasdens array!")

    if len(zgrid) != gasdens.shape[2]:
        print("Warning: Number of entries in z grid does not match gasdens array!")


    #exit()
# ============================== #
#        DATA PLOTTING SECTION    #
# ============================== #

    # Plot radial profile
    plot_initial_profiles(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path)

    if sptd:
        plot_azimuthal_velocity_deviation(data_arrays, xgrid, time_array, subdir_path)

    if not test:
        if nsteps == 1:
            plot_alpha_r(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, nsteps,dust=dust)
 
    if movie:
        if ny > 1:
            create_simulation_movie(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path,dust=dust)
        else:
            create_simulation_movie_axi(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path,dust=dust)



# ============================== #
#        EXAMPLE EXECUTION        #
# ============================== #


# Example command to run the script:
#python3 plot_fargo.py subdirectory_name --movie --sptd


