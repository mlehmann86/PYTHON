#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_idefix.py: Visualize IDEFIX simulation output with options for movies, diagnostics, and space-time plots.
"""

# === Configuration ===
gas = True        # Whether to read gas component
test = False      # If True, read only gasdens
noniso = True     # True = adiabatic, False = isothermal
movie = False     # Enable movie creation
sptd = True       # Enable space-time diagnostics
planet = True     # Whether a planet is present
nsteps = 1        # Use every n-th time step
parallel = True   # Parallel reading of files
itstart = 0
itend = 10000

import argparse
#import numpy as np
import os
import sys
#import subprocess
import time
#import matplotlib.pyplot as plt
#from tqdm import tqdm

from data_reader import (
    read_idefix_vtk,
    read_idefix_vtk_parallel_filetype,
    determine_nt,
    read_parameters,
    reconstruct_grid,
    check_nfluids,
    check_planet_presence
)

from plotting_functions import (
    plot_radial_profiles_over_time,
    plot_metallicity,
    plot_pressure_gradient_deviation,
    plot_alpha_r,
    plot_vorticity_difference,
    create_simulation_movie,
    create_simulation_movie_noz,
    create_simulation_movie_axi,
    create_combined_movie
)


def determine_base_path(setup, subdirectory):
    """
    Determine the base path of the IDEFIX simulation given a setup and subdirectory.
    If setup is None, search all setups in known base paths.
    Returns (setup, full_path, base_path)
    """
    print(f"Searching for subdirectory: {subdirectory} (setup: {setup})")
    base_paths = [
        "/theory/lts/mlehmann/idefix-mkl/outputs",
        "/tiara/home/mlehmann/data/idefix-mkl/outputs"
    ]

    matches = []
    for base in base_paths:
        if setup is not None:
            candidate_path = os.path.join(base, setup, subdirectory)
            if os.path.isdir(candidate_path):
                return setup, candidate_path, base
        else:
            for possible_setup in os.listdir(base):
                candidate_path = os.path.join(base, possible_setup, subdirectory)
                if os.path.isdir(candidate_path):
                    matches.append((possible_setup, candidate_path, base))

    if not matches:
        print(f"❌ Could not find subdirectory '{subdirectory}' in any known setup folders.")
        raise FileNotFoundError(f"Subdirectory '{subdirectory}' not found.")
    elif len(matches) == 1:
        setup, path, base = matches[0]
        print(f"✅ Found simulation: setup = '{setup}', path = {path}")
        return setup, path, base
    else:
        print(f"🔍 Found multiple matches for subdirectory '{subdirectory}':")
        for i, (s, path, _) in enumerate(matches):
            print(f"  [{i}] setup = '{s}', path = {path}")
        idx = input("Enter index of setup to use: ")
        try:
            idx = int(idx)
            return matches[idx]
        except (ValueError, IndexError):
            print("❌ Invalid selection.")
            raise RuntimeError("Setup selection failed.")
# ============================== #
#       SCRIPT ENTRY POINT       #
# ============================== #

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:")
        print("   python3 plot_idefix.py <subdirectory> [--movie] [--sptd] [--quick]")
        print("Example:")
        print("   python3 plot_idefix.py test_noniso --movie --sptd")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Plot IDEFIX simulation results")
    parser.add_argument("subdirectory", type=str, help="Subdirectory containing the simulation results")
    parser.add_argument("--movie", action="store_true", help="Generate a movie")
    parser.add_argument("--sptd", action="store_true", help="Generate space time plots")
    parser.add_argument("--quick", action="store_true", help="Skip combined movie creation")
    args = parser.parse_args()

    movie = args.movie
    sptd = args.sptd
    quick = args.quick

    try:
        setup, subdir_path, base_path = determine_base_path(None, args.subdirectory)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # === Continue original logic ===
    global dust
    dust = False  # Temporarily disable dust detection
    summary_file = os.path.join(subdir_path, "idefix.0.log")

    planet = check_planet_presence(summary_file, IDEFIX=True)
    print("Planet present: Yes" if planet else "Planet present: No")

    print("***************************************")
    print(f"Running evaluation with gas={gas}")
    print(f"Running evaluation with dust={dust}")
    print(f"Running evaluation with test={test}")
    print(f"Running evaluation with movie={movie}")
    print(f"Running evaluation with sptd={sptd}")
    print(f"Running evaluation with noniso={noniso}")
    print(f"Running evaluation with planet={planet}")
    print(f"Quick mode is set to {quick}")
    print("***************************************")

    parameters = read_parameters(summary_file, IDEFIX=True)

    with open(summary_file, 'r') as f:
        for line in f:
            if line.strip().startswith("gamma"):
                gammas = float(line.strip().split()[1])
                break

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=True)

    start_time = time.time()

    if not parallel:
        data_arrays, data_types, xgrid, ygrid, zgrid, time_array = read_idefix_vtk(
        subdir_path, gas=gas, dust=dust, noniso=noniso, itstart=itstart, itend=itend,
        nsteps=nsteps, test=test
        )
    else:
        data_arrays, data_types, xgrid, ygrid, zgrid, time_array = read_idefix_vtk_parallel_filetype(
        subdir_path, gas=gas, dust=dust, noniso=noniso,
        itstart=itstart, itend=itend, nsteps=nsteps, test=test
        )

    print(f'nx = {nx}, ny = {ny}, nz = {nz}')
    print(f"Number of outputs (time steps): {len(time_array)}")
    print(f"Simulation time = {max(time_array)} ORB")

    gasdens = data_arrays['gasdens']
    if len(time_array) != gasdens.shape[-1]:
        print("Warning: Number of time steps does not match in time array and gasdens array!")
    if len(xgrid) != gasdens.shape[1]:
        print("Warning: Number of entries in x grid does not match gasdens array!")
    if len(ygrid) != gasdens.shape[0]:
        print("Warning: Number of entries in y grid does not match gasdens array!")
    if len(zgrid) != gasdens.shape[2]:
        print("Warning: Number of entries in z grid does not match gasdens array!")

    plot_radial_profiles_over_time(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, IDEFIX=True)
 
    if sptd:
        plot_pressure_gradient_deviation(data_arrays, xgrid, time_array, subdir_path, planet=planet)

    if dust and sptd:
        plot_metallicity(data_arrays, xgrid, time_array, subdir_path, planet=planet)
        plot_vorticity_difference(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, planet=planet)

    if not test:
        if nsteps == 1:
            plot_alpha_r(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, nsteps, dust=dust, planet=planet, IDEFIX=True)

    if movie:
        if ny > 1:
            create_simulation_movie_noz(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, planet=planet, IDEFIX=True)
            if len(zgrid) > 1:
                create_simulation_movie(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, dust=dust, planet=planet, IDEFIX=True)
            if planet and not quick:
                create_combined_movie(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, IDEFIX=True)
        else:
            create_simulation_movie_axi(data_arrays, xgrid, ygrid, zgrid, time_array, subdir_path, dust=dust, planet=planet, IDEFIX=True)



    # ============================== #
    #        EXAMPLE EXECUTION      #
    # ============================== #

    # Example command to run the script:
    #   python3 plot_idefix.py my_setup my_subdir --movie --sptd --quick
