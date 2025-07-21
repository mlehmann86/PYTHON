# data_reader.py

import os
import re
import math
import time
import ctypes
import numpy as np
from tqdm import tqdm
from multiprocessing import Array, cpu_count, Process, Lock, shared_memory, Pool, Manager
from joblib import Parallel, delayed
import meshio


########################################################################################################################################

def check_nfluids(summary_file, dust):
    # Open and read the summary0.dat file
    with open(summary_file, 'r') as file:
        contents = file.read()

    # Check if the line contains -DNFLUIDS=2 or -DNFLUIDS=1
    if '-DNFLUIDS=2' in contents:
        dust = True  # Override dust to True if NFLUIDS=2
        print("Detected -DNFLUIDS=2: Overriding dust=True")
    elif '-DNFLUIDS=1' in contents:
        dust = False  # Override dust to False if NFLUIDS=1
        print("Detected -DNFLUIDS=1: Overriding dust=False")

    return dust

########################################################################################################################################

def check_planet_presence(summary_file, IDEFIX=False):
    """
    Checks if a planet is present in the simulation.

    If IDEFIX=True, it uses IDEFIX log file format for detection.
    Otherwise, it assumes Fargo format and looks for standard headers.
    """
    with open(summary_file, 'r') as file:
        contents = file.readlines()

    planet_present = False

    if IDEFIX:
        for line in contents:
            if "PlanetarySystem: have" in line and "planets" in line:
                try:
                    num = int(line.strip().split()[-2])
                    if num > 0:
                        planet_present = True
                        break
                except Exception as e:
                    print(f"Could not parse number of planets: {e}")
                    return False
    else:
        for i, line in enumerate(contents):
            if "# Planet Name" in line:
                try:
                    planet_line = contents[i + 1].strip()
                    mass_value = float(planet_line.split()[2])
                    if mass_value > 0:
                        planet_present = True
                        break
                except (IndexError, ValueError) as e:
                    print(f"Error reading planet mass: {e}")
                    return False

    if not planet_present:
        print("No planet detected in the simulation.")
    return planet_present


########################################################################################################################################





def determine_nt(path, IDEFIX=False):
    """
    Determine the number of available snapshot files.
    For IDEFIX, this counts *.vtk files.
    For Fargo3D, it checks specific .dat file types.
    """
    if IDEFIX:
        vtk_files = [f for f in os.listdir(path) if f.endswith('.vtk') and re.match(r'data\.\d{4}\.vtk$', f)]
        return len(vtk_files)
    
    file_types = ['summary', 'gasdens', 'gasenergy', 'gasvx', 'gasvy']
    max_indices = []

    for file_type in file_types:
        snapshot_numbers = []
        for f in os.listdir(path):
            match = re.match(rf'{file_type}(\d+)\.dat$', f)
            if match:
                snapshot_numbers.append(int(match.group(1)))

        if snapshot_numbers:
            max_indices.append(max(snapshot_numbers))

    return min(max_indices) + 1 if max_indices else 0


########################################################################################################################################

def read_parameters(filename, IDEFIX=False):
    """
    Read and parse simulation parameters.
    Supports FARGO3D (default) and IDEFIX (if IDEFIX=True).
    """
    print('READING SIMULATION PARAMETERS')
    parameters = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if IDEFIX:
        in_params = False
        for line in lines:
            line = line.strip()
            if "Input Parameters using input file" in line:
                in_params = True
                continue
            if in_params:
                if line.startswith("---") or line == "":
                    continue
                if line.startswith("["):  # Section header
                    current_section = line.strip("[]")
                    continue

                # Special handling for smoothing line with plummer
                if line.startswith("smoothing") and "plummer" in line:
                    parts = line.split()
                    try:
                        smoothing_val = float(parts[2])  # This is the 0.02 value
                        parameters['smoothing'] = smoothing_val
                    except (IndexError, ValueError):
                        parameters['smoothing'] = None
                    continue

                # Split into key-value, special case for grid lines
                if 'grid' in line.lower():
                    parts = line.split()
                    key = parts[0].strip().lower()
                    parameters[key] = parts[1:]  # store the full list of values
                elif "=" in line:
                    key, val = map(str.strip, line.split("=", 1))
                    key = key.lower()
                    try:
                        parameters[key] = float(val)
                    except ValueError:
                        parameters[key] = val
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].strip().lower()
                        val = parts[1]
                        try:
                            parameters[key] = float(val)
                        except ValueError:
                            parameters[key] = val

    else:
        # FARGO3D parser logic (left unchanged)
        in_parameters_section = False
        for line in lines:
            line = line.strip()
            if "PARAMETERS SECTION:" in line:
                in_parameters_section = True
            elif in_parameters_section and line:
                if line.startswith('=') or line == "":
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    value = value.strip().split()[0]  # Extract the numeric value, strip any comment
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    parameters[key.upper()] = value

    return parameters

########################################################################################################################################

def reconstruct_grid(parameters, IDEFIX=False):
    """Reconstruct grid based on parameters. Supports both FARGO3D and IDEFIX modes."""
    print('RECONSTRUCTING GRIDS')

    if IDEFIX:
        # Extract from x1-grid, x2-grid, etc. using lowercase keys
        if 'x1-grid' not in parameters:
            raise ValueError("x1-grid not found in parameters.")
        if 'x2-grid' not in parameters:
            parameters['x2-grid'] = ['1', '0.0', '1', 'u', '1.0']
        if 'x3-grid' not in parameters:
            parameters['x3-grid'] = ['1', '0.0', '1', 'u', '1.0']

        x1grid = parameters['x1-grid']
        x2grid = parameters['x2-grid']
        x3grid = parameters['x3-grid']

        nx = int(x1grid[2])
        ny = int(x2grid[2])
        nz = int(x3grid[2])

        r_min = float(x1grid[1])
        r_max = float(x1grid[4])
        y_min = float(x2grid[1])
        y_max = float(x2grid[4])
        z_min = float(x3grid[1])
        z_max = float(x3grid[4])
    else:
        ny = int(parameters['NX'])
        nx = int(parameters['NY'])
        nz = int(parameters['NZ'])

        y_min = parameters['XMIN']
        y_max = parameters['XMAX']
        r_min = parameters['YMIN']
        r_max = parameters['YMAX']
        z_min = parameters['ZMIN']
        z_max = parameters['ZMAX']

    xgrid = np.linspace(r_min, r_max, nx)
    ygrid = np.linspace(y_min, y_max, ny)
    zgrid = np.linspace(z_min, z_max, nz)

    print(f"xgrid: [{r_min:.3f}, {r_max:.3f}], nx={nx}")
    print(f"ygrid: [{y_min:.3f}, {y_max:.3f}], ny={ny}")
    print(f"zgrid: [{z_min:.3f}, {z_max:.3f}], nz={nz}")

    return xgrid, ygrid, zgrid, ny, nx, nz


########################################################################################################################################

def read_hydro_fields_serial(path, gas=True, dust=False, noniso=False, itstart=0, nsteps=1, test=False):
    print('READING HYDRO FIELDS - SERIAL')

    # Determine the number of summary files
    nt = determine_nt(path)
    nts = math.ceil((nt - itstart) / nsteps)

    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    gammas = parameters['GAMMA']
    ninterm = parameters['NINTERM']

    # Time array based on dt=8 and nt
    dt = ninterm / 20
    time_array = np.arange(itstart, itstart + nts * nsteps * dt, nsteps * dt)

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    if ny==1:
        print(f"Simulation is axisymmetric 2D")
    if ny>1:
        print(f"Simulation is 3D")

    data_types = []
    if gas:
        data_types = ['gasvx', 'gasvy', 'gasvz', 'gasdens']
    if dust:
        data_types += ['dust1vx', 'dust1vy', 'dust1vz', 'dust1dens']
    if noniso:
        data_types += ['gasenergy']
    if test:
        if dust:
            data_types = ['gasdens', 'dust1dens']
        else:
            data_types = ['gasdens']

    print(f"Data types to be read: {data_types}")

    # Allocate memory for each data type and store it in a dictionary
    data_arrays = {array_name: np.zeros((ny, nx, nz, nts)) for array_name in data_types}

    def read_single_time_step(array_name, it):
        file_prefix = os.path.join(path, array_name)
        file = file_prefix + str(it) + '.dat'
        if os.path.exists(file):
            return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
        else:
            return np.zeros((ny, nx, nz))

    def read_and_store(array_name, data_array):
        progress_bar = tqdm(range(itstart, nt, nsteps), desc=f"Reading {array_name}", unit="files", leave=True)
        for i, it in enumerate(progress_bar):
            data_array[..., i] = read_single_time_step(array_name, it)
        progress_bar.close()

    # Measure time for reading and storing data types
    start_time = time.time()

    # Read each file type sequentially and store it in the data_arrays dictionary
    for array_name, data_array in data_arrays.items():
        read_and_store(array_name, data_array)

    # Apply the scaling to gasenergy if it exists
    if 'gasenergy' in data_types:
        print("Applying gamma scaling to gasenergy.")
        data_arrays['gasenergy'] *= (gammas - 1)

    end_time = time.time()
    print(f"Total time taken to read and store all data types: {end_time - start_time:.2f} seconds")

    print('All fields read')

    # Return the data_arrays dictionary along with the grid and time arrays
    return data_arrays, data_types, xgrid, ygrid, zgrid, time_array


########################################################################################################################################

def read_hydro_fields_parallel_optimized(path, gas=True, dust=False, noniso=False, itstart=0, nsteps=1, test=False, n_jobs=8, backend='loky'):
    print('READING HYDRO FIELDS')

    # Determine the number of summary files
    nt = determine_nt(path)
    nts = math.ceil((nt - itstart) / nsteps)

    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    gammas = parameters['GAMMA']
    ninterm = parameters['NINTERM']

    # Time array based on dt=8 and nt
    dt = ninterm / 20
    time_array = np.arange(itstart, itstart + nts * nsteps * dt, nsteps * dt)

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    data_types = []
    if gas:
        data_types = ['gasvx', 'gasvy', 'gasvz', 'gasdens']
    if dust:
        data_types += ['dust1vx', 'dust1vy', 'dust1vz', 'dust1dens']
    if noniso:
        data_types += ['gasenergy']
    if test:
        if dust:
            data_types = ['gasdens', 'dust1dens']
        else:
            data_types = ['gasdens']

    print(f"Data types to be read: {data_types}")

    def read_single_time_step(array_name, it):
        file_prefix = os.path.join(path, array_name)
        file = file_prefix + str(it) + '.dat'
        if os.path.exists(file):
            return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)
        else:
            return np.zeros((ny, nx, nz))

    def process_single_time_step(array_name, it):
        return read_single_time_step(array_name, it)

    def read_full_array(array_name):
        # This function reads the entire array for a given data type in parallel across time steps
        time_steps = range(itstart, nt, nsteps)
        array_full = np.array(
            Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(process_single_time_step)(array_name, it) for it in tqdm(time_steps, desc=f"Reading {array_name}")
            )
        )

        return array_full.transpose(1, 2, 3, 0)

    start_time = time.time()

    # Parallel processing of each full array and storing directly in the final structure
    data_arrays = {array_name: read_full_array(array_name) for array_name in data_types}

    if 'gasenergy' in data_arrays:
        print("Applying gamma scaling to gasenergy.")
        data_arrays['gasenergy'] *= (gammas - 1)

    end_time = time.time()
    print(f"Total time taken to read and store all data types: {end_time - start_time:.2f} seconds")
    print('All fields read')

    return data_arrays, data_types, xgrid, ygrid, zgrid, time_array

########################################################################################################################


import numpy as np
import os
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import numpy as np

# Function to replace NaNs and plot azimuthal average for the problematic time step
def visualize_nan_issue(data_arrays, xgrid, zgrid, problematic_timestep):
    # Replace NaN values with a large number (1e10)
    gasdens = data_arrays['gasdens'][:, :, :, problematic_timestep]
    dust1dens = data_arrays['dust1dens'][:, :, :, problematic_timestep]

    gasdens_nan_mask = np.isnan(gasdens)
    dust1dens_nan_mask = np.isnan(dust1dens)

    gasdens[np.isnan(gasdens)] = 1e10
    dust1dens[np.isnan(dust1dens)] = 1e10

    # Azimuthally average (along the y-direction, axis 0)
    gasdens_avg = np.mean(gasdens, axis=0)  # Shape (nx, nz)
    dust1dens_avg = np.mean(dust1dens, axis=0)  # Shape (nx, nz)

    # Create x-z imshow plot for gasdens and dust1dens
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for gasdens
    im1 = axs[0].imshow(gasdens_avg.T, extent=[xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]], 
                        aspect='auto', origin='lower', cmap='coolwarm', vmin=0, vmax=1e10)
    axs[0].set_title('Azimuthally Averaged Gas Density')
    axs[0].set_xlabel('Radial coordinate (x)')
    axs[0].set_ylabel('Vertical coordinate (z)')
    fig.colorbar(im1, ax=axs[0])

    # Highlight NaN locations with contour lines or scatter points
    nan_gasdens_avg = np.mean(gasdens_nan_mask, axis=0)
    axs[0].contour(nan_gasdens_avg.T, levels=[0.5], colors='red', extent=[xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]])

    # Plot for dust1dens
    im2 = axs[1].imshow(dust1dens_avg.T, extent=[xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]], 
                        aspect='auto', origin='lower', cmap='coolwarm', vmin=0, vmax=1e10)
    axs[1].set_title('Azimuthally Averaged Dust Density')
    axs[1].set_xlabel('Radial coordinate (x)')
    axs[1].set_ylabel('Vertical coordinate (z)')
    fig.colorbar(im2, ax=axs[1])

    # Highlight NaN locations with contour lines or scatter points
    nan_dust1dens_avg = np.mean(dust1dens_nan_mask, axis=0)
    axs[1].contour(nan_dust1dens_avg.T, levels=[0.5], colors='red', extent=[xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]])

    plt.tight_layout()
    plt.show()


########################################################################################################################

# Main function for reading hydro fields
def read_hydro_fields_parallel_filetype(path, gas=True, dust=False, noniso=False, itstart=0, itend=None, nsteps=1, test=False, backend='loky'):
    print('READING HYDRO FIELDS')

    # Determine the number of summary files
    nt = determine_nt(path)
    if itend is None:
        itend = nt
    
    nts = math.ceil((min(itend, nt) - itstart) / nsteps)

    summary_file = os.path.join(path, "summary0.dat")
    parameters = read_parameters(summary_file)
    gammas = parameters['GAMMA']
    ninterm = parameters['NINTERM']

    # Time array based on dt=8 and nt
    dt = ninterm / 20
    time_array = np.arange(itstart, itstart + nts * nsteps * dt, nsteps * dt)

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    if ny == 1:
        print(f"Simulation is axisymmetric 2D")
    if ny > 1:
        print(f"Simulation is 3D")

    data_types = []
    if gas:
        data_types = ['gasvx', 'gasvy', 'gasvz', 'gasdens']
    if dust:
        data_types += ['dust1vx', 'dust1vy', 'dust1vz', 'dust1dens']
    if noniso:
        data_types += ['gasenergy']
    if test:
        if dust:
            data_types = ['gasdens', 'dust1dens']
        else:
            data_types = ['gasdens']

    print(f"Data types to be read: {data_types}")

    def read_single_time_step(array_name, it):
        file_prefix = os.path.join(path, array_name)
        file = file_prefix + str(it) + '.dat'
        if os.path.exists(file):
            arr = np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny))
            return arr.transpose(2, 1, 0)  # (ny, nx, nz)
        else:
            return np.zeros((ny, nx, nz))

    def process_filetype(array_name):
        # This function handles all time steps for a given data type
        time_steps = range(itstart, min(itend, nt), nsteps)
        shape = (len(time_steps), ny, nx, nz)
        array_full = np.empty(shape, dtype=np.float64)

        for i, it in enumerate(tqdm(time_steps, desc=f"Reading {array_name}")):
            array_full[i] = read_single_time_step(array_name, it)  # (ny, nx, nz)

        array_full = array_full.transpose(1, 2, 3, 0)  # Final shape: (ny, nx, nz, nt)

        # Perform checks for gasdens (for both zeros and NaNs) and dust1dens (for NaNs)
        first_nan_timestep = None
        if array_name in ['gasdens', 'dust1dens']:
            for i, it in enumerate(time_steps):
                problematic_nans = np.isnan(array_full[:, :, :, i])
                if np.any(problematic_nans):
                    print(f"WARNING: NaN values detected in {array_name} at file number {it} (index {i}).")
                    if first_nan_timestep is None:
                        first_nan_timestep = it  # Record the first timestep with NaN

        return array_full, first_nan_timestep

    start_time = time.time()

    # Parallel processing across filetypes
    results = Parallel(n_jobs=len(data_types), backend=backend)(
        delayed(process_filetype)(array_name) for array_name in data_types
    )

    # Separate the data arrays and first NaN timestep for each type
    data_arrays = {data_types[i]: results[i][0] for i in range(len(data_types))}
    nan_timesteps = [results[i][1] for i in range(len(data_types)) if results[i][1] is not None]

    if 'gasenergy' in data_arrays:
        print("Applying gamma scaling to gasenergy.")
        data_arrays['gasenergy'] *= (gammas - 1)

    end_time = time.time()
    print(f"Total time taken to read and store all data types: {end_time - start_time:.2f} seconds")
    print('All fields read')

    # If any NaN values were detected, visualize the first problematic timestep
    if nan_timesteps:
        first_problematic_timestep = min(nan_timesteps)  # Get the earliest timestep with NaNs
        print(f"Visualizing first problematic timestep: {first_problematic_timestep}")
        visualize_nan_issue(data_arrays, xgrid, zgrid, first_problematic_timestep)

        print("\nFinal array shapes:")
    for key in data_arrays:
        print(f"{key}: shape = {data_arrays[key].shape}")

    return data_arrays, data_types, xgrid, ygrid, zgrid, time_array




###########################################################################################################################

import os
import subprocess
import numpy as np

def read_single_snapshot(path, snapshot,
                         read_gasvx=False, read_gasvy=False, read_gasvz=False,
                         read_gasenergy=False, read_gasdens=False, read_dust1vx=False,
                         read_dust1vy=False, read_dust1vz=False, read_dust1dens=False,
                         IDEFIX=False, params=None, grid_info=None):
    """
    Reads a single snapshot from either a FARGO3D or IDEFIX simulation.

    This function unifies the reading logic. It can optionally accept pre-read
    parameters and grid info to avoid redundant reads in loops.

    Args:
        path (str): The local path to the simulation directory.
        snapshot (int): The snapshot number.
        read_gasvx, ... (bool): Flags to read specific fields.
        IDEFIX (bool): Set to True for an IDEFIX simulation.
        params (dict, optional): Pre-loaded parameters dictionary.
        grid_info (tuple, optional): Pre-loaded grid info (x, y, z, ny, nx, nz).

    Returns:
        tuple: (data_arrays, xgrid, ygrid, zgrid, parameters)
    """
    if params is None:
        param_file = os.path.join(path, "idefix.0.log" if IDEFIX else "summary0.dat")
        params = read_parameters(param_file, IDEFIX=IDEFIX)

    if grid_info is None:
        xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(params, IDEFIX=IDEFIX)
    else:
        xgrid, ygrid, zgrid, ny, nx, nz = grid_info

    data_arrays = {}

    if IDEFIX:
        # --- IDEFIX VTK Reading Logic ---
        import glob
        from vtk import vtkStructuredGridReader
        from vtk.util.numpy_support import vtk_to_numpy

        vtk_files = sorted(glob.glob(os.path.join(path, "data.*.vtk")))
        if snapshot >= len(vtk_files):
            raise IndexError(f"Snapshot index {snapshot} exceeds available snapshots ({len(vtk_files)})")
        vtk_file = vtk_files[snapshot]

        reader = vtkStructuredGridReader()
        reader.SetFileName(vtk_file)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        cell_data = reader.GetOutput().GetCellData()

        field_map = {
            'RHO': ('gasdens', read_gasdens), 'PRS': ('gasenergy', read_gasenergy),
            'VX1': ('gasvy', read_gasvy), 'VX2': ('gasvx', read_gasvx), 'VX3': ('gasvz', read_gasvz),
        }
        gamma = float(params.get("gamma", 1.6667))

        for vtk_name, (internal_name, read_flag) in field_map.items():
            if read_flag and cell_data.GetArray(vtk_name):
                raw_array = vtk_to_numpy(cell_data.GetArray(vtk_name))
                reshaped = raw_array.reshape((nz, ny, nx)).transpose(1, 2, 0)
                if vtk_name == 'PRS':
                    reshaped /= (gamma - 1.0)
                data_arrays[internal_name] = reshaped
    else:
        # --- FARGO3D .dat Reading Logic ---
        gammas = params['GAMMA']
        # Read requested gas fields
        if read_gasvx:
            data_arrays['gasvx'] = read_field_file(path, 'gasvx', snapshot, nx, ny, nz)
        if read_gasvy:
            data_arrays['gasvy'] = read_field_file(path, 'gasvy', snapshot, nx, ny, nz)
        if read_gasvz:
            data_arrays['gasvz'] = read_field_file(path, 'gasvz', snapshot, nx, ny, nz)
        if read_gasdens:
            data_arrays['gasdens'] = read_field_file(path, 'gasdens', snapshot, nx, ny, nz)
        if read_gasenergy:
            data_arrays['gasenergy'] = read_field_file(path, 'gasenergy', snapshot, nx, ny, nz)
            data_arrays['gasenergy'] *= (gammas - 1)
        # Read requested dust fields
        if read_dust1vx:
            data_arrays['dust1vx'] = read_field_file(path, 'dust1vx', snapshot, nx, ny, nz)
        if read_dust1vy:
            data_arrays['dust1vy'] = read_field_file(path, 'dust1vy', snapshot, nx, ny, nz)
        if read_dust1vz:
            data_arrays['dust1vz'] = read_field_file(path, 'dust1vz', snapshot, nx, ny, nz)
        if read_dust1dens:
            data_arrays['dust1dens'] = read_field_file(path, 'dust1dens', snapshot, nx, ny, nz)

    return data_arrays, xgrid, ygrid, zgrid, params




def read_field_file(path, field_name, snapshot, nx, ny, nz):
    """
    Reads a field file, fetching it from a remote server if it does not exist locally.

    Args:
        path (str): The local directory path.
        field_name (str): Name of the field (e.g., 'gasvx', 'gasvy').
        snapshot (int): Snapshot number.
        nx, ny, nz (int): Dimensions of the data.

    Returns:
        np.ndarray: The field data.
    """
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    
    if not os.path.exists(file):
        print(f"File not found locally: {file}")
        # Attempt to fetch from remote backup
        remote_user_host = "mlehmann@slurm-ui03.twgrid.org"
        remote_backup_base = "/ceph/sharedfs/users/m/mlehmann/FARGO3D/outputs/backup"
        remote_subdir = os.path.basename(path)
        remote_file = os.path.join(remote_backup_base, remote_subdir, f"{field_name}{snapshot}.dat")

        try:
            print(f"Checking remote file: {remote_file}")
            result = subprocess.run(
                ["ssh", remote_user_host, f"ls \"{remote_file}\""],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"Remote file found: {remote_file}. Transferring...")
                os.makedirs(path, exist_ok=True)
                scp_command = f"scp {remote_user_host}:\"{remote_file}\" \"{file}\""
                print(f"Executing SCP: {scp_command}")
                subprocess.run(scp_command, shell=True, check=True)
                print(f"Transferred {field_name}{snapshot}.dat to {path}")
            else:
                print(f"Remote file {field_name}{snapshot}.dat not found: {result.stderr.strip()}")
                return np.zeros((ny, nx, nz))
        except subprocess.CalledProcessError as e:
            print(f"Error during remote file transfer: {e}")
            return np.zeros((ny, nx, nz))
        except Exception as e:
            print(f"Unexpected error during transfer: {e}")
            return np.zeros((ny, nx, nz))

    print(f"Reading file: {file}")
    return np.fromfile(file, dtype=np.float64).reshape((nz, nx, ny)).transpose(2, 1, 0)


###########################################################################################################################

# Function to read the initial snapshot (timestep 0)
def read_initial_snapshot(path, read_gasvx=False, read_gasvy=False, read_gasvz=False,
                          read_gasenergy=False, read_gasdens=False, read_dust1vx=False,
                          read_dust1vy=False, read_dust1vz=False, read_dust1dens=False, IDEFIX=False):
    """
    Reads the initial snapshot (timestep 0) for either FARGO3D or IDEFIX.
    This is a convenient wrapper around read_single_snapshot.
    """
    return read_single_snapshot(
        path, 0,
        read_gasvx=read_gasvx, read_gasvy=read_gasvy, read_gasvz=read_gasvz,
        read_gasenergy=read_gasenergy, read_gasdens=read_gasdens,
        read_dust1vx=read_dust1vx, read_dust1vy=read_dust1vy,
        read_dust1vz=read_dust1vz, read_dust1dens=read_dust1dens,
        IDEFIX=IDEFIX
    )[0] # Returns only the data_arrays dictionary




###########################################################################################################################

def read_idefix_vtk(path, gas=True, dust=False, noniso=False, itstart=0, itend=None, nsteps=1, test=False):
    import os
    import glob
    import numpy as np
    from vtk import vtkStructuredGridReader
    from vtk.util.numpy_support import vtk_to_numpy
    from data_reader import read_parameters, reconstruct_grid

    print("READING IDEFIX VTK FIELDS")

    parameters = read_parameters(os.path.join(path, "idefix.0.log"), IDEFIX=True)
    gamma = parameters.get("gamma", 1.6667)

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=True)
    print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

    vtk_files = sorted(glob.glob(os.path.join(path, "data.*.vtk")))
    if itend is None:
        itend = len(vtk_files)

    vtk_files = vtk_files[itstart:itend:nsteps]
    nt = len(vtk_files)

    interval = float(parameters.get("vtk", 50.265))
    time_array = np.arange(0, nt * interval, interval) / (2 * np.pi)

    reader = vtkStructuredGridReader()
    reader.SetFileName(vtk_files[0])
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    cell_data = reader.GetOutput().GetCellData()
    n_cells = reader.GetOutput().GetNumberOfCells()
    available_fields = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
    print(f"Available fields in VTK files: {available_fields}")

    field_map = {
        'RHO': 'gasdens',
        'VX1': 'gasvy',
        'VX2': 'gasvx',
        'VX3': 'gasvz',
        'PRS': 'gasenergy',
    }

    selected_fields = [f for f in field_map if f in available_fields]
    data_types = [field_map[f] for f in selected_fields]
    print(f"Selected data types: {data_types}")

    total_grid_cells = nx * ny * nz
    if n_cells != total_grid_cells:
        raise ValueError(f"Mismatch between VTK cell count ({n_cells}) and grid size ({total_grid_cells})")

    # ✅ Output shape: [ny, nx, nz, nt]
    data_arrays = {key: np.zeros((ny, nx, nz, nt)) for key in data_types}

    for t_index, vtk_file in enumerate(vtk_files):
        reader.SetFileName(vtk_file)
        reader.Update()
        cell_data = reader.GetOutput().GetCellData()

        for vtk_name in selected_fields:
            key = field_map[vtk_name]  # ✅ Define key first
            raw_array = vtk_to_numpy(cell_data.GetArray(vtk_name))
            reshaped = raw_array.reshape((nz, ny, nx)).transpose(1, 2, 0)  # → [ny, nx, nz]
            #print(f"{key} shape after transpose: {reshaped.shape}")

            if vtk_name == 'PRS':
                reshaped = reshaped / (gamma - 1.0)

            data_arrays[key][..., t_index] = reshaped

    print(f"Finished reading {nt} VTK snapshots.")
    return data_arrays, data_types, xgrid, ygrid, zgrid, time_array

########################################################################################################

def read_idefix_vtk_parallel_filetype(path, gas=True, dust=False, noniso=False,
                                      itstart=0, itend=None, nsteps=1, test=False, backend='loky'):
    import os
    import glob
    import numpy as np
    import time
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from vtk import vtkStructuredGridReader
    from vtk.util.numpy_support import vtk_to_numpy
    from data_reader import read_parameters, reconstruct_grid

    print("READING IDEFIX VTK FIELDS (parallel by file type)")

    parameters = read_parameters(os.path.join(path, "idefix.0.log"), IDEFIX=True)
    gamma = parameters.get("gamma", 1.6667)

    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=True)
    print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

    vtk_files = sorted(glob.glob(os.path.join(path, "data.*.vtk")))
    if itend is None:
        itend = len(vtk_files)
    vtk_files = vtk_files[itstart:itend:nsteps]
    nt = len(vtk_files)

    interval = float(parameters.get("vtk", 50.265))
    time_array = np.arange(0, nt * interval, interval) / (2 * np.pi)

    isothermal = False
    def_file = os.path.join(path, "definitions.hpp")
    if os.path.exists(def_file):
        with open(def_file, 'r') as f:
            for line in f:
                # Remove inline comments
                line_clean = re.sub(r'//.*', '', line).strip()
                # Match only active #define ISOTHERMAL directives
                if re.fullmatch(r'#\s*define\s+ISOTHERMAL', line_clean):
                    isothermal = True
                    print("✅ ISOTHERMAL mode detected via definitions.hpp")
                    break

    # Check available fields in the first snapshot
    reader = vtkStructuredGridReader()
    reader.SetFileName(vtk_files[0])
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    cell_data = reader.GetOutput().GetCellData()
    n_cells = reader.GetOutput().GetNumberOfCells()
    available_fields = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
    print(f"Available fields in VTK files: {available_fields}")

    field_map = {
        'RHO': 'gasdens',
        'VX1': 'gasvy',
        'VX2': 'gasvx',
        'VX3': 'gasvz',
        'PRS': 'gasenergy',
    }

    selected_fields = [f for f in field_map if f in available_fields]

    if isothermal and 'PRS' not in selected_fields:
        selected_fields.append('PRS')
        print("🔧 Reconstructing 'PRS' from density and cs² profile (isothermal run)")
    data_types = [field_map[f] for f in selected_fields]
    print(f"Selected data types: {data_types}")

    total_grid_cells = nx * ny * nz
    if n_cells != total_grid_cells:
        raise ValueError(f"Mismatch between VTK cell count ({n_cells}) and grid size ({total_grid_cells})")

    # Precompute cs2(r) if needed
    if isothermal:
        h0 = float(parameters.get("h0", 0.05))
        flaringIndex = float(parameters.get("flaringIndex", 0.0))
        rvals = xgrid  # 1D radial array
        cs2_profile = h0 ** 2 * rvals ** (2 * flaringIndex - 1)

    # Preallocate arrays for all selected fields
    data_arrays = {field_map[f]: np.empty((ny, nx, nz, nt), dtype=np.float64) for f in selected_fields}

    def read_single_vtk_snapshot(t_index, vtk_file):
        reader = vtkStructuredGridReader()
        reader.SetFileName(vtk_file)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        cell_data = reader.GetOutput().GetCellData()

        snapshot = {}
        for vtk_name in selected_fields:
            key = field_map[vtk_name]

            if vtk_name == 'PRS':
                if isothermal:
                    rho_array = cell_data.GetArray("RHO")
                    if rho_array is None:
                        raise ValueError(f"Missing 'RHO' array needed to reconstruct 'PRS' at t_index={t_index}")
                    rho_array = vtk_to_numpy(rho_array)
                    rho_reshaped = rho_array.reshape((nz, ny, nx)).transpose(1, 2, 0)
                    reshaped = rho_reshaped * cs2_profile[None, :, None] / (gamma - 1.0)
                    snapshot[key] = reshaped
                    continue
                else:
                    prs_array = cell_data.GetArray("PRS")
                    if prs_array is None:
                        raise ValueError(f"'PRS' not found in non-isothermal run at t_index={t_index}")
                    raw_array = vtk_to_numpy(prs_array)
                    reshaped = raw_array.reshape((nz, ny, nx)).transpose(1, 2, 0)
                    reshaped = reshaped / (gamma - 1.0)
                    snapshot[key] = reshaped
                    continue

            # All other fields
            raw = cell_data.GetArray(vtk_name)
            if raw is None:
                raise ValueError(f"Field '{vtk_name}' missing in VTK file at t_index={t_index}")
            raw_array = vtk_to_numpy(raw)
            reshaped = raw_array.reshape((nz, ny, nx)).transpose(1, 2, 0)
            snapshot[key] = reshaped

        return t_index, snapshot
    nproc=4
    print(f"Running File reading with {nproc} Threads!!")
    results = Parallel(n_jobs=nproc, backend=backend)(
        delayed(read_single_vtk_snapshot)(i, vtk_file)
        for i, vtk_file in enumerate(tqdm(vtk_files, desc="Reading VTK snapshots"))
    )

    # Insert time slices into the preallocated arrays
    for t_index, snapshot in results:
        for key, slice_3d in snapshot.items():
            data_arrays[key][..., t_index] = slice_3d
    return data_arrays, data_types, xgrid, ygrid, zgrid, time_array

#####################################################################################################################

def read_single_snapshot_idefix(path, snapshot_idx,
                                read_gasvx=False, read_gasvy=False, read_gasvz=False,
                                read_gasenergy=False, read_gasdens=False):
    """
    Reads a single VTK snapshot from an IDEFIX simulation.

    Args:
        path (str): Path to the simulation output folder.
        snapshot_idx (int): Index of the snapshot to read.
        read_gasvx, read_gasvy, read_gasvz, read_gasenergy, read_gasdens (bool): Flags for each field.

    Returns:
        tuple: (data_arrays, xgrid, ygrid, zgrid, parameters)
    """
    import os
    import glob
    from vtk import vtkStructuredGridReader
    from vtk.util.numpy_support import vtk_to_numpy
    from data_reader import read_parameters, reconstruct_grid

    parameters = read_parameters(os.path.join(path, "idefix.0.log"), IDEFIX=True)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=True)

    # Find VTK file
    vtk_files = sorted(glob.glob(os.path.join(path, "data.*.vtk")))
    if snapshot_idx >= len(vtk_files):
        raise IndexError(f"Snapshot index {snapshot_idx} exceeds available snapshots ({len(vtk_files)})")

    vtk_file = vtk_files[snapshot_idx]

    reader = vtkStructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    cell_data = reader.GetOutput().GetCellData()
    available_fields = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]

    # Mapping from VTK to internal field names
    field_map = {
        'RHO': ('gasdens', read_gasdens),
        'PRS': ('gasenergy', read_gasenergy),
        'VX1': ('gasvy', read_gasvy),
        'VX2': ('gasvx', read_gasvx),
        'VX3': ('gasvz', read_gasvz),
    }

    gamma = float(parameters.get("gamma", 1.6667))
    data_arrays = {}

    for vtk_name, (internal_name, read_flag) in field_map.items():
        if not read_flag:
            continue
        if vtk_name in available_fields:
            raw_array = vtk_to_numpy(cell_data.GetArray(vtk_name))
            reshaped = raw_array.reshape((nz, ny, nx)).transpose(1, 2, 0)  # [ny, nx, nz]
            if vtk_name == 'PRS':
                reshaped = reshaped / (gamma - 1.0)
            data_arrays[internal_name] = reshaped

    return data_arrays, xgrid, ygrid, zgrid, parameters



