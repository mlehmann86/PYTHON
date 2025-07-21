# data_storage.py

import os
import numpy as np
from tqdm import tqdm
import subprocess

###########################################################################################################################

def get_remote_ip(file_path):
    with open(file_path, 'r') as file:
        ip_address = file.read().strip()
    return ip_address

###########################################################################################################################

def determine_base_path(subdirectory, IDEFIX=False):
    """
    Determines the full path to a simulation output directory based on a list of base paths.

    Parameters
    ----------
    subdirectory : str
        The name of the simulation subdirectory (usually the actual simulation folder name).
    IDEFIX : bool
        If True, search IDEFIX-specific base paths (and auto-detect setup). If False, use FARGO3D paths.

    Returns
    -------
    subdir_path : str
        The full resolved path to the simulation output directory.
    """
    if IDEFIX:
        base_paths = [
            "/theory/lts/mlehmann/idefix-mkl/outputs",
            "/tiara/home/mlehmann/data/idefix-mkl/outputs"
        ]
    else:
        base_paths = [
            "/tiara/home/mlehmann/data/FARGO3D/outputs",
            "/theory/lts/mlehmann/FARGO3D/outputs"
        ]

    print(f"[INFO] Searching for subdirectory: '{subdirectory}' (IDEFIX: {IDEFIX})")

    matches = []
    for base in base_paths:
        if IDEFIX:
            # For IDEFIX, loop over all setups
            try:
                for setup in os.listdir(base):
                    potential_path = os.path.join(base, setup, subdirectory)
                    if os.path.isdir(potential_path):
                        matches.append((setup, potential_path, base))
            except FileNotFoundError:
                continue
        else:
            potential_path = os.path.join(base, subdirectory)
            if os.path.isdir(potential_path):
                matches.append((None, potential_path, base))

    if not matches:
        raise FileNotFoundError(f"❌ Subdirectory '{subdirectory}' not found in any known base path.")
    elif len(matches) == 1:
        setup, subdir_path, base_path = matches[0]
        if IDEFIX:
            print(f"✅ Found IDEFIX simulation: setup = '{setup}', path = {subdir_path}")
        else:
            print(f"✅ Found FARGO3D simulation: path = {subdir_path}")
        return subdir_path
    else:
        print(f"🔍 Multiple matches found for subdirectory '{subdirectory}':")
        for i, (setup, path, _) in enumerate(matches):
            print(f"  [{i}] setup = '{setup}', path = {path}")
        idx = input("Enter index of setup to use: ")
        try:
            idx = int(idx)
            _, subdir_path, _ = matches[idx]
            return subdir_path
        except (ValueError, IndexError):
            raise RuntimeError("❌ Invalid selection.")


###########################################################################################################################


import subprocess
import time
import os

def scp_transfer(remote_filepath, local_directory, username, max_retries=2, timeout=30):
    """
    Transfer a file from the remote server to the local machine using scp, with retry logic.

    Parameters:
    - remote_filepath: Path to the file on the remote server.
    - local_directory: Directory on the local machine where the file will be saved.
    - username: Username for the local machine.
    - max_retries: Maximum number of retries if the transfer fails (default: 5).
    - timeout: Maximum time (seconds) before forcibly killing a stuck transfer (default: 60).
    """

    # Define the path to the external IP address file
    ip_file_path = "/theory/lts/mlehmann/PYTHON/external_ip.txt"

    # Retrieve the IP address from the file
    try:
        with open(ip_file_path, 'r') as file:
            ip_address = file.read().strip()
    except Exception as e:
        print(f"❌ [ERROR] Failed to read IP address from {ip_file_path}. Error: {e}")
        return

    # Skip SCP transfer if the IP address is a local network IP (e.g., 192.168.x.x)
    if ip_address.startswith("192.168"):
        print(f"[INFO] Local IP address detected ({ip_address}). Skipping SCP transfer.")
        return

    # Extract the filename from the remote filepath
    filename = os.path.basename(remote_filepath)

    # Construct the SCP command
    scp_command = f"scp -o ConnectTimeout=30 -o TCPKeepAlive=yes -o ServerAliveInterval=30 {remote_filepath} {username}@{ip_address}:{local_directory}"
    
    print(f"[INFO] Executing SCP command: {scp_command}")

    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Attempt {attempt}/{max_retries} to transfer {filename}...")

            # Start SCP process
            process = subprocess.Popen(scp_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait for completion or timeout
            start_time = time.time()
            while process.poll() is None:
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"⚠️ [WARNING] SCP transfer timeout ({timeout} sec). Killing process...")
                    process.kill()
                    process.wait()
                    break
                time.sleep(1)  # Avoid excessive CPU usage

            # Check the result
            return_code = process.poll()
            if return_code == 0:
                print(f"✅ [SUCCESS] File transferred successfully to {local_directory}/{filename}")
                return  # Exit function after successful transfer
            else:
                print(f"❌ [ERROR] SCP failed with exit code {return_code}. Retrying...")

        except Exception as e:
            print(f"❌ [ERROR] Unexpected error: {e}. Retrying...")

        time.sleep(5)  # Wait before retrying

    print(f"🚨 [CRITICAL] SCP transfer failed after {max_retries} attempts.")
###########################################################################################################################

def save_simulation_quantities(output_path, time_array, alpha_r, alpha_r_HS, p_HS, q_HS, rms_vr, rms_vphi, rms_vz, max_vz, min_vz, max_epsilon, H_d_array, roche_times, m_gas, m_dust, avg_metallicity, vort_min, rms_vr_profile, rms_vphi_profile, rms_vz_profile, vortr_avg, vortz_avg):
    """
    Save the computed quantities (alpha_r, RMS of vertical dust velocity, max epsilon, scale height H_d, and Roche exceed times, m_gas, m_dust) to a file.

    Parameters:
    - output_path: Directory where the simulation was run, included in the file name.
    - time_array: Array of time steps.
    - alpha_r: Array of turbulent alpha values over time.
    - rms_dust1vz: Array of RMS of vertical dust velocity over time.
    - max_epsilon: Array of maximum epsilon values over time.
    - H_d_array: Array of dust scale height values over time.
    - roche_times: Array of times where the Roche density was exceeded.
    """

    # Create the file name based on output_path
    file_name = os.path.join(output_path, f"{os.path.basename(output_path)}_quantities.npz")

    # Save the data including the Roche exceed times
    np.savez(file_name, time=time_array, alpha_r=alpha_r, alpha_r_HS=alpha_r_HS, p_HS=p_HS, q_HS=q_HS, rms_vr=rms_vr, rms_vphi=rms_vphi, rms_vz=rms_vz, max_vz=max_vz, min_vz=min_vz, max_epsilon=max_epsilon, H_d=H_d_array, roche_times=roche_times, m_gas=m_gas, m_dust=m_dust, Z_avg=avg_metallicity, vort_min=vort_min, rms_vr_profile=rms_vr_profile, rms_vphi_profile=rms_vphi_profile, rms_vz_profile=rms_vz_profile, vortr_avg=vortr_avg, vortz_avg=vortz_avg)
    print(f"#######################################")
    print(f"Simulation quantities saved to {file_name}")
    print(f"#######################################")





