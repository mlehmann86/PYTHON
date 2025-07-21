import os
import numpy as np
from tqdm import tqdm
import subprocess


###########################################################################################################################

def save_simulation_quantities(output_path, time_array, alpha_r, alpha_z, rms_vr, rms_vphi, rms_vz, max_vz, min_vz, max_epsilon, H_d_array, roche_times, m_gas, m_dust, avg_metallicity, vort_min, xgrid_masked, rms_vr_profile, rms_vphi_profile, rms_vz_profile, vortr_avg, vortz_avg):
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
    np.savez(file_name, time=time_array, alpha_r=alpha_r, alpha_z=alpha_z, rms_vr=rms_vr, rms_vphi=rms_vphi, rms_vz=rms_vz, max_vz=max_vz, min_vz=min_vz, max_epsilon=max_epsilon, H_d=H_d_array, roche_times=roche_times, m_gas=m_gas, m_dust=m_dust, Z_avg=avg_metallicity, vort_min=vort_min, xgrid_masked=xgrid_masked, rms_vr_profile=rms_vr_profile, rms_vphi_profile=rms_vphi_profile, rms_vz_profile=rms_vz_profile, vortr_avg=vortr_avg, vortz_avg=vortz_avg)
    print(f"#######################################")
    print(f"Simulation quantities saved to {file_name}")
    print(f"#######################################")

