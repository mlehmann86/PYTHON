# FILE: torque_evolution.py
import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

# Import the definitive, corrected torque calculation function
from torque_profile import compute_torque_gravity

# Import helper functions from your project
from data_reader import read_single_snapshot, read_parameters, reconstruct_grid, determine_nt
from data_storage import determine_base_path, scp_transfer
from planet_data import read_alternative_torque, extract_planet_mass_and_migration


def compute_torque_for_snapshot(args):
    """
    Worker function that calls the torque calculation.
    """
    i, base_path, xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index, imin, use_hill_cut = args
    
    data = read_single_snapshot(base_path, i, read_gasdens=True)[0]
    
    # Call the corrected function with the radially-dependent smoothing and grid limits
    grav_profile = compute_torque_gravity(data["gasdens"], xgrid, ygrid, zgrid, 
                                          1.0, 0.0, qp, h0, thick_smooth, flaring_index,
                                          use_hill_cut=use_hill_cut)
    
    # We still only sum over the specified range (imin to end)
    dr = xgrid[1] - xgrid[0]
    imax = xgrid.shape[0] # The calculation in torque_profile respects imin/imax, but summation is simpler
    torque_on_disk = np.sum(grav_profile[imin:imax]) * dr
    
    # Return the torque ON THE PLANET (Action-Reaction)
    return i, -torque_on_disk


def compute_total_gravitational_torque(simname, imin=0, use_hill_cut=False, parallel=True):
    """
    This function sets up and runs the parallel torque calculation over the specified domain.
    """
    base_path = determine_base_path(simname)
    summary_file = os.path.join(base_path, "summary0.dat")
    params = read_parameters(summary_file)

    h0 = float(params.get("ASPECTRATIO"))
    qp, _ = extract_planet_mass_and_migration(summary_file)
    thick_smooth = float(params.get("THICKNESSSMOOTHING"))
    flaring_index = float(params.get("FLARINGINDEX"))
    
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(params)[0:6]
    nt = determine_nt(base_path)
    time_array = np.arange(nt) * (float(params['NINTERM']) / 20.0)

    print(f"Calculating torque over radial index range: [{imin}, {nx-1}]")
    if use_hill_cut:
        print("HILL CUT is ENABLED.")

    grav_torque_planet = np.zeros(nt)
    args_list = [(i, base_path, xgrid, ygrid, zgrid, qp, h0, thick_smooth, flaring_index, imin, use_hill_cut) for i in range(nt)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_torque_for_snapshot, args_list), total=nt, desc="Calculating Torque"))
        for i, torque_value in results:
            grav_torque_planet[i] = torque_value

    return grav_torque_planet, time_array, params, qp, h0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare computed torque with internal tqwk0.dat file.")
    parser.add_argument("simname", type=str, help="Simulation subdirectory")
    parser.add_argument("--imin", type=int, default=0, help="Minimum radial index to start torque calculation from.")
    parser.add_argument("--hillcut", action="store_true", help="Enable the Hill Cut in the torque calculation.")
    args = parser.parse_args()

    # 1. Perform the snapshot-based calculation
    grav_torque_planet, time_array, params, qp, h0 = compute_total_gravitational_torque(args.simname, imin=args.imin, use_hill_cut=args.hillcut)

    # 2. Define Normalization Factors
    SIGMA0 = float(params.get("SIGMA0"))
    Gamma0 = (qp / h0)**2 * SIGMA0
    
    # --- FIX: Define and use two separate, correct normalization factors ---
    norm_factor_grav = 1.0 / Gamma0 if Gamma0 > 0 else 0.0
    norm_factor_tqwk = qp / Gamma0 if Gamma0 > 0 else 0.0

    # 3. Load and Normalize Internal Torque from tqwk0.dat
    try:
        base_path = determine_base_path(args.simname)
        tqwk_file = os.path.join(base_path, "tqwk0.dat")
        date_torque, torque_on_planet_raw, _ = read_alternative_torque(tqwk_file)
        internal_time = date_torque / (2.0 * np.pi)
        
        # Apply the correct normalization for tqwk data
        internal_torque_planet_norm = torque_on_planet_raw * norm_factor_tqwk
    except Exception as e:
        print(f"Warning: Could not process tqwk0.dat: {e}")
        internal_time = internal_torque_planet_norm = None
        
    # 4. Normalize the snapshot-based torque
    grav_torque_planet_norm = grav_torque_planet * norm_factor_grav

    # 5. Plotting
    plt.figure(figsize=(12, 7))
    plt.axhline(0, color='k', lw=0.5, zorder=0)

    if internal_time is not None:
        dt_internal = np.mean(np.diff(internal_time))
        smoothing_window = max(1, int(round(0.1 / dt_internal)))
        internal_torque_smoothed = uniform_filter1d(internal_torque_planet_norm, size=smoothing_window, mode='nearest')
        plt.plot(internal_time, internal_torque_smoothed, label="Internal Torque (tqwk0, smoothed)", color='darkcyan', lw=2.5, zorder=5)
    
    plt.plot(time_array, grav_torque_planet_norm, label="Gravitational Torque (Snapshots)", color='orangered', alpha=0.9, zorder=10)

    plt.xlabel("Time (orbits)"); plt.ylabel(r"$\Gamma_{\mathrm{planet}} / \Gamma_0$")
    plt.title(f"Torque on Planet Comparison for {args.simname}")
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.tight_layout()

    outname = f"torque_evolution_comparison_{args.simname}.pdf"
    plt.savefig(outname)
    print(f"\nSaved comparison plot to {outname}")

    scp_transfer(outname, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

if __name__ == "__main__":
    main()
