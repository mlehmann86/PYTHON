import numpy as np
import os

# Import your utility functions (adjust the path as necessary)
from data_reader import read_parameters, reconstruct_grid

def inject_velocity_noise(sim_path, strength=0.1, verbose=True):
    logfile = os.path.join(sim_path, "inject_cos_noise.debug.log")
    sys.stdout = open(logfile, "a")
    sys.stderr = sys.stdout
    print(f"\n--- New injection run ---")
    print(f"[inject_velocity_noise] sim_path = {sim_path}")
    print(f"[inject_velocity_noise] strength = {strength}")
    """
    Injects white noise (strength * c_s(r)) into the initial velocity fields of a Fargo3D simulation.
    Requires data_reader.py utilities for param and grid reading.
    """

    print(f"[inject_velocity_noise] sim_path = {sim_path}")
    print(f"[inject_velocity_noise] strength = {strength}")

    # === Read parameters and grid ===
    summaryfile = os.path.join(sim_path, "summary0.dat")
    if not os.path.isfile(summaryfile):
        raise FileNotFoundError(f"Parameter file not found: {summaryfile}")
    parameters = read_parameters(summaryfile)
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters)

    shape_disk = (nz, nx, ny)  # FARGO3D binary order

    # === File paths ===
    vxfile = os.path.join(sim_path, "gasvx0.dat")
    vyfile = os.path.join(sim_path, "gasvy0.dat")
    vzfile = os.path.join(sim_path, "gasvz0.dat")
    densfile = os.path.join(sim_path, "gasdens0.dat")
    enefile = os.path.join(sim_path, "gasenergy0.dat")
    for f in [vxfile, vyfile, vzfile, densfile, enefile]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"{f} not found!")

    # === Read arrays and transpose to (ny, nx, nz) ===
    dens = np.fromfile(densfile, dtype=np.float64).reshape(shape_disk).transpose(2, 1, 0)
    ene  = np.fromfile(enefile, dtype=np.float64).reshape(shape_disk).transpose(2, 1, 0)
    vx   = np.fromfile(vxfile, dtype=np.float64).reshape(shape_disk).transpose(2, 1, 0)
    vy   = np.fromfile(vyfile, dtype=np.float64).reshape(shape_disk).transpose(2, 1, 0)
    vz   = np.fromfile(vzfile, dtype=np.float64).reshape(shape_disk).transpose(2, 1, 0)

    # === Compute c_s(r) profile ===
    cs2 = ene / dens
    cs2_profile = np.mean(cs2, axis=(0, 2))  # (nx,)
    cs_profile = np.sqrt(cs2_profile)

    if verbose:
        print(f"Sound speed profile shape: {cs_profile.shape} (nx={nx})")
        print(f"Example c_s values: min={cs_profile.min():.3e}, max={cs_profile.max():.3e}")

    # === Inject noise ===
    rng = np.random.default_rng()
    for vfield, fname in zip([vx, vy, vz], ["gasvx0.dat", "gasvy0.dat", "gasvz0.dat"]):
        noise = (2.0 * rng.random(size=vfield.shape) - 1.0)  # shape (ny, nx, nz)
        for ix in range(nx):
            noise[:, ix, :] *= strength * cs_profile[ix]
        vfield += noise
        # Write back as (nz, nx, ny)
        outarr = vfield.transpose(2, 1, 0)
        outpath = os.path.join(sim_path, fname)
        outarr.astype(np.float64).tofile(outpath)
        if verbose:
            print(f"Injected noise and overwrote {fname}")

    print("All velocity fields updated with random noise.")

if __name__ == "__main__":
    import sys
    print(f"[DEBUG] sys.argv = {sys.argv}")
    if len(sys.argv) < 2:
        print("Usage: python3 inject_cos_noise.py <sim_path>")
        sys.exit(1)
    sim_path = sys.argv[1]
    inject_velocity_noise(sim_path, strength=0.005)
