import os
import sys
import numpy as np  # Used globally in math, array ops

# Data handling modules (used globally)
from data_storage import determine_base_path, scp_transfer
from data_reader import (
    read_parameters, determine_nt,
    read_single_snapshot, read_single_snapshot_idefix,
    check_planet_presence
)

# These should be imported *inside* functions that use them:
import matplotlib.pyplot as plt        # Used only in plotting functions
import matplotlib.cm as cm            # Not used directly – remove if unused
import matplotlib.colors as mcolors   # Not used directly – remove if unused
from scipy.ndimage import uniform_filter1d  # Used for torque smoothing → move inside plotting/torque fn
from scipy.interpolate import griddata      # Used for polar-to-Cartesian interpolation → move inside plotting fn


def uses_thermal_diffusion(summary_file, IDEFIX=False):
    """
    Returns True if the simulation used thermal diffusion.
    - For FARGO: checks for "-DTHERMALDIFFUSION" in summary file.
    - For IDEFIX: checks for "Thermal Diffusion: ENABLED" in log file.
    """
    try:
        with open(summary_file, "r") as f:
            contents = f.read()
            if not IDEFIX:
                return "-DTHERMALDIFFUSION" in contents
            else:
                return "Thermal Diffusion: ENABLED" in contents
    except Exception as e:
        print(f"Could not check for thermal diffusion: {e}")
        return False



def compute_theoretical_torques_PK11(parameters, qp, simulation_name, IDEFIX=False, summary_file=None, avg_start_orbit=None, avg_end_orbit=None, nu_threshold=1e-9):
    """
    Computes the total PK11 (Paardekooper+2011) model torque and Lindblad torque,
    plus the normalization factor GAM0.

    If the laminar viscosity is below a threshold, it attempts to read a time-averaged
    turbulent alpha from a corresponding _quantities.npz file and use that to calculate
    a turbulent viscosity instead.

    Returns
    -------
    total_torque_PK11 : float
        Total PK11 model torque (corotation + Lindblad).
    lindblad_torque_PK11 : float
        Lindblad torque (PK11, non-isothermal).
    GAM0 : float
        Normalization factor.
    gamma_eff : float
        The effective gamma value considering thermal effects.
    nu_used : float
        The viscosity value (laminar or turbulent) used for the calculation.
    nu_type : str
        A string indicating the type of viscosity used ('Laminar' or 'Turbulent').
    """
    import numpy as np
    import os
    from corotation_torque_pnu import compute_theoretical_corotation_paarde2011, gamma_eff_theory, gamma_eff_diffusion
    from data_storage import determine_base_path

    # --- Parameter parsing (as in your conventions) ---
    if IDEFIX:
        gam = parameters.get("gamma", 1.4)
        FLARINGINDEX = parameters.get("flaringindex", 0.5)
        ASPECTRATIO = parameters.get("h0", 0.05)
        SIGMASLOPE = parameters.get("sigmaslope", 1.0)
        SIGMA0 = parameters.get("sigma0", 1.0)
        THICKNESSSMOOTHING = parameters.get("smoothing", 0.4) / ASPECTRATIO
        beta = parameters.get("beta", 1.0)
        chi = parameters.get("kappa0", 1.0e-5)
        rhog0 = parameters.get("rhog0", 1.0)
        nu_laminar = float(parameters.get('nu', 1e-5))
        if "x3-grid" in parameters:
            grid_entry = parameters["x3-grid"]
            zmin = float(grid_entry[1])
            zmax = float(grid_entry[4])
        else:
            zmin = 0.0
            zmax = 1.0
    else:
        gam = float(parameters['GAMMA'])
        FLARINGINDEX = float(parameters['FLARINGINDEX'])
        ASPECTRATIO = float(parameters['ASPECTRATIO'])
        THICKNESSSMOOTHING = float(parameters['THICKNESSSMOOTHING'])
        SIGMASLOPE = float(parameters['SIGMASLOPE'])
        SIGMA0 = float(parameters['SIGMA0'])
        zmin = float(parameters['ZMIN'])
        zmax = float(parameters['ZMAX'])
        beta = float(parameters['BETA'])
        chi = float(parameters.get('CHI', 1e-5))
        nu_laminar = float(parameters.get('NU', 1e-5))

    # --- Viscosity determination ---
    h = ASPECTRATIO
    nu_used = nu_laminar
    nu_type = 'Laminar'  # Default type
    print("\n--- Theoretical Torque Viscosity ---")
    if nu_laminar < nu_threshold and avg_start_orbit is not None and avg_end_orbit is not None:
        print(f"Laminar viscosity {nu_laminar:.2e} is below threshold {nu_threshold:.2e}. Checking for turbulent alpha.")
        base_path = determine_base_path(simulation_name, IDEFIX)
        quantities_file = os.path.join(base_path, f"{simulation_name}_quantities.npz")
        
        try:
            if os.path.exists(quantities_file):
                data = np.load(quantities_file)
                if 'alpha_r' in data and 'time' in data:
                    alpha_r = data['alpha_r']
                    alpha_time = data['time']
                    
                    alpha_avg_mask = (alpha_time >= avg_start_orbit) & (alpha_time <= avg_end_orbit)
                    
                    if np.sum(alpha_avg_mask) > 0:
                        mean_alpha = np.mean(alpha_r[alpha_avg_mask])
                        nu_turbulent = mean_alpha * h**2
                        print(f"--> Found turbulent alpha data in {os.path.basename(quantities_file)}.")
                        print(f"    Averaged alpha over [{avg_start_orbit:.1f} - {avg_end_orbit:.1f} orbits] = {mean_alpha:.2e}")
                        print(f"--> \033[1mOverwriting laminar Nu. Using turbulent Nu = alpha * h^2 = {nu_turbulent:.2e}\033[0m")
                        nu_used = nu_turbulent
                        nu_type = 'Turbulent'
                    else:
                        print(f"--> Could not find overlapping time window for turbulent alpha in {os.path.basename(quantities_file)}. Using laminar Nu.")
                else:
                    print(f"--> 'alpha_r' or 'time' not found in {os.path.basename(quantities_file)}. Using laminar Nu.")
            else:
                print(f"--> Quantities file not found: {os.path.basename(quantities_file)}. Using laminar Nu.")
        except Exception as e:
            print(f"--> An error occurred while reading turbulent alpha: {e}. Using laminar Nu.")
    else:
        print(f"\033[1mUsing laminar viscosity Nu = {nu_laminar:.2e}\033[0m")
    print("-------------------------------------\n")


    # Debug print
    print(f"GAMMA: {gam}")
    print(f"FLARINGINDEX: {FLARINGINDEX}")
    print(f"ASPECTRATIO: {h}")
    print(f"THICKNESSSMOOTHING: {THICKNESSSMOOTHING}")
    print(f"SIGMASLOPE: {SIGMASLOPE}")
    print(f"ZMIN: {zmin}")
    print(f"ZMAX: {zmax}")
    print(f"qp: {qp}")
    print(f"beta (cooling): {beta}")
    print(f"Nu (viscosity used for theory): {nu_used}")


    # --- 3D/2D normalization ---
    bh = THICKNESSSMOOTHING
    fi = FLARINGINDEX

    H0         = h
    omega_kep0 = 1.0
    cs0        = omega_kep0 * H0
    rhog0 = SIGMA0 / np.sqrt(2 * np.pi) / H0

    if zmin == 0:
        SIG0 = SIGMA0
        ss = SIGMASLOPE
        print(f"ORIGINAL SIGMA0: {SIG0}")
    else:
        SIG0 = rhog0 * (zmax - zmin)
        ss = SIGMASLOPE + FLARINGINDEX +1 # In a 3D unstratified simulation the theoretical torque fomula should use p instead of ss 
        print(f"ACTUAL SIGMA0: {SIG0}")

    GAM0 = (qp / h) ** 2 * SIG0
    print(f"GAM0: {GAM0}")

    # --- PK11: physical setup ---
    rp = 1.0
    Omega_p = 1.0
    qindex = 1 - 2 * fi
    G_Ls = -2.5 - 1.7 * qindex + 0.1 * ss

    if summary_file is not None:
        uses_td = uses_thermal_diffusion(summary_file, IDEFIX)
    else:
        uses_td = False

    if uses_td:
        print("Thermal diffusion detected: using gamma_eff_diffusion")
        gamma_eff = gamma_eff_diffusion(gam, chi, h)
    else:
        print("Thermal diffusion NOT detected: using gamma_eff_theory (beta cooling)")
        gamma_eff = gamma_eff_theory(gam, beta, h)


    xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / bh) ** 0.25 * np.sqrt(qp / h)
    p_nu = (2.0 / 3.0) * np.sqrt((rp ** 2 * Omega_p * xs ** 3) / (2 * np.pi * nu_used))

    print(f"p_nu: {p_nu}")

    if uses_td:
        Gamma_C = compute_theoretical_corotation_paarde2011(
            p_nu, h=h, gamma=gam, q=qindex, s=ss, beta=None, qp=qp, chi=chi
        )
    else:
        Gamma_C = compute_theoretical_corotation_paarde2011(
            p_nu, h=h, gamma=gam, q=qindex, s=ss, beta=beta, qp=qp, chi=None
        )
    Gamma_L = G_Ls / gamma_eff
    total_torque_PK11 = Gamma_C + Gamma_L

    print(f"Gamma_L: {Gamma_L}")
    print(f"Gamma_TOT: {total_torque_PK11}")
    print(f"gamma_eff: {gamma_eff}")

    return total_torque_PK11, Gamma_L, GAM0, gamma_eff, nu_used, nu_type



def compute_theoretical_torques(parameters, qp, eq_label=None, IDEFIX=False):
    """
    Computes the theoretical adiabatic and isothermal torque values.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the relevant simulation parameters.
    qp : float
        Planet-to-star mass ratio (often M_p / M_*).
    eq_label : str, optional
        Selects which Paardekooper formula to use for the adiabatic torque.
    IDEFIX : bool, optional
        If True, extract parameters using IDEFIX-style keys (from idefix.0.log).
    """

    if IDEFIX:
        gam = parameters.get("gamma", 1.4)
        FLARINGINDEX = parameters.get("flaringIndex", 0.5)
        ASPECTRATIO = parameters.get("h0", 0.05)
        SIGMASLOPE = parameters.get("sigmaSlope", 1.0)
        SIGMA0 = parameters.get("sigma0", 1.0)
        THICKNESSSMOOTHING = parameters.get("smoothing", 0.6) / ASPECTRATIO
        betc = parameters.get("beta", 1.0)

        # Handle ZMIN and ZMAX depending on whether X3-grid exists
        if "X3-grid" in parameters:
            # IDEFIX X3-grid entry: [1, zmin, nz, 'u', zmax]
            grid_entry = parameters["X3-grid"]
            zmin = float(grid_entry[1])
            zmax = float(grid_entry[4])
        else:
            zmin = 0.0
            zmax = 1.0
    else:
        gam = float(parameters['GAMMA'])
        FLARINGINDEX = float(parameters['FLARINGINDEX'])
        ASPECTRATIO = float(parameters['ASPECTRATIO'])
        THICKNESSSMOOTHING = float(parameters['THICKNESSSMOOTHING'])
        SIGMASLOPE = float(parameters['SIGMASLOPE'])
        SIGMA0 = float(parameters['SIGMA0'])
        zmin = float(parameters['ZMIN'])
        zmax = float(parameters['ZMAX'])
        betc = float(parameters['BETA'])
 

    # Debug print
    print(f"GAMMA: {gam}")
    print(f"FLARINGINDEX: {FLARINGINDEX}")
    print(f"ASPECTRATIO: {ASPECTRATIO}")
    print(f"THICKNESSSMOOTHING: {THICKNESSSMOOTHING}")
    print(f"SIGMASLOPE: {SIGMASLOPE}")
    print(f"ZMIN: {zmin}")
    print(f"ZMAX: {zmax}")
    print(f"qp: {qp}")
    print(f"beta (cooling): {betc}")

    # [Continue with torque formula computations...]

    # Define local variables for convenience
    h = ASPECTRATIO
    fi = FLARINGINDEX
    q = 1 - 2 * fi
    bh = THICKNESSSMOOTHING
    ss = SIGMASLOPE

    # Sound speed, scale height, etc.
    H0         = h
    omega_kep0 = 1.0
    cs0        = omega_kep0 * H0
    rhog0      = SIGMA0 / np.sqrt(2 * np.pi) / H0

    # Convert SIGMA0 if zmin > 0 (i.e., 3D integration)
    if zmin == 0: #2D simulation
        SIG0 = SIGMA0  # default
        print(f"ORIGINAL SIGMA0: {SIG0}")
    else: #3D simulation
        SIG0 = rhog0 * (zmax - zmin)  # actual vertically integrated mass density
        print(f"ACTUAL SIGMA0: {SIG0}")

    # Normalization factor
    GAM0 = (qp / h) ** 2 * SIG0

    print(f"GAM0: {GAM0}")

    # Paardekooper definitions
    fac = 0.4 / bh
    bet = q

    if zmin == 0:     #2D simulation
        xi = q - (gam - 1) * ss
        alp = ss
    else:     #3D simulation
        p = ss + fi +1
        xi = q - (gam - 1) * p
        alp = p




    #----------------------
    # Common factor GAML
    #----------------------
    # Lindblad Torque
    GAML = -(2.5 + 1.7 * bet - 0.1 * alp) * fac**0.71

    # -------------------------------------------------------
    # "Adiabatic" torque formulas from Paardekooper et al. (2010)
    # -------------------------------------------------------
    # Variables:
    #   GAML  -> Lindblad torque  (see eq. (14))
    #   fac   -> (b/h)-dependent factor, etc.
    #   alp   -> alpha (surface density slope)
    #   bet   -> beta  (temperature slope)
    #   xi    -> beta - (gamma - 1) * alpha
    #   gam   -> gamma
    #
    # eq. (45):  Lindblad + NONLINEAR horseshoe drag (barotropic + entropy)
    eq45 = (
        GAML                                          # Lindblad (L)
        + 1.1 * fac * (1.5 - alp)                     # Nonlinear barotropic HSD
        + (xi / gam) * fac * (10.1 * fac**0.5 - 2.2)   # Nonlinear ENTROPY-related HSD
    )

    # eq. (18):  Lindblad + LINEAR corotation (barotropic + entropy)
    eq18 = (
        GAML                                                           # Lindblad (L)
        + 0.7 * (1.5 - alp - 2.0 * xi / gam) * fac**1.26               # Linear barotropic CT
        + 2.2 * xi * fac**0.71                                         # Linear ENTROPY-related CT
    )


    # Eq. (14)
    eq14 = GAML

    #----------------------
    # Select adiabatic torque formula
    #----------------------
    # Default (currently eq14)
    predicted_torque_adi = eq14
    print("Using DEFAULT (Eq. 14) for ADIABATIC torque.")

    if eq_label == "Paardekooper6" or eq_label == "Equation45":
        # Eq. (45)
        predicted_torque_adi = eq45
        print(f"Using Paardekooper Eq. (45) for ADIABATIC torque with label {eq_label}.")
    if eq_label == "Paardekooper9"  or eq_label == "Equation18":
        # Eq. (18)
        predicted_torque_adi = eq18
        print("Using Paardekooper Eq. (18) for ADIABATIC torque.")
    if eq_label in ["Paardekooper2", "Paardekooper1", "Equation14"]:
        # Eq. (14)
        predicted_torque_adi = eq14
        print("Using Paardekooper Eq. (14) for ADIABATIC torque.")


    print(f"GAM_ADI: {predicted_torque_adi}")
    # Multiply by overall factor
    #if betc > 0.1:
    predicted_torque_adi *= (GAM0 / gam)
    #else:
    #    predicted_torque_adi *= GAM0
    # -------------------------------------------------------
    # "Locally isothermal" torque formula from Paardekooper (2010)
    # -------------------------------------------------------
    # eq. (49) (assuming gamma=1) => Lindblad + linear entropy + nonlinear barotropic
    predicted_torque_iso = (
        GAML                                           # Linear Lindblad
        + 2.2 * xi * fac**0.71 - 1.4 * bet * fac**1.26   # Linear ENTROPY corotation (iso)
        + 1.1 * (1.5 - alp) * fac                      # NONLINEAR barotropic HSD
    )

    print(f"GAM_ISO: {predicted_torque_iso}")
    #if betc > 0.1:
    #    predicted_torque_iso *= (GAM0 / gam) 
    #else:
    predicted_torque_iso *= GAM0  # (no division by gam here)




    return predicted_torque_adi, predicted_torque_iso, GAM0



# planet_data.py

def extract_planet_mass_and_migration(summary_file, IDEFIX=False):
    """
    Extracts the planet-to-star mass ratio (qp) and whether the planet feels the disk (migration status).

    Parameters:
        summary_file (str): Path to the summary or log file.
        IDEFIX (bool): Set to True if reading from an IDEFIX simulation.

    Returns:
        tuple: (planet_mass, migration_status)
               planet_mass is float (planet-to-star mass ratio)
               migration_status is bool (True if migration is enabled, False otherwise)
    """
    # --- FIX: Add encoding='utf-8' to handle special characters in log files ---
    with open(summary_file, 'r', encoding='utf-8') as file:
        contents = file.readlines()

    if IDEFIX:
        in_planet_block = False
        qp = None
        migration = None
        for line in contents:
            if "[Planet]" in line:
                in_planet_block = True
                continue
            if in_planet_block:
                if line.strip() == "" or line.startswith("["):
                    break  # Exit if block ends
                if "planetToPrimary" in line:
                    try:
                        qp = float(line.split()[1])
                    except (IndexError, ValueError):
                        raise ValueError("Could not parse 'planetToPrimary' from IDEFIX log.")
                if "feelDisk" in line:
                    try:
                        val = line.split()[1].strip().lower()
                        migration = val == "true"
                    except IndexError:
                        raise ValueError("Could not parse 'feelDisk' from IDEFIX log.")
        if qp is None or migration is None:
            raise ValueError("Missing planet mass or migration info in IDEFIX log.")
        return qp, migration

    else:
        for i, line in enumerate(contents):
            if "# Planet Name" in line:
                try:
                    planet_line = contents[i + 1].strip()
                    planet_parameters = planet_line.split()
                    mass_value = float(planet_parameters[2])
                    feels_disk = planet_parameters[4].upper()
                    migration = feels_disk == "YES"
                    return mass_value, migration
                except (IndexError, ValueError):
                    raise ValueError("Error reading planet mass or migration status from summary file.")
        raise ValueError("Planet section not found in summary file.")





def plot_combined_density(output_path, simulation_name, torque, time_in_orbits, qp, parameters, IDEFIX=False):
    """
    Produces a two-panel plot:
    1. Top panel: Cartesian-transformed imshow plot of gas density deviation.
    2. Bottom panel: Radial profiles of initial and final gas density (azimuthally and vertically averaged),
       with smoothed torque as a function of time on a secondary axis.
    """


        # Extract parameters for torque prediction
    if IDEFIX:
        h = float(parameters.get('h0', 0.05))
        SIGMA0 = float(parameters.get('Sigma0', 1.0))  # IDEFIX uses "Sigma0"
    else:
        h = float(parameters['ASPECTRATIO'])
        SIGMA0 = float(parameters['SIGMA0'])          # FARGO uses "SIGMA0"

    GAM0 = (qp / h) ** 2 * SIGMA0  # Note: qp is the planet-to-star mass ratio

    # Determine the total number of snapshots
    nt = determine_nt(output_path, IDEFIX=IDEFIX)

    # Ensure nt is the last valid snapshot index
    nt = max(0, nt - 1)  # Ensure nt is always valid

    print(f"Using last snapshot index: {nt}")

    # Read the first and final snapshots
    if IDEFIX:
        data_arrays_initial, xgrid, ygrid, zgrid, parameters = read_single_snapshot_idefix(
            output_path, 0, read_gasdens=True
        )
        data_arrays_final, _, _, _, _ = read_single_snapshot_idefix(
            output_path, nt, read_gasdens=True
        )
    else:
        data_arrays_initial, xgrid, ygrid, zgrid, parameters = read_single_snapshot(
            output_path, 0, read_gasdens=True
        )
        data_arrays_final, _, _, _, _ = read_single_snapshot(
            output_path, nt, read_gasdens=True
        )

    # Extract the initial and final gas density
    gas_density_initial = data_arrays_initial['gasdens']
    gas_density_final = data_arrays_final['gasdens']

    # Compute the deviation at each point in the 3D grid
    nx = gas_density_initial.shape[1]
    ny = gas_density_initial.shape[0]
    nz = gas_density_initial.shape[2]
    gas_density_deviation_3d = (
        gas_density_final - gas_density_initial
    ) / gas_density_initial[ny // 2, nx // 2, nz // 2]

    # Average the deviation vertically (z-axis)
    gas_density_deviation_avg = np.mean(gas_density_deviation_3d, axis=2)

    # Pad the radial grid down to r=0
    dr = xgrid[1] - xgrid[0]
    r_padded = np.insert(xgrid, 0, np.arange(xgrid[0] - dr, 0, -dr)[::-1])

    # Pad the gas density deviation array with zeros for the inner cavity
    zero_padding = np.zeros((gas_density_deviation_avg.shape[0], len(r_padded) - len(xgrid)))
    gas_density_deviation_padded = np.hstack((zero_padding, gas_density_deviation_avg))

    # Generate Cartesian coordinates for the polar grid
    r_mesh, phi_mesh = np.meshgrid(r_padded, ygrid, indexing='ij')
    x_polar = r_mesh * np.cos(phi_mesh)
    y_polar = r_mesh * np.sin(phi_mesh)

    # Define a high-resolution regular Cartesian grid
    x_cartesian = np.linspace(x_polar.min(), x_polar.max(), 1500)
    y_cartesian = np.linspace(y_polar.min(), y_polar.max(), 1500)
    x_grid, y_grid = np.meshgrid(x_cartesian, y_cartesian)

    # Interpolate polar data onto Cartesian grid
    cartesian_data = griddata(
        (x_polar.flatten(), y_polar.flatten()),
        gas_density_deviation_padded.T.flatten(),
        (x_grid, y_grid),
        method='linear',
        fill_value=np.nan
    )

    # Calculate smoothed torque
    max_time = time_in_orbits.max()

    # Apply a rolling average (time-averaged torque)
    rolling_window_size = 100  # Define the window size for averaging (number of data points)
    time_averaged_torque = uniform_filter1d(torque, size=rolling_window_size, mode='nearest')

    global_min = np.min(cartesian_data)
    global_max = np.max(cartesian_data)

    print(f"qp = {qp}")
   
    if qp < 2e-5:
        vmin = global_min
        vmax = global_max
    else:
        vmin = -0.5
        vmax = 1.0



    # Create a two-panel figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [3, 1]})

    # Top Panel: Cartesian imshow plot
    im = axs[0].imshow(
        cartesian_data,
        origin='lower',
        extent=[x_cartesian.min(), x_cartesian.max(), y_cartesian.min(), y_cartesian.max()],
        aspect='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[0], orientation='vertical', pad=0.1)
    cbar.set_label('Gas Density Deviation (z-averaged)')

    # Top panel labels and title
    axs[0].set_xlabel('X (Cartesian)')
    axs[0].set_ylabel('Y (Cartesian)')
    axs[0].set_title(f'Time = {(nt - 1) * parameters["NINTERM"] / 20:.1f} orbits')

    # Add guiding lines and circle
    axs[0].axhline(0, color='white', linestyle='--', linewidth=0.5)
    axs[0].axvline(0, color='white', linestyle='--', linewidth=0.5)
    circle = plt.Circle((0, 0), 1.0, color='white', linestyle='--', fill=False)
    axs[0].add_artist(circle)

    # Bottom Panel: Radial density profiles (scaled) and smoothed torque
    r = xgrid
    initial_profile = np.mean(np.mean(gas_density_initial, axis=0), axis=1)
    final_profile = np.mean(np.mean(gas_density_final, axis=0), axis=1)

    # Scale the radial profiles using the value at r=1
    scale_factor = initial_profile[np.searchsorted(r, 1)]
    scaled_initial_profile = initial_profile / scale_factor
    scaled_final_profile = final_profile / scale_factor

    # Main axis: radial distance vs. scaled density
    ax_density = axs[1]

    density_line_1, = ax_density.plot(
        r, scaled_initial_profile,
        label="Scaled Initial Gas Density",
        color='blue', linewidth=2
    )
    density_line_1.set_zorder(10)

    density_line_2, = ax_density.plot(
        r, scaled_final_profile,
        label="Scaled Final Gas Density",
        color='red', linewidth=2
    )
    density_line_2.set_zorder(10)

    # Ensure density axis is on top
    ax_density.set_zorder(2)
    ax_density.patch.set_visible(False)  # Hide density axis background to avoid overlap

    ax_density.set_xlabel("Radial Distance (r)")
    ax_density.set_ylabel("Gas Density (Scaled)")
    ax_density.set_xlim([r[0], r[-1]])
    ax_density.set_ylim([0, 3.5])
    ax_density.grid(True)

    # ---------------------
    # Create top axis for time
    # ---------------------
    ax_time = ax_density.twiny()
    ax_time.spines["top"].set_visible(True)
    ax_time.spines["bottom"].set_visible(False)
    ax_time.spines["left"].set_visible(False)
    ax_time.spines["right"].set_visible(False)

    # Turn off y-ticks on this time axis (since it's just an x-axis for time)
    ax_time.tick_params(
        axis='y', which='both',
        left=False, right=False,
        labelleft=False, labelright=False
    )

    # Configure the time axis
    ax_time.set_xlim(0, time_in_orbits.max())
    ax_time.set_xlabel("Time (orbits)")
    num_ticks = 5
    time_ticks = np.linspace(0, time_in_orbits.max(), num_ticks)
    ax_time.set_xticks(time_ticks)
    ax_time.set_xticklabels([f"{t:.1f}" for t in time_ticks])

    # ---------------------
    # Right axis for torque
    # ---------------------
    ax_torque = ax_time.twinx()
    ax_torque.spines["right"].set_visible(True)
    ax_torque.spines["left"].set_visible(False)
    ax_torque.spines["top"].set_visible(False)
    ax_torque.spines["bottom"].set_visible(False)

    # Ensure torque axis is below density axis
    ax_torque.set_zorder(1)

    ax_torque.set_ylabel(r"$\Gamma / \Gamma_0$")
    if qp >1e-4:
        ax_torque.set_ylim([-7, 3])  # Set the desired y-axis range for torque
    
    ax_torque_min = np.min(qp * time_averaged_torque/GAM0)
    ax_torque_max = np.max(qp * time_averaged_torque/GAM0)
    #ax_torque.set_ylim(ax_torque_min, ax_torque_max)

    # Plot torque vs. time
    torque_line, = ax_torque.plot(
        time_in_orbits,
        qp * time_averaged_torque/GAM0,
        label="Smoothed Torque (simulation)",
        color='green', linestyle='--', linewidth=2
    )
    torque_line.set_zorder(1)  # behind the density lines

    # ---------------------
    # Combine legends
    # ---------------------
    lines_density, labels_density = ax_density.get_legend_handles_labels()
    lines_torque, labels_torque = ax_torque.get_legend_handles_labels()
    legend = ax_density.legend(
        lines_density + lines_torque,
        labels_density + labels_torque,
        loc='lower right',
        framealpha=0.5  # Make the legend semi-transparent
    )
    legend.set_zorder(9999)  # Ensure legend is always on top

    # Title and layout
    #ax_density.set_title("Radial Profiles of Gas Density and Torque")
    plt.tight_layout()

    # Save the plot
    combined_output_filename = os.path.join(
        output_path, f"{simulation_name}_combined_density_plot.pdf"
    )
    plt.savefig(combined_output_filename)
    plt.close()
    print(f"Combined density plot saved to {combined_output_filename}")

    # Transfer the plot to the local directory
    local_directory = "/Users/mariuslehmann/Downloads/Contours/planet_evolution"
    scp_transfer(combined_output_filename, local_directory, "mariuslehmann")



def read_bigplanet_data(filepath):
    """
    Reads the bigplanet0.dat file and extracts the required data.
    """
    data = np.loadtxt(filepath)
    date = data[:, 8]
    position = data[:, 1:4]  # x, y, z
    velocity = data[:, 4:7]  # vx, vy, vz
    mass = data[:, 7]
    return date, position, velocity, mass



def read_planet_data_idefix(filepath):
    """
    Reads the IDEFIX planet0.dat file and extracts position, velocity, mass,
    and computes eccentricity and semi-major axis.

    Returns:
        date, position, velocity, mass, eccentricity, semi_major_axis
    """
    data = np.loadtxt(filepath)
    date = data[:, 0]
    position = data[:, 1:4]     # x, y, z
    velocity = data[:, 4:7]     # vx, vy, vz
    mass = data[:, 7]

    # Compute magnitudes
    r = np.linalg.norm(position, axis=1)
    v = np.linalg.norm(velocity, axis=1)

    # Gravitational parameter mu = GM, assumed to be 1 in code units
    mu = 1.0

    # Specific orbital energy
    energy = 0.5 * v**2 - mu / r
    semi_major_axis = -mu / (2 * energy)

    # Specific angular momentum
    h_vec = np.cross(position, velocity)

    # Eccentricity vector and magnitude
    e_vec = (np.cross(velocity, h_vec) / mu) - (position / r[:, None])
    eccentricity = np.linalg.norm(e_vec, axis=1)

    return date, position, velocity, mass, eccentricity, semi_major_axis

def read_orbit_data(filepath):
    """
    Reads the orbit0.dat file and extracts the required data.
    """
    data = np.loadtxt(filepath)
    date = data[:, 0]
    eccentricity = data[:, 1]
    semi_major_axis = data[:, 2]
    return date, semi_major_axis, eccentricity


def read_torque_data(filepath):
    """
    Reads the torq_planet_0.dat file and extracts the required data.
    """
    data = np.loadtxt(filepath)
    time = data[:, 0]
    torque = data[:, 1]
    return time, torque

# Helper function to read tqwk0.dat as an alternative
#def read_alternative_torque(file_path):
#    """
#    Reads tqwk0.dat and computes the torque as the sum of columns 2 and 3.
#    Returns the time and total torque arrays.
#    """
#    data = np.loadtxt(file_path)
#    time = data[:, -1]  # Time is in the last column
#    torque = data[:, 1] + data[:, 2]  # Sum of columns 2 and 3
#    return time, torque

# Modify the read_alternative_torque function to create separate segments at orbit changes

def read_alternative_torque(file_path, IDEFIX=False):
    """
    Reads torque file from either IDEFIX (tqwk0.dat) or FARGO (torq_planet_0.dat).

    Parameters:
        file_path (str): Path to the torque file.
        IDEFIX (bool): Set to True for IDEFIX-style tqwk0.dat.

    Returns:
        time (np.ndarray): Time array.
        torque (np.ndarray): Torque array.
        orbit_numbers (np.ndarray or None): Orbit number array (if available).
    """

    print(f"📂 Reading torque file: {file_path}")

    data = np.loadtxt(file_path)

    if IDEFIX:
        # Use last column as time, and sum columns 1–4 for torque
        time = data[:, -1]  # typically column 18 in IDEFIX tqwk0.dat
        torque = data[:, 1] + data[:, 2] + data[:, 3] + data[:, 4]
        orbit_numbers = data[:, 0]  # use first column as orbit_numbers (may be float)
    else:
        # FARGO format: time in last column, torque in columns 1 + 2
        time = data[:, -1]
        torque = data[:, 1] + data[:, 2]
        orbit_numbers = data[:, 0]

    # Remove NaN or infinite values
    valid = np.isfinite(time) & np.isfinite(torque)
    time = time[valid]
    torque = torque[valid]
    if orbit_numbers is not None:
        orbit_numbers = orbit_numbers[valid]

    # Sort by time
    idx = np.argsort(time)
    time = time[idx]
    torque = torque[idx]
    if orbit_numbers is not None:
        orbit_numbers = orbit_numbers[idx]

    # Remove duplicate time entries
    unique = np.concatenate(([True], np.diff(time) > 1e-10))
    time = time[unique]
    torque = torque[unique]
    if orbit_numbers is not None:
        orbit_numbers = orbit_numbers[unique]

    return time, torque, orbit_numbers


# Helper function to debug tqwk0.dat
def debug_alternative_torque(file_path):
    """
    Reads tqwk0.dat and computes the torque as the sum of columns 2 and 3.
    Iterates through all rows to identify mismatches in column lengths or data inconsistencies.
    """
    try:
        data = np.loadtxt(file_path)  # Load the file
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None

    n_rows, n_cols = data.shape  # Get the number of rows and columns
    print(f"File shape: {n_rows} rows, {n_cols} columns")

    # Initialize lists to store the time and torque values
    time = []
    torque = []

    # Loop through each row to detect the issue
    for i in range(n_rows):
        try:
            # Attempt to access the time and torque columns
            current_time = data[i, -1]  # Time is in the last column
            current_torque = data[i, 1] + data[i, 2]  # Torque = column 2 + column 3

            # Append to the lists if no error
            time.append(current_time)
            torque.append(current_torque)

        except IndexError as e:
            # If an IndexError occurs, print debug information and break the loop
            print(f"IndexError at row {i + 1}: {e}")
            print(f"Data shape: {data.shape}")
            break

    # Print the lengths of the arrays
    print(f"Length of time array: {len(time)}")
    print(f"Length of torque array: {len(torque)}")

    # Check for mismatched lengths at the end
    if len(time) != len(torque):
        print(f"Mismatch in lengths: Time ({len(time)}) vs Torque ({len(torque)})")

    return np.array(time), np.array(torque)



def save_data_to_npz(output_path, simulation_name, **kwargs):
    """
    Saves the extracted data to a .npz file.
    """
    npz_filename = os.path.join(output_path, f"{simulation_name}_planet_data.npz")
    np.savez(npz_filename, **kwargs)
    print(f"Data saved to {npz_filename}")


def plot_planet_data(time_in_orbits, semi_major_axis, eccentricity, mass, torque, output_path, 
                    simulation_name, parameters, qp, migration, gam, 
                    orbit_numbers=None, date_torque=None, original_torque=None, IDEFIX=False, summary_file=None):
    """
    Plots the planet data and saves the figure as a PDF.
    If migration=False, only the torque panel is plotted.
    Overplots PK11 total and Lindblad torques and the full-time-averaged simulation torque.
    """
    torque_time = date_torque if date_torque is not None else time_in_orbits

    actual_avg_start = np.nan
    actual_avg_end = np.nan
    avg_mask = None

    if len(torque_time) > 0:
        available_orbits = torque_time
        final_orbit = available_orbits[-1]

        avg_start_orbit = final_orbit - 200
        current_mask = (available_orbits >= avg_start_orbit)
        if np.sum(current_mask) >= 2:
            avg_times = available_orbits[current_mask]
            actual_avg_start = avg_times[0]
            actual_avg_end = avg_times[-1]
            avg_mask = current_mask
        else:
            avg_start_orbit = final_orbit - 100
            current_mask = (available_orbits >= avg_start_orbit)
            if np.sum(current_mask) >= 2:
                avg_times = available_orbits[current_mask]
                actual_avg_start = avg_times[0]
                actual_avg_end = avg_times[-1]
                avg_mask = current_mask
                print(f"Warning: Not enough data for 500 orbits. Using last 200 orbits [{actual_avg_start:.1f}–{actual_avg_end:.1f}] for averaging.")
            else:
                if len(available_orbits) >= 2:
                    actual_avg_start = available_orbits[0]
                    actual_avg_end = final_orbit
                    avg_mask = (available_orbits >= actual_avg_start) & (available_orbits <= actual_avg_end)
                    print(f"Warning: Less than 200 orbits available. Using full range [{actual_avg_start:.1f}–{actual_avg_end:.1f}] for averaging.")

    print("Computing PK11 torques...")
    Total_torque_PK11, Lindblad_torque_PK11, GAM0, gamma_eff, nu_used, nu_type = compute_theoretical_torques_PK11(
        parameters, qp, simulation_name, IDEFIX=IDEFIX, summary_file=summary_file,
        avg_start_orbit=actual_avg_start, avg_end_orbit=actual_avg_end
    )

    smoothing_timescale = 10.
    if len(torque_time) > 1:
        dt = np.mean(np.diff(torque_time))
    else:
        dt = 1.0
    rolling_window_size = max(1, int(smoothing_timescale / dt))

    if original_torque is not None and date_torque is not None:
        time_averaged_torque_for_plot = uniform_filter1d(original_torque, size=rolling_window_size, mode='nearest')
        torque_scaled_for_plot = qp * time_averaged_torque_for_plot / GAM0
    else:
        time_averaged_torque_for_plot = uniform_filter1d(torque, size=rolling_window_size, mode='nearest')
        torque_scaled_for_plot = qp * time_averaged_torque_for_plot / GAM0
        
    mean_sim_torque = np.nan
    if avg_mask is not None and np.sum(avg_mask) >= 2:
        mean_sim_torque = np.mean(torque_scaled_for_plot[avg_mask]) * gamma_eff

    # ============= PLOTTING =============
    plt.style.use('default')

    if migration:
        # Placeholder for your existing migration plot code
        pass 
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(torque_time, torque_scaled_for_plot * gamma_eff, color='green', linewidth=2, label='Time-Averaged Torque')
        ax.axhline(Total_torque_PK11 * gamma_eff, color='black', linestyle='--',
                   label=fr'$\gamma_{{\mathrm{{eff}}}} \Gamma_{{\mathrm{{tot}}}} / \Gamma_0$ (PK11) [{Total_torque_PK11 * gamma_eff:.3f}]')
        ax.axhline(Lindblad_torque_PK11 * gamma_eff, color='red', linestyle='--',
                   label=fr'$\gamma_{{\mathrm{{eff}}}} \Gamma_{{L}} / \Gamma_0$ (PK11) [{Lindblad_torque_PK11 * gamma_eff:.3f}]')
        
        if not np.isnan(mean_sim_torque):
            # Updated legend label
            legend_label = (
                fr'Mean Simulation Torque ({mean_sim_torque:.3f}) '
                fr'[{actual_avg_start:.0f}–{actual_avg_end:.0f} orbits] '
                fr'($\nu_{{{nu_type.lower()}}}={nu_used:.1e}$)'
            )
            ax.axhline(mean_sim_torque, color='blue', linestyle=':', linewidth=2,
                       label=legend_label)

        finite_values = torque_scaled_for_plot[np.isfinite(torque_scaled_for_plot)]
        all_plot_values = np.concatenate([
            finite_values * gamma_eff,
            [val for val in [Total_torque_PK11 * gamma_eff, Lindblad_torque_PK11 * gamma_eff, mean_sim_torque] if np.isfinite(val)]
        ])
        
        if len(all_plot_values) > 0:
            y_min_val = np.min(all_plot_values)
            y_max_val = np.max(all_plot_values)
            padding = (y_max_val - y_min_val) * 0.1
            ax.set_ylim([y_min_val - padding, y_max_val + padding])

        ax.set_ylabel(r'$\gamma_{\mathrm{eff}} \Gamma / \Gamma_0$')
        ax.set_xlabel('Time (orbits)')
        if len(torque_time)>0:
            ax.set_xlim([0, np.max(torque_time)])
        ax.legend(fontsize='small')
        title = f'Torque Evolution (Smoothing: {smoothing_timescale:.1f} orbits)'

    plt.suptitle(title)
    output_filename = os.path.join(output_path, f"{simulation_name}_planet_evolution.pdf")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_filename}")

    local_directory = "/Users/mariuslehmann/Downloads/Profiles/planet_evolution/"
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    return output_filename



def plot_adiabatic_torque(parameters, output_path, qp):
    """
    Plots the adiabatic torque in p-q space using imshow and saves the figure as a PDF.
    Includes demarcations for COS occurrence, adiabatic torque = 0, and isothermal torque = 0.
    """
    FONT_SIZE = 18  # Define a global font size parameter
    plt.rcParams.update({'font.size': FONT_SIZE})  # Apply font size globally
    
    gam = parameters['GAMMA']
    bh = parameters['THICKNESSSMOOTHING']
    fac = 0.4 / (bh)
    
    # Define p and q grid with increased resolution
    p_vals = np.linspace(-2, 5, 300)
    q_vals = np.linspace(0.5, 2.5, 300)
    P, Q = np.meshgrid(p_vals, q_vals)

    
    # Compute torque
    fi = 0.5 * (1.0 - Q)
    ss = P - fi - 1.0
    xi = Q - (gam - 1) * ss

    GAML = -(2.5 + 1.7 * Q -0.1 * ss) * fac**0.71
    # Eq. (45)
    #predicted_torque_adi = GAML + 1.1 * fac * (1.5 - ss) + xi / gam * fac * (10.1 * fac**0.5 -2.2)
    # Eq. (18)
    predicted_torque_adi = GAML + 0.7*(1.5 - ss - 2.*xi / gam)*fac**1.26 + 2.2*xi*fac**0.71

    predicted_torque_adi *= 1 / gam 

    # Compute isothermal torque
    predicted_torque_iso = -(2.5 - 0.5 * Q - 0.1 * ss) * fac ** 0.71 - 1.4 * Q * fac ** 1.26 + 1.1 * (1.5 - ss) * fac
   
    # Compute COS occurrence condition
    COS_condition = (P + Q) * (Q + (1 - gam) * P) < 0
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))  # Increased figure size for better spacing
    
    # Use imshow instead of contourf
    # Note: imshow expects data in [y, x] format, so we need to transpose the torque array
    # Also, imshow uses the array indices for extent by default, so we need to specify the extent
    extent = [q_vals.min(), q_vals.max(), p_vals.min(), p_vals.max()]
    im = ax.imshow(
        predicted_torque_adi.T,  # Transpose to match [y, x] format expected by imshow
        origin='lower',  # Sets the origin to bottom left
        aspect='auto',   # Adjusts the aspect ratio to fill the plot box
        extent=extent,   # Sets the limits in data coordinates
        cmap='RdYlBu_r'  # Same colormap as before
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Torque (Adiabatic)')
    
    # COS demarcation
    cs1 = ax.contour(Q, P, COS_condition, levels=[0.5], colors='black', linewidths=2)
    
    # Determine a representative point for COS region annotation
    cos_x = np.mean(Q[COS_condition]) + 0.5
    cos_y = np.mean(P[COS_condition]) - 1.0
    ax.text(cos_x, cos_y, "COS region", color='black', fontsize=FONT_SIZE + 3, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Torque zero demarcations - use a small range around zero instead of exactly zero
    zero_tolerance = 0.0001  # Adjust this value as needed
    zero_range = [-zero_tolerance, 0, zero_tolerance]  # Create levels around zero

    # For adiabatic torque
    cs2 = ax.contour(Q, P, predicted_torque_adi, levels=zero_range, colors='black', linestyles='dashed')
    if cs2.collections:
        # Only label specific locations to avoid clutter
        ax.clabel(cs2, fmt="$\\Gamma_{ADI} = 0$", inline=True, fontsize=FONT_SIZE + 3, manual=[(1.5, 3.0), (2.5, 4.5)])
    else:
        print("No adiabatic torque near zero contour found")
        ax.text(1.2, 2.5, "$\\Gamma_{ADI} = 0$", color='black', fontsize=FONT_SIZE + 3, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    # For isothermal torque
    cs3 = ax.contour(Q, P, predicted_torque_iso, levels=zero_range, colors='blue', linestyles='dotted')
    if cs3.collections:
        # Add label at specific location to avoid clutter
        ax.clabel(cs3, fmt="$\\Gamma_{ISO} = 0$", inline=True, fontsize=FONT_SIZE + 3, manual=[(1.0, -1.0)])
        
        # Check isothermal torque values directly to determine which side is positive and negative
        # Sample points on either side of the contour line
        sample_q1, sample_p1 = 1.0, 0.0  # Point above the contour line (based on the plot)
        sample_q2, sample_p2 = 1.0, -1.5  # Point below the contour line
        
        # Calculate fi, ss, and predicted_torque_iso for these points
        fi1 = 0.5 * (1.0 - sample_q1)
        ss1 = sample_p1 - fi1 - 1.0
        torque_iso1 = -(2.5 - 0.5 * sample_q1 - 0.1 * ss1) * fac ** 0.71 - 1.4 * sample_q1 * fac ** 1.26 + 1.1 * (1.5 - ss1) * fac
        
        fi2 = 0.5 * (1.0 - sample_q2)
        ss2 = sample_p2 - fi2 - 1.0
        torque_iso2 = -(2.5 - 0.5 * sample_q2 - 0.1 * ss2) * fac ** 0.71 - 1.4 * sample_q2 * fac ** 1.26 + 1.1 * (1.5 - ss2) * fac
        
        # Place labels based on calculated values
        # The point with positive torque gets ">0" and negative gets "<0"
        if torque_iso1 > 0:
            ax.text(0.8, 0.2, '>0', color='blue', fontsize=FONT_SIZE + 3, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            ax.text(1.2, -1.7, '<0', color='blue', fontsize=FONT_SIZE + 3, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        else:
            ax.text(0.8, 0.2, '<0', color='blue', fontsize=FONT_SIZE + 3, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            ax.text(1.2, -1.7, '>0', color='blue', fontsize=FONT_SIZE + 3, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    else:
        print("No isothermal torque near zero contour found")
        ax.text(1.4, 2.0, "$\\Gamma_{ISO} = 0$", color='blue', fontsize=FONT_SIZE + 3,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue', boxstyle='round,pad=0.5'))
    
    # Mark specific p-q points
    special_points = [(2.0, 2.5)]
    for q_point, p_point in special_points:
        ax.plot(q_point, p_point, marker='o', color='red', markersize=8, label='Special Point')
    
    # Labels and aesthetics
    ax.set_xlabel('q', fontsize=FONT_SIZE+2)
    ax.set_ylabel('p', fontsize=FONT_SIZE+2)

    # **Updated Title with Smoothing Length b**
    ax.set_title(f'Adiabatic Torque (b/h = {bh:.3f}, γ = {gam:.2f})', fontsize=FONT_SIZE+4)
    
    # Adjust layout to make more room
    plt.tight_layout()
    
    # Save the figure
    output_filename = os.path.join(output_path, "adiabatic_torque.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_filename}")
    
    # SCP Transfer (as per template)
    local_directory = "/Users/mariuslehmann/Downloads/Profiles/planet_evolution/"
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    
    return output_filename




import argparse


def main(simulation_name, quick_mode=False, IDEFIX=False):
    # Determine the base path
    base_path = determine_base_path(simulation_name, IDEFIX=IDEFIX)
    output_path = base_path

    # Define file paths
    if not IDEFIX:
        bigplanet_file = os.path.join(output_path, "bigplanet0.dat")
        orbit_file = os.path.join(output_path, "orbit0.dat")
        summary_file = os.path.join(output_path, "summary0.dat")
    else:
        bigplanet_file = os.path.join(output_path, "planet0.dat")
        summary_file = os.path.join(output_path, "idefix.0.log")
        log_file = summary_file
  
    tqwk_file = os.path.join(output_path, "tqwk0.dat")


    # Planet presence check
    planet_file = log_file if IDEFIX else summary_file
    planet = check_planet_presence(planet_file, IDEFIX=IDEFIX)
    if not planet:
        print("Planet present: No")
        sys.exit("Exiting: No planet found in the simulation.")
    print("Planet found")

    # Read parameters
    param_file = log_file if IDEFIX else summary_file
    parameters = read_parameters(param_file, IDEFIX=IDEFIX)

    # Planet mass and migration
    if IDEFIX:
        gam = float(parameters.get('gamma', 1.4))
    else:
        gam = float(parameters.get('GAMMA', 1.4))
    qp, migration = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)


    print(f"Planet-to-star mass ratio (qp): {qp}")
    print(f"Migration Enabled: {migration}")



######################
    # Read main simulation output files
    if IDEFIX:
        date_bp, position, velocity, mass, eccentricity, semi_major_axis = read_planet_data_idefix(bigplanet_file)
    else:
        date_bp, position, velocity, mass = read_bigplanet_data(bigplanet_file)
        date_orbit, eccentricity, semi_major_axis = read_orbit_data(orbit_file)

    # Read torque data
    if os.path.exists(tqwk_file):
        date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file, IDEFIX=IDEFIX)
        print(f"Length of time array AFTER CALL: {len(date_torque)}")
        print(f"Length of torque array AFTER CALL: {len(torque)}")
        if orbit_numbers is not None:
            print(f"Length of orbit_numbers AFTER CALL: {len(orbit_numbers)}")
    else:
        raise FileNotFoundError("tqwk0.dat was not found.")

    # Store original torque data
    original_date_torque = date_torque.copy()
    original_torque = torque.copy()
    original_orbit_numbers = orbit_numbers.copy() if orbit_numbers is not None else None

    # Interpolate torque to match planet time steps
    if not np.array_equal(date_bp, date_torque):
        print("Warning: Torque time data does not align with other data. Interpolating torque data.")
        torque = np.interp(date_bp, date_torque, torque)
        date_torque = date_bp

    # Time in orbits (conversion depends on code)
    if IDEFIX:
        time_in_orbits = date_torque  # Already in orbital units
    else:
        time_in_orbits = date_torque / (2 * np.pi)

    # Interpolate orbital data to match (FARGO only)
    if not IDEFIX and not np.array_equal(date_bp, date_orbit):
        print("Warning: Orbit data does not align with bigplanet data. Interpolating orbital data.")
        semi_major_axis = np.interp(date_bp, date_orbit, semi_major_axis)
        eccentricity = np.interp(date_bp, date_orbit, eccentricity)
        date_orbit = date_bp

    # Save interpolated data
    save_data_to_npz(output_path, simulation_name,
                     time_in_orbits=time_in_orbits,
                     position=position,
                     velocity=velocity,
                     mass=mass,
                     semi_major_axis=semi_major_axis,
                     eccentricity=eccentricity,
                     torque=torque)

    # Plot theory curve if appropriate
    if not IDEFIX:
        if gam == 1.4:
            plot_adiabatic_torque(parameters, output_path, qp)

    # Planet evolution plot (use original non-interpolated torque/orbit_number data)
    plot_planet_data(time_in_orbits, semi_major_axis, eccentricity, mass, torque, output_path,
                     simulation_name, parameters, qp, migration, gam,
                     orbit_numbers=original_orbit_numbers,
                     date_torque=(original_date_torque / (2 * np.pi) if IDEFIX else original_date_torque / (2 * np.pi)),
                     original_torque=original_torque, IDEFIX=IDEFIX,
                     summary_file=summary_file)
###################

    if not quick_mode:
        plot_combined_density(output_path, simulation_name, torque, time_in_orbits, qp, parameters, IDEFIX=IDEFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run planet evolution analysis.")
    parser.add_argument("simulation_name", help="Name of the simulation to process.")
    parser.add_argument("--quick", action="store_true", help="Skip the combined density plot.")
    parser.add_argument("--idefix", action="store_true", help="Flag to indicate IDEFIX simulation")

    args = parser.parse_args()
    main(args.simulation_name, quick_mode=args.quick, IDEFIX=args.idefix)
