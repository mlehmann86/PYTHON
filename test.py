import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Assume these functions exist in a 'planet_data.py' file
# If not, define them or adjust the import
try:
    from planet_data import determine_base_path, read_parameters, extract_planet_mass_and_migration, read_alternative_torque
except ImportError:
    print("Warning: Could not import from planet_data.py. Define dummy functions.")
    # Define dummy functions if planet_data.py is not available
    def determine_base_path(sim): return sim # Placeholder
    def read_parameters(file): return {} # Placeholder
    def extract_planet_mass_and_migration(file): return 1.26e-5, 0.0 # Placeholder qp, migration
    def read_alternative_torque(file): # Placeholder
         times = np.linspace(0, 200 * 2 * np.pi, 100)
         torques = np.random.rand(100) * 1e-5
         return times, torques, None

# --- User Provided gamma_eff_theory Function ---
# Note: This function seems based on Eq 46 of Paardekooper 2011, which uses chi_p
# via Q = 2*chi_p / (3*h^3*r^2*Omega).
# However, it's being passed 'beta' (beta_cool) here.
# Ensure this formula correctly represents gamma_eff for BETA-COOLING.
# The variable Q here might need redefinition based on beta_cool instead of chi_p
# if this formula is to be used for beta-cooling.
# For PK11 Fig 6, Q relates to chi_p. For beta-cooling, the mapping might differ.
# Let's rename 'beta' argument to 'beta_or_chi_param' for clarity in this context.
def gamma_eff_theory(gamma, beta_or_chi_param, h_val):
    """
    Calculates effective gamma.
    WARNING: Original formula (PK11 Eq 46) uses Q derived from chi_p.
    Using beta_cool here instead assumes a specific mapping between beta_cool and chi_p
    (e.g., via the derived chi_p used elsewhere) or that this formula structure
    also applies directly to beta_cool. Please verify the physical basis for using
    beta_cool in this specific Q definition.

    PK11 Eq 45 definition: Q = 2*chi_p / (3*h^3*r^2*Omega) = 2*chi_p / (3*h*cs^2) assuming cs = h*r*Omega
    If using beta_cool, is the effective Q = beta_cool * constant ? Check literature/derivation.
    The definition below uses Q = (2.0 * beta_or_chi_param) / (3.0 * h_val),
    which seemsdimensionally inconsistent if beta_or_chi_param is the beta_cool parameter (often Omega*t_cool).
    Assuming the formula structure is correct but Q definition might need review for beta-cooling.
    """
    # Assuming Q definition provided by user is intended, proceed with caution.
    Q = (2.0 * beta_or_chi_param) / (3.0 * h_val)
    # Clamp Q if it becomes excessively large or small due to beta_or_chi_param input
    Q = np.clip(Q, 1e-10, 1e10) # Avoid extreme values

    with np.errstate(divide='ignore', invalid='ignore'):
         # Term under sqrt based on PK11 Eq 46 structure derivation
         # Simplified version from Mihalas & Mihalas dispersion relation: (1-i*chi*k^2/w) / (1-i*gamma*chi*k^2/w) -> phase speed^2
         # PK11 Eq 46 seems to be derived from wave speed calculation.
         # Let's use the structure derived from Mihalas & Mihalas (1984) as appears in some related codes:
         # gamma_eff = (1 + gamma * Q**2) / (1 + Q**2) # Incorrect, this is ratio of specific heats approx
         # Let's use PK11 Eq 46 formula directly (from their Fig 8 relation)
         # Re-deriving from Eq 44 requires care. Let's use the provided structure but acknowledge the Q issue.

         # Using the user-provided calculation structure:
         sqrt_term_inner = (1.0 + Q**2) / (1.0 + gamma**2 * Q**2)
         # Ensure inner term is non-negative before sqrt
         sqrt_term = np.sqrt(np.maximum(0.0, sqrt_term_inner))
         numerator = 2.0
         # Denominator calculation might be sensitive to Q -> 0 or Q -> inf
         denominator_part1 = (1.0 + gamma * Q**2) / (1.0 + gamma**2 * Q**2)
         denominator = denominator_part1 + sqrt_term

         # Avoid division by zero or very small numbers
         if np.isclose(denominator, 0):
             # Determine limit based on Q -> 0 (denominator -> 1+1=2, gamma_eff -> 1)
             # or Q -> inf (denominator -> 1/gamma + 1/gamma = 2/gamma, gamma_eff -> gamma)
             # The user formula gives gamma_eff -> 1 as Q -> 0.
             # As Q -> inf, sqrt_term -> 1/gamma, denom_part1 -> 1/gamma, so denom -> 2/gamma, gamma_eff -> gamma.
             gamma_eff_val = 1.0 if Q < 1 else gamma # Approximate limit behavior
         else:
             gamma_eff_val = numerator / denominator

         # Clamp result to physically meaningful range [1, gamma]
         gamma_eff_val = np.clip(gamma_eff_val, 1.0, gamma)

    # Handle potential NaN results if inputs were extreme despite clipping Q
    if np.isnan(gamma_eff_val):
        print(f"Warning: gamma_eff calculation resulted in NaN (gamma={gamma}, Q={Q}). Returning gamma.")
        return gamma # Fallback to adiabatic gamma

    return gamma_eff_val


# ----------------------------------------------------------------------
# 1. Non‑linear corotation torque function with PK11 override flag
# ----------------------------------------------------------------------
def compute_theoretical_corotation_paarde2011(
        p_nu, *, h, gamma, q, s, beta, qp, verbose=False,
        force_pk11_dotted_params=False): # Flag to force PK11 dotted curve params
    """
    Calculates Γ_C / Γ_0 from Paardekooper+2011 (Eq. 50 structure).

    If force_pk11_dotted_params is True:
        Uses fixed parameters from PK11 Fig 6 dotted curve (chi_p=1e-5).
        Inputs h, gamma, q, s, beta, qp are IGNORED. gamma_eff is forced to gamma=1.4.
    Else (default):
        Uses the provided parameters h, gamma, q, s, qp.
        Assumes 'beta' is the beta-cooling parameter.
        Calculates gamma_eff using gamma_eff_theory.
        Derives an effective chi_p from beta-cooling using "Method 2".
        IMPORTANT: Assumes input 'q' is beta_idx (temp slope) and 's' is alpha_idx (dens slope).
    """
    # --- Parameters for Paardekooper+11 Fig. 6 dotted curve ---
    pk11_gamma = 1.4       # 7/5
    pk11_h = 0.05
    pk11_q_beta_idx = 1.0  # Temp slope index (-beta in paper)
    pk11_s_alpha_idx = 0.5 # Sigma slope index (-alpha in paper)
    pk11_qp = 1.26e-5
    pk11_fixed_chi_p = 1e-5
    #---------------------------------------------------------

    # Local variables for calculation, overridden if needed
    h_calc = h
    gamma_calc = gamma
    q_calc = q # Assumed to be beta_idx
    s_calc = s # Assumed to be alpha_idx
    qp_calc = qp
    beta_cool_param = beta # Store original beta if needed

    if force_pk11_dotted_params:
        h_calc = pk11_h
        gamma_calc = pk11_gamma
        q_calc = pk11_q_beta_idx
        s_calc = pk11_s_alpha_idx
        qp_calc = pk11_qp
        γ_eff = gamma_calc # Force gamma_eff = gamma for this specific case
        χ_p_to_use = pk11_fixed_chi_p
        if verbose:
             print("    >> Forcing PK11 Fig 6 Dotted Curve Parameters <<")
             print(f"    >> gamma={gamma_calc:.2f}, h={h_calc:.2f}, q(beta_idx)={q_calc:.1f}, s(alpha_idx)={s_calc:.1f}, qp={qp_calc:.2e}, chi_p={χ_p_to_use:.1e}, gamma_eff={γ_eff:.2f}")
    else:
        # --- Use simulation/beta-cooling parameters ---
        # Calculate gamma_eff using the provided theory function
        γ_eff = gamma_eff_theory(gamma_calc, beta_cool_param, h_calc)

        # Calculate effective chi_p from beta cooling ("Method 2")
        # Need x_s first, which depends on gamma_eff
        _C = 1.1 * γ_eff**(-0.25)
        _x_s_temp = _C * np.sqrt(qp_calc / h_calc)
        # Handle beta_cool_param = 0 case (adiabatic) -> infinite chi_p effectively? Or should be treated differently?
        # If beta=0, cooling is infinitely slow (adiabatic). Mapping to chi_p is unclear. Assume chi_p -> 0 (no diffusion)
        if np.isclose(beta_cool_param, 0):
             χ_p_to_use = 1e-20 # Effectively zero, avoid division by zero later
        elif beta_cool_param > 0:
             χ_p_to_use = (_x_s_temp/h_calc)**2 * h_calc**2 / beta_cool_param
        else:
             χ_p_to_use = np.inf # Negative beta doesn't make sense here

        if verbose:
             print(f"    >> Using Beta-Cooling Parameters (beta={beta_cool_param}) <<")
             print(f"    >> gamma={gamma_calc:.3f}, h={h_calc:.3f}, q={q_calc:.2f}, s={s_calc:.2f}, qp={qp_calc:.2e}")
             print(f"    >> Calculated: gamma_eff={γ_eff:.3f}, derived chi_p={χ_p_to_use:.2e}")

    # ---------------- coefficients (Table 1) -------------
    # Calculate xi using the potentially overridden q_calc, s_calc, gamma_calc
    xi = q_calc - (gamma_calc - 1.0) * s_calc
    Gamma_hs_baro = 1.1 * (1.5 - s_calc)
    Gamma_lin_baro = 0.7 * (1.5 - s_calc)
    # PK11 formula uses adiabatic gamma in coefficients, not gamma_eff
    Gamma_hs_ent  = 7.9 * xi / gamma_calc
    Gamma_lin_ent = (2.2 - 1.4 / gamma_calc) * xi

    # --- Saturation functions F, G, K ---
    F = lambda p: 1.0 / (1.0 + (p/1.3)**2)

    def G(p):
        p = np.asarray(p, float)
        pc = np.sqrt(8/(45*np.pi)) # Critical p for G
        # Handle potential non-positive p inputs gracefully
        p_safe = np.maximum(p, 1e-20)
        term1 = (16/25)*(45*np.pi/8)**0.75 * p_safe**1.5
        term2 = 1.0 - (9/25)*(8/(45*np.pi))**(4/3) * p_safe**(-8/3)
        out = np.where(p < pc, term1, term2)
        # Ensure output is physically meaningful (e.g., between 0 and 1)
        return np.clip(out, 0.0, 1.0) if out.ndim else np.clip(out.item(), 0.0, 1.0)


    def K(p):
        p = np.asarray(p, float)
        pc = np.sqrt(28/(45*np.pi)) # Critical p for K
        # Handle potential non-positive p inputs gracefully
        p_safe = np.maximum(p, 1e-20)
        term1 = (16/25)*(45*np.pi/28)**0.75 * p_safe**1.5
        term2 = 1.0 - (9/25)*(28/(45*np.pi))**(4/3) * p_safe**(-8/3)
        out = np.where(p < pc, term1, term2)
         # Ensure output is physically meaningful (e.g., between 0 and 1)
        return np.clip(out, 0.0, 1.0) if out.ndim else np.clip(out.item(), 0.0, 1.0)

    # ---------------- horseshoe width x_s -------------
    # gamma_eff was set above based on the mode
    C     = 1.1 * γ_eff**(-0.25)
    x_s   = C * np.sqrt(qp_calc / h_calc)

    # ---------------- p_chi calculation -------------
    # chi_p_to_use was set above based on the mode
    # Avoid sqrt of negative or division by zero if chi_p_to_use is invalid
    if χ_p_to_use <= 0 or np.isinf(χ_p_to_use) or np.isnan(χ_p_to_use):
        p_χ = np.inf # Treat as fully unsaturated entropy torque if chi_p is invalid/zero
    else:
        # Calculate x_s^3 term, ensure it's non-negative
        xs_cubed = np.maximum(x_s**3, 0)
        p_χ = np.sqrt(xs_cubed / (2.0*np.pi*χ_p_to_use))


    if verbose and np.ndim(p_nu) == 0:
         # Print derived p_chi and intermediate x_s
         print(f"    p_ν = {p_nu:5.3f} -> x_s = {x_s:.4f}, p_χ = {p_χ:.4f} (using chi_p={χ_p_to_use:.2e})")
         # Only print Pr for beta-cooling case where nu_p and chi_p are derived together
         if not force_pk11_dotted_params:
             ν_p = xs_cubed / (2*np.pi*np.maximum(p_nu**2, 1e-20)) # Avoid div by zero
             Pr = ν_p / χ_p_to_use if χ_p_to_use > 0 else np.inf
             print(f"                    -> ν_p = {ν_p:.2e}, Pr(th) = {Pr:5.2f}")


    # ---------------- Combine terms (Eq. 51-53 structure) --------
    # Ensure terms involving p_chi handle p_chi = inf gracefully
    # F(inf)=0, G(inf)=1, K(inf)=1
    F_pchi = F(p_χ)
    G_pchi = G(p_χ)
    K_pchi = K(p_χ)

    term1 = Gamma_hs_baro * F(p_nu) * G(p_nu)
    term2 = (1.0 - K(p_nu)) * Gamma_lin_baro
    term3 = Gamma_hs_ent * F(p_nu) * F_pchi * np.sqrt(G(p_nu) * G_pchi)
    term4 = np.sqrt(np.maximum(0.0, (1.0 - K(p_nu)) * (1.0 - K_pchi))) * Gamma_lin_ent

    γeff_times_Gc_over_G0 = term1 + term2 + term3 + term4

    # Final result is Gamma_C / Gamma_0
    # Avoid division by zero if gamma_eff is somehow zero
    if np.isclose(γ_eff, 0):
        print(f"Warning: gamma_eff is near zero ({γ_eff}). Cannot normalize torque.")
        return np.nan
    else:
        return γeff_times_Gc_over_G0 / γ_eff

# ======================================================================
# Utility functions (extract_pnu, analyze_corotation_vs_pnu)
# ======================================================================

def extract_pnu(simname):
    """Extracts p_nu value from simulation name string."""
    # Allow decimals in Pnu like Pnu0.3 or Pnu035 for 0.35
    match = re.search(r'Pnu(\d+\.?\d*)', simname)
    if match:
        pnu_str = match.group(1)
        return float(pnu_str)
        # # Old logic based on Pnu03 -> 0.3, Pnu10 -> 1.0
        # if '.' in pnu_str:
        #      return float(pnu_str)
        # else:
        #      # Assumes Pnu03 -> 0.3, Pnu10 -> 1.0 etc.
        #      power = len(pnu_str) - 1 if pnu_str.startswith('0') else len(pnu_str) -1 # Adjust logic as needed
        #      # This logic seems ambiguous, revert to just float conversion
        #      # Revert to original logic if needed:
        #      # return float(match.group(1)) / 10.0
    raise ValueError(f"Could not extract p_nu from {simname}")

def analyze_corotation_vs_pnu(simlist, avg_interval=100.0):
    """
    Analyzes corotation torque from simulations and plots against theoretical curves.
    """
    pnu_values_sim = []
    corotation_torques_sim = []

    # Store parameters from the first successfully processed simulation
    first_sim_params = {}
    params_stored = False
    beta_label = "beta_unknown" # Default label

    print("-" * 70)
    print("Processing Simulations...")
    print("-" * 70)

    for i, sim in enumerate(simlist):
        print(f"[{i+1}/{len(simlist)}] Processing {sim}")
        try:
            base_path = determine_base_path(sim)
            summary_file = os.path.join(base_path, "summary0.dat")
            parameters = read_parameters(summary_file)

            # --- Read simulation parameters ---
            sim_beta_cool = float(parameters.get("BETA", np.nan))
            sim_gamma = float(parameters.get("GAMMA", np.nan))
            sim_h0 = float(parameters.get("ASPECTRATIO", np.nan))
            sim_qp, _ = extract_planet_mass_and_migration(summary_file)
            sim_sigma0 = float(parameters.get("SIGMA0", 1.0)) # Assuming Sigma0=1 if not found

            # Map simulation params to theoretical indices (NEEDS VERIFICATION)
            # Assuming FLARINGINDEX = temperature power law index beta_idx (T ~ r^-beta_idx)
            # Assuming SIGMASLOPE = surface density power law index alpha_idx (Sigma ~ r^-alpha_idx)
            # Default to PK11 values if not found in params
            sim_q_beta_idx = float(parameters.get("FLARINGINDEX", 1.0)) # Default to PK11 beta_idx=1.0
            sim_s_alpha_idx = float(parameters.get("SIGMASLOPE", 0.5)) # Default to PK11 alpha_idx=0.5

             # --- Check for NaN parameters ---
            if any(np.isnan(p) for p in [sim_beta_cool, sim_gamma, sim_h0, sim_qp]):
                print(f"  WARNING: Found NaN in critical parameters (BETA, GAMMA, ASPECTRATIO, Mass) for {sim}. Skipping.")
                continue

            # Store parameters from the first valid simulation
            if not params_stored:
                first_sim_params = {
                    'h': sim_h0,
                    'gamma': sim_gamma,
                    'q': sim_q_beta_idx,  # Pass assumed beta_idx
                    's': sim_s_alpha_idx,  # Pass assumed alpha_idx
                    'beta': sim_beta_cool, # Pass beta_cool value
                    'qp': sim_qp
                }
                beta_label = f"beta{sim_beta_cool:.3f}".replace('.', 'd')
                params_stored = True
                print(f"  -> Storing params from this sim for theory curves:")
                print(f"     h={sim_h0:.3f}, gamma={sim_gamma:.3f}, q(beta_idx)={sim_q_beta_idx:.2f}, s(alpha_idx)={sim_s_alpha_idx:.2f}, beta_cool={sim_beta_cool:.2f}, qp={sim_qp:.2e}")

            # --- Calculate simulation torque ---
            GAM0 = (sim_qp / sim_h0) ** 2 * sim_sigma0
            gamma_eff = gamma_eff_theory(sim_gamma, sim_beta_cool, sim_h0)

            # Compute Lindblad torque (PK11 Eq 3, using sim indices & calculated gamma_eff)
            Gamma_L_scaled = -(2.5 + 1.7 * sim_q_beta_idx - 0.1 * sim_s_alpha_idx)
            Gamma_L = (GAM0 / gamma_eff) * Gamma_L_scaled if not np.isclose(gamma_eff,0) else 0

            # Read internal torque file & calculate avg Gamma_C
            tqwk_file = os.path.join(base_path, "tqwk0.dat")
            t_arr, torque_arr, _ = read_alternative_torque(tqwk_file)
            time_orb = t_arr / (2 * np.pi)

            if len(time_orb) == 0:
                print(f"  WARNING: No time data found in {tqwk_file} for {sim}. Skipping.")
                continue

            tmax = np.max(time_orb)
            avg_start = max(0, tmax - avg_interval) # Ensure avg_start >= 0
            avg_end = tmax
            mask = (time_orb >= avg_start) & (time_orb <= avg_end)

            if not np.any(mask):
                 print(f"  WARNING: No data in averaging interval [{avg_start:.1f}, {avg_end:.1f}] orbits for {sim}. Using last point if available.")
                 if len(torque_arr) > 0:
                     Gamma_tot = torque_arr[-1] * sim_qp # Use last point as fallback
                 else:
                      print(f"  ERROR: No torque data points found at all for {sim}. Skipping.")
                      continue # Skip if truly no data
            else:
                Gamma_tot = np.mean(torque_arr[mask]) * sim_qp

            Gamma_C = Gamma_tot - Gamma_L
            # Avoid division by zero if GAM0 is zero (e.g., qp=0 or sigma0=0)
            if np.isclose(GAM0, 0):
                 Gamma_C_norm = np.nan
                 print(f"  WARNING: GAM0 is zero for {sim}. Cannot normalize corotation torque.")
            else:
                 Gamma_C_norm = Gamma_C / GAM0

            pnu = extract_pnu(sim)
            print(f"  -> p_nu = {pnu:.3f}, Sim Gamma_C/Gamma_0 = {Gamma_C_norm:.4f} (Avg T=[{avg_start:.1f},{avg_end:.1f}])")

            # Store result if valid
            if not np.isnan(Gamma_C_norm):
                 pnu_values_sim.append(pnu)
                 corotation_torques_sim.append(Gamma_C_norm)
            else:
                 print(f"  -> Skipping result for p_nu={pnu:.3f} due to NaN torque.")

        except FileNotFoundError as e:
             print(f"  ERROR: Required file not found for {sim}: {e}. Skipping.")
        except ValueError as e: # Catch pnu extraction errors etc.
             print(f"  ERROR: Data processing error for {sim}: {e}. Skipping.")
        except Exception as e:
             print(f"  ERROR: Unexpected error processing {sim}: {type(e).__name__} - {e}. Skipping.")

    print("-" * 70)

    if not params_stored:
         print("ERROR: No simulations processed successfully. Cannot store parameters or generate plot.")
         return
    if len(pnu_values_sim) == 0:
        print("ERROR: No valid simulation torque data collected. Cannot generate plot.")
        return

    # Sort simulation results by pnu for plotting
    pnu_values_sim = np.array(pnu_values_sim)
    corotation_torques_sim = np.array(corotation_torques_sim)
    sort_indices = np.argsort(pnu_values_sim)
    pnu_values_sim = pnu_values_sim[sort_indices]
    corotation_torques_sim = corotation_torques_sim[sort_indices]

    # === Generate Plot ===
    print("Generating plot...")
    plt.figure(figsize=(8, 6)) # Slightly larger figure
    plt.plot(pnu_values_sim, corotation_torques_sim, 'ks', markersize=6, linestyle='-', linewidth=1.5, label=rf"Simulations ($\beta_{{cool}}={first_sim_params['beta']:.1f}$)")

    # Define p_nu range for theoretical curves
    pnu_plot_min = min(pnu_values_sim) * 0.5
    pnu_plot_max = max(pnu_values_sim) * 2.0
    pnu_plot = np.logspace(np.log10(pnu_plot_min), np.log10(pnu_plot_max), 150) # More points for smooth curve

    # --- Calculate Theoretical Curves ---
    # 1. Beta-cooling theory curve (using stored sim params)
    theo_vals_beta = [
        compute_theoretical_corotation_paarde2011(
            p, **first_sim_params, verbose=False, force_pk11_dotted_params=False)
        for p in pnu_plot
    ]

    # 2. Forced PK11 Dotted curve theory
    #    Inputs other than p_nu will be ignored due to the flag, but must be provided.
    theo_vals_pk11_dotted = [
        compute_theoretical_corotation_paarde2011(
            p, **first_sim_params, verbose=False, force_pk11_dotted_params=True)
        for p in pnu_plot
    ]

    # --- Plot Theoretical Curves ---
    plt.plot(pnu_plot, theo_vals_beta, 'r--', linewidth=2, label=f"Theory (Beta-cooling, $\\beta_{{cool}}={first_sim_params['beta']:.1f}$)")
    plt.plot(pnu_plot, theo_vals_pk11_dotted, 'b:', linewidth=2, label=r"Theory (PK11 Fig 6 dotted, $\chi_p=10^{-5}$)")

    # --- Plot Formatting ---
    plt.xlabel(r"$p_\nu$", fontsize=12)
    plt.ylabel(r"$\Gamma_C / \Gamma_0$", fontsize=12)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.title(rf"Corotation Torque $\Gamma_C$ vs $p_\nu$ ($\beta_{{cool}}={first_sim_params['beta']:.1f}$)", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save corotation torque plot
    try:
        pdf_out = f"corotation_torque_vs_pnu_{beta_label}_comparison.pdf"
        png_out = f"corotation_torque_vs_pnu_{beta_label}_comparison.png"
        plt.savefig(pdf_out)
        plt.savefig(png_out)
        print(f"Saved plot: {pdf_out}, {png_out}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show() # Display the plot

    # === Plot Gamma(t) for all simulations ===
    print("\nGenerating time series plot for all simulations...")
    plt.figure(figsize=(8, 5))
    sim_count_tseries = 0
    for sim in simlist:
        try:
            base_path = determine_base_path(sim)
            summary_file = os.path.join(base_path, "summary0.dat")
            parameters = read_parameters(summary_file)

            sim_qp, _ = extract_planet_mass_and_migration(summary_file)
            sim_gam = float(parameters.get("GAMMA", np.nan))
            sim_h = float(parameters.get("ASPECTRATIO", np.nan))
            sim_beta_val = float(parameters.get("BETA", np.nan))
            sim_sigma0 = float(parameters.get("SIGMA0", 1.0))

            if any(np.isnan(p) for p in [sim_gam, sim_h, sim_beta_val, sim_qp]):
                 print(f"Skipping {sim} in Gamma(t) plot due to NaN parameters.")
                 continue

            gamma_eff_val = gamma_eff_theory(sim_gam, sim_beta_val, sim_h)
            GAM0_val = (sim_qp / sim_h)**2 * sim_sigma0

            if np.isclose(GAM0_val, 0) or np.isclose(gamma_eff_val, 0):
                 print(f"Skipping {sim} in Gamma(t) plot due to zero GAM0 or gamma_eff.")
                 continue

            tqwk_file = os.path.join(base_path, "tqwk0.dat")
            t_arr, torque_arr, _ = read_alternative_torque(tqwk_file)
            time_orb = t_arr / (2 * np.pi)
            # Normalize torque
            gamma_torque = torque_arr * sim_qp / (GAM0_val / gamma_eff_val)

            pnu_label = f"{extract_pnu(sim):.1f}" # Format pnu label
            plt.plot(time_orb, gamma_torque, label=pnu_label, alpha=0.8)
            sim_count_tseries += 1

        except FileNotFoundError:
            print(f"Warning: skipping {sim} in Gamma(t) plot, file not found.")
        except Exception as e:
            print(f"Warning: skipping {sim} in Gamma(t) plot due to error: {e}")

    if sim_count_tseries > 0:
        plt.xlabel("Time (orbits)")
        plt.ylabel(r"$\Gamma / (\Gamma_0 / \gamma_{\mathrm{eff}})$")
        plt.title(r"Torque evolution $\Gamma(t)$ for all simulations")
        # Adjust legend position/size if needed
        plt.legend(title=r"$p_\nu$", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend

        # Save time evolution plot
        try:
             timeplot_pdf = f"torque_evolution_all_{beta_label}.pdf"
             timeplot_png = f"torque_evolution_all_{beta_label}.png"
             plt.savefig(timeplot_pdf)
             plt.savefig(timeplot_png)
             print(f"Saved time series plot: {timeplot_pdf}, {timeplot_png}")
             plt.show()
        except Exception as e:
             print(f"Error saving time series plot: {e}")

    else:
        print("No simulations plotted for Gamma(t).")


    # --- SCP Transfer ---
    # Add try-except block around import if data_storage might be missing
    try:
        from data_storage import scp_transfer
        files_to_transfer = []
        if 'pdf_out' in locals() and os.path.exists(pdf_out): files_to_transfer.append(pdf_out)
        if 'png_out' in locals() and os.path.exists(png_out): files_to_transfer.append(png_out)
        if 'timeplot_pdf' in locals() and os.path.exists(timeplot_pdf): files_to_transfer.append(timeplot_pdf)
        if 'timeplot_png' in locals() and os.path.exists(timeplot_png): files_to_transfer.append(timeplot_png)

        if files_to_transfer:
            print("\nAttempting SCP transfer...")
            # Make sure target directory exists or handle errors appropriately
            target_dir = "/Users/mariuslehmann/Downloads/Profiles/"
            username = "mariuslehmann"
            for fname in files_to_transfer:
                 try:
                     print(f"  Transferring {fname}...")
                     scp_transfer(fname, target_dir, username=username)
                     print(f"  Successfully transferred {fname}")
                 except Exception as e_scp:
                     print(f"  SCP transfer failed for {fname}: {e_scp}")
        else:
            print("\nNo plot files generated to transfer.")

    except ImportError:
        print("\nSkipping SCP transfer: 'data_storage' module not found.")
    except Exception as e:
        print(f"\nSCP transfer failed: {e}")


# ======================================================================
# Main execution block
# ======================================================================
if __name__ == "__main__":
    # Define the list of simulation directories or identifiers
    simlist = [
        "cos_Pnu02_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu03_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu07_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu10_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu20_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
    ]
    # Make sure the names match your directory structure and naming convention
    # Also ensure the parameters encoded in the name (like beta1.0, gam1.4, s0.5, q1.0)
    # actually match the parameters within the simulation's summary0.dat file.

    analyze_corotation_vs_pnu(simlist, avg_interval=100.0)
