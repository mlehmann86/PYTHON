#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse

# --- Import functions from your existing script ---
try:
    # Adjust 'planet_data' if your script file has a different name
    from planet_data import (
        determine_base_path,
        read_parameters,
        extract_planet_mass_and_migration, # For FARGO qp extraction
        read_alternative_torque, # Reads total torque from tqwk0.dat
        compute_theoretical_torques # Import the original function
    )
    print("Successfully imported functions from planet_data.py")
except ImportError as e:
    print(f"Error: Could not import functions from 'planet_data.py': {e}")
    print("Please ensure the file exists and is in the Python path.")
    exit()

# --- Main Analysis Function ---

def analyze_corotation_torque_estimate(simulation_list,
                                       peak_time_window_orbits=(5, 100),
                                       late_time_window_orbits=(800, 1000), # Window for Gamma_L
                                       smoothing_time_orbits=1.0,
                                       IDEFIX=False):
    """
    Estimates peak corotation torque by subtracting late-time (Lindblad) torque
    from early peak total torque. Plots estimated gamma*CT/Gamma0 vs beta.
    """
    # Imports needed within this function
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    beta_values = []
    peak_ct_estimate_plot_values = [] # Store gamma * Gamma_C_est / Gamma0
    processed_sims = []

    hline_lin_ct = np.nan # gamma * Gamma_c,lin / Gamma0
    hline_nl_hsd = np.nan # gamma * Gamma_hs / Gamma0
    GAM0_ref = np.nan
    ref_gam = np.nan # Store gamma from first sim
    first_sim = True

    for sim_name in simulation_list:
        print(f"\nProcessing: {sim_name}")
        try:
            base_path = determine_base_path(sim_name, IDEFIX=IDEFIX)
            output_path = base_path
            print(f"  Base path: {base_path}")

            param_file = os.path.join(output_path, "idefix.0.log") if IDEFIX else os.path.join(output_path, "summary0.dat")
            torque_file = os.path.join(output_path, "tqwk0.dat") # Use this for both
            summary_file = os.path.join(output_path, "summary0.dat") # Needed for FARGO qp

            print(f"  Checking param file: {param_file}")
            print(f"  Checking torque file: {torque_file}")
            if not IDEFIX: print(f"  Checking summary file: {summary_file}")

            if not os.path.exists(param_file) or not os.path.exists(torque_file):
                raise FileNotFoundError("Missing parameter or torque file.")
            if not IDEFIX and not os.path.exists(summary_file):
                 raise FileNotFoundError("Missing FARGO summary file.")

            parameters = read_parameters(param_file, IDEFIX=IDEFIX)
            beta_str = parameters.get("beta") if IDEFIX else parameters.get("BETA")
            if beta_str is None: raise ValueError("Cooling time not found.")
            beta = float(beta_str)

            # --- Extract planet mass qp correctly ---
            if IDEFIX:
                qp_str = parameters.get("planetToPrimary")
                if qp_str is None: raise ValueError("Planet mass 'planetToPrimary' not found.")
                qp = float(qp_str)
            else: # FARGO
                qp, _ = extract_planet_mass_and_migration(summary_file)
            if qp == 0.0: raise ValueError("Planet mass qp is zero.")
            print(f"  Extracted qp: {qp}")

            # --- Theoretical critical cooling time β_c (t_lib = t_cool) ---
            if IDEFIX:
                h = parameters.get("h0", 0.05)
                gam = parameters.get("gamma", 1.4)
                b_over_h = parameters.get("smoothing", 0.6) / ASPECTRATIO

            else:
                h = float(parameters.get("ASPECTRATIO"))        # H/r_p
                gam = float(parameters['GAMMA'])
                b_over_h = float(parameters['THICKNESSSMOOTHING'])

            C = 1.1 * gam**(-0.25) * (0.4/b_over_h)**(-0.25)
            x_s = C * np.sqrt(qp / h)                      # horseshoe half–width (units of r_p)
            beta_c_theory = 4*np.pi / (3 * x_s)            # from t_lib = 4π/(3 Ω_p x_s), Ω_p=1
            print(f"  Theoretical β_c ≃ {beta_c_theory:.2f} (t_lib = t_cool)")

            # --- End Extract planet mass ---

            # Read TOTAL Torque Data using imported function
            # Pass IDEFIX flag to read_alternative_torque as it might affect column summing
            time, total_torque, orbit_numbers = read_alternative_torque(torque_file, IDEFIX=IDEFIX)
            if time.size < 2: raise ValueError("Not enough torque data points.")

            time_in_orbits = time if IDEFIX else time / (2.0 * np.pi)

            # Calculate Theoretical Values & Gamma0 (only once)
            if first_sim:
                print("  Calculating theoretical reference values...")
                # Use imported compute_theoretical_torques
                try:
                    # Get Gamma_L
                    Gamma_L_theory, _, GAM0_ref = compute_theoretical_torques(parameters, qp, eq_label="Equation14", IDEFIX=IDEFIX)
                    # Get Gamma_L + Gamma_c,lin
                    Gamma_Lin_total, _, _ = compute_theoretical_torques(parameters, qp, eq_label="Equation18", IDEFIX=IDEFIX)
                    # Get Gamma_L + Gamma_hs
                    Gamma_NL_total, _, _ = compute_theoretical_torques(parameters, qp, eq_label="Equation45", IDEFIX=IDEFIX)

                    if np.isnan(GAM0_ref) or GAM0_ref == 0: raise ValueError(f"GAM0_ref invalid: {GAM0_ref}")

                    ref_gam = float(parameters.get("gamma", 1.666667)) if IDEFIX else float(parameters.get("GAMMA", 1.666667))
                    if ref_gam == 0: raise ValueError("Gamma is zero.")

                    # Calculate corotation components (raw torque units)
                    Gamma_c_lin_theory = Gamma_Lin_total - Gamma_L_theory
                    Gamma_hs_theory = Gamma_NL_total - Gamma_L_theory

                    # Calculate normalized values for horizontal lines (gamma * Gamma / Gamma0)
                    hline_lin_ct = Gamma_c_lin_theory / GAM0_ref * ref_gam
                    hline_nl_hsd = Gamma_hs_theory / GAM0_ref * ref_gam

                    if not (np.isfinite(hline_lin_ct) and np.isfinite(hline_nl_hsd)):
                        print("Warning: Could not calculate finite theoretical lines.")
                        hline_lin_ct = np.nan; hline_nl_hsd = np.nan
                    else:
                        print(f"  Theoretical Gamma0 = {GAM0_ref:.4e}")
                        print(f"  Theoretical gamma*CT_lin/Gamma0 = {hline_lin_ct:.4f}")
                        print(f"  Theoretical gamma*HSD_nl/Gamma0 = {hline_nl_hsd:.4f}")

                except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                    print(f"Warning: Error calculating theoretical values: {e}. No theory lines plotted.")
                    hline_lin_ct = np.nan; hline_nl_hsd = np.nan; GAM0_ref = np.nan
                first_sim = False

            if np.isnan(GAM0_ref) or GAM0_ref == 0:
                 print(f"Skipping {sim_name} due to invalid GAM0_ref."); continue

            # --- Estimate Lindblad Torque (Late Time Average) ---
            t_late_start, t_late_end = late_time_window_orbits
            mask_late = (time_in_orbits >= t_late_start) & (time_in_orbits <= t_late_end)
            if not np.any(mask_late):
                print(f"Warning: No data in late window [{t_late_start}, {t_late_end}]. Using last point for Gamma_L.")
                # Fallback: use the last value? or skip? Let's use last smoothed value
                gamma_L_late = uniform_filter1d(total_torque, size=10, mode='nearest')[-1] if total_torque.size>0 else np.nan
            else:
                torque_late = total_torque[mask_late]
                if torque_late.size == 0: gamma_L_late = np.nan
                else: gamma_L_late = np.mean(torque_late)

            if not np.isfinite(gamma_L_late):
                print("Warning: Could not estimate late-time Lindblad torque. Skipping.")
                continue
            # print(f"  Estimated Gamma_L (late avg): {gamma_L_late:.4e} (Raw Units)")

            # --- Estimate Peak Total Torque (Early Window, Smoothed) ---
            dt_orbit = np.mean(np.diff(time_in_orbits)) if len(time_in_orbits) > 1 else 0
            if dt_orbit <= 0 or not np.isfinite(dt_orbit): smoothing_window_size = 10 # Fallback
            else: smoothing_window_size = max(10, int(smoothing_time_orbits / dt_orbit))
            smoothed_torque = uniform_filter1d(total_torque, size=smoothing_window_size, mode='nearest')
            # Normalize to gamma * Gamma / Gamma0 for plotting
            smoothed_torque = smoothed_torque / GAM0_ref * qp * ref_gam

            # ---- DEBUG PLOT HERE ----
            if sim_name == "cos_bet1d2_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D":
                plt.figure(figsize=(8, 4))
                plt.plot(time_in_orbits, smoothed_torque, label='Smoothed Torque')
                plt.xlabel('Time [orbits]')
                plt.ylabel('Smoothed Total Torque')
                plt.title(f'Smoothed Torque vs Time: {sim_name}')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()
                debug_fname = f"{sim_name}_smoothed_torque.pdf"
                plt.tight_layout()
                plt.savefig(debug_fname)
                print(f"Debug: saved smoothed-torque plot to {debug_fname}")
                plt.close()
            # ---- end debug plot ----

            t_peak_start, t_peak_end = peak_time_window_orbits
            # --- Estimate Peak Corotation Torque via extremum method ---
            # (A) = max(smoothed torque) in early window; (B) = min(...)
            t_peak_start, t_peak_end = peak_time_window_orbits
            mask_peak = (time_in_orbits >= t_peak_start) & (time_in_orbits <= t_peak_end)
            if not np.any(mask_peak):
                print(f"Warning: No data in early window [{t_peak_start}, {t_peak_end}]. Skipping.")
                continue

            early_torque = smoothed_torque[mask_peak]
            if early_torque.size == 0:
                continue

            A = np.max(early_torque)
            B = np.min(early_torque)

            # make sure Γ_L is normalized the same way as smoothed_torque:
            # (raw Lindblad average) → (γ Γ_L / Γ0)
            gamma_L_norm = gamma_L_late / GAM0_ref * qp * ref_gam

            diff_A = A - gamma_L_norm
            diff_B = B - gamma_L_norm

            # pick whichever has larger absolute deviation
            if abs(diff_A) >= abs(diff_B):
                gamma_C_estimate = diff_A
                which = 'max'
            else:
                gamma_C_estimate = diff_B
                which = 'min'

            print(f"  Chose '{which}' extreme: Δ = {gamma_C_estimate:.4f}")

            # store (already in normalized units)
            beta_values.append(beta)
            peak_ct_estimate_plot_values.append(gamma_C_estimate)
            processed_sims.append(sim_name)
            print(f"  Beta = {beta:.2e}, Est. Peak gamma*CT/Gamma0 = {gamma_C_estimate:.4f}")

        except FileNotFoundError as e: print(f"FNF Error for {sim_name}: {e}")
        except ValueError as e: print(f"Value Error for {sim_name}: {e}")
        except KeyError as e: print(f"Key Error processing {sim_name}: Missing key {e}")
        except Exception as e: print(f"General Error for {sim_name}: {e}")

    # --- Plotting ---
    if not beta_values:
        print("\nNo simulations processed successfully. Cannot create plot.")
        return

    # Imports for plotting
    import matplotlib.pyplot as plt

    beta_values = np.array(beta_values)
    peak_ct_estimate_plot_values = np.array(peak_ct_estimate_plot_values)
    sort_indices = np.argsort(beta_values)
    beta_values_sorted = beta_values[sort_indices]
    peak_ct_estimate_plot_values_sorted = peak_ct_estimate_plot_values[sort_indices]


 # ——— Estimate measured β_c from 50% point ———
 betas = beta_values_sorted
 ct_vals = peak_ct_estimate_plot_values_sorted
 plateau = np.max(np.abs(ct_vals))
 norm_ct = np.abs(ct_vals) / plateau

 if np.any(norm_ct >= 0.5):
     # interpolate to find β where normalized CT = 0.5
     beta_c_measured = np.interp(0.5, norm_ct, betas)
     print(f"Measured   β_c (50% of max HSD) ≃ {beta_c_measured:.2f}")


    plt.figure(figsize=(10, 6))
    plt.plot(beta_values_sorted, peak_ct_estimate_plot_values_sorted, marker='s', linestyle='--', color='purple',
             label=f'Est. Peak Sim. CT/HSD ($\gamma (\Gamma_{{peak}} - \Gamma_{{late}}) / \Gamma_0$)')

    # Plot theoretical lines (gamma * Gamma / Gamma0)
    if np.isfinite(hline_lin_ct):
        plt.axhline(hline_lin_ct, color='red', linestyle='--',
                    label=f'Theory Linear CT ($\gamma \Gamma / \Gamma_0 = {hline_lin_ct:.3f}$)')
        print(f"Plotting Theory Linear CT line at: {hline_lin_ct:.3f}")
    if np.isfinite(hline_nl_hsd):
        plt.axhline(hline_nl_hsd, color='green', linestyle=':',
                    label=f'Theory Non-linear HSD ($\gamma \Gamma / \Gamma_0 = {hline_nl_hsd:.3f}$)')
        print(f"Plotting Theory Non-linear HSD line at: {hline_nl_hsd:.3f}")

    plt.xscale('log')
    plt.xlabel(r'Cooling Time Parameter ($\beta = t_{cool} \Omega_p$)')
    plt.ylabel(r'Est. Peak Norm. Corotation Torque ($\gamma \Gamma_C / \Gamma_0$)') # Corrected Label
    plt.title('Estimated Peak Corotation Torque vs. Cooling Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plot_filename = "peak_corotation_torque_estimate_vs_beta.pdf"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    # plt.show()


# --- Script Execution ---
if __name__ == "__main__":
    simulation_list = [
        "cos_bet1d2_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
        "cos_bet1d4_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
        "cos_bet1d3_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
        "cos_bet1dm2_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
        "cos_bet1dm1_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
        "cos_bet1d1_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
        "cos_bet1d0_gam53_ss15_q2_r0516_nu1dm11_COR_HR150_2D",
    ]

    parser = argparse.ArgumentParser(description="Estimate peak corotation torque vs cooling time.")
    parser.add_argument('-s', '--simulations', nargs='+', default=simulation_list,
                        help="List of simulation directory names.")
    parser.add_argument('--idefix', action='store_true', default=False,
                        help="Flag if simulations use IDEFIX format.")
    parser.add_argument('--t_peak_start', type=float, default=5.0,
                        help="Start time (orbits) for peak torque window.")
    parser.add_argument('--t_peak_end', type=float, default=100.0,
                        help="End time (orbits) for peak torque window.")
    parser.add_argument('--t_late_start', type=float, default=400.0,
                        help="Start time (orbits) for late (Lindblad) torque window.")
    parser.add_argument('--t_late_end', type=float, default=800.0,
                        help="End time (orbits) for late (Lindblad) torque window.")
    parser.add_argument('--smooth', type=float, default=20.0,
                        help="Smoothing timescale (orbits) for torque.")

    args = parser.parse_args()

    # Call the correct analysis function name
    analyze_corotation_torque_estimate(
        args.simulations,
        peak_time_window_orbits=(args.t_peak_start, args.t_peak_end),
        late_time_window_orbits=(args.t_late_start, args.t_late_end),
        smoothing_time_orbits=args.smooth,
        IDEFIX=args.idefix
    )
