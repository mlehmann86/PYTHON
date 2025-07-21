#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse

# --- Import functions from your existing script ---
try:
    from planet_data import (
        determine_base_path,
        read_parameters,
        extract_planet_mass_and_migration,
        read_alternative_torque,
        compute_theoretical_torques
    )
    print("Successfully imported functions from planet_data.py")
except ImportError as e:
    print(f"Error: Could not import functions from 'planet_data.py': {e}")
    exit()

# --- Main Analysis Function ---

def analyze_corotation_torque_estimate(simulation_list,
                                       peak_time_window_orbits=(5, 100),
                                       late_time_window_orbits=(800, 1000),
                                       smoothing_time_orbits=1.0,
                                       IDEFIX=False):
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    beta_values = []
    peak_ct_values = []   # normalized corotation torque
    amplitude_values = [] # late-time oscillation amplitude

    # store theoretical reference lines
    hline_lin_ct = np.nan
    hline_nonlin_ct = np.nan
    GAM0_ref = np.nan
    ref_gam = np.nan
    first_sim = True

    for sim_name in simulation_list:
        print(f"\nProcessing: {sim_name}")
        try:
            base_path = determine_base_path(sim_name, IDEFIX=IDEFIX)
            parameters = read_parameters(
                os.path.join(base_path, "summary0.dat"), IDEFIX=IDEFIX)

            # cooling-time beta
            beta = float(parameters.get('BETA', parameters.get('beta', np.nan)))

            # planet mass qp
            qp, _ = extract_planet_mass_and_migration(
                os.path.join(base_path, "summary0.dat"))
            if qp == 0:
                raise ValueError("Planet mass qp is zero.")

            # Theoretical β_c estimate
            h = float(parameters.get('ASPECTRATIO', 0.05))
            gam = float(parameters.get('GAMMA', parameters.get('gamma', 1.6667)))
            b_over_h = float(parameters.get('THICKNESSSMOOTHING', 0.4))
            C = 1.1 * gam**(-0.25) * (0.4 / b_over_h)**(-0.25)
            x_s = C * np.sqrt(qp / h)
            beta_c_theory = 4 * np.pi / (3 * x_s)
            print(f"  Extracted qp: {qp:.3e}, Theoretical β_c ≃ {beta_c_theory:.2f}")

            # read torque
            t, torque, _ = read_alternative_torque(
                os.path.join(base_path, 'tqwk0.dat'), IDEFIX=IDEFIX)
            time_orb = t if IDEFIX else t / (2.0 * np.pi)

            # compute reference lines once
            if first_sim:
                _, _, GAM0_ref = compute_theoretical_torques(
                    parameters, qp, eq_label='Equation14', IDEFIX=IDEFIX)
                ref_gam = gam
                # linear corotation
                Gam_L, _, _ = compute_theoretical_torques(
                    parameters, qp, eq_label='Equation14', IDEFIX=IDEFIX)
                Gam_lin, _, _ = compute_theoretical_torques(
                    parameters, qp, eq_label='Equation18', IDEFIX=IDEFIX)
                # nonlinear horseshoe drag
                Gam_nl, _, _ = compute_theoretical_torques(
                    parameters, qp, eq_label='Equation45', IDEFIX=IDEFIX)
                hline_lin_ct    = (Gam_lin - Gam_L) / GAM0_ref * ref_gam
                hline_nonlin_ct = (Gam_nl  - Gam_L) / GAM0_ref * ref_gam
                first_sim = False

            # late-time Lindblad average
            t_late_start, t_late_end = late_time_window_orbits
            mask_late = (time_orb >= t_late_start) & (time_orb <= t_late_end)
            gamma_L = np.mean(torque[mask_late])

            # smooth torque
            dt = np.mean(np.diff(time_orb))
            win = max(1, int(smoothing_time_orbits / dt))
            smooth = uniform_filter1d(torque, size=win, mode='nearest')
            norm_smooth = smooth / GAM0_ref * qp * ref_gam

            # early-time extremum corotation
            t_peak_start, t_peak_end = peak_time_window_orbits
            mask_peak = (time_orb >= t_peak_start) & (time_orb <= t_peak_end)
            A = np.max(norm_smooth[mask_peak])
            B = np.min(norm_smooth[mask_peak])
            gamma_L_norm = gamma_L / GAM0_ref * qp * ref_gam
            diffA = A - gamma_L_norm
            diffB = B - gamma_L_norm
            gamma_C = diffA if abs(diffA) >= abs(diffB) else diffB

            # late-time oscillation amplitude (e.g. 40 to late end)
            amp_start = 40.0
            mask_amp = (time_orb >= amp_start) & (time_orb <= t_late_end)
            if np.any(mask_amp):
                resid = norm_smooth[mask_amp] - gamma_L_norm
                amplitude = 0.5 * (resid.max() - resid.min())
            else:
                amplitude = np.nan

            beta_values.append(beta)
            peak_ct_values.append(gamma_C)
            amplitude_values.append(amplitude)
            print(f"  β={beta:.1f}, CT_est={gamma_C:.3f}, Amp={amplitude:.3f}")

        except Exception as e:
            print(f"Error with {sim_name}: {e}")

    # sort results
    idx = np.argsort(beta_values)
    betas = np.array(beta_values)[idx]
    CTs   = np.array(peak_ct_values)[idx]
    amps  = np.array(amplitude_values)[idx]

    # measured β_c
    plateau = np.max(np.abs(CTs))
    normCT = np.abs(CTs) / plateau
    beta_c_meas = np.interp(0.5, normCT, betas)
    print(f"Measured β_c ≃ {beta_c_meas:.2f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

    # Top: CT vs β
    ax1.plot(betas, CTs, 's--', color='purple', label='Simulated CT')
    ax1.axvline(beta_c_meas, color='black', ls=':', label=r'Measured $\beta_c$')
    ax1.axvline(beta_c_theory, color='grey', ls='--', label=r'Theoretical $\beta_c$')
    ax1.axhline(hline_lin_ct, color='red', ls='--', label=r'Linear corotation')
    ax1.axhline(hline_nonlin_ct, color='green', ls='--', label=r'Nonlinear corotation')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$\gamma\Gamma_C/\Gamma_0$')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()

    # Bottom: oscillation amplitude vs β
    ax2.plot(betas, amps, 'o-', color='blue',
             label=r'Amplitude of $\gamma\Gamma_C/\Gamma_0$')
    ax2.axhline(hline_lin_ct, color='red', ls='--', label=r'Linear corotation')
    ax2.axhline(hline_nonlin_ct, color='green', ls='--', label=r'Nonlinear corotation')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Cooling time $\beta$')
    ax2.set_ylabel(r'Amplitude of $\gamma\Gamma_C/\Gamma_0$')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    fig.savefig('peak_corotation_vs_beta.png', dpi=150)
    fig.savefig('peak_corotation_vs_beta.pdf')
    print("Plots saved: peak_corotation_vs_beta.{png,pdf}")


if __name__ == '__main__':
    sims = [
        "cos_bet1dm4_gam53_ssm20_q16_r0516_nur1dm6_COR_HR150_2D"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--simulations', nargs='+', default=sims)
    parser.add_argument('--idefix', action='store_true')
    parser.add_argument('--t_peak_start', type=float, default=5.0)
    parser.add_argument('--t_peak_end', type=float, default=100.0)
    parser.add_argument('--t_late_start', type=float, default=400.0)
    parser.add_argument('--t_late_end', type=float, default=800.0)
    parser.add_argument('--smooth', type=float, default=20.0)
    args = parser.parse_args()

    analyze_corotation_torque_estimate(
        args.simulations,
        peak_time_window_orbits=(args.t_peak_start, args.t_peak_end),
        late_time_window_orbits=(args.t_late_start, args.t_late_end),
        smoothing_time_orbits=args.smooth,
        IDEFIX=args.idefix
    )

