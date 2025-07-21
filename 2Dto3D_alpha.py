import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters
from corotation_torque_pnu import gamma_eff_theory, gamma_eff_diffusion

def extract_pnu_from_alpha(simname, alpha_measured, IDEFIX=False, chi_simulation=False, beta_simulation=True):
    base = determine_base_path(simname, IDEFIX=IDEFIX)
    summary_file = os.path.join(base, "idefix.0.log" if IDEFIX else "summary0.dat")
    par = read_parameters(summary_file, IDEFIX=IDEFIX)

    if IDEFIX:
        qp = float(par.get("planetToPrimary", 0.0))
        h = float(par.get("h0", 0.05))
        gamma = float(par.get("gamma", 1.4))
        b_over_h = float(par.get("smoothing", 0.4)) / h
        β = float(par.get("beta", 1.0))
        chi = float(par.get("chi", 1.0e-5))
    else:
        try:
            qp, _ = extract_planet_mass_and_migration(summary_file)
            if qp == 0.0:
                qp = 1.26e-5
        except Exception:
            qp = 1.26e-5
        h = float(par.get("ASPECTRATIO", 0.05))
        gamma = float(par.get("GAMMA", 1.4))
        b_over_h = float(par.get("THICKNESSSMOOTHING", 0.4))
        β = float(par.get("BETA", 1.0))
        chi = float(par.get("CHI", 1.0e-5))

    # Compute nu_p from measured alpha
    nu_p = alpha_measured * h**2

    rp = 1.0
    Omega_p = 1.0

    # Compute gamma_eff
    if chi_simulation:
        gamma_eff = gamma_eff_diffusion(gamma, chi, h)
    elif beta_simulation:
        gamma_eff = gamma_eff_theory(gamma, β, h)
    else:
        gamma_eff = gamma

    # Compute horseshoe half-width
    xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / b_over_h) ** 0.25 * np.sqrt(qp / h)

    # Compute p_nu
    p_nu = (2.0 / 3.0) * np.sqrt((rp ** 2 * Omega_p * xs ** 3) / (2 * np.pi * nu_p))

    # === Debug output ===
    print(f"{simname:60s}")
    print(f"  gamma        = {gamma:.3f}")
    print(f"  beta         = {β:.3e}")
    print(f"  chi          = {chi:.3e}")
    print(f"  gamma_eff    = {gamma_eff:.5f}")
    print(f"  h            = {h:.5f}")
    print(f"  qp           = {qp:.3e}")
    print(f"  b_over_h     = {b_over_h:.3f}")
    print(f"  alpha_meas   = {alpha_measured:.3e}")
    print(f"  nu_p         = {nu_p:.3e}")
    print(f"  x_s          = {xs:.5f}")
    print(f"  p_nu         = {p_nu:.3f}")
    print()

    return p_nu


# ====== User-defined ======
simnames = [
    "cos_beta05_gam75_p05_q1_r0516_z05_NOPLANET_HR150_h005_2DAXI_nosth2Dto3D_NOISE",
    "cos_beta075_gam75_p05_q1_r0516_z05_NOPLANET_HR150_h0075_2DAXI_nosth2Dto3D_NOISE",
    "cos_beta1_gam75_p05_q1_r0516_z05_NOPLANET_HR150_h01_2DAXI_nosth2Dto3D_NOISE"
]
colors = ['blue', 'orange', 'green']
avg_orbit_start = None
avg_orbit_end = None
output_filename = "2Dto3D_compare_alpha_r_rmsvz_rmsvr_scaled.pdf"
# ===========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
alpha_vals = []
rms_vz_vals = []
rms_vr_vals = []

from corotation_torque_pnu import extract_pnu  # Add this near other imports

# Inside your for loop:
for simname, color in zip(simnames, colors):
    base_path = determine_base_path(simname)
    npz_path = os.path.join(base_path, f"{simname}_quantities.npz")
    summary_path = os.path.join(base_path, "summary0.dat")
    data = np.load(npz_path)
    params = read_parameters(summary_path)

    h = float(params['ASPECTRATIO'])
    time_array = data['time']
    alpha_r = data['alpha_r']
    rms_vz = data['rms_vz'] / h
    rms_vr = data['rms_vr'] / h

    alpha_vals.append(alpha_r)
    rms_vz_vals.append(rms_vz)
    rms_vr_vals.append(rms_vr)

    label = simname.split("HR150_")[-1].replace("_2DAXI_nosth2Dto3D_NOISE", "")

    # Determine averaging mask
    mask = ((time_array >= avg_orbit_start) & (time_array <= avg_orbit_end)) if avg_orbit_start and avg_orbit_end else np.ones_like(time_array, dtype=bool)
    mean_alpha = np.mean(alpha_r[mask])

    # NEW: extract p_nu value
    p_nu_val = extract_pnu_from_alpha(simname, mean_alpha, IDEFIX=False, beta_simulation=True)
    
    # α_r panel with p_nu in label
    ax1.plot(time_array, alpha_r, color=color,
             label=fr"$\alpha_r$ ({label}, $\langle\alpha\rangle$={mean_alpha:.2e}, $\langle p_\nu\rangle$={p_nu_val:.1f})")
    ax1.axhline(mean_alpha, color=color, linestyle='dashed')

    # RMS(vz) and RMS(vr) panel
    ax2.plot(time_array, rms_vz, color=color, linestyle='-', label=fr'RMS($v_z$)/$H_g$ ({label})')
    ax2.plot(time_array, rms_vr, color=color, linestyle='--', label=fr'RMS($v_r$)/$H_g$ ({label})')

# === Dynamic y-limits ===
def compute_log_range(data_list):
    vals = np.concatenate(data_list)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return vals.min() * 0.8, vals.max() * 1.2

alpha_min, alpha_max = compute_log_range(alpha_vals)
rms_all_vals = rms_vz_vals + rms_vr_vals
rms_min, rms_max = compute_log_range(rms_all_vals)

for ax, ymin, ymax, ylabel in zip((ax1, ax2),
                                  (alpha_min, rms_min),
                                  (alpha_max, rms_max),
                                  [r"$\alpha_r$", r"RMS($v_z$), RMS($v_r$) / $H_g$"]):
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='major', ls=':', color='gray', alpha=0.7)
    ax.grid(False, which='minor')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))

ax1.set_xlabel("Time [orbits]")
ax2.set_xlabel("Time [orbits]")
ax1.set_title(r"$\alpha_r$ evolution")
ax2.set_title(r"Radial & Vertical velocity RMS (scaled)")
ax1.legend(fontsize=9)
ax2.legend(fontsize=9, loc='lower left')

plt.tight_layout()
plt.savefig(output_filename)

# === SCP transfer ===
print(f"\nTransferring {output_filename} to local machine...")
scp_transfer(output_filename, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
