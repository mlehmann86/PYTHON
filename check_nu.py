import os
import re
import numpy as np

# Constants
# Compute xs from Equation (11)
gamma_eff = 1.4
qp = 1.26e-5
h = 0.05
b_over_h = 0.4

xs = 1.1  / (gamma_eff ** 0.25) * (0.4 / b_over_h) ** 0.25 * np.sqrt(qp / h)

print(f"Computed x_s = {xs:.6f}")
rp = 1.0
Omega_p = 1.0
prefactor = (2.0 / 3.0) * np.sqrt((rp**2 * Omega_p * xs**3) / (2 * np.pi))

# Simulation sets
pnu_values = [2, 3, 5, 7, 10, 14, 20]
betas = [1, 10, 100]

# Base paths to try
base_paths = [
    "/theory/lts/mlehmann/FARGO3D/outputs",
    "/tiara/home/mlehmann/data/FARGO3D/outputs"
]

def get_existing_path(subdir):
    for base in base_paths:
        full_path = os.path.join(base, subdir)
        if os.path.isdir(full_path):
            return full_path
    return None

def extract_nu_from_summary(filepath):
    try:
        with open(filepath) as f:
            for line in f:
                if line.strip().startswith("Nu"):
                    parts = line.split()
                    return float(parts[1])
    except Exception:
        return None
    return None

# Main loop
print(f"{'Sim Name':<65} {'Nu (from file)':>15} {'p_nu (expected)':>20} {'p_nu (calc)':>15}")
print("-" * 115)

for beta in betas:
    for p in pnu_values:
        pnu_target = p / 10  # because Pnu02 = 0.2, etc.
        simname = f"cos_Pnu{p:02d}_beta{beta}_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
        sim_path = get_existing_path(simname)
        if sim_path is None:
            print(f"{simname:<65} {'MISSING':>15}")
            continue
        summary_file = os.path.join(sim_path, "summary0.dat")
        nu_val = extract_nu_from_summary(summary_file)
        if nu_val is None:
            print(f"{simname:<65} {'NO SUMMARY':>15}")
            continue
        pnu_actual = prefactor * nu_val**(-0.5)
        print(f"{simname:<65} {nu_val:15.3e} {pnu_actual:20.3f} {pnu_actual:15.3e}")
