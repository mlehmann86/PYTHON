import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# Function: torque from given chi_p and p_nu
# ------------------------------------------------------------------------
def compute_theoretical_corotation_given_chip(p_nu, chi_p, *, h, gamma, q, s, qp, verbose=False):
    gamma_eff = gamma

    xi = q - (gamma - 1.0) * s
    Gamma_hs_baro = 1.1 * (1.5 - s)
    Gamma_lin_baro = 0.7 * (1.5 - s)
    Gamma_hs_ent = 7.9 * xi / gamma
    Gamma_lin_ent = (2.2 - 1.4 / gamma) * xi

    C = 1.1 * gamma_eff**(-0.25)
    x_s = C * np.sqrt(qp / h)
    xs_cubed = x_s**3

    denom = 2.0 * np.pi * chi_p
    p_chi = np.sqrt(xs_cubed / denom) if denom > 0 else np.inf

    F = lambda p: 1.0 / (1.0 + (p / 1.3)**2)

    def G(p):
        p = np.asarray(p, float)
        pc = np.sqrt(8 / (45 * np.pi))
        p_safe = np.maximum(p, 1e-20)
        term1 = (16 / 25) * (45 * np.pi / 8)**0.75 * p_safe**1.5
        term2 = 1.0 - (9 / 25) * (8 / (45 * np.pi))**(4 / 3) * p_safe**(-8 / 3)
        return np.where(p < pc, term1, term2).clip(0.0, 1.0)

    def K(p):
        p = np.asarray(p, float)
        pc = np.sqrt(28 / (45 * np.pi))
        p_safe = np.maximum(p, 1e-20)
        term1 = (16 / 25) * (45 * np.pi / 28)**0.75 * p_safe**1.5
        term2 = 1.0 - (9 / 25) * (28 / (45 * np.pi))**(4 / 3) * p_safe**(-8 / 3)
        return np.where(p < pc, term1, term2).clip(0.0, 1.0)

    F_pchi = F(p_chi)
    G_pchi = G(p_chi)
    K_pchi = K(p_chi)

    term1 = Gamma_hs_baro * F(p_nu) * G(p_nu)
    term2 = (1.0 - K(p_nu)) * Gamma_lin_baro
    G_prod_sqrt = np.sqrt(np.maximum(0.0, G(p_nu) * G_pchi))
    K_prod_sqrt = np.sqrt(np.maximum(0.0, (1.0 - K(p_nu)) * (1.0 - K_pchi)))
    term3 = Gamma_hs_ent * F(p_nu) * F_pchi * G_prod_sqrt
    term4 = K_prod_sqrt * Gamma_lin_ent

    result = (term1 + term2 + term3 + term4) / gamma_eff

    if verbose:
        print(f"χ_p = {chi_p:.3e}, p_χ = {p_chi:.3f}, Γ_C/Γ₀ = {result:.3f}")

    return result

# ------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------
gamma = 1.4
h = 0.05
q = 1.0
s = 0.5
qp = 1.26e-5

# Horseshoe width
C = 1.1 * gamma**(-0.25)
x_s = C * np.sqrt(qp / h)
xs_sq = x_s**2

# chi_p scan and corresponding beta
chi_p_array = np.logspace(-6, -4, 100)
pnu_scan = np.logspace(-1, 1.2, 100)  # from 0.1 to ~16
peak_torques = []
beta_array = []

for chi_p in chi_p_array:
    beta = xs_sq / chi_p
    beta_array.append(beta)
    torque_vals = [
        compute_theoretical_corotation_given_chip(
            pnu, chi_p, h=h, gamma=gamma, q=q, s=s, qp=qp)
        for pnu in pnu_scan
    ]
    peak_torques.append(np.max(torque_vals))

# Identify max
peak_index = int(np.argmax(peak_torques))
chip_max = chi_p_array[peak_index]
gamma_max = peak_torques[peak_index]

# Find beta required for chi_p = 1e-5
target_chi_p = 1e-5
beta_target = xs_sq  / target_chi_p

print(f"\n→ Maximum torque: Γ_C/Γ₀ = {gamma_max:.3f} at χ_p = {chip_max:.3e}")
print(f"→ β required for χ_p = 1e-5: β = {beta_target:.3f}")

# ------------------------------------------------------------------------
# Plot: Two panels
# ------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

# Top: max torque vs chi_p
ax1.plot(chi_p_array, peak_torques, 'r-')
ax1.axvline(target_chi_p, color='gray', linestyle='--', label=r"$\chi_p = 10^{-5}$")
ax1.set_xscale('log')
ax1.set_ylabel(r"max $\Gamma_C / \Gamma_0$")
ax1.set_title(r"Peak corotation torque vs $\chi_p$")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend()

# Bottom: chi_p vs beta (inverted mapping)
ax2.plot(chi_p_array, beta_array, 'k-')
ax2.axvline(target_chi_p, color='gray', linestyle='--')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r"$\chi_p$")
ax2.set_ylabel(r"$\beta = x_s^2 h^2 / \chi_p$")
ax2.set_title(r"Mapping from $\chi_p$ to $\beta$")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("peak_torque_vs_chip_with_beta.pdf")
plt.show()
