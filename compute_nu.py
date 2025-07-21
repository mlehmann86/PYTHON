"""
Compute viscosity parameter nu_p for given disk scale heights (h)

This script computes the required kinematic viscosity nu_p for a series of protoplanetary
disk models with different aspect ratios (h), assuming a fixed cooling parameter (beta),
adiabatic index (gamma), smoothing parameter (b_over_h), and a fixed value of the
dimensionless viscous parameter p_nu.

For each h, it calculates:
- The effective adiabatic index gamma_eff (including beta-cooling effects)
- The half-width of the horseshoe region xs
- The corresponding nu_p required to achieve the specified p_nu

Results are printed as a table.

References:
  - Paardekooper et al. 2011 (MNRAS, 410, 293)

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import numpy as np

def gamma_eff_theory(gamma, beta, h_val):
    Q = (2.0 * beta) / (3.0 * h_val)
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_term_inner = (1.0 + Q**2) / (1.0 + gamma**2 * Q**2)
        sqrt_term = np.sqrt(np.maximum(0.0, sqrt_term_inner))
        numerator = 2.0
        denominator = (1.0 + gamma * Q**2) / (1.0 + gamma**2 * Q**2) + sqrt_term
        gamma_eff_val = numerator / denominator
    return gamma_eff_val

# Fixed physical and model parameters
beta = 1.0
gamma = 1.4
b_over_h = 0.4
rp = 1.0
Omega_p = 1.0
qp = 1.26e-5               # Planet-to-star mass ratio
p_nu = 0.33                # Fixed p_nu

# Varying scale heights
h_values = [0.05, 0.075, 0.1]

print(f"{'h':>8} {'gamma_eff':>12} {'xs':>12} {'nu_p':>12}")
for h in h_values:
    gamma_eff = gamma_eff_theory(gamma, beta, h)
    xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / b_over_h) ** 0.25 * np.sqrt(qp / h)
    nu_p = (rp**2 * Omega_p * xs**3) / (2 * np.pi * (3.0 / 2.0) * p_nu**2)
    print(f"{h:8.3f} {gamma_eff:12.6f} {xs:12.6e} {nu_p:12.6e}")
