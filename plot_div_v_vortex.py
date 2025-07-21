import numpy as np
import matplotlib.pyplot as plt
from data_storage import scp_transfer
import os

# Constants
tau = 0.1
Omega = 1.0

# Define the functions for Eq. (9) and Eq. (15)
def omega_v(chi, Omega):
    return 3 * Omega / (2 * (chi - 1))

def div_v(chi, tau, Omega):
    omega_v_val = omega_v(chi, Omega)
    term1 = 2 * omega_v_val * ((chi**2 + 1) / chi)
    term2 = 2 * omega_v_val**2 + 3
    return -tau * Omega**2 * (term1 - term2)

# Generate chi values (avoiding singularity at chi = 1)
chi_values_left = np.linspace(1.01, 1.99, 250)  # chi < 2
chi_values_right = np.linspace(2.01, 10, 250)  # chi > 2

# Compute div v for each chi
div_v_values_left = div_v(chi_values_left, tau, Omega)
div_v_values_right = -div_v(chi_values_right, tau, Omega)

# Plot div v against chi
plt.figure(figsize=(8, 6))
plt.plot(chi_values_left, div_v_values_left, label=r"$\nabla \cdot v$ ($\chi < 2$)", color="blue")
plt.plot(chi_values_right, div_v_values_right, label=r"$-\nabla \cdot v$ ($\chi > 2$)", color="red", linestyle="--")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xscale("linear")
plt.yscale("log")
plt.xlabel(r"$\chi$", fontsize=14)
plt.ylabel(r"$\nabla \cdot v$", fontsize=14)
plt.title(r"Divergence of velocity vs $\chi$", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Save plot and transfer PDF
output_dir = "contours"
os.makedirs(output_dir, exist_ok=True)
pdf_filename = os.path.join(output_dir, "div_v_vs_chi_branches.pdf")
plt.savefig(pdf_filename)
plt.close()

print(f"Divergence of velocity plot saved to {pdf_filename}")
scp_transfer(pdf_filename, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")

