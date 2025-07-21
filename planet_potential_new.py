import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
G = 1.0  # Gravitational constant
H = 0.1  # Scale height
m_p = 2.86e-4  # Planet mass (Saturn mass in code units)
r_planet = 1.0  # Radial position of the planet
epsilon = 0.1 * H  # Smoothing length
zmin, zmax = -0.25 * H, 0.25 * H  # Vertical domain
r = 1.0  # Radial distance
z_t = 0.15 * H  # Transition height
sigma = 0.04 * H  # Decay scale
n = 2  # Steepness of the transition

# Point-Mass Potential
def point_mass_potential(z, r, r_planet, m_p, epsilon):
    dist = np.sqrt((r - r_planet)**2 + z**2 + epsilon**2)
    return -G * m_p / dist

# Composite Potential with Exponential Transition
def composite_potential_exp(z, r, r_planet, m_p, epsilon, z_t, sigma, n):
    phi_pm = point_mass_potential(z, r, r_planet, m_p, epsilon)
    phi_pm_max = point_mass_potential(zmax, r, r_planet, m_p, epsilon)*1.2
    if abs(z) < z_t:
        return phi_pm
    else:
        transition = np.exp(-((abs(z) - z_t) / sigma)**n)
        return transition * phi_pm + (1 - transition) * phi_pm_max

# Derivative of the Composite Potential (Force)
def composite_force_exp(z, r, r_planet, m_p, epsilon, z_t, sigma, n):
    phi_pm = point_mass_potential(z, r, r_planet, m_p, epsilon)
    phi_pm_max = point_mass_potential(zmax, r, r_planet, m_p, epsilon)
    dist = np.sqrt((r - r_planet)**2 + z**2 + epsilon**2)
    force_pm = G * m_p * z / dist**3  # Force from point-mass potential
    if abs(z) < z_t:
        return force_pm
    else:
        transition = np.exp(-((abs(z) - z_t) / sigma)**n)
        d_transition = -n * ((abs(z) - z_t) / sigma)**(n - 1) * transition / sigma
        return (
            transition * force_pm
            + d_transition * np.sign(z) * (phi_pm - phi_pm_max)
        )

# Generate z values for plotting
z_values = np.linspace(zmin, zmax, 500)

# Compute potentials
point_mass_potentials = [point_mass_potential(z, r, r_planet, m_p, epsilon) for z in z_values]
composite_potentials = [
    composite_potential_exp(z, r, r_planet, m_p, epsilon, z_t, sigma, n) for z in z_values
]

# Compute forces
point_mass_forces = [
    G * m_p * z / (np.sqrt((r - r_planet)**2 + z**2 + epsilon**2))**3 for z in z_values
]
composite_forces = [
    composite_force_exp(z, r, r_planet, m_p, epsilon, z_t, sigma, n) for z in z_values
]

# Plot the results
fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Top panel: Potentials
axs[0].plot(z_values, point_mass_potentials, label="Point-Mass Potential", linestyle="--", color="blue")
axs[0].plot(z_values, composite_potentials, label="Composite Potential", linestyle="-", color="green")
axs[0].axhline(point_mass_potential(zmax, r, r_planet, m_p, epsilon), color="red", linestyle=":", label="Constant Potential")
axs[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[0].set_ylabel("Potential (Φ)")
axs[0].set_title(f"Composite Potential with Smooth Exponential Transition (r = {r})")
axs[0].legend()
axs[0].grid(True)

# Bottom panel: Forces
axs[1].plot(z_values, point_mass_forces, label="Point-Mass Force", linestyle="--", color="blue")
axs[1].plot(z_values, composite_forces, label="Composite Force", linestyle="-", color="green")
axs[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[1].set_xlabel("z (Vertical Distance)")
axs[1].set_ylabel("Force (dΦ/dz)")
axs[1].set_title("Force Derived from Composite Potential")
axs[1].legend()
axs[1].grid(True)

# Save the plot as a PDF
output_pdf_path = "exponential_transition_potential.pdf"
plt.savefig(output_pdf_path)
plt.close()

print(f"Plot saved as: {output_pdf_path}")
