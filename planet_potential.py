import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Constants and parameters
G = 1.0  # Gravitational constant
H = 0.1  # Scale height
m_p = 2.86e-4  # Planet mass (Saturn mass in code units)
r_planet = 1.0  # Radial position of the planet
epsilon = 0.1 * H  # Smoothing length
zmin, zmax = -0.25 * H, 0.25 * H  # Vertical domain
r = 1.0  # Radial distance
k = 10  # Steepness of the transition
phi_constant = -G * m_p / np.sqrt((r - r_planet)**2 + epsilon**2)  # Potential value at z=0

# Point-Mass Potential
def point_mass_potential(z, r, r_planet, m_p, epsilon):
    dist = np.sqrt((r - r_planet)**2 + z**2 + epsilon**2)
    return -G * m_p / dist

# Derivative of the Point-Mass Potential
def point_mass_force(z, r, r_planet, m_p, epsilon):
    dist = np.sqrt((r - r_planet)**2 + z**2 + epsilon**2)
    return G * m_p * z / dist**3

# Transition Function (Numerically Stable)
def transition_function(z, z_smooth, k):
    exp_value = np.exp(-k * (np.abs(z) - np.abs(z_smooth)))
    return 1 / (1 + np.clip(exp_value, 1e-10, 1e10))

# Derivative of the Transition Function
def transition_function_derivative(z, z_smooth, k):
    T = transition_function(z, z_smooth, k)
    return k * T * (1 - T) * np.sign(z)

# Composite Potential
def composite_potential(z, r, r_planet, m_p, epsilon, z_smooth, k, phi_constant):
    phi_point = point_mass_potential(z, r, r_planet, m_p, epsilon)
    T = transition_function(z, z_smooth, k)
    return (1 - T) * phi_point + T * phi_constant

# Derivative of the Composite Potential
def composite_force(z, r, r_planet, m_p, epsilon, z_smooth, k, phi_constant):
    phi_point = point_mass_potential(z, r, r_planet, m_p, epsilon)
    force_point = point_mass_force(z, r, r_planet, m_p, epsilon)
    T = transition_function(z, z_smooth, k)
    T_derivative = transition_function_derivative(z, z_smooth, k)
    return (1 - T) * force_point + T_derivative * (phi_constant - phi_point)

# Function to solve for z_smooth
def force_zero_condition(z_smooth, zmax, r, r_planet, m_p, epsilon, k, phi_constant):
    force = composite_force(zmax, r, r_planet, m_p, epsilon, np.abs(z_smooth), k, phi_constant)
    print(f"z_smooth: {z_smooth}, Force at zmax: {force}")
    return force

# Solve for z_smooth using Newton's method
z_smooth_initial_guess = 0.01
try:
    z_smooth_solution = newton(
        force_zero_condition, 
        z_smooth_initial_guess, 
        args=(zmax, r, r_planet, m_p, epsilon, k, phi_constant),
        maxiter=100  # Limit iterations
    )
    z_smooth_solution = np.abs(z_smooth_solution)  # Ensure positive z_smooth
    print(f"Solved z_smooth: {z_smooth_solution}")
except RuntimeError as e:
    print(f"Solver failed: {e}")
    z_smooth_solution = z_smooth_initial_guess  # Use initial guess if solver fails
    print("Using initial guess for z_smooth.")

# Generate z values for plotting
z_values = np.linspace(zmin, zmax, 500)

# Compute potentials for the initial guess
composite_potentials_initial = [
    composite_potential(z, r, r_planet, m_p, epsilon, z_smooth_initial_guess, k, phi_constant)
    for z in z_values
]

# Compute potentials for the solved z_smooth
composite_potentials_solution = [
    composite_potential(z, r, r_planet, m_p, epsilon, z_smooth_solution, k, phi_constant)
    for z in z_values
]

# Compute point-mass potentials
point_mass_potentials = [point_mass_potential(z, r, r_planet, m_p, epsilon) for z in z_values]

# Compute forces for the initial guess
composite_forces_initial = [
    composite_force(z, r, r_planet, m_p, epsilon, z_smooth_initial_guess, k, phi_constant)
    for z in z_values
]

# Compute forces for the solved z_smooth
composite_forces_solution = [
    composite_force(z, r, r_planet, m_p, epsilon, z_smooth_solution, k, phi_constant)
    for z in z_values
]

# Create the figure with two panels
fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Top panel: Potentials
axs[0].plot(z_values, point_mass_potentials, label="Point-Mass Potential", linestyle="--", color="blue")
axs[0].plot(z_values, composite_potentials_initial, label="Composite Potential (Initial)", linestyle="--", color="orange")
axs[0].plot(z_values, composite_potentials_solution, label="Composite Potential (Solved)", linestyle="-", color="green")
axs[0].axhline(phi_constant, color="red", linestyle=":", label="Constant Potential")
axs[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[0].set_ylabel("Potential (Φ)")
axs[0].set_title(f"Composite Potentials for Initial and Solved z_smooth (r = {r})")
axs[0].legend()
axs[0].grid(True)

# Bottom panel: Forces
axs[1].plot(z_values, composite_forces_initial, label="Composite Force (Initial)", linestyle="--", color="orange")
axs[1].plot(z_values, composite_forces_solution, label="Composite Force (Solved)", linestyle="-", color="green")
axs[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[1].set_xlabel("z (Vertical Distance)")
axs[1].set_ylabel("Force (dΦ/dz)")
axs[1].set_title("Forces Derived from Composite Potentials")
axs[1].legend()
axs[1].grid(True)

# Save the plot as a PDF
output_pdf_path = "composite_potential_with_forces_comparison.pdf"
plt.savefig(output_pdf_path)
plt.close()

print(f"Plot saved as: {output_pdf_path}")
