import numpy as np
import matplotlib.pyplot as plt

# Set constants
gamma = 1.6667
h = 0.05
Omega = 1.0

# Create ss and q grids
ss = np.linspace(-2.0, 2.0, 500)
q = np.linspace(-2.0, 2.5, 500)
SS, Q = np.meshgrid(ss, q, indexing='ij')

# Compute xi
xi = Q - (gamma - 1) * SS

# Compute N_r^2
Nr2 = -(1/gamma) * h**2 * Omega**2 * (SS + Q) * (Q + (1 - gamma) * SS)

# Compute vortensity gradient
vort_grad = 1.5 + (1 - 2/gamma) * SS - 2 * Q / gamma

# Define masks
stable_mask = Nr2 > 0
unstable_mask = Nr2 <= 0

# Mask xi where unstable
xi_masked = np.ma.masked_where(~stable_mask, np.abs(xi))

# Find ss(q) and xi(q) that maximize |xi| for each q
optimal_ss = []
optimal_xi = []
for j in range(len(q)):
    xi_column = xi[:, j]
    mask_column = stable_mask[:, j]
    if np.any(mask_column):
        idx = np.argmax(np.abs(xi_column * mask_column))
        optimal_ss.append(ss[idx])
        optimal_xi.append(xi_column[idx])
    else:
        optimal_ss.append(np.nan)
        optimal_xi.append(np.nan)

optimal_ss = np.array(optimal_ss)
optimal_xi = np.array(optimal_xi)

# Find global max and min xi along black curve
global_max_idx = np.nanargmax(optimal_xi)
global_min_idx = np.nanargmin(optimal_xi)

global_max_q = q[global_max_idx]
global_max_ss = optimal_ss[global_max_idx]
global_max_xi = optimal_xi[global_max_idx]

global_min_q = q[global_min_idx]
global_min_ss = optimal_ss[global_min_idx]
global_min_xi = optimal_xi[global_min_idx]

# --- Find intersection with vortensity zero line ---
# Compute ss_vort0(q) from analytic formula
ss_vort0 = (2/gamma) * q - 1.5
ss_vort0 = ss_vort0 / (1 - 2/gamma)

# Compute the difference between optimal_ss and vort0_ss
diff = optimal_ss - ss_vort0

# Find where the absolute difference is minimal
intersection_idx = np.nanargmin(np.abs(diff))
intersection_q = q[intersection_idx]
intersection_ss = optimal_ss[intersection_idx]
intersection_xi = optimal_xi[intersection_idx]

# --- Plot ---
plt.figure(figsize=(10, 8))

# Plot |xi| where N_r^2>0
c = plt.pcolormesh(q, ss, xi_masked, shading='auto', cmap='viridis')

# Overlay unstable region (gray)
plt.contourf(q, ss, unstable_mask, levels=[0.5, 1], colors='lightgray', alpha=0.8)

# Add colorbar
plt.colorbar(c, label=r'$|\xi|$')

# Contour where N_r^2 = 0
plt.contour(q, ss, Nr2, levels=[0], colors='red', linewidths=2, linestyles='solid')

# Contour where vortensity gradient = 0
plt.contour(q, ss, vort_grad, levels=[0], colors='white', linewidths=2, linestyles='dashed')

# Plot optimal ss(q) line: plot long segments
start_idx = 0
current_sign = np.sign(optimal_xi[start_idx])

for j in range(1, len(q)):
    if np.sign(optimal_xi[j]) != current_sign or j == len(q)-1:
        if current_sign > 0:
            plt.plot(q[start_idx:j+1], optimal_ss[start_idx:j+1], 'k-', linewidth=2)
        elif current_sign < 0:
            plt.plot(q[start_idx:j+1], optimal_ss[start_idx:j+1], 'k--', linewidth=2)
        start_idx = j
        current_sign = np.sign(optimal_xi[j])

# Mark global max and min xi
plt.plot(global_max_q, global_max_ss, 'ko', markersize=10, label=f'Max $\\xi$ = {global_max_xi:.2f}')
plt.plot(global_min_q, global_min_ss, 'ko', markersize=10, markerfacecolor='none', label=f'Min $\\xi$ = {global_min_xi:.2f}')

# Mark intersection with red circle
plt.plot(intersection_q, intersection_ss, 'ro', markersize=12, markeredgewidth=2, label='Intersection point')

# Labels and title
plt.xlabel(r'$q$ (temperature slope)')
plt.ylabel(r'$\mathrm{ss}$ (surface density slope)')
plt.title(r'Maximizing $\xi=q-(\gamma-1)\,\mathrm{ss}$ (only where $N_r^2>0$)')
plt.legend()
plt.grid(True)
plt.xlim(-2, 2.5)
plt.ylim(-2, 2)
plt.tight_layout()
plt.show()

# --- Print important values ---
print(f"Global maximum xi = {global_max_xi:.3f} at q = {global_max_q:.3f}, ss = {global_max_ss:.3f}")
print(f"Global minimum xi = {global_min_xi:.3f} at q = {global_min_q:.3f}, ss = {global_min_ss:.3f}")
print(f"Intersection with vortensity zero:")
print(f"q = {intersection_q:.3f}, ss = {intersection_ss:.3f}, xi = {intersection_xi:.3f}")

# Compute flaring index
flaring_index = 0.5 * (1 - intersection_q)
print(f"Corresponding flaring index: fi = {flaring_index:.3f}")

# --- Compute predicted torques from Paardekooper+2010 formulas ---

# Assumed softening parameter
bh = 0.4  # b/h

# Linear entropy-related corotation torque (second term of Eq. 17)
gamma_Gamma_c_lin_entropy_over_Gamma0 = 2.2 * intersection_xi * (0.4 / bh)**0.71

# Nonlinear entropy-related horseshoe drag (second term of Eq. 45)
gamma_Gamma_c_hs_entropy_over_Gamma0 = (intersection_xi / gamma) * (0.4 / bh) * (10.1 * np.sqrt(0.4 / bh) - 2.2)

# Lindblad torque
GAML = -(2.5 + 1.7 * intersection_q - 0.1 * intersection_ss) * (0.4 / bh)**0.71

print(f"\nPredicted normalized torques (gamma*Gamma/Gamma0):")
print(f"Linear entropy-related corotation torque: {gamma_Gamma_c_lin_entropy_over_Gamma0:.3f}")
print(f"Nonlinear entropy-related horseshoe drag (unsaturated): {gamma_Gamma_c_hs_entropy_over_Gamma0:.3f}")
print(f"Lindblad torque: {GAML:.3f}")


