import numpy as np
import matplotlib.pyplot as plt
import os
from corotation_torque_pnu import compute_theoretical_corotation_paarde2011, gamma_eff_theory
from data_storage import scp_transfer
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO

# --- Constants and Global Settings ---
FONT_SIZE = 18
plt.rcParams.update({'font.size': FONT_SIZE})
LOCAL_DEST_DIR = "/Users/mariuslehmann/Downloads/Profiles/"
USERNAME = "mariuslehmann"

def plot_torque_beta_nu_grid(s_fixed, q_fixed):
    """
    Plots total PK11 torque in (beta, p_nu) space for fixed p and q.
    Also calculates and prints critical values and the optimal beta for outward migration.
    """
    # === Fixed simulation parameters ===
    gam = 1.4
    bh = 0.4
    h = 0.1
    qp = 2.52e-5
    rp = 1.0
    Omega_p = 1.0

    G_Ls = -2.5 - 1.7 * q_fixed + 0.1 * s_fixed

    # === Grids ===
    beta_vals = np.logspace(-2, 4, 400)
    nu_vals = np.logspace(-8, -3, 400)
    BETA, NU = np.meshgrid(beta_vals, nu_vals)

    total_torque = np.zeros_like(BETA)
    PNU = np.zeros_like(BETA)

    for i in range(BETA.shape[0]):
        for j in range(BETA.shape[1]):
            beta = BETA[i, j]
            nu = NU[i, j]

            gamma_eff = gamma_eff_theory(gam, beta, h)
            xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / bh) ** 0.25 * np.sqrt(qp / h)
            p_nu = (2.0 / 3.0) * np.sqrt((rp ** 2 * Omega_p * xs ** 3) / (2 * np.pi * nu))
            PNU[i, j] = p_nu

            Gamma_C = compute_theoretical_corotation_paarde2011(
                p_nu, h=h, gamma=gam, q=q_fixed, s=s_fixed, beta=beta, qp=qp
            )
            Gamma_L = G_Ls / gamma_eff
            total_torque[i, j] = gamma_eff * (Gamma_C + Gamma_L)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 8))
    log_beta = np.log10(BETA)
    log_nu = np.log10(NU)

    vmin, vmax = np.min(total_torque), np.max(total_torque)

    im = ax.imshow(
        total_torque.T,
        origin='lower',
        extent=[
            np.min(log_nu), np.max(log_nu),
            np.min(log_beta), np.max(log_beta)
        ],
        aspect='auto',
        cmap='RdYlBu_r',
        vmin=vmin,
        vmax=vmax
    )

    # === Contour ===
    cs = ax.contour(
        log_nu, log_beta, total_torque,
        levels=[0.0],
        colors='black',
        linewidths=2,
        linestyles='dashed'
    )
    
    if cs.allsegs[0]:
        contour_coords = cs.allsegs[0][0]
        log_pnu_coords = contour_coords[:, 0]
        log_beta_coords = contour_coords[:, 1]

        # Find indices of the four tangent points
        idx_pnu_min = np.argmin(log_pnu_coords)
        idx_pnu_max = np.argmax(log_pnu_coords)
        idx_beta_min = np.argmin(log_beta_coords)
        idx_beta_max = np.argmax(log_beta_coords)

        # Extract the (p_nu, beta) pairs at these tangent points
        pair_pnu_min = (10**log_pnu_coords[idx_pnu_min], 10**log_beta_coords[idx_pnu_min])
        pair_pnu_max = (10**log_pnu_coords[idx_pnu_max], 10**log_beta_coords[idx_pnu_max])
        pair_beta_min = (10**log_pnu_coords[idx_beta_min], 10**log_beta_coords[idx_beta_min])
        pair_beta_max = (10**log_pnu_coords[idx_beta_max], 10**log_beta_coords[idx_beta_max])

        print("\n" + "="*60)
        print("Critical Tangent Points on Gamma_tot = 0 Contour (Beta-Cooling)")
        print("="*60)
        print("Point of Minimum p_nu (Left-most tangent):")
        print(f"  - p_nu = {pair_pnu_min[0]:.4f}, beta = {pair_pnu_min[1]:.3f}")
        
        print("\nPoint of Maximum p_nu (Right-most tangent):")
        print(f"  - p_nu = {pair_pnu_max[0]:.4f}, beta = {pair_pnu_max[1]:.3f}")

        print("\nPoint of Minimum beta (Bottom tangent):")
        print(f"  - p_nu = {pair_beta_min[0]:.4f}, beta = {pair_beta_min[1]:.3f}")

        print("\nPoint of Maximum beta (Top tangent):")
        print(f"  - p_nu = {pair_beta_max[0]:.4f}, beta = {pair_beta_max[1]:.3f}")
        print("="*60 + "\n")

        # === Analysis for widest positive torque range ===
        max_delta_pnu = 0
        optimal_beta = None
        optimal_pnu_range = [0, 0]

        # Iterate through each unique beta value (each row of the grid)
        for i, beta_val in enumerate(beta_vals):
            # Find where the torque is positive for this beta
            positive_torque_mask = total_torque[:, i] > 0
            
            if np.any(positive_torque_mask):
                # Get the p_nu values for this beta where torque is positive
                pnu_for_beta = PNU[positive_torque_mask, i]
                pnu_min = np.min(pnu_for_beta)
                pnu_max = np.max(pnu_for_beta)
                delta_pnu = pnu_max - pnu_min

                if delta_pnu > max_delta_pnu:
                    max_delta_pnu = delta_pnu
                    optimal_beta = beta_val
                    optimal_pnu_range = [pnu_min, pnu_max]

        print("="*60)
        print("Analysis of Positive Torque Region")
        print("="*60)
        print("The widest range of p_nu with positive torque occurs at:")
        print(f"  - beta          = {optimal_beta:.3f}")
        print(f"  - Delta p_nu    = {max_delta_pnu:.4f}")
        print(f"  - p_nu range    = [{optimal_pnu_range[0]:.4f}, {optimal_pnu_range[1]:.4f}]")
        print("="*60 + "\n")

    else:
        print("Could not find the Gamma_tot = 0 contour to infer critical values.")

    if cs.collections:
        ax.clabel(cs, fmt=r"$\Gamma_{\mathrm{tot}} = 0$", inline=True, fontsize=FONT_SIZE)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\gamma_{eff} \Gamma_{tot}/\Gamma_0$ (PK11)", fontsize=FONT_SIZE)

    ax.set_xlabel(r"$\log_{10}(\nu)$", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$\log_{10}(\beta)$", fontsize=FONT_SIZE)
    ax.set_title(f"Total PK11 Torque (p={s_fixed}, q={q_fixed})", fontsize=FONT_SIZE)

    plt.tight_layout()

    # Save and Transfer
    output_path = os.getcwd()
    output_filename = os.path.join(output_path, f"total_torque_p{s_fixed}_q{q_fixed}_betapnu.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_filename}")

    try:
        scp_transfer(output_filename, LOCAL_DEST_DIR, USERNAME)
    except Exception as e:
        print(f"SCP transfer failed for {os.path.basename(output_filename)}: {e}")

    return output_filename


def plot_total_torque_pk11():
    """
    Plots the total PK11 torque (corotation + Lindblad) in p-q space for fixed beta and p_nu.
    """
    output_path = os.getcwd()

    # === Fixed parameters ===
    gam = 1.4         # Adiabatic index
    bh = 0.4          # Disk aspect ratio h/smoothing
    qp = 2.526e-5     # Planet-to-star mass ratio
    beta = 1.0        # Cooling time
    nu_p=1e-5         # Viscosity
    h = 0.1           # Disk aspect ratio h
    rp = 1.0
    Omega_p = 1.0

    gamma_eff = gamma_eff_theory(gam, beta, h)

    xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / bh) ** 0.25 * np.sqrt(qp / h)
    p_nu = (2.0 / 3.0) * np.sqrt((rp ** 2 * Omega_p * xs ** 3) / (2 * np.pi * nu_p))
    
    # Grid
    p_vals = np.linspace(-2, 5, 300)
    q_vals = np.linspace(0.5, 2, 300)
    P, Q = np.meshgrid(p_vals, q_vals)

    # Correctly use p as the surface density slope 's'
    ss = P 

    # Lindblad torque
    G_Ls = -2.5 - 1.7 * Q + 0.1 * ss
    torque_lindblad = G_Ls / gamma_eff

    # Total torque = corotation + Lindblad
    total_torque = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            q_val = Q[i, j]
            s_val = ss[i, j]
            Gamma_C = compute_theoretical_corotation_paarde2011(
                p_nu, h=h, gamma=gam, q=q_val, s=s_val, beta=beta, qp=qp
            )
            Gamma_L = torque_lindblad[i, j]
            total_torque[i, j] = gamma_eff * (Gamma_C + Gamma_L)

    # COS condition
    COS_condition = (P + Q) * (Q + (1 - gam) * P) < 0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [q_vals.min(), q_vals.max(), p_vals.min(), p_vals.max()]
    im = ax.imshow(
        total_torque.T,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='RdYlBu_r'
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\gamma_{eff} \Gamma_{tot}/\Gamma_0$ (PK11)", fontsize=FONT_SIZE)

    # COS boundary
    ax.contour(Q, P, COS_condition, levels=[0.5], colors='black', linewidths=2)
    ax.text(1.25, 1.25, "COS region", color='black', fontsize=FONT_SIZE + 3, ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Zero contour
    cs2 = ax.contour(Q, P, total_torque, levels=[0], colors='black', linestyles='dashed')
    if cs2.collections:
        ax.clabel(cs2, fmt="$\\Gamma_{\\mathrm{tot}} = 0$", inline=True, fontsize=FONT_SIZE)

    nu_p_latex = fr"$\nu_p = {nu_p:.2e}$"
    ax.set_xlabel('q (Temp. Slope)', fontsize=FONT_SIZE)
    ax.set_ylabel('p (Density Slope)', fontsize=FONT_SIZE)
    ax.set_title(f"Total PK11 Torque (β={beta}, {nu_p_latex}, b/h={bh:.2f})", fontsize=FONT_SIZE)

    output_filename = os.path.join(os.getcwd(), f"total_torque_beta{beta}_nup{nu_p:.2e}.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_filename}")
    
    try:
        scp_transfer(output_filename, LOCAL_DEST_DIR, USERNAME)
    except Exception as e:
        print(f"SCP transfer failed for {os.path.basename(output_filename)}: {e}")
    
    return output_filename


def merge_pdfs_side_by_side(pdf1_path, pdf2_path, output_path="combined_torque_plots_betacooling.pdf"):
    """Merges two PDFs side-by-side."""
    try:
        images1 = convert_from_path(pdf1_path, dpi=200)
        images2 = convert_from_path(pdf2_path, dpi=200)
    except Exception as e:
        print(f"Error converting PDF to image: {e}. Please ensure poppler is installed.")
        return

    if not images1 or not images2:
        raise ValueError("Could not convert one or both PDFs.")

    img1, img2 = images1[0], images2[0]
    
    total_width = img1.width + img2.width
    total_height = max(img1.height, img2.height)

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(total_width, total_height))
    c.drawImage(ImageReader(img1), 0, 0, width=img1.width, height=img1.height)
    c.drawImage(ImageReader(img2), img1.width, 0, width=img2.width, height=img2.height)
    c.save()

    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())
    print(f"Combined PDF saved to {output_path}")

    try:
        scp_transfer(output_path, LOCAL_DEST_DIR, USERNAME)
    except Exception as e:
        print(f"SCP transfer failed for {os.path.basename(output_path)}: {e}")


if __name__ == "__main__":
    file1 = plot_total_torque_pk11()
    file2 = plot_torque_beta_nu_grid(s_fixed=0.5, q_fixed=1.0)
    if file1 and file2:
        merge_pdfs_side_by_side(file1, file2)

