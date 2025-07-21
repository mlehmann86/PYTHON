import numpy as np
import matplotlib.pyplot as plt
import os
from corotation_torque_pnu import compute_theoretical_corotation_paarde2011, gamma_eff_diffusion
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

def plot_torque_pq_chi_grid(chi_fixed, p_nu_fixed):
    """
    Plots the total PK11 torque in (p, q) space for a fixed thermal diffusivity (chi)
    and viscosity parameter (p_nu). This generates the LEFT panel.
    """
    output_path = os.getcwd()

    # === Fixed physical parameters ===
    gam = 1.4         # Adiabatic index
    bh = 0.4          # Gravitational smoothing length b/h
    qp = 2.52e-5      # Planet-to-star mass ratio
    h = 0.1           # Disk aspect ratio h
    
    # --- Calculate effective gamma for the fixed thermal diffusivity ---
    # This value will be constant across the p-q grid.
    gamma_eff = gamma_eff_diffusion(gam, chi_fixed, h)

    # === Setup p and q grids ===
    p_vals = np.linspace(-2, 5, 300)
    q_vals = np.linspace(0.5, 2, 300)
    P, Q = np.meshgrid(p_vals, q_vals)

    # --- Calculate torque components over the grid ---
    # Here, 'ss' is the surface density power-law index 's', and 'P' is the corresponding grid.
    ss = P 
    
    # Lindblad torque (depends on s and q)
    G_Ls = -2.5 - 1.7 * Q + 0.1 * ss
    torque_lindblad = G_Ls / gamma_eff

    # Initialize total torque array
    total_torque = np.zeros_like(P)

    # Loop through grid to calculate corotation torque
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            q_val = Q[i, j]
            s_val = ss[i, j]
            
            # Calculate Corotation Torque for thermal diffusion
            # Note: beta=None and chi is specified
            Gamma_C = compute_theoretical_corotation_paarde2011(
                p_nu_fixed, h=h, gamma=gam, q=q_val, s=s_val, 
                beta=None, qp=qp, chi=chi_fixed
            )
            
            Gamma_L = torque_lindblad[i, j]
            total_torque[i, j] = gamma_eff * (Gamma_C + Gamma_L)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [q_vals.min(), q_vals.max(), p_vals.min(), p_vals.max()]
    im = ax.imshow(
        total_torque.T,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='RdYlBu_r',
        vmin=-5, # Set a symmetric color range if desired
        vmax=5
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\gamma_{eff} \Gamma_{tot}/\Gamma_0$ (PK11)", fontsize=FONT_SIZE)

    # --- Overlays (COS region and Zero Torque Contour) ---
    COS_condition = (P + Q) * (Q + (1 - gam) * P) < 0
    ax.contour(Q, P, COS_condition, levels=[0.5], colors='black', linewidths=2)
    ax.text(1.25, 1.25, "COS region", color='black', fontsize=FONT_SIZE, ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
            
    cs2 = ax.contour(Q, P, total_torque, levels=[0], colors='black', linestyles='dashed')
    if cs2.collections:
        ax.clabel(cs2, fmt="$\\Gamma_{\\mathrm{tot}} = 0$", inline=True, fontsize=FONT_SIZE)

    # --- Labels and Title ---
    chi_latex = fr"$\chi = {chi_fixed:.1e}$"
    p_nu_latex = fr"$p_\nu = {p_nu_fixed:.2f}$"
    ax.set_xlabel('q (Temp. Slope)', fontsize=FONT_SIZE)
    ax.set_ylabel('p (Density Slope)', fontsize=FONT_SIZE)
    ax.set_title(f"Total PK11 Torque ({chi_latex}, {p_nu_latex})", fontsize=FONT_SIZE)

    # --- Save and Return Filename ---
    output_filename = os.path.join(output_path, f"torque_pq_chi_{chi_fixed:.1e}_pnu_{p_nu_fixed:.2f}.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_filename}")

    try:
        scp_transfer(output_filename, LOCAL_DEST_DIR, USERNAME)
    except Exception as e:
        print(f"SCP transfer failed for {os.path.basename(output_filename)}: {e}")

    return output_filename


def plot_torque_chi_pnu_grid(s_fixed, q_fixed):
    """
    Plots total PK11 torque in (chi, p_nu) space for fixed p and q.
    This generates the RIGHT panel and prints critical values.
    """
    # === Fixed physical parameters ===
    gam = 1.4
    bh = 0.4
    h = 0.1
    qp = 2.52e-5
    rp = 1.0
    Omega_p = 1.0

    # Lindblad torque component, depends on fixed s and q
    G_Ls = -2.5 - 1.7 * q_fixed + 0.1 * s_fixed

    # === Setup grids for chi (thermal diffusivity) and nu (viscosity) ===
    chi_vals = np.logspace(-7, -4, 400)
    nu_vals = np.logspace(-8, -3, 400)
    CHI, NU = np.meshgrid(chi_vals, nu_vals)

    # Initialize arrays for total torque and p_nu
    total_torque = np.zeros_like(CHI)
    PNU = np.zeros_like(CHI)

    # --- Loop through grids to calculate torque ---
    for i in range(CHI.shape[0]):
        for j in range(CHI.shape[1]):
            chi = CHI[i, j]
            nu = NU[i, j]
            gamma_eff = gamma_eff_diffusion(gam, chi, h)
            xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / bh) ** 0.25 * np.sqrt(qp / h)
            p_nu = (2.0 / 3.0) * np.sqrt((rp ** 2 * Omega_p * xs ** 3) / (2 * np.pi * nu))
            PNU[i, j] = p_nu
            Gamma_C = compute_theoretical_corotation_paarde2011(
                p_nu, h=h, gamma=gam, q=q_fixed, s=s_fixed, beta=None, qp=qp, chi=chi
            )
            Gamma_L = G_Ls / gamma_eff
            total_torque[i, j] = gamma_eff * (Gamma_C + Gamma_L)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    log_chi = np.log10(CHI)
    log_nu = np.log10(NU)
    vmin, vmax = np.min(total_torque), np.max(total_torque)

    im = ax.imshow(
        total_torque.T, origin='lower',
        extent=[np.min(log_nu), np.max(log_nu), np.min(log_chi), np.max(log_chi)],
        aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax
    )

    # --- Find Contour and Infer Critical Values ---
    cs = ax.contour(
        log_nu, log_chi, total_torque, levels=[0.0],
        colors='black', linewidths=2, linestyles='dashed'
    )
    
    if cs.allsegs[0]:
        contour_coords = cs.allsegs[0][0]
        log_pnu_coords = contour_coords[:, 0]
        log_chi_coords = contour_coords[:, 1]

        # Find indices of the four tangent points
        idx_pnu_min = np.argmin(log_pnu_coords)
        idx_pnu_max = np.argmax(log_pnu_coords)
        idx_chi_min = np.argmin(log_chi_coords)
        idx_chi_max = np.argmax(log_chi_coords)

        # Extract the (p_nu, chi) pairs at these tangent points
        pair_pnu_min = (10**log_pnu_coords[idx_pnu_min], 10**log_chi_coords[idx_pnu_min])
        pair_pnu_max = (10**log_pnu_coords[idx_pnu_max], 10**log_chi_coords[idx_pnu_max])
        pair_chi_min = (10**log_pnu_coords[idx_chi_min], 10**log_chi_coords[idx_chi_min])
        pair_chi_max = (10**log_pnu_coords[idx_chi_max], 10**log_chi_coords[idx_chi_max])

        print("\n" + "="*60)
        print("Critical Tangent Points on Gamma_tot = 0 Contour (Diffusion)")
        print("="*60)
        print("Point of Minimum p_nu (Left-most tangent):")
        print(f"  - p_nu = {pair_pnu_min[0]:.4f}, chi = {pair_pnu_min[1]:.3e}")
        
        print("\nPoint of Maximum p_nu (Right-most tangent):")
        print(f"  - p_nu = {pair_pnu_max[0]:.4f}, chi = {pair_pnu_max[1]:.3e}")

        print("\nPoint of Minimum chi (Bottom tangent):")
        print(f"  - p_nu = {pair_chi_min[0]:.4f}, chi = {pair_chi_min[1]:.3e}")

        print("\nPoint of Maximum chi (Top tangent):")
        print(f"  - p_nu = {pair_chi_max[0]:.4f}, chi = {pair_chi_max[1]:.3e}")
        print("="*60 + "\n")

        # === Analysis for widest positive torque range ===
        max_delta_pnu = 0
        optimal_chi = None
        optimal_pnu_range = [0, 0]

        # Iterate through each unique chi value (each column of the grid)
        for j, chi_val in enumerate(chi_vals):
            # Find where the torque is positive for this chi
            positive_torque_mask = total_torque[j, :] > 0
            
            if np.any(positive_torque_mask):
                # Get the p_nu values for this chi where torque is positive
                pnu_for_chi = PNU[j, positive_torque_mask]
                pnu_min = np.min(pnu_for_chi)
                pnu_max = np.max(pnu_for_chi)
                delta_pnu = pnu_max - pnu_min

                if delta_pnu > max_delta_pnu:
                    max_delta_pnu = delta_pnu
                    optimal_chi = chi_val
                    optimal_pnu_range = [pnu_min, pnu_max]
        
        print("="*60)
        print("Analysis of Positive Torque Region (Diffusion)")
        print("="*60)
        print("The widest range of p_nu with positive torque occurs at:")
        print(f"  - chi           = {optimal_chi:.3e}")
        print(f"  - Delta p_nu    = {max_delta_pnu:.4f}")
        print(f"  - p_nu range    = [{optimal_pnu_range[0]:.4f}, {optimal_pnu_range[1]:.4f}]")
        print("="*60 + "\n")

    else:
        print("Could not find the Gamma_tot = 0 contour to infer critical values.")
    
    # Add contour label to plot
    if cs.collections:
        ax.clabel(cs, fmt=r"$\Gamma_{\mathrm{tot}} = 0$", inline=True, fontsize=FONT_SIZE)

    # --- Finalize Plot ---
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\gamma_{eff} \Gamma_{tot}/\Gamma_0$ (PK11)", fontsize=FONT_SIZE)
    ax.set_xlabel(r"$\log_{10}(\nu)$", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$\log_{10}(\chi)$", fontsize=FONT_SIZE)
    ax.set_title(f"Total PK11 Torque (p={s_fixed}, q={q_fixed})", fontsize=FONT_SIZE)
    plt.tight_layout()

    # --- Save and Return Filename ---
    output_path = os.getcwd()
    output_filename = os.path.join(output_path, f"torque_chipnu_p{s_fixed}_q{q_fixed}.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_filename}")
    
    try:
        scp_transfer(output_filename, LOCAL_DEST_DIR, USERNAME)
    except Exception as e:
        print(f"SCP transfer failed for {os.path.basename(output_filename)}: {e}")

    return output_filename


def merge_pdfs_side_by_side(pdf1_path, pdf2_path, output_path="combined_torque_plots_diffusion.pdf"):
    """
    Merges two single-page PDFs into one side-by-side PDF.
    """
    try:
        images1 = convert_from_path(pdf1_path, dpi=200)
        images2 = convert_from_path(pdf2_path, dpi=200)
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        print("Please ensure poppler is installed and in your PATH.")
        return

    if not images1 or not images2:
        raise ValueError("Could not convert one or both PDFs to images.")

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
    file1 = plot_torque_pq_chi_grid(chi_fixed=1e-5, p_nu_fixed=0.32)
    file2 = plot_torque_chi_pnu_grid(s_fixed=0.5, q_fixed=1.0)
    
    if file1 and file2:
        merge_pdfs_side_by_side(file1, file2)

