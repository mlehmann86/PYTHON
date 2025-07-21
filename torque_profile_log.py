import os
import re
import numpy as np
import matplotlib.pyplot as plt
from data_reader import read_single_snapshot, read_single_snapshot_idefix, read_parameters, reconstruct_grid, determine_nt
from data_storage import determine_base_path, scp_transfer
from planet_data import extract_planet_mass_and_migration, compute_theoretical_torques, read_alternative_torque

def plot_gasdens_2d(gasdens_avg, gasdens0, xgrid, ygrid, simname, zgrid=None):
    """
    Plot vertically averaged gas density perturbation (gasdens - gasdens0) in r-phi space.
    """
    if zgrid is not None and gasdens_avg.ndim == 3:
        gasdens_perturb = np.mean(gasdens_avg - gasdens0, axis=2)
    else:
        gasdens_perturb = gasdens_avg - gasdens0  # 2D

    r, phi = np.meshgrid(xgrid, ygrid, indexing='ij')

    plt.figure(figsize=(8, 5))
    plt.pcolormesh(r, phi, gasdens_perturb.T, shading='auto', cmap='RdBu_r')
    plt.xlabel('r')
    plt.ylabel(r'$\phi$')
    plt.title('Gas Density Perturbation (Vertically Averaged)')
    cbar = plt.colorbar()
    cbar.set_label(r'$\langle \rho - \rho_0 \rangle_z$')
    outname = f"gasdens_perturb_map_{simname}.pdf"
    plt.tight_layout()
    plt.savefig(outname)
    print(f"Saved gas density perturbation map as: {outname}")

def plot_fphi_2d(gasdens, xgrid, ygrid, r_p, phi_p, qp, h0, thick_smooth, output_filename="fphi_map.pdf"):
    """
    Plot the vertically averaged torque integrand rho * r^2 * f_phi in (r, phi) space.
    """
    ny, nx, nz = gasdens.shape
    dphi = ygrid[1] - ygrid[0]

    # Average over vertical direction
    rho_2d = np.mean(gasdens, axis=2)
    fphi = np.zeros((ny, nx))

    for j in range(ny):
        phi = ygrid[j]
        # Correct delta_phi with wrapping
        delta_phi = (phi - phi_p + np.pi) % (2 * np.pi) - np.pi
        for i in range(nx):
            r = xgrid[i]
            dx = r - r_p
            dy = r * delta_phi
            smooth = (h0 * thick_smooth) ** 2
            denom = (dx * dx + dy * dy + smooth) ** 1.5
            fphi[j, i] = -qp * dy / denom

    integrand = rho_2d * (xgrid[np.newaxis, :] ** 2) * fphi

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(integrand, origin='lower', aspect='auto',
                   extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]],
                   cmap='seismic', vmin=-np.max(np.abs(integrand)), vmax=np.max(np.abs(integrand)))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\rho \, r^2 \, f_\phi$ (vert. avg)", fontsize=13)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("Torque Integrand (Vertically Averaged)")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved 2D torque integrand map to {output_filename}")

def compute_torque_gravity(gasdens, xgrid, ygrid, zgrid, r_p, phi_p, qp, h0, thick_smooth):
    nx = len(xgrid)
    ny = len(ygrid)
    nz = len(zgrid)
    dphi_phi = ygrid[1] - ygrid[0]  # azimuthal spacing in radians
    dz = zgrid[1] - zgrid[0] if nz > 1 else 1.0  # vertical spacing; use 1.0 if 2D
    torque_density = np.zeros(nx)

    for j in range(ny):
        phi = ygrid[j]
        delta_phi = (phi - phi_p + np.pi) % (2 * np.pi) - np.pi  # angle wrapping
        for i in range(nx):
            r = xgrid[i]
            dx = r - r_p
            dy = r * delta_phi
            smooth = (h0 * thick_smooth) ** 2
            denom = (dx * dx + dy * dy + smooth) ** 1.5
            fphi = -qp * dy / denom
            for k in range(nz):
                rho = gasdens[j, i, k]
                torque_density[i] += rho * r ** 2 * fphi * dphi_phi * dz

    return torque_density

def compute_torque_flux(gasdens, gasvx, gasvy, xgrid, ygrid, zgrid):
    ny, nx, nz = gasdens.shape
    dphi = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0] if nz > 1 else 1.0
    torque_density = np.zeros(nx)

    for i in range(nx):
        r = xgrid[i]
        for j in range(ny):
            for k in range(nz):
                rho = gasdens[j, i, k]
                vphi = gasvx[j, i, k]
                vr = gasvy[j, i, k]
                torque_density[i] += rho * vphi * vr * r ** 2 * dphi * dz

    return torque_density

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute radial torque density profile.")
    parser.add_argument("snapshot", type=int, help="Snapshot number")
    parser.add_argument("simname", nargs="?", type=str,
                        help="Simulation subdirectory (required if --comparison is not set)")
    parser.add_argument("--IDEFIX", action="store_true", help="Enable IDEFIX mode")
    parser.add_argument("--t_avg", type=float, default=50.0, help="Averaging time in orbits (default: 50)")
    parser.add_argument("--comparison", action="store_true", help="Compare multiple simulations instead of one")
    args = parser.parse_args()

    # In single simulation mode, simname must be provided.
    if (not args.comparison) and (args.simname is None):
        parser.error("Simulation subdirectory must be provided when not using --comparison.")

    # List of simulations to compare (only used if --comparison is set).
    simlist = [
        "cos_bet1d4_gam53_ss15_q1_r0416_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0516_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0616_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0515_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0516_z05_nu1dm11_COR_LR_2D",
        "cos_bet1d4_gam53_ss15_q1_r0517_z05_nu1dm11_COR_LR_2D"
    ]

    snap = args.snapshot
    IDEFIX = args.IDEFIX
    t_avg = args.t_avg

    if args.comparison:
        print("Comparison mode enabled. Processing multiple simulations...")
        fig, ax = plt.subplots(figsize=(8, 5))

        # We'll store these for later so we can compute overall min and max of the plotted torque.
        all_grav_profiles = []
        all_xgrid = None

        # We'll also store each simulation's domain boundaries and line color,
        # so we can plot them once we know the final y-limits.
        domain_boundaries = []  # list of tuples: (r_min, r_max, color)

        for sim in simlist:
            print(f"\nProcessing simulation: {sim}")
            base_path = determine_base_path(sim, IDEFIX=IDEFIX)
            summary_file = os.path.join(base_path, "summary0.dat" if not IDEFIX else "idefix.0.log")

            parameters = read_parameters(summary_file, IDEFIX=IDEFIX)
            h0 = parameters.get("h0") if IDEFIX else parameters.get("ASPECTRATIO")
            SIG0 = parameters.get("sigma0") if IDEFIX else parameters.get("SIGMA0")
            gam = parameters.get("gamma") if IDEFIX else parameters.get("GAMMA")
            thick_smooth = parameters.get("smoothing", 0.02) / h0 if IDEFIX else parameters.get("THICKNESSSMOOTHING")
            if IDEFIX:
                qp = parameters.get("PlanetMass", parameters.get("PlanetMass0", 1e-5))
            else:
                qp, _ = extract_planet_mass_and_migration(summary_file)
            GAM0 = (qp / h0) ** 2 * SIG0

            rmin_str = re.search(r"r(\d{2})(\d{2})", sim)
            label = sim
            if rmin_str:
                rmin_val = int(rmin_str.group(1)) / 10.0
                rmax_val = int(rmin_str.group(2)) / 10.0
                label = f"r ∈ [{rmin_val:.1f}, {rmax_val:.1f}]"
            else:
                # Fallback if for some reason the string isn't matched
                rmin_val, rmax_val = 0.4, 1.6  # or some default

            snapshot_data = read_single_snapshot(base_path, 0, read_gasdens=True)
            xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=IDEFIX)[0:6]
            # reconstruct_grid might also return extra info depending on your version, so adjust as needed.

            spacing = parameters.get("vtk", 50.265) / (2 * np.pi) if IDEFIX else parameters.get("NINTERM", 20) / 20.0
            nt = determine_nt(base_path, IDEFIX=IDEFIX)
            n_half = int(round((t_avg / spacing) / 2))
            idx_start = max(0, snap - n_half)
            idx_end = min(nt, snap + n_half + 1)
            if (idx_end - idx_start) * spacing < t_avg and idx_start > 0:
                missing = int(round((t_avg / spacing))) - (idx_end - idx_start)
                idx_start = max(0, idx_start - missing)

            grav = np.zeros(nx)
            for i in range(idx_start, idx_end):
                data = read_single_snapshot(base_path, i, read_gasdens=True)[0]
                gasdens = data["gasdens"]
                grav += compute_torque_gravity(gasdens, xgrid, ygrid, zgrid, 1.0, 0.0, qp, h0, thick_smooth)
            num_snap = (idx_end - idx_start)
            if num_snap > 0:
                grav /= num_snap
            grav /= (GAM0 / gam)

            # Plot absolute value in log scale
            abs_grav = np.abs(grav)
            line, = ax.plot(xgrid, abs_grav, label=label)
            all_grav_profiles.append(abs_grav)
            if all_xgrid is None:
                all_xgrid = xgrid

            domain_boundaries.append((rmin_val, rmax_val, line.get_color()))

            # Compute total torque as well
            total_grav = np.sum(grav) * (xgrid[1] - xgrid[0])
            print(f"  → Total gravitational torque (from snapshots): {total_grav:.6e}")
            # Plot as horizontal line: absolute value
            ax.plot([xgrid[0], xgrid[-1]], [np.abs(total_grav)]*2, color=line.get_color(),
                    linestyle='--', linewidth=2, alpha=0.7)

            # --- Internal torque from tqwk0.dat ---
            tqwk_file = os.path.join(base_path, "tqwk0.dat")
            try:
                date_torque, torque, _ = read_alternative_torque(tqwk_file, IDEFIX=IDEFIX)
                internal_time = date_torque if IDEFIX else date_torque / (2 * np.pi)
                internal_torque = -torque * qp / (GAM0 / gam)

                # Time average over last 200 orbits
                t_start = internal_time[-1] - 200
                mask = internal_time >= t_start
                if np.any(mask):
                    total_internal = np.mean(internal_torque[mask])
                    print(f"  → Total internal torque (from tqwk0.dat, avg last 200 orbits): {total_internal:.6e}")
                    ax.plot([xgrid[0], xgrid[-1]], [np.abs(total_internal)]*2, color=line.get_color(),
                            linestyle=':', linewidth=2, alpha=0.9)
                else:
                    print("  → Not enough data points in tqwk0.dat to average over last 200 orbits.")
            except Exception as e:
                print(f"  → Warning reading tqwk0.dat: {e}")

        # Overplot theoretical Lindblad torque using the first simulation's parameters.
        # (Assumes domain changes won't affect the typical dimensionless Lindblad torque.)
        first_sim = simlist[0]
        base_path = determine_base_path(first_sim, IDEFIX=IDEFIX)
        summary_file = os.path.join(base_path, "summary0.dat" if not IDEFIX else "idefix.0.log")
        parameters = read_parameters(summary_file, IDEFIX=IDEFIX)
        h0 = parameters.get("h0") if IDEFIX else parameters.get("ASPECTRATIO")
        SIG0 = parameters.get("sigma0") if IDEFIX else parameters.get("SIGMA0")
        gam = parameters.get("gamma") if IDEFIX else parameters.get("GAMMA")
        if IDEFIX:
            qp = parameters.get("PlanetMass", parameters.get("PlanetMass0", 1e-5))
        else:
            qp, _ = extract_planet_mass_and_migration(summary_file)
        thick_smooth = parameters.get("smoothing", 0.02) / h0 if IDEFIX else parameters.get("THICKNESSSMOOTHING")
        GAM0 = (qp / h0) ** 2 * SIG0
        lindblad, _, _ = compute_theoretical_torques(parameters, qp, eq_label="Paardekooper2", IDEFIX=IDEFIX)
        lindblad /= (GAM0 / gam)
        ax.axhline(-lindblad, color='black', linestyle=':', linewidth=2, alpha=0.9, label='Lindblad Torque')

        # Collect all grav arrays in all_grav_profiles; let all_grav be their concatenation.
        all_grav = np.concatenate(all_grav_profiles)
        #y_min, y_max = all_grav.min(), all_grav.max()
        y_min, y_max = 1, all_grav.max()

        ax.set_yscale("log")
        ax.set_ylim([max(y_min * 0.8, 1e-3), y_max * 1.5])  # avoid log(0) and adjust headroom

        # Decide on how to draw domain boundary lines:
        # - base_domain_y: vertical location of the first boundary lines
        # - seg_height: how tall each line segment is
        # - domain_offset: how far to offset each simulation's boundary lines from the previous
        base_domain_y = 5.0   # Closer to zero line
        seg_height   = 6.0    # ~2x bigger than before (was 3)
        domain_offset = 6.0   # ~2x bigger spacing

        # Plot each simulation's domain boundaries
        for i, (rmin_val, rmax_val, color) in enumerate(domain_boundaries):
            # Vertical extent for the i-th simulation's boundaries
            y1 = base_domain_y + i * domain_offset
            y2 = y1 + seg_height
            ax.vlines(rmin_val, y1, y2, color=color, linewidth=3, alpha=0.8)
            ax.vlines(rmax_val, y1, y2, color=color, linewidth=3, alpha=0.8)

        # Adjust y-limits so the boundary lines are clearly visible.
        # We pick the upper limit to be at least as high as the top boundary line, or the max torque, whichever is greater.
        lowest_y = min(y_min, 0) - 5   # a bit of margin below
        top_boundaries_y = base_domain_y + domain_offset * (len(domain_boundaries) - 1) + seg_height + 5
        highest_y = max(y_max + 5, top_boundaries_y)
        ax.set_ylim([lowest_y, highest_y])

        ax.set_xlabel("r")
        ax.set_ylabel(r"$|\gamma \Gamma(r)/\Gamma_{0}|$")
        ax.set_title("Radial Torque Profiles Comparison")
        ax.legend()
        ax.grid()
        outname = "torque_profile_comparison.pdf"
        plt.tight_layout()
        plt.savefig(outname)
        print(f"Saved {outname}")

        # Transfer the comparison plot.
        scp_transfer(outname, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        return

    else: 
        # --- SINGLE SIMULATION ANALYSIS ---
        simname = args.simname
        print(f"Processing simulation: {simname}")
        base_path = determine_base_path(simname, IDEFIX=IDEFIX)
        summary_file = os.path.join(base_path, "summary0.dat" if not IDEFIX else "idefix.0.log")

        parameters = read_parameters(summary_file, IDEFIX=IDEFIX)
        h0 = parameters.get("h0") if IDEFIX else parameters.get("ASPECTRATIO")
        SIG0 = parameters.get("sigma0") if IDEFIX else parameters.get("SIGMA0")
        gam = parameters.get("gamma") if IDEFIX else parameters.get("GAMMA")
        thick_smooth = parameters.get("smoothing", 0.02) / h0 if IDEFIX else parameters.get("THICKNESSSMOOTHING")
        if IDEFIX:
            qp = parameters.get("PlanetMass", parameters.get("PlanetMass0", 1e-5))
        else:
            qp, _ = extract_planet_mass_and_migration(summary_file)
        print(f"h0 = {h0}, qp = {qp:.2e}, thick_smooth = {thick_smooth:.3f}")
        GAM0 = (qp / h0) ** 2 * SIG0

        snapshot_data = read_single_snapshot(base_path, 0, read_gasdens=True)
        data_arrays = snapshot_data[0]
        xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=IDEFIX)

        spacing = parameters.get("vtk", 50.265) / (2 * np.pi) if IDEFIX else parameters.get("NINTERM", 20) / 20.0
        nt = determine_nt(base_path, IDEFIX=IDEFIX)
        n_half = int(round((t_avg / spacing) / 2))
        idx_start = max(0, snap - n_half)
        idx_end = min(nt, snap + n_half + 1)
        if (idx_end - idx_start) * spacing < t_avg and idx_start > 0:
            missing = int(round((t_avg / spacing))) - (idx_end - idx_start)
            idx_start = max(0, idx_start - missing)

        gasvx0_data = read_single_snapshot(base_path, 0, read_gasvx=True)[0]
        gasvx0 = gasvx0_data['gasvx']

        grav = np.zeros(nx)
        flux = np.zeros(nx)
        gasdens_avg = np.zeros((ny, nx, nz))
        for i in range(idx_start, idx_end):
            print(f"Reading snapshot {i}...")
            data = read_single_snapshot(base_path, i, read_gasdens=True)[0]
            gasdens = data["gasdens"]
            gasdens_avg += gasdens
            print(f"Shape of gasdens: {gasdens.shape}")
            print(f"Shape of xgrid: {xgrid.shape}")
            print(f"Shape of ygrid: {ygrid.shape}")
            print(f"Shape of zgrid: {zgrid.shape}")
            grav += compute_torque_gravity(gasdens, xgrid, ygrid, zgrid, 1.0, 0.0, qp, h0, thick_smooth)
        grav /= (idx_end - idx_start)
        flux /= (idx_end - idx_start)
        gasdens_avg /= (idx_end - idx_start)
        grav /= GAM0 / gam
        flux /= GAM0 / gam

        x = xgrid
        i_min = np.argmin(grav)
        i_max = np.argmax(grav)
        print(f"max(grav) = {grav[i_max]:.3e} at r = {x[i_max]:.3f}")
        print(f"min(grav) = {grav[i_min]:.3e} at r = {x[i_min]:.3f}")

        total_grav = np.sum(grav) * (x[1] - x[0])
        print(f"Total gravitational torque (normalized): {total_grav:.3e}")

        lindblad, _, _ = compute_theoretical_torques(parameters, qp, eq_label="Paardekooper2", IDEFIX=IDEFIX)
        lindblad /= GAM0 / gam
        print(f"Lindblad Torque: {lindblad:.3e}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, grav, label='Gravitational Torque')
        ax.axhline(total_grav, color='gray', linestyle='--', linewidth=2, alpha=0.7,
                   label='Total Torque (avg)', zorder=5)
        ax.axhline(-lindblad, color='black', linestyle=':', linewidth=2, alpha=0.9,
                   label='Lindblad Torque', zorder=6)

        ax.set_xlabel("r")
        ax.set_ylabel(r"$\gamma \Gamma(r)/\Gamma_{\mathrm{0}}$", fontsize=14)
        y_min = min(np.min(grav), total_grav, lindblad)
        y_max = max(np.max(grav), total_grav, lindblad)
        margin = 0.1 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.legend()
        ax.grid()

        x_s_over_rp = 1.1 * (thick_smooth / 0.4)**0.25 * (1.0 / gam)**0.25 * np.sqrt(qp / h0)
        r_p = 1.0
        x_s = x_s_over_rp * r_p
        r_in = r_p - x_s
        r_out = r_p + x_s
        print(f"Horseshoe region: r ∈ [{r_in:.3f}, {r_out:.3f}]")
        ax.axvline(r_in, color='purple', linestyle=':', label=r"$r_p \pm x_s$")
        ax.axvline(r_out, color='purple', linestyle=':')

        outname = f"torque_profile_{simname}.pdf"
        plt.tight_layout()
        plt.savefig(outname)
        print(f"Saved {outname}")

        print(f"\nAttempting to transfer {outname} to /Users/mariuslehmann/Downloads/Profiles/ for user mariuslehmann...")
        scp_transfer(outname, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")

        # Optionally, additional plots may be generated:
        # plot_fphi_2d(...)
        # plot_gasdens_2d(...)

if __name__ == "__main__":
    main()
