import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_reader import read_parameters, reconstruct_grid, read_single_snapshot, read_single_snapshot_idefix, determine_nt
from data_storage import determine_base_path, scp_transfer

def compute_N2(gasdens, gasenergy, xgrid, gamma):
    pressure = gasenergy * (gamma - 1)
    S = np.log(pressure / gasdens**gamma)
    dP_dr = np.gradient(pressure, xgrid, axis=1)
    dS_dr = np.gradient(S, xgrid, axis=1)
    N2 = -(1 / gamma) * (1 / gasdens) * dP_dr * dS_dr
    return N2

def compute_vortensity_like(gasdens, gasenergy, gasvx, xgrid, gamma):
    x = xgrid[np.newaxis, :, np.newaxis]  # for broadcasting
    pressure = gasenergy * (gamma - 1)
    Omega = gasvx / x + 1.0  # inertial frame
    dOmega_dr = np.gradient(Omega, xgrid, axis=1)
    kappa2 = 4 * Omega**2 + 2 * Omega * x * dOmega_dr
    vortensity_like = gasdens * kappa2 / (Omega * pressure**(2/gamma))
    return vortensity_like, Omega, kappa2

def compute_shock_radius(gamma, q_p, h):
    coeff = (gamma + 1) / (12.0 / 5.0)
    lsh = 0.8 * (coeff * q_p / h**3)**(-2.0 / 5.0) * h
    return 1.0 - lsh, 1.0 + lsh  # r_p = 1.0 assumed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=int)
    parser.add_argument("simname", type=str)
    parser.add_argument("--IDEFIX", action="store_true")
    args = parser.parse_args()

    base_path = determine_base_path(args.simname, IDEFIX=args.IDEFIX)
    summary_file = f"{base_path}/{'idefix.0.log' if args.IDEFIX else 'summary0.dat'}"
    parameters = read_parameters(summary_file, IDEFIX=args.IDEFIX)
    gamma = parameters.get("gamma") if args.IDEFIX else parameters.get("GAMMA")
    h0 = parameters.get("h0") if args.IDEFIX else parameters.get("ASPECTRATIO")
    q_p = parameters.get("PlanetMass", parameters.get("PlanetMass0", 1e-5)) if args.IDEFIX else parameters.get("PlanetMass", 1e-5)

    rsh_in, rsh_out = compute_shock_radius(gamma, q_p, h0)

    # Estimate time in orbits
    dt = parameters.get("vtk", 50.265) / (2 * np.pi) if args.IDEFIX else parameters.get("NINTERM", 20) / 20.0
    time_orbit = args.snapshot * dt

    # Load snapshot=0 for scaling reference
    if args.IDEFIX:
        data0 = read_single_snapshot_idefix(base_path, 0,
                                            read_gasdens=True, read_gasenergy=True, read_gasvx=True)[0]
    else:
        data0 = read_single_snapshot(base_path, 0,
                                     read_gasdens=True, read_gasenergy=True, read_gasvx=True)[0]
    gasdens0 = data0["gasdens"]
    gasenergy0 = data0["gasenergy"]
    gasvx0 = data0["gasvx"]
    xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(parameters, IDEFIX=args.IDEFIX)

    vort0, _, _ = compute_vortensity_like(gasdens0, gasenergy0, gasvx0, xgrid, gamma)
    i_r = np.argmin(np.abs(xgrid - 1.0))
    if nz > 1:
        vref = vort0[0, i_r, nz // 2]  # phi=0, r~1, z=0
    else:
        vref = vort0[0, i_r]  # phi=0, r~1

    # Load target snapshot
    if args.IDEFIX:
        data = read_single_snapshot_idefix(base_path, args.snapshot,
                                           read_gasdens=True, read_gasenergy=True, read_gasvx=True)[0]
    else:
        data = read_single_snapshot(base_path, args.snapshot,
                                    read_gasdens=True, read_gasenergy=True, read_gasvx=True)[0]

    gasdens = data["gasdens"]
    gasenergy = data["gasenergy"]
    gasvx = data["gasvx"]

    N2 = compute_N2(gasdens, gasenergy, xgrid, gamma)
    vortensity_like, Omega, kappa2 = compute_vortensity_like(gasdens, gasenergy, gasvx, xgrid, gamma)

    # Azimuthal and vertical averaging
    axis_avg = (0, 2) if nz > 1 else 0
    N2_avg = np.mean(N2, axis=axis_avg)
    vort_avg = np.mean(vortensity_like, axis=axis_avg) / vref
    Omega_avg = np.mean(Omega, axis=axis_avg)
    kappa_avg = np.sqrt(np.mean(kappa2, axis=axis_avg))
    OmegaK = xgrid**(-1.5)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(xgrid, N2_avg)
    axs[0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[0].set_ylabel(r'$\langle N^2 \rangle$')

    axs[1].plot(xgrid, vort_avg)
    axs[1].set_ylabel(r'$\left\langle \frac{\Sigma \kappa^2}{\Omega P^{2/\gamma}} \right\rangle / \mathrm{ref}$')

    axs[2].plot(xgrid, Omega_avg, label=r'$\langle \Omega \rangle$')
    axs[2].plot(xgrid, kappa_avg, label=r'$\langle \kappa \rangle$')
    axs[2].plot(xgrid, OmegaK, '--', label=r'$\Omega_K = r^{-3/2}$')
    axs[2].set_ylabel(r'$\Omega, \kappa$')
    axs[2].set_xlabel("r")
    axs[2].legend()

    # Mark shock location
    for ax in axs:
        ax.axvline(rsh_in, color='gray', linestyle=':', linewidth=1.5)
        ax.axvline(rsh_out, color='gray', linestyle=':', linewidth=1.5)
        ax.grid(True)

    fig.suptitle(f"Snapshot {args.snapshot}, Time = {time_orbit:.1f} orbits", fontsize=14)

    outname = f"radial_N2_vortensity_{args.simname}.pdf"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outname)
    print(f"Saved plot: {outname}")

    scp_transfer(outname, "/Users/mariuslehmann/Downloads/Profiles/", "mariuslehmann")

if __name__ == "__main__":
    main()

