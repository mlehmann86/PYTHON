#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from planet_data import (
    determine_base_path,
    read_parameters,
    extract_planet_mass_and_migration,
    read_alternative_torque
)

##################################################################################
##################################################################################

def gamma_eff_theory(gamma, beta, h):
    Q = (2.0 * beta) / (3.0 * h)
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_term_inner = (1.0 + Q**2) / (1.0 + gamma**2 * Q**2)
        sqrt_term = np.sqrt(np.maximum(0.0, sqrt_term_inner))
        numerator = 2.0
        denominator = (1.0 + gamma * Q**2) / (1.0 + gamma**2 * Q**2) + sqrt_term
        gamma_eff_val = numerator / denominator
    return gamma_eff_val

##################################################################################
##################################################################################

def gamma_eff_diffusion(gamma, chi, h):
    Q = 2.0 * chi / (3.0 * h**3)
    Q2 = Q**2
    γ2 = gamma**2

    with np.errstate(divide='ignore', invalid='ignore'):
        inner_root = np.sqrt((γ2 * Q2 + 1)**2 - 16 * Q2 * (gamma - 1))
        sqrt_term = np.sqrt(np.maximum(0.0, 2.0 * inner_root + 2 * γ2 * Q2 - 2))
        denom = gamma * Q + 0.5 * sqrt_term
        gamma_eff = np.where(denom > 0, 2 * Q * gamma / denom, 1.0)

    return gamma_eff

##################################################################################
##################################################################################

def compute_theoretical_corotation_paarde2011(
        p_nu, *, h, gamma, q, s, beta, qp, chi=None, verbose=False):
    """
    Non-linear corotation torque Γ_C/Γ₀ from Paardekooper+2011.
    Uses dynamic χ_p with ptrans={1e5 if β≈1; 0.12 otherwise}.
    """
    # PK11 dotted–curve override parameters
    pk11_gamma = 1.4
    pk11_h     = 0.05
    pk11_qβ    = 1.0
    pk11_sα    = 0.5
    pk11_qp    = 1.26e-5

    # switch to force PK11 dotted curve?
    Paardekooper = False

    # default = function arguments
    h_calc, γ_calc, q_calc, s_calc, qp_calc = h, gamma, q, s, qp
    β_param = beta

    if Paardekooper:
        h_calc, γ_calc, q_calc, s_calc, qp_calc = (
            pk11_h, pk11_gamma, pk11_qβ, pk11_sα, pk11_qp
        )
        γ_eff = γ_calc

    if chi is not None:
        γ_eff = gamma_eff_diffusion(γ_calc, chi, h_calc)
    else:
        γ_eff = gamma_eff_theory(γ_calc, β_param, h_calc)


    # Table 1 coefficients
    ξ = q_calc - (γ_calc - 1.0)*s_calc
    Γ_hs_b  = 1.1*(1.5 - s_calc)
    Γ_lin_b = 0.7*(1.5 - s_calc)
    Γ_hs_e  = 7.9*ξ/γ_eff
    Γ_lin_e = (2.2 - 1.4/γ_eff)*ξ

    # saturation functions F, G, K
    F = lambda p: 1.0/(1.0 + (p/1.3)**2)

    def G(p):
        p = np.asarray(p, float)
        pc = np.sqrt(8/(45*np.pi))
        ps = np.maximum(p, 1e-20)
        t1 = (16/25)*(45*np.pi/8)**0.75 * ps**1.5
        t2 = 1 - (9/25)*(8/(45*np.pi))**(4/3)*ps**(-8/3)
        out = np.where(p<pc, t1, t2)
        return np.clip(out, 0, 1)

    def K(p):
        p = np.asarray(p, float)
        pc = np.sqrt(28/(45*np.pi))
        ps = np.maximum(p, 1e-20)
        t1 = (16/25)*(45*np.pi/28)**0.75 * ps**1.5
        t2 = 1 - (9/25)*(28/(45*np.pi))**(4/3)*ps**(-8/3)
        out = np.where(p<pc, t1, t2)
        return np.clip(out, 0, 1)



    # horseshoe width
    C   = 1.1*γ_eff**(-0.25)
    x_s = C*np.sqrt(qp_calc/h_calc)
    xs3 = np.maximum(x_s**3, 0)

    # dynamic χ_p
    if Paardekooper:
        χ_p = pk11_chi_p
    else:
        if chi is not None:
            χ_p = chi
            #print(f"  USING CHI={χ_p}")
        else:
            ptrans=0.0
            χ_p = ((h_calc**2*ptrans**2) + ((x_s)**2*p_nu**2)) / (ptrans**2 + p_nu**2) / (β_param/(h_calc/0.05)) 
    



    # p_chi
    if χ_p<=0 or np.isinf(χ_p) or np.isnan(χ_p):
        pχ = np.inf
    else:
        denom = 2*np.pi*χ_p
        pχ = np.sqrt(xs3/np.maximum(denom, 1e-20))

    # combine (Eq 51–53)
    F_pχ  = F(pχ)
    G_pν  = G(p_nu)
    G_pχ  = G(pχ)
    K_pν  = K(p_nu)
    K_pχ  = K(pχ)

    term1 = Γ_hs_b  * F(p_nu) * G_pν
    term2 = (1-K_pν) * Γ_lin_b
    term3 = Γ_hs_e  * F(p_nu) * F_pχ * np.sqrt(np.maximum(0, G_pν*G_pχ))
    term4 = Γ_lin_e * np.sqrt(np.maximum(0, (1-K_pν)*(1-K_pχ)))

    return (term1 + term2 + term3 + term4) / γ_eff

##################################################################################
##################################################################################

def extract_pnu(simname, IDEFIX=False,chi_simulation=False, beta_simulation=True):
    base = determine_base_path(simname, IDEFIX=IDEFIX)
    summary_file = os.path.join(base, "idefix.0.log" if IDEFIX else "summary0.dat")
    par = read_parameters(summary_file, IDEFIX=IDEFIX)

    if IDEFIX:
        nu_p = float(par.get("nu", 1.0e-5))
        qp, _ = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)
        h = float(par.get("h0", 0.05))
        gamma = float(par.get("gamma", 1.4))
        b_over_h = float(par.get("smoothing", 0.4)) / h
        β = float(par.get("beta", 1.0))
        chi = float(par.get("chi", 1.0e-5))
    else:
        nu_p = float(par.get("NU", 1.0e-5))
        try:
            qp, _ = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)
            if qp == 0.0:
                print(f"\033[91m[WARNING]\033[0m qp = 0.0 in '{simname}'. Falling back to qp = 1.26e-5.")
                qp = 1.26e-5
        except Exception:
            print(f"\033[91m[WARNING]\033[0m Failed to extract qp from '{summary_file}'. Falling back to qp = 1.26e-5.")
            qp = 1.26e-5
        h = float(par.get("ASPECTRATIO", 0.05))
        gamma = float(par.get("GAMMA", 1.4))
        b_over_h = float(par.get("THICKNESSSMOOTHING", 0.4))
        β = float(par.get("BETA", 1.0))
        chi = float(par.get("CHI", 1.0e-5))

    rp = 1.0
    Omega_p = 1.0

    if chi_simulation==True:
        gamma_eff = gamma_eff_diffusion(gamma, chi, h)
    if beta_simulation==True:
        gamma_eff = gamma_eff_theory(gamma, β, h)



    xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / b_over_h) ** 0.25 * np.sqrt(qp / h)
    p_nu = (2.0 / 3.0) * np.sqrt((rp ** 2 * Omega_p * xs ** 3) / (2 * np.pi * nu_p))

    print(f"{simname:60s}  NU = {nu_p:.3e}  =>  p_nu = {p_nu:.3f}")

    return p_nu

##################################################################################
##################################################################################

def extract_xs(simname, IDEFIX=False, chi_simulation=False, beta_simulation=True):
    base = determine_base_path(simname, IDEFIX=IDEFIX)
    summary_file = os.path.join(base, "idefix.0.log" if IDEFIX else "summary0.dat")
    par = read_parameters(summary_file, IDEFIX=IDEFIX)

    if IDEFIX:
        qp, _ = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)
        h = float(par.get("h0", 0.05))
        gamma = float(par.get("gamma", 1.4))
        b_over_h = float(par.get("smoothing", 0.4)) / h
        β = float(par.get("beta", 1.0))
        chi = float(par.get("chi", 1.0e-5))
    else:
        qp, _ = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)
        h = float(par.get("ASPECTRATIO", 0.05))
        gamma = float(par.get("GAMMA", 1.4))
        b_over_h = float(par.get("THICKNESSSMOOTHING", 0.4))
        β = float(par.get("BETA", 1.0))
        chi = float(par.get("CHI", 1.0e-5))

    if chi_simulation:
        gamma_eff = gamma_eff_diffusion(gamma, chi, h)
    elif beta_simulation:
        gamma_eff = gamma_eff_theory(gamma, β, h)
    else:
        gamma_eff = gamma

    xs = 1.1 / (gamma_eff ** 0.25) * (0.4 / b_over_h) ** 0.25 * np.sqrt(qp / h)
    return xs


##################################################################################
##################################################################################
def get_sim_data(simlist, avg_interval=100.0, IDEFIX=False, read_chi=False,
                 chi_simulation=False, beta_simulation=True):
    xvals, ΓC_vals = [], []
    params = None

    for sim in simlist:
        try:
            base = determine_base_path(sim, IDEFIX=IDEFIX)
            param_file = os.path.join(base, "idefix.0.log" if IDEFIX else "summary0.dat")
            par = read_parameters(param_file, IDEFIX=IDEFIX)
            #if IDEFIX:
                #print(f"[DEBUG] Parameters read from {param_file}:")
                #for k, v in par.items():
                    #print(f"  {k!r}: {v!r}")


            if IDEFIX:
                β = float(par.get("beta", 1.0))
                γ = float(par.get("gamma", 1.4))
                h0 = float(par.get("h0", 0.05))
                qp, _ = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)
                q = 1.0 - 2.0 * float(par.get("flaringindex", 0.0))
                s = float(par.get("sigmaslope", 0.5))
                SIGMA0 = float(par.get("sigma0", 1.0))
                chi = float(par.get("kappa0", 1.0e-5))
            else:
                β = float(par.get("BETA", 1.0))
                γ = float(par.get("GAMMA", 1.6667))
                h0 = float(par.get("ASPECTRATIO", 0.05))
                qp, _ = extract_planet_mass_and_migration(summary_file, IDEFIX=IDEFIX)
                q = 1.0 - 2.0 * float(par.get("FLARINGINDEX", 0.5))
                s = float(par.get("SIGMASLOPE", 1.0))
                SIGMA0 = float(par.get("SIGMA0", 1.0))
                chi = float(par.get("CHI", 1.0e-5))

            GAM0 = (qp / h0) ** 2 * SIGMA0

            if beta_simulation:
                γ_eff = gamma_eff_theory(γ, β, h0)
            elif chi_simulation:
                γ_eff = gamma_eff_diffusion(γ, chi, h0)
            else:
                raise ValueError("Specify either beta_simulation=True or chi_simulation=True")

            G_Ls = -2.5 - 1.7 * q + 0.1 * s
            Γ_L = (GAM0 / γ_eff) * G_Ls

            tqwk_file = os.path.join(base, "tqwk0.dat")
            tarr, Tarr, _ = read_alternative_torque(tqwk_file, IDEFIX=IDEFIX)
            torb = tarr / (2 * np.pi)
            msk = torb >= (torb.max() - avg_interval)
            Γ_tot = np.mean(Tarr[msk]) * qp
            ΓC = (Γ_tot - Γ_L) / GAM0

            print(f"{sim}:  ΓC / Γ0 = {ΓC:.3f}")

            # Use CHI or P_nu as x-axis value
            if read_chi:
                try:
                    if IDEFIX:
                        raw_chi = par["kappa0"]
                    else:
                        raw_chi = par["CHI"]
                    chi_val = float(str(raw_chi).replace("d", "e").replace("D", "E"))
                except (KeyError, ValueError):
                    print(f"[WARNING] {sim}: Could not read CHI, using 1e-5")
                    chi_val = 1e-5
                xvals.append(chi_val)
            else:
                pnu = extract_pnu(sim, IDEFIX=IDEFIX, chi_simulation=chi_simulation, beta_simulation=beta_simulation)
                xvals.append(pnu)

            ΓC_vals.append(ΓC)

            if params is None:
                params = dict(beta=β, chi=chi, gamma=γ, h0=h0, qp=qp, q=q, s=s)

        except Exception as e:
            print(f"  Warning: skipping {sim}: {e}")

    return np.array(xvals), np.array(ΓC_vals), params




##################################################################################
##################################################################################


if __name__ == "__main__":


    all_data = {}
    all_params = {}


##################################################################################
##################################################################################


    # === BETA Cooling Plot ===


    # define your three simlists
    betas    = [0.001,1.0, 10.0, 100.0,1000.0]
    simlists = {
        0.001:  ["cos_Pnu013_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
                  "cos_Pnu020_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
                  "cos_Pnu033_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
                  "cos_Pnu046_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
                  "cos_Pnu066_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
                  "cos_Pnu093_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
                  "cos_Pnu13_beta1dm3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"],

        1.0:  [f"cos_Pnu{p:02d}_beta1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
               for p in (2,3,5,7,10,14,20)],
        10.0: [f"cos_Pnu{p:02d}_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
               for p in (2,3,5,7,10,14,20)],
        100.0: [f"cos_Pnu{p:02d}_beta100_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
                for p in (2,3,5,7,10,14,20)],
        1000.0: [f"cos_Pnu{p:02d}_beta1000_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
                 for p in (2,3,5,7,10,14,20)]
    }

    for beta_key in betas: # Renamed 'beta' to 'beta_key' to avoid conflict
        pnu, GC, params = get_sim_data(simlists[beta_key], beta_simulation=True)
        all_data[beta_key] = (pnu, GC)
        all_params[beta_key] = params


    # pnu_plot was used for beta cooling plot, can redefine for chi plot if range differs
    pnu_plot_beta = np.logspace(np.log10(0.1), np.log10(10), 400)


    plt.figure(figsize=(7, 5))
    styles = {
        0.001: ("ks", "k-"),      # black square, solid line
        1.0:   ("ro", "r--"),
        10.0:  ("gs", "g-."),
        100.0: ("b^", "b:"),
        1000.0:("md", "m-")
    }
    for key in betas:
        pnu_data, GC_data = all_data[key] # Renamed variables for clarity
        mk, ls = styles[key]
        plt.plot(pnu_data, GC_data, mk, label=f"Sim, β={key}")

        pr_data = all_params[key] # Renamed for clarity
        theo_data = np.array([ # Renamed for clarity
            compute_theoretical_corotation_paarde2011(
                p_val, h=pr_data['h0'], gamma=pr_data['gamma'], q=pr_data['q'], s=pr_data['s'],
                beta=key, qp=pr_data['qp']
            ) for p_val in pnu_plot_beta # Use pnu_plot_beta
        ])
        plt.plot(pnu_plot_beta, theo_data, ls, label=f"Theory, β={key}")

    plt.xscale('log')
    plt.xlabel(r"$p_\nu$")
    plt.ylabel(r"$\Gamma_C / \Gamma_0$")
    plt.title(r"Corotation torque for varying $p_\nu$ (cooling time $\beta$)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("corotation_beta.pdf")
    print("Beta plot saved as corotation_beta.pdf")

    try:
        from data_storage import scp_transfer
        scp_transfer("corotation_beta.pdf", "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        print("Successfully transferred corotation_beta.pdf")
    except Exception as e:
        print(f"SCP transfer for corotation_beta.pdf failed: {e}")




##################################################################################
##################################################################################




    # === Simulation Lists ===
    fargo_list = [
        "cos_Pnu033_Chir1dm7_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm7_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm7_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm5_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm5_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir4dm5_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
    ]

    idefix_list = [
        "cos_Pnu033_Chir1dm7_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX",
        "cos_Pnu033_Chir2dm7_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX",
        "cos_Pnu033_Chir5dm7_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX",
        "cos_Pnu033_Chir1dm6_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX",
        "cos_Pnu033_Chir5dm6_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX",
        "cos_Pnu033_Chir1dm5_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX",
        "cos_Pnu033_Chir4dm5_gam75_ss05_q1_r0416_PK11Fig6_2D_IDEFIX"
    ]

    # === Load Simulation Data ===
    chi_fargo, gc_fargo, params = get_sim_data(fargo_list, IDEFIX=False, read_chi=True, chi_simulation=True)
    chi_idefix, gc_idefix, _ = get_sim_data(idefix_list, IDEFIX=True, read_chi=True, chi_simulation=True)

    # === Theoretical Curve ===
    chi_vals = np.logspace(-7.5, -4, 3000)
    theo_gc = [
        compute_theoretical_corotation_paarde2011(
            0.33, h=params["h0"], gamma=params["gamma"],
            q=params["q"], s=params["s"],
            beta=None, qp=params["qp"], chi=chi_val
        ) for chi_val in chi_vals
    ]

    # === Plot ===
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.plot(chi_vals, theo_gc, 'k--', label=r"Theory ($p_\nu = 0.33$)")
    ax.plot(chi_fargo, gc_fargo, 'bo', label="FARGO3D")
    ax.plot(chi_idefix, gc_idefix, 'ms', label="IDEFIX")

    ax.set_xscale("log")
    ax.set_xlim(1e-7, 5e-5)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"$\Gamma_C / \Gamma_0$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    # === Save and Transfer ===
    fname = "corotation_fargo_vs_idefix.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"✅ Plot saved as {fname}")

    try:
        scp_transfer(fname, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        print("✅ SCP transfer complete")
    except Exception as e:
        print(f"❌ SCP transfer failed: {e}")


##################################################################################
##################################################################################


#========== VARYING CHI WITH FIXED PNU PLOT ===========

    # === Define Simulation Series ===
    fargo_chi_CN = [
        "cos_Pnu033_Chir1dm7_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm7_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm7_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm6_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm6_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm5_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm5_S_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir4dm5_E_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
    ]

    fargo_chi_EF = [
        "cos_Pnu033_Chir1dm7_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm7_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm7_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm5_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm5_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir4dm5_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
    ]

    fargo_chi_PC = [
        "cos_Pnu033_Chir1dm7_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm7_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm7_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir5dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir1dm5_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir2dm5_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
        "cos_Pnu033_Chir4dm5_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
    ]

    fargo_pnuvar_CN = [
"cos_Pnu013_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu020_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu033_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu046_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu066_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu093_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu13_Chir3dm6_S_CN_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
]

    fargo_pnuvar_EF = [
"cos_Pnu012_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob", #awkward sim naming
"cos_Pnu020_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu033_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu046_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu066_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu093_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu13_Chir3dm6_T_EF_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
]


    fargo_pnuvar_PC = [
"cos_Pnu012_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob", #awkward sim naming
"cos_Pnu020_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu033_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu046_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu066_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu093_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob",
"cos_Pnu13_Chir3dm6_T_PC_gam75_ss05_q1_r0416_PK11Fig6_2D_glob"
]




    # === Load Simulation Data ===
    chi_CN, gc_CN, params = get_sim_data(fargo_chi_CN, IDEFIX=False, read_chi=True, chi_simulation=True)
    chi_EF, gc_EF, _ = get_sim_data(fargo_chi_EF, IDEFIX=False, read_chi=True, chi_simulation=True)
    chi_PC, gc_PC, _ = get_sim_data(fargo_chi_PC, IDEFIX=False, read_chi=True, chi_simulation=True)
    pnu_CN, gc_pnu_CN, _ = get_sim_data(fargo_pnuvar_CN, IDEFIX=False, chi_simulation=True)
    pnu_EF, gc_pnu_EF, _ = get_sim_data(fargo_pnuvar_EF, IDEFIX=False, chi_simulation=True)
    pnu_PC, gc_pnu_PC, _ = get_sim_data(fargo_pnuvar_PC, IDEFIX=False, chi_simulation=True)

    # === Theoretical Curves ===
    chi_vals = np.logspace(-7.5, -4, 3000)
    theo_gc_pnu033 = [
        compute_theoretical_corotation_paarde2011(
            0.33, h=params["h0"], gamma=params["gamma"],
            q=params["q"], s=params["s"],
            beta=None, qp=params["qp"], chi=chi_val
        ) for chi_val in chi_vals
    ]

    all_pnu_vals = np.concatenate([pnu_CN, pnu_EF, pnu_PC])
    pnu_range = np.logspace(np.log10(min(all_pnu_vals)*0.8), np.log10(max(all_pnu_vals)*1.2), 3000)
    theo_gc_chi2em6 = [
        compute_theoretical_corotation_paarde2011(
            pnu_val, h=params["h0"], gamma=params["gamma"],
            q=params["q"], s=params["s"],
            beta=None, qp=params["qp"], chi=3e-6
        ) for pnu_val in pnu_range
    ]


    plt.rcParams.update({
        "font.size": 18,            # base font size
        "axes.titlesize": 18,       # title size
        "axes.labelsize": 18,       # x/y label size
        "xtick.labelsize": 16,      # tick label size
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    # === Plot: Two Panels Side-by-Side ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Panel 1: Varying chi --
    ax1.plot(chi_vals, theo_gc_pnu033, 'k--', label=r"Theory ($p_\nu = 0.33$)")
    ax1.plot(chi_CN, gc_CN, 'bo', label=r"FARGO3D CN (Entropy)")
    #ax1.plot(chi_EF, gc_EF, 'ms', label=r"FARGO3D EF (Temperature)")
    ax1.plot(chi_PC, gc_PC, 'ms', label=r"FARGO3D PC (Temperature)")
    ax1.set_xscale("log")
    ax1.set_xlim(1e-7, 5e-5)
    ax1.set_ylim(0.5, 4)
    ax1.set_xlabel(r"$\chi$")
    ax1.set_ylabel(r"$\Gamma_C / \Gamma_0$")
    #ax1.set_title(r"Corotation Torque vs $\chi$ at $p_\nu = 0.33$")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # -- Panel 2: Varying p_nu at fixed chi=2e-6 --
    ax2.plot(pnu_range, theo_gc_chi2em6, 'k--', label=r"Theory ($\chi = 3{\times}10^{-6}$)")
    ax2.plot(pnu_CN, gc_pnu_CN, 'bo', label=r"FARGO3D CN (Entropy)")
    #ax2.plot(pnu_EF, gc_pnu_EF, 'ms', label=r"FARGO3D EF (Temperature)")
    ax2.plot(pnu_PC, gc_pnu_PC, 'ms', label=r"FARGO3D PC (Temperature)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.1, 1.5)
    ax2.set_ylim(0.5, 4)
    ax2.set_xlabel(r"$p_\nu$")
    ax2.set_ylabel("")  # remove y-axis label
    ax2.tick_params(labelleft=False)  # keep ticks and grid, hide labels
    #ax2.set_title(r"Corotation Torque vs $p_\nu$ at $\chi = 2{\times}10^{-6}$")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()

    # === Save and Transfer ===
    fname = "corotation_diffusion_variation_chi_and_pnu.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"✅ Plot saved as {fname}")

    try:
        scp_transfer(fname, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        print("✅ SCP transfer complete")
    except Exception as e:
        print(f"❌ SCP transfer failed: {e}")

##################################################################################
##################################################################################


    # === FINAL BETA-COOLING PLOT ===

    fargo_beta = [
        "cos_Pnu033_beta1dm2_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta1dm1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta3d0_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta30_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta50_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta100_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta300_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta1000_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta3000_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
    ]
    beta_vals = [0.01,0.1, 1, 3, 10, 30,50, 100, 300, 1000, 3000]

    fargo_pnu = [
        f"cos_Pnu{p:02d}_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
        for p in (2, 3, 5, 7, 10, 14, 20)
    ]

    # === Load Datasets ===
    pnu_beta, gc_beta, params = get_sim_data(fargo_beta, IDEFIX=False, beta_simulation=True)
    pnu_pnu, gc_pnu, _        = get_sim_data(fargo_pnu, IDEFIX=False, chi_simulation=False, beta_simulation=True)

    # === Print Torques for pnu set ===
    print("\n" + "="*80)
    print("🌊 FARGO3D Beta=10: P_nu values and Corotation Torques")
    print("="*80)
    for name, pnu, gc in zip(fargo_pnu, pnu_pnu, gc_pnu):
        print(f"{name:65s}  p_nu = {pnu:.2f},  ΓC / Γ0 = {gc:.3f}")

    # === Compute Theory Curve for left panel ===
    beta_range = np.logspace(-3, 4, 3000)
    theo_gc_beta = [
        compute_theoretical_corotation_paarde2011(
            0.33, h=params["h0"], gamma=params["gamma"],
            q=params["q"], s=params["s"],
            beta=beta_val, qp=params["qp"], chi=None
        ) for beta_val in beta_range
    ]

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 18,            # base font size
        "axes.titlesize": 18,       # title size
        "axes.labelsize": 18,       # x/y label size
        "xtick.labelsize": 16,      # tick label size
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    # === Plotting: Two Panels Side-by-Side ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Panel 1: Varying β at fixed p_nu = 0.33 --
    ax1.plot(beta_range, theo_gc_beta, 'k--', label=r"Theory ($p_\nu = 0.33$)")
    ax1.plot(beta_vals, gc_beta, 'ro', label=r"FARGO3D (Beta-cooling)")
    ax1.set_xscale("log")
    ax1.set_xlim(0.005, 3200)
    ax1.set_ylim(0.5, 4)
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$\Gamma_C / \Gamma_0$")
    #ax1.set_title(r"Corotation Torque vs $\beta$ at $p_\nu = 0.33$")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # -- Panel 2: Varying p_nu at fixed β = 10 --
    pnu_range = np.logspace(np.log10(min(pnu_pnu)*0.8), np.log10(max(pnu_pnu)*1.2), 3000)
    theo_gc_pnu_range = [
        compute_theoretical_corotation_paarde2011(
            pnu_val, h=params["h0"], gamma=params["gamma"],
            q=params["q"], s=params["s"],
            beta=10.0, qp=params["qp"], chi=None
        ) for pnu_val in pnu_range
    ]

    ax2.plot(pnu_range, theo_gc_pnu_range, 'k--', label=r"Theory ($\beta = 10$)")
    ax2.plot(pnu_pnu, gc_pnu, 'ro', label=r"FARGO3D (Beta-cooling)")
    ax2.set_xscale("log")
    ax2.set_xlim(min(pnu_range), max(pnu_range))
    ax2.set_ylim(0.5, 4)
    ax2.set_xlabel(r"$p_\nu$")
    ax2.tick_params(labelleft=False)  # <-- keep ticks/grid but hide numbers
    ax2.set_ylabel("")                # <-- remove y-axis label
    #ax2.set_title(r"Corotation Torque vs $p_\nu$ at $\beta = 10$")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()


    # --- Critical p_nu values (vertical dashed lines) ---

    pnu_crit_nl = 1.0
    pnu_crit_lin = np.sqrt(3.0 / 20.0)  # ≈ 0.387

    for x, style, lab in zip(
        [pnu_crit_nl, pnu_crit_lin],
        ['--', ':'],
        [r"$p_{\nu, \rm crit,NL}$", r"$p_{\nu, \rm crit,lin}$"]
    ):
        ax2.axvline(x, color='k', linestyle=style, lw=2)
        ax2.text(
            x * 1.05,
            0.55 if lab.endswith('lin}$') else 0.7,
            lab, color='k', rotation=90, va='bottom', ha='left', fontsize=14,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
        )

    # --- Critical cooling times (vertical dashed lines) ---

    # Import extract_xs at the top if needed:
    # from your_module import extract_xs

    # Dynamically obtain xs from a representative simulation (here: beta=1)
    sim_for_xs = fargo_beta[7]  # e.g., index 2 for beta=1; change if needed!
    xs = extract_xs(sim_for_xs, IDEFIX=False, chi_simulation=False, beta_simulation=True)

    beta_crit_nl = 4 * np.pi / (3 * xs)
    beta_crit_lin = np.pi / (5 * xs)

    for x, style, lab in zip(
        [beta_crit_nl, beta_crit_lin],
        ['--', ':'],
        [r"$\beta_{\rm crit,NL}$", r"$\beta_{\rm crit,lin}$"]
    ):
        ax1.axvline(x, color='k', linestyle=style, lw=2)
        ax1.text(
            x * 1.05,
            0.55 if lab.endswith('lin}$') else 0.7,
            lab, color='k', rotation=90, va='bottom', ha='left', fontsize=14,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
        )


    fig.tight_layout()

    # === Save and Transfer ===
    fname = "corotation_torque_variation_beta_and_pnu.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"✅ Plot saved as {fname}")

    try:
        scp_transfer(fname, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        print("✅ SCP transfer complete")
    except Exception as e:
        print(f"❌ SCP transfer failed: {e}")



##################################################################################
##################################################################################


    # === Simulation Lists ===
    fargo_beta = [
        "cos_Pnu033_beta1dm2_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta1dm1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta3d0_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta10_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta30_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta50_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta100_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta300_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu05_beta1000_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D",
        "cos_Pnu033_beta3000_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D"
    ]
    beta_vals_fargo = [0.01, 0.1, 1, 3, 10, 30, 50, 100, 300, 1000, 3000]

    idefix_beta = [
        "cos_Pnu033_beta1dm2_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX",
        "cos_Pnu033_beta1dm1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX",
        "cos_Pnu033_beta1d0_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX",
        "cos_Pnu033_beta1d1_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX",
        "cos_Pnu033_beta50_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX",
        "cos_Pnu033_beta1d2_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX",
        "cos_Pnu033_beta1d3_gam75_ss05_q1_r0516_PK11Fig6_HR150_2D_IDEFIX"
    ]
    beta_vals_idefix = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 1000.0]  # Match ordering above

    # === Load Torques ===
    _, gc_fargo, params = get_sim_data(fargo_beta, IDEFIX=False, beta_simulation=True)
    _, gc_idefix, _     = get_sim_data(idefix_beta, IDEFIX=True,  beta_simulation=True)

    # === Theory Curve ===
    beta_range = np.logspace(-3, 4, 3000)
    theo_gc = [
        compute_theoretical_corotation_paarde2011(
            0.33, h=params["h0"], gamma=params["gamma"],
            q=params["q"], s=params["s"], beta=bval, qp=params["qp"], chi=None
        ) for bval in beta_range
    ]

    # === Plot ===
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.plot(beta_range, theo_gc, 'k--', label=r"Theory ($p_\nu = 0.33$)")
    ax.plot(beta_vals_fargo, gc_fargo, 'ro', label="FARGO3D")
    ax.plot(beta_vals_idefix, gc_idefix, 'ms', label="IDEFIX")

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e4)
    ax.set_ylim(0.5, 4)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Gamma_C / \Gamma_0$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()

    # === Save and Transfer ===
    fname = "corotation_beta_fargo_vs_idefix.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"✅ Plot saved as {fname}")

    try:
        scp_transfer(fname, "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        print("✅ SCP transfer complete")
    except Exception as e:
        print(f"❌ SCP transfer failed: {e}")

##################################################################################
##################################################################################

    # === Append CHI panel to Γ(t) figure ===
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8, 15), sharex=True)
    for i, β in enumerate(betas):
        ax = axs[i]
        for sim in simlists[β]:
            try:
                base = determine_base_path(sim)
                pfile = os.path.join(base, "summary0.dat")
                par   = read_parameters(pfile)
                qp, _ = extract_planet_mass_and_migration(pfile)
                gam   = float(par.get("GAMMA", 1.6667))
                h0    = float(par.get("ASPECTRATIO", 0.05))
                beta_val = float(par.get("BETA", β))
                γ_eff = gamma_eff_theory(gam, beta_val, h0)
                GAM0  = (qp/h0)**2 * float(par.get("SIGMA0", 1.0))

                tfile = os.path.join(base, "tqwk0.dat")
                t_arr, torque_arr, _ = read_alternative_torque(tfile)
                torb = t_arr / (2*np.pi)
                gamma_torque = torque_arr * qp / (GAM0 / γ_eff)
                pnu_label = extract_pnu(sim)
                ax.plot(torb, gamma_torque, label=rf"$p_\nu={pnu_label:.1f}$")
            except Exception as e:
                print(f"  Warning: skipping {sim}: {e}")
        ax.set_title(rf"$\beta = {β:g}$")
        ax.set_ylabel(r"$\Gamma / (\Gamma_0 / \gamma_{\mathrm{eff}})$")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize="small", loc="upper right")

    # Panel for CHI
    ax = axs[-1]
    for sim in fargo_list_CN1dm5:
        try:
            base = determine_base_path(sim)
            pfile = os.path.join(base, "summary0.dat")
            par   = read_parameters(pfile)
            qp, _ = extract_planet_mass_and_migration(pfile)
            gam   = float(par.get("GAMMA", 1.6667))
            h0    = float(par.get("ASPECTRATIO", 0.05))
            chi_val = float(par.get("CHI", 1e-5))
            γ_eff = gamma_eff_diffusion(gam, chi_val, h0)
            GAM0  = (qp/h0)**2 * float(par.get("SIGMA0", 1.0))

            tfile = os.path.join(base, "tqwk0.dat")
            t_arr, torque_arr, _ = read_alternative_torque(tfile)
            torb = t_arr / (2*np.pi)
            gamma_torque = torque_arr * qp / (GAM0 / γ_eff)
            pnu_label = extract_pnu(sim)
            ax.plot(torb, gamma_torque, label=rf"$p_\nu={pnu_label:.1f}$")
        except Exception as e:
            print(f"  Warning: skipping {sim}: {e}")
    ax.set_title(r"Heat diffusion ($\chi = 10^{-5}$)")
    ax.set_xlabel("Time (orbits)")
    ax.set_ylabel(r"$\Gamma / (\Gamma_0 / \gamma_{\mathrm{eff}})$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize="small", loc="upper right")
    ax.set_xlim(1e-7, 5e-5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("torque_evolution_with_chi.pdf")
    plt.savefig("torque_evolution_with_chi.png")
    print("Saved CHI panel torque evolution plot.")

    try:
        scp_transfer("torque_evolution_with_chi.pdf", "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
        scp_transfer("torque_evolution_with_chi.png", "/Users/mariuslehmann/Downloads/Profiles/", username="mariuslehmann")
    except Exception as e:
        print(f"SCP transfer failed: {e}")
