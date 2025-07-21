import argparse
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from data_storage import determine_base_path, scp_transfer
from planet_data import debug_alternative_torque, read_alternative_torque, read_torque_data, extract_planet_mass_and_migration, compute_theoretical_torques
from data_reader import read_parameters

"""
This script compares the torques from multiple hydrodynamic simulations 
and plots their time-averaged values for analysis. Theoretical torque 
values are also computed and plotted for reference.

### Command-line Flags:
  --LowMass       : Compare torques for a set of low-mass planet simulations.
  --simple        : Compare only two simulations (isothermal vs adiabatic).
  --Isothermal    : Compare only isothermal cooling times (β < 1).
  --Adiabatic     : Compare only adiabatic cooling times (β > 1).
  --zdomain       : Compare simulations with different vertical domain sizes.
  --SmoothingLength : Compare simulations with different smoothing lengths.
  --Paardekooper9 : Reproduce torque evolution from Paardekooper et al. (2010), Fig. 1.
  --Paardekooper2 : Reproduce torque evolution from Paardekooper et al. (2010), Fig. 2.
  --Paardekooper6 : Compare entropy-related torques from Paardekooper et al. (2010), Fig. 6.
  --Numerics      : Compare torques of 2D and 3D simulations with different numerical parameters.
  --fix-yrange    : Fix the vertical axis range to [-5e-11, 1e-11] for consistency.

Each flag selects a different set of simulations and modifies the output filename
to indicate the selected comparison type.
"""

def extract_eq_label(args):
    """
    Extracts the equation label dynamically based on which Paardekooper flag is set in args.
    """
    eq_label_map = {
        "Paardekooper9": "Paardekooper9",
        "Paardekooper2": "Paardekooper2",
        "Paardekooper1": "Paardekooper1",
        "Paardekooper6": "Paardekooper6"
    }
    
    return next((eq_label_map[key] for key in eq_label_map if getattr(args, key, False)), None)

def dynamic_smoothing(torque, window_size):
    """
    Smooths the torque data with a dynamic window size near the boundaries.
    """
    smoothed = np.zeros_like(torque)
    n = len(torque)

    for i in range(n):
        left = max(0, i - window_size // 2)
        right = min(n, i + window_size // 2 + 1)
        smoothed[i] = np.mean(torque[left:right])

    return smoothed

def plot_torque_comparison(
        simulation_names, labels, output_path,
        high_mass=False, low_mass=False, simple=False, 
        isothermal=False, adiabatic=False, zdomain=False, 
        smoothing_length=False, Paardekooper9=False, Paardekooper2=False, 
        Paardekooper1=False, Paardekooper6=False, Numerics=False, fix_yrange=False
):
    """
    Plots the time-averaged torques of multiple simulations in one figure for comparison.
    """
    # Apply different figure size if Paardekooper case is selected
    fig_size = (6, 6) if args.Paardekooper9 else (10, 6)
    fig, ax = plt.subplots(figsize=fig_size)

    # Compute theoretical torques only once, using the first simulation
    first_simulation = simulation_names[0]
    first_base_path = determine_base_path(first_simulation)
    first_summary_file = f"{first_base_path}/summary0.dat"
    first_parameters = read_parameters(first_summary_file)
    qp, _ = extract_planet_mass_and_migration(first_summary_file)

    # Compute theoretical torques only for the first simulation
    predicted_torque_adi, predicted_torque_iso, GAM0 = compute_theoretical_torques(first_parameters, qp)

    # Define colormap
    colormap = cm.get_cmap("viridis", len(simulation_names))
    colors = [colormap(i) for i in range(len(simulation_names))]

    # For Paardekooper2/3/6 storage
    if args.Paardekooper2 or args.Paardekooper1 or args.Paardekooper6:
        theo_values = []
        theo_colors = []
    
    # For Numerics case, store theoretical values for each simulation
    if args.Numerics:
        # For Lindblad torques (Paardekooper2)
        lindblad_theo_values = []
        lindblad_theo_colors = []
        lindblad_labels = []
        
        # For full unsaturated torques (Equation45)
        full_theo_values = []
        full_theo_colors = []
        full_labels = []
        
        # Track which types we've shown in legends
        shown_2d_lindblad = False
        shown_3d_lindblad = False
        shown_2d_full = False
        shown_3d_full = False
    
    # Set rolling window size based on case
    rolling_window_size = 1  # Default: 1 = 1/20 ORBIT!
    if args.Paardekooper9 or args.Paardekooper6:
        rolling_window_size = 200 #10
    if args.Paardekooper2 or args.Numerics:
        rolling_window_size = 400
    if args.Paardekooper1:
        rolling_window_size = 1
        
    # Calculate smoothing timescale in orbits for the title
    smoothing_timescale = rolling_window_size / 20  # Convert to orbits (1 = 1/20 ORBIT)

    for i, (simulation_name, label) in enumerate(zip(simulation_names, labels)):
        base_path = determine_base_path(simulation_name)
        summary_file = f"{base_path}/summary0.dat"
        parameters = read_parameters(summary_file)

        # Extract ZMAX and compute scaling factor for zdomain
        delta_z = parameters['ZMAX']
        scaling_factor = 1 / (2 * delta_z / 0.00625) if args.zdomain else 1.0

        # Extract individual smoothing length
        b = parameters['THICKNESSSMOOTHING']
        gam = parameters['GAMMA']

        # Compute theoretical torque for each simulation when needed
        if args.SmoothingLength:
            _, predicted_torque_iso, GAM0 = compute_theoretical_torques(parameters, qp)
        if args.Paardekooper9 or args.Paardekooper2 or args.Paardekooper1 or args.Paardekooper6:
            predicted_torque_adi, predicted_torque_iso, GAM0 = compute_theoretical_torques(parameters, qp, eq_label=extract_eq_label(args))
        
        # For Numerics case, compute two different theoretical torques
        if args.Numerics:
            # Determine if simulation is 2D or 3D
            zmin = parameters['ZMIN']
            sim_type = "2D" if zmin == 0 else "3D"
            
            # 1. Compute Lindblad torque (Paardekooper2)
            lindblad_torque_adi, lindblad_torque_iso, GAM0 = compute_theoretical_torques(parameters, qp, eq_label="Paardekooper2")
            
            # Store Lindblad theoretical values for later plotting
            normalized_lindblad_adi = (lindblad_torque_adi / GAM0) * gam
            lindblad_theo_values.append(normalized_lindblad_adi)
            lindblad_theo_colors.append(colors[i])
            lindblad_labels.append(f"Lindblad Torque ({sim_type})" if 
                                 (sim_type == "2D" and not shown_2d_lindblad) or 
                                 (sim_type == "3D" and not shown_3d_lindblad) else None)
            
            # Track which types we've shown in the legend for Lindblad
            if sim_type == "2D":
                shown_2d_lindblad = True
            else:
                shown_3d_lindblad = True
                
            # 2. Compute full unsaturated torque (Equation45)
            full_torque_adi, full_torque_iso, _ = compute_theoretical_torques(parameters, qp, eq_label="Equation45")
            
            # Store full theoretical values for later plotting
            normalized_full_adi = (full_torque_adi / GAM0) * gam
            full_theo_values.append(normalized_full_adi)
            full_theo_colors.append(colors[i])
            full_labels.append(f"Full Unsaturated Torque ({sim_type})" if 
                              (sim_type == "2D" and not shown_2d_full) or 
                              (sim_type == "3D" and not shown_3d_full) else None)
            
            # Track which types we've shown in the legend for full torque
            if sim_type == "2D":
                shown_2d_full = True
            else:
                shown_3d_full = True
            
            # Use Lindblad torque values for the primary calculation
            predicted_torque_adi = lindblad_torque_adi
            predicted_torque_iso = lindblad_torque_iso
        
        # For Numerics case, we need to compute GAM0 per simulation
        sim_GAM0 = GAM0

        torque_file = f"{base_path}/monitor/gas/torq_planet_0.dat"
        tqwk_file = f"{base_path}/tqwk0.dat"

        if os.path.exists(torque_file):
            date_torque, torque = read_torque_data(torque_file)
        elif os.path.exists(tqwk_file):
            date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file)
        else:
            print(f"Torque file not found for simulation {simulation_name}")
            continue

        time_in_orbits = date_torque / (2 * np.pi)
        time_averaged_torque = dynamic_smoothing(torque, rolling_window_size)
        #time_averaged_torque = torque #DEBUG

        # Scale torques for zdomain comparison
        scaled_torque = qp * time_averaged_torque
        if args.zdomain:
            scaled_torque *= scaling_factor

        # Handle different cases
        if args.Paardekooper2 or args.Paardekooper1 or args.Paardekooper6:
            # Plot (Γ/Γ₀) × γ without theoretical lines yet
            ax.plot(time_in_orbits, (scaled_torque / GAM0) * gam, label=label, color=colors[i])
            
            # Calculate and store the theoretical value for later
            normalized_predicted_adi = (predicted_torque_adi / GAM0) * gam
            theo_values.append(normalized_predicted_adi)
            theo_colors.append(colors[i])
        elif args.zdomain:
            # For zdomain case, scale with absolute theoretical isothermal torque
            abs_iso_torque = abs(predicted_torque_iso)
            ax.plot(time_in_orbits, scaled_torque / abs_iso_torque, label=label, color=colors[i])
        elif args.Numerics:
            # For Numerics case, use the simulation-specific GAM0 for scaling
            ax.plot(time_in_orbits, (scaled_torque / sim_GAM0) * gam, label=label, color=colors[i])
        else:
            # For all other cases, plot Γ/Γ₀
            ax.plot(time_in_orbits, scaled_torque / GAM0, label=label, color=colors[i])
            
            # Plot theoretical torque values if needed for this simulation
            if args.SmoothingLength:
                ax.axhline(predicted_torque_iso / GAM0, color=colors[i], linestyle='dashed', 
                          label=f"Theoretical Torque (ISO) b={b:.2f}")
            
            # For Paardekooper9, add the theoretical line for each simulation
            if args.Paardekooper9:
                ax.axhline(predicted_torque_adi / GAM0, color=colors[i], linestyle='dashed', 
                          label=f"Theoretical Torque (ISO)")

    # After the loop, plot all theoretical lines for Paardekooper2/3
    if args.Paardekooper2 or args.Paardekooper1:
        # Add a single "Theoretical Torque (AD)" entry to the legend
        for i, (value, color) in enumerate(zip(theo_values, theo_colors)):
            if i == 0:
                # Plot the first one with the label (will appear in legend)
                ax.axhline(value, color=color, linestyle='dotted', 
                          label='Theoretical Torque (AD)')
            else:
                # Plot subsequent lines without label (won't appear in legend)
                ax.axhline(value, color=color, linestyle='dotted')
    
    # For Paardekooper6, only plot the first and last theoretical lines
    if args.Paardekooper6 and len(theo_values) >= 2:
        # Plot only the first and last theoretical values with labels
        ax.axhline(theo_values[0], color=theo_colors[0], linestyle='dotted', 
                   label='Predicted adiabatic torque')
        ax.axhline(theo_values[-1], color=theo_colors[-1], linestyle='dotted')
    
    # For Numerics case, plot both types of theoretical lines for each simulation
    if args.Numerics:
        # 1. Plot Lindblad torque lines (dotted)
        for value, color, label in zip(lindblad_theo_values, lindblad_theo_colors, lindblad_labels):
            ax.axhline(value, color=color, linestyle='dotted', label=label)
            
        # 2. Plot full unsaturated torque lines (dashed)
        for value, color, label in zip(full_theo_values, full_theo_colors, full_labels):
            ax.axhline(value, color=color, linestyle='dashed', label=label)

    # Apply theoretical torque lines based on the selected flag
    # Fixed: Now correctly handling high_mass and low_mass cases
    if (args.simple or args.Adiabatic or args.HighMass or args.LowMass) and not args.Paardekooper2 and not args.Paardekooper1 and not args.Paardekooper6 and not args.Numerics:
        ax.axhline(predicted_torque_adi / GAM0, color='black', linestyle='--', label='Theoretical Torque (AD)')
    
    # Handle zdomain case separately
    if args.zdomain:
        # For zdomain, set horizontal line at -1 (normalized by |Γ_ISO|)
        ax.axhline(-1, color='blue', linestyle='--', label='Theoretical Torque (ISO)')
    elif (args.simple or args.Isothermal or args.HighMass or args.LowMass) and not args.Adiabatic and not args.Paardekooper9 and not args.Paardekooper2 and not args.Paardekooper1 and not args.Paardekooper6 and not args.Numerics:
        ax.axhline(predicted_torque_iso / GAM0, color='blue', linestyle='--', label='Theoretical Torque (ISO)')

    # Set labels and formatting
    ax.set_xlabel(r"Time (orbits)", fontsize=14)
    
    # Set y-label based on case
    if args.Paardekooper2 or args.Paardekooper1 or args.Paardekooper6 or args.Numerics:
        ax.set_ylabel(r"$\gamma \Gamma/\Gamma_{0}$", fontsize=14)
    elif args.zdomain:
        ax.set_ylabel(r"$\Gamma/|\Gamma_{\mathrm{ISO}}|$", fontsize=14)
    else:
        ax.set_ylabel(r"$\Gamma/\Gamma_{0}$", fontsize=14)

    # Add title with smoothing timescale
    title = f"Torque Comparison (Smoothing: {smoothing_timescale:.1f} orbits)"
    plt.suptitle(title, fontsize=16)

    # Adjust axis limits for different cases
    if args.Paardekooper9:
        #ax.set_xlim([0, 20])
        ax.set_xlim([0, 1000])
        ax.set_ylim([-6, 1])
    if args.Paardekooper2:
        ax.set_ylim([-8, 4])
    if args.Paardekooper1:
        ax.set_ylim([-6, 4])
        ax.set_xlim([0, 50])
    if args.Paardekooper6:
        ax.set_ylim([-13, 4])
        ax.set_xlim([0, 20])
    if args.Adiabatic:
        ax.set_ylim([-80, 40])
    if args.Isothermal:
        ax.set_ylim([-40, 20])
        ax.set_xlim([0, 400])
    if args.zdomain:
        ax.set_ylim([-2, 2.5])
    if args.Numerics:
        ax.set_ylim([-8, 4])
        ax.set_xlim([0, 600])

    if adiabatic:
        # Place legend in upper left for Adiabatic case
        ax.legend(fontsize=12, loc="upper left", frameon=False)
    else:
        # Keep default position for all other cases
        ax.legend(fontsize=12, loc="upper right", frameon=False)
    ax.grid(False)

    # Apply fixed y-range if the flag is enabled
    if fix_yrange:
        ax.set_ylim([-5e-11, 1e-11])

    # Incorporate flag into filename
    flag_suffix = ""

    if args.Paardekooper9:
        flag_suffix += "_Paardekooper9"
    if args.Paardekooper2:
        flag_suffix += "_Paardekooper2"
    if args.Paardekooper1:
        flag_suffix += "_Paardekooper1"
    if args.Paardekooper6:
        flag_suffix += "_Paardekooper6"
    if args.LowMass:
        flag_suffix += "_LowMass"
    if args.simple:
        flag_suffix += "_simple"
    if args.Isothermal:
        flag_suffix += "_Isothermal"
    if args.Adiabatic:
        flag_suffix += "_Adiabatic"
    if args.zdomain:
        flag_suffix += "_zdomain"
    if args.SmoothingLength:
        flag_suffix += "_SmoothingLength"
    if args.HighMass:
        flag_suffix += "_HighMass"
    if args.Numerics:
        flag_suffix += "_Numerics"

    output_filename = f"{output_path}/torque_comparison{flag_suffix}.pdf"

    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Torque comparison plot saved to {output_filename}")

    local_directory = "/Users/mariuslehmann/Downloads/Profiles/planet_evolution/"
    scp_transfer(output_filename, local_directory, "mariuslehmann")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare torques from multiple simulations.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--LowMass", action="store_true", help="Use the LowMassPlanet simulation set.")
    group.add_argument("--HighMass", action="store_true", help="Use the HighMassPlanet simulation set.")
    group.add_argument("--simple", action="store_true", help="Compare only two simulations (isothermal vs adiabatic).")
    group.add_argument("--Isothermal", action="store_true", help="Compare only isothermal cooling times (β < 1).")
    group.add_argument("--Adiabatic", action="store_true", help="Compare only adiabatic cooling times (β > 1).")
    group.add_argument("--zdomain", action="store_true", help="Compare simulations with different vertical domain sizes.")
    group.add_argument("--SmoothingLength", action="store_true", help="Compare simulations with different smoothing lengths.")
    group.add_argument("--Paardekooper9", action="store_true", help="Reproduce Fig. 1 from Paardekooper et al. (2010).")
    group.add_argument("--Paardekooper2", action="store_true", help="Reproduce Fig. 2 from Paardekooper et al. (2010).")
    group.add_argument("--Paardekooper1", action="store_true", help="Compare smoothing length variations from Paardekooper cases.")
    group.add_argument("--Paardekooper6", action="store_true", help="Compare entropy-related torques from Paardekooper et al. (2010), Fig. 6.")
    group.add_argument("--Numerics", action="store_true", help="Compare torques of 2D and 3D simulations with different numerical parameters.")
    parser.add_argument("--fix-yrange", action="store_true", help="Fix the vertical axis range to [-5e-11, 1e-11].")

    args = parser.parse_args()

    simulation_names = []
    labels = []

    if args.Paardekooper9:
        simulation_names.extend([
            "cos_bet1dm6_ss0_fi0_gam1001_r0416_PaardekooperFig9_nu1dm11_COR",
            "cos_bet1dm6_ss0_fi0_gam1001_r0416_PaardekooperFig9_nu1dm11_3Dthin",  
            "cos_bet1dm6_ss15_fim05_gam1001_r0416_PaardekooperFig9_nu1dm11_COR",
            "cos_bet1dm6_ss15_fim05_gam1001_r0416_PaardekooperFig9_nu1dm11_3Dthin" 
        ])
        labels.extend([
            r"$\alpha=0, \beta=1$",
            r"$\alpha=0, \beta=1$ (3D thin)",
            r"$\alpha=3/2, \beta=2$",
            r"$\alpha=3/2, \beta=2$ (3D thin)"
        ])
        
    if args.Paardekooper2:
        simulation_names.extend([
            #"cos_bet1d6_ss0_fi0_nodust_r0416_sth_unstrat_PaardekooperFig2_nu1dm11_pramp10",
            "cos_bet1d6_ss0_fi0_r0416_PaardekooperFig2_nu1dm11_COR",
            "cos_bet1d6_ss15_fim05_r0416_PaardekooperFig2_nu1dm11_COR",
            #"cos_bet1d6_ss15_fim05_nodust_r0416_z05_PaardekooperFig2_nu1dm11_pramp10_3Dthin",
            #"cos_bet1d6_ss15_fim05_nodust_r0416_z05_PaardekooperFig2_nu1dm11_pramp10_3D",
            #"cos_bet1d6_ss15_fi05_nodust_r0416_sth_unstrat_PaardekooperFig2_nu1dm11_pramp10"
            "cos_bet1d6_ss15_fi05_r0416_PaardekooperFig2_nu1dm11_COR"
        ])
        labels.extend([
            #r"$\alpha=0, \beta=1$",
            r"$\alpha=0, \beta=1$",
            r"$\alpha=3/2, \beta=2$",
            #r"$\alpha=3/2, \beta=2$ (3D)",
            #r"$\alpha=3/2, \beta=2$ (3D thin)",
            r"$\alpha=3/2, \beta=0$"
        ])

    # Add to the simulation definitions section:
    if args.Paardekooper1:
        simulation_names.extend([
            "cos_bet1d6_ss15_fi0_r0416_PaardekooperFig1_nu1dm11_bh100_COR",
            "cos_bet1d6_ss15_fi0_r0416_PaardekooperFig1_nu1dm11_bh60_COR",
            "cos_bet1d6_ss15_fi0_r0416_PaardekooperFig1_nu1dm11_bh30_COR"
        ])
        labels.extend([
            r"$b/h=1.0$",
            r"$b/h=0.6$", 
            r"$b/h=0.3$"
        ])

    #
    # NEW CASE: Paardekooper6
    #
    if args.Paardekooper6:
        # Order them so that alpha=0,beta=1 is first (top line), alpha=3/2,beta=3/2 second (bottom line)
        simulation_names.extend([
            "cos_bet1d6_ss0_fi0_r0416_PaardekooperFig2_nu1dm11_COR",
            "cos_bet1d6_ss15_fi125_r0416_PaardekooperFig6_nu1dm11_COR"
        ])
        labels.extend([
            r"$\alpha=0, \beta=1$",
            r"$\alpha=3/2, \beta=3/2$"
        ])
        
    #
    # NEW CASE: Numerics
    #
    if args.Numerics:
        simulation_names.extend([
            "cos_bet1d6_ss0_fi0_r0416_PaardekooperFig2_nu1dm11_COR",
            #"cos_bet1d6_ss0_fi0_r0416_PaardekooperFig2_nu1dm11_COR_LRY",
            #"cos_bet1d6_ss0_fi0_r0416_PaardekooperFig2_nu1dm11_COR_LR",
            "cos_bet1d6_ss0_fi0_r0615_PaardekooperFig2_nu1dm11_COR_LR",
            "cos_bet1d6_ss0_fi0_r0615_PaardekooperFig2_nu1dm11_COR_LR_3Dthin_2",
            #"cos_bet1d6_ss0_fi0_r0615_z05_PaardekooperFig2_nu1dm11_COR_LR_3D",
            "cos_bet1d6_ss0_fi0_r0615_z05_PaardekooperFig2_nu1dm11_COR_LR_3D_2dbnd",
            #"cos_bet1d6_ss0_fi0_r0615_z0125_PKFig2_nu1dm11_COR_LR_3D_2dbnd_vbcunp",
            "cos_bet1d6_ss0_fi0_r0615_z05_PKFig2_nu1dm11_COR_LR_3D_2dbnd_reszx2",
            "cos_bet1d6_ss0_fi0_r0615_z1_PKFig2_nu1dm11_COR_LR_3D_2dbnd"
        ])
        labels.extend([
            r"$\alpha=0, \beta=1$",  
            #r"$\alpha=0, \beta=1$ (LRY)",    
            #r"$\alpha=0, \beta=1$ (LR)",     
            r"$\alpha=0, \beta=1$ (<$\Delta R$, LR)",                                                           
            r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, 3D thin)",
            #r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, 3D $\Delta z = 1 H$)",
            r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, 3D $\Delta z = 1 H$ & 2D bndrs)",
            #r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, 3D $\Delta z = 1 H$ & 2D bndrs & v. unp. bndrs.)",
            r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, 3D $\Delta z = 1 H$ & 2D bndrs & zresx2)",
            r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, 3D $\Delta z = 2 H$ & 2D bndrs)"
            #r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, $\Delta z=0.5H$)",
            #r"$\alpha=0, \beta=1$ (LR, $<\Delta R$, $\Delta z=H$)"
        ])

    if args.Isothermal:
        simulation_names.extend([
            "cos_bet1dm4_nodust_r0615_z05_sth_SEarth_unstrat_corr_gam10_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_corr_gam10_gpu",
            "cos_bet1dm2_nodust_r0615_z05_sth_SEarth_unstrat_corr_gam10_gpu",
            "cos_bet1dm1_nodust_r0615_z05_sth_SEarth_unstrat_corr_gam10_gpu"
        ])
        labels.extend([
            r"$\beta=10^{-4}$",
            r"$\beta=10^{-3}$",
            r"$\beta=10^{-2}$",
            r"$\beta=10^{-1}$"
        ])

    if args.Adiabatic:
        simulation_names.extend([
            "cos_bet5d2_nodust_r0615_z05_sth_SEarth_unstrat_corr_gpu",  
            "cos_bet1d3_nodust_r0615_z05_sth_SEarth_unstrat_corr_gpu",  
            "cos_bet1d4_nodust_r0615_z05_sth_SEarth_unstrat_corr_gpu",  
            "cos_bet1d5_nodust_r0615_z05_sth_SEarth_unstrat_corr_gpu",  
            "cos_bet1d6_nodust_r0615_z05_sth_SEarth_unstrat_corr_gpu"
        ])
        labels.extend([
            r"$\beta=5 \times 10^2$",
            r"$\beta=10^3$",
            r"$\beta=10^4$",
            r"$\beta=10^5$",
            r"$\beta=10^6$"
        ])

    if args.SmoothingLength:
        simulation_names = [
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_gpu_corr",  
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_corr_smo02_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_corr_smo03_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_smo04_corr_gpu"
        ]
        labels = [
            r"fiducial",
            r"$b=0.2H$",
            r"$b=0.3H$",
            r"$b=0.4H$"
        ]

    if args.LowMass:
        simulation_names = [
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_gpu_corr",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_corr_mp07_gpu",
            "cos_bet1dm3_nodust_r0415_z05_sth_SEarth_unstrat_corr_gpu",
            "cos_bet1dm3_nodust_r0620_z05_sth_SEarth_unstrat_corr_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_corr_resx2_gpu"
        ]
        labels = [
            r"fiducial",
            r"$\Phi_{p} \times 0.7$",
            r"$r_\mathrm{min}=0.4$",
            r"$r_\mathrm{max}=2.0$",
            r"resolution $\times 2$"
        ]

    if args.simple:
        simulation_names = [
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_gpu_corr",
            "cos_bet1d3_nodust_r0615_z05_sth_SEarth_unstrat_corr_gpu"
        ]
        labels = [
            "isothermal",
            "adiabatic"
        ]

    if args.zdomain:
        simulation_names = [
            "cos_bet1dm3_nodust_r0615_z00625_sth_SEarth_unstrat_HR150_gpu",
            "cos_bet1dm3_nodust_r0615_z0125_sth_SEarth_unstrat_HR150_gpu",
            "cos_bet1dm3_nodust_r0615_z025_sth_SEarth_unstrat_HR150_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_gpu_corr",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_gpu_corr_phipx12",
            "cos_bet1dm3_nodust_r0615_z05_sth_SEarth_unstrat_corr_gam10_gpu",
            "cos_bet1dm3_nodust_r0615_z1_sth_SEarth_unstrat_corr_gpu",
            "cos_bet1dm3_nodust_r0615_z2_sth_SEarth_unstrat_HR150_gpu",
            "cos_bet1dm3_nodust_r0615_z4_sth_SEarth_unstrat_HR150_gpu"
        ]
        labels = [
            r"$\Delta z = 0.125$ (2D equivalent)",
            r"$\Delta z \times 2$",
            r"$\Delta z \times 4$",
            r"$\Delta z \times 8$ (fiducial)",
            r"$\Delta z \times 8$ (fiducial, $\phi_P \times 1.2$)",
            r"$\Delta z \times 16$",
            r"$\Delta z \times 32$",
            r"$\Delta z \times 64$"
        ]

    if args.HighMass:
        simulation_names = [
            "cos_bet1dm3_nodust_r0615_z1_sth_saturn_unstrat_gpu",
            "cos_bet1dm3_nodust_r0620_z05_sth_saturn_unstrat_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_saturn_unstrat_smo04_gpu",
            "cos_bet1dm3_nodust_r0515_z05_sth_saturn_unstrat_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_saturn_unstrat_resx2",
            "cos_bet1dm3_nodust_r0615_z05_sth_saturn_unstrat_resx4_gpu",
            "cos_bet1dm3_nodust_r0615_z05_sth_saturn_unstrat_gpu",
        ]
        labels = [
            r"$\Delta z \times 2$",
            r"$r_\mathrm{max}=2$",
            r"$b=0.4H$",
            r"$r_\mathrm{min}=0.5$",
            r"resolution $\times 2$",
            r"resolution $\times 4$",
            r"fiducial"
        ]

    output_path = "profiles"
    plot_torque_comparison(
        simulation_names, 
        labels, 
        output_path, 
        high_mass=args.HighMass,
        low_mass=args.LowMass, 
        simple=args.simple, 
        isothermal=args.Isothermal, 
        adiabatic=args.Adiabatic, 
        zdomain=args.zdomain,  
        smoothing_length=args.SmoothingLength,
        Paardekooper9=args.Paardekooper9,
        Paardekooper2=args.Paardekooper2,
        Paardekooper1=args.Paardekooper1,
        Paardekooper6=args.Paardekooper6,
        Numerics=args.Numerics,
        fix_yrange=args.fix_yrange
    )
