import argparse
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import time

# Import the functions from the provided modules
from data_storage import determine_base_path, scp_transfer
from planet_data import debug_alternative_torque, read_alternative_torque, read_torque_data, extract_planet_mass_and_migration, compute_theoretical_torques
from data_reader import read_parameters

def dynamic_smoothing(torque, window_size):
    """Smooths the torque data with a dynamic window size near the boundaries."""
    print(f"Applying smoothing with window size {window_size}...")
    start_time = time.time()
    
    smoothed = np.zeros_like(torque)
    n = len(torque)

    for i in range(n):
        left = max(0, i - window_size // 2)
        right = min(n, i + window_size // 2 + 1)
        smoothed[i] = np.mean(torque[left:right])
    
    elapsed_time = time.time() - start_time
    print(f"Smoothing completed in {elapsed_time:.2f} seconds")
    return smoothed

# Function to patch the compute_theoretical_torques function to use separate if statements
def patched_compute_theoretical_torques(parameters, qp, eq_label=None):
    """
    A wrapper for the compute_theoretical_torques function that ensures 
    correct equation selection by printing debug info and forcing the eq_label.
    """
    print(f"Computing theoretical torques with eq_label: {eq_label}")
    
    # Call the original function
    predicted_torque_adi, predicted_torque_iso, GAM0 = compute_theoretical_torques(parameters, qp, eq_label)
    
    # Verify which equation was actually used
    print(f"Returned torque values: adi={predicted_torque_adi}, iso={predicted_torque_iso}")
    
    return predicted_torque_adi, predicted_torque_iso, GAM0

def plot_isothermal_torque_comparison(output_path="profiles", short_mode=False):
    """
    Creates a plot comparing 3D and 3D thin isothermal simulation torques.
    
    Parameters:
    output_path (str): Directory to save the output files
    short_mode (bool): If True, only plot [0,40] orbits with early torque predictions
    """
    print("Starting isothermal torque comparison...")
    if short_mode:
        print("Running in SHORT MODE: Will plot first 40 orbits with early torque predictions only")
    
    start_time = time.time()
    
    # Define the minimum orbit number for y-axis range calculations
    min_orbit_for_y_range = 20  # Only consider data after 20 orbits for y-range
    print(f"Will filter data for y-range calculations to include only t > {min_orbit_for_y_range} orbits")
    
    # Define simulations based on the provided list (isothermal simulations)
    # Updated 3D thin simulation list with first simulation changed
    simulation_3d_thin = [
        "cos_bet1dm6_gam1001_ss0_fi0_r0615_z05_PaardekooperFig2_nu1dm11_COR_LR_3Dthin",
        "cos_bet1dm6_gam1001_ss15_fi05_nodust_r0615_PaardekooperFig2_nu1dm11_pramp10_3Dthin",
        "cos_bet1dm6_gam1001_ss15_fim05_nodust_r0615_PaardekooperFig2_nu1dm11_pramp10_3Dthin"
    ]
    
    simulation_3d = [
        "cos_bet1dm6_gam1001_ss0_fi0_r0615_z05_PaardekooperFig2_nu1dm11_COR_LR_3D",
        "cos_bet1dm6_gam1001_ss15_fi05_r0615_z05_PaardekooperFig2_nu1dm11_COR_LR_3D_CHECK",
        "cos_bet1dm6_gam1001_ss15_fim05_r0615_z05_PaardekooperFig2_nu1dm11_COR_LR_3D_CHECK"
    ]
    
    print(f"Processing {len(simulation_3d_thin)} isothermal 3D thin simulations and {len(simulation_3d)} isothermal 3D simulations")
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 8))
    print("Figure created")
    
    # Define colors for parameter sets and line styles for simulation types
    param_colors = {
        "a0_b1": "#1f77b4",  # Blue for α=0, β=1
        "a15_b2": "#ff7f0e",  # Orange for α=3/2, β=2
        "a15_b0": "#2ca02c"   # Green for α=3/2, β=0
    }
    
    linestyles = {
        "3D": "-",
        "3Dthin": "--"
    }
    
    # Set smoothing window size and calculate smoothing time in orbits
    # Use different smoothing parameters for short vs. normal mode
    if short_mode:
        # In short mode, use smaller window for better early-time resolution
        rolling_window_size = 10  # For short mode (0.5 orbits smoothing)
    else:
        rolling_window_size = 400  # Normal mode - 20 orbits smoothing
        
    smoothing_time_orbits = rolling_window_size / 20  # Convert to orbits (1 = 1/20 ORBIT)
    
    # Get x-axis range based on mode
    if short_mode:
        plot_x_max = 40  # Short mode: Show only first 40 orbits
        theo_line_length = plot_x_max * 0.3  # 30% of the total time range
    else:
        plot_x_max = 1000  # Normal mode: Show full 1000 orbits
        theo_line_length = plot_x_max * 0.2  # 20% of the total time range
    
    # Lists to store min and max values for determining plot range
    all_data_min = []
    all_data_max = []
    
    # Process 3D thin simulations
    print("Processing 3D thin simulations...")
    for i, sim in enumerate(simulation_3d_thin):
        print(f"Processing 3D thin simulation {i+1}/{len(simulation_3d_thin)}: {sim}")
        base_path = determine_base_path(sim)
        print(f"Base path: {base_path}")
        
        summary_file = f"{base_path}/summary0.dat"
        print(f"Reading parameters from {summary_file}")
        parameters = read_parameters(summary_file)
        
        # Extract alpha and beta for coloring
        if 'ss0' in sim:
            alpha = 0
            beta = 1  # Added this to fix the bug
            color_key = "a0_b1"
            label = r"$\alpha=0, \beta=1$ (3D thin)"
        elif 'ss15' in sim and 'fim05' in sim:
            alpha = 1.5
            beta = 2
            color_key = "a15_b2"
            label = r"$\alpha=3/2, \beta=2$ (3D thin)"
        else:  # ss15 and fi05
            alpha = 1.5
            beta = 0
            color_key = "a15_b0"
            label = r"$\alpha=3/2, \beta=0$ (3D thin)"
        
        print(f"Simulation parameters: alpha={alpha}, beta={beta}")
        
        # Read torque data
        torque_file = f"{base_path}/monitor/gas/torq_planet_0.dat"
        tqwk_file = f"{base_path}/tqwk0.dat"
        
        print(f"Checking for torque files...")
        if os.path.exists(torque_file):
            print(f"Reading torque data from {torque_file}")
            date_torque, torque = read_torque_data(torque_file)
            print(f"Read {len(torque)} data points")
        elif os.path.exists(tqwk_file):
            print(f"Reading torque data from {tqwk_file}")
            date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file)
            print(f"Read {len(torque)} data points")
        else:
            print(f"Torque file not found for simulation {sim}")
            continue
        
        # Extract planet mass and compute normalization
        print("Extracting planet mass and computing normalization...")
        qp, _ = extract_planet_mass_and_migration(summary_file)
        print(f"Planet mass ratio q = {qp}")
        
        # Get gamma value for scaling
        gamma = parameters['GAMMA']
        print(f"Gamma = {gamma}")
        
        # Compute theoretical torques for each parameter set
        # For early time: Use default isothermal torque (eq 49)
        print("Computing theoretical torques for early time (unsaturated)...")
        _, predicted_torque_early, GAM0 = patched_compute_theoretical_torques(parameters, qp, eq_label=None)
        
        # For late time: Use Paardekooper2/Equation14 (saturated, Lindblad only)
        # Only compute if not in short mode
        if not short_mode:
            print("Computing theoretical torques for late time (saturated)...")
            predicted_torque_late, _, _ = patched_compute_theoretical_torques(parameters, qp, eq_label="Paardekooper2")
            print(f"Theoretical torques: early (unsaturated, iso)={predicted_torque_early}, late (saturated)={predicted_torque_late}")
        else:
            print(f"Theoretical torque (unsaturated, iso)={predicted_torque_early}")
        
        print(f"Normalization factor GAM0={GAM0}")
        
        # Convert to orbits and smooth
        print("Converting to orbits and applying smoothing...")
        time_in_orbits = date_torque / (2 * np.pi)
        time_averaged_torque = dynamic_smoothing(torque, rolling_window_size)
        
        # Apply time limit for short mode
        if short_mode:
            # Create a mask for data within the time range
            time_mask = time_in_orbits <= plot_x_max
            plot_time = time_in_orbits[time_mask]
            plot_torque = time_averaged_torque[time_mask]
        else:
            plot_time = time_in_orbits
            plot_torque = time_averaged_torque
        
        # Scale and plot
        print("Scaling and plotting the data...")
        scaled_torque = qp * plot_torque
        normalized_torque = (scaled_torque / GAM0) * gamma
        ax.plot(plot_time, normalized_torque, 
                label=label, color=param_colors[color_key], linestyle=linestyles["3Dthin"])
        
        # Store min and max values for determining plot range, but only consider data after min_orbit_for_y_range
        # and before plot_x_max (for short mode)
        time_mask = (time_in_orbits > min_orbit_for_y_range)
        if short_mode:
            time_mask = time_mask & (time_in_orbits <= plot_x_max)
            
        if np.any(time_mask):  # Make sure there's data after our time cutoff
            filtered_torque = (qp * time_averaged_torque[time_mask] / GAM0) * gamma
            all_data_min.append(np.min(filtered_torque))
            all_data_max.append(np.max(filtered_torque))
            print(f"Y-range data for filtered time range: min={np.min(filtered_torque)}, max={np.max(filtered_torque)}")
        
        # Calculate normalized theoretical torques (with gamma)
        early_normalized = (predicted_torque_early/GAM0) * gamma
        
        # Store these for y-axis range calculation
        all_data_min.append(early_normalized)
        all_data_max.append(early_normalized)
        
        # Plot theoretical predictions (early time - unsaturated)
        # Include gamma multiplication for the theoretical torques!
        print("Plotting theoretical predictions for early time (unsaturated)...")
        ax.plot([0, theo_line_length], 
                [early_normalized, early_normalized], 
                color=param_colors[color_key], linestyle=':', linewidth=2.0)
        
        # Plot theoretical predictions (late time - saturated) - only if not in short mode
        if not short_mode:
            late_normalized = (predicted_torque_late/GAM0) * gamma
            all_data_min.append(late_normalized)
            all_data_max.append(late_normalized)
            
            print("Plotting theoretical predictions for late time (saturated)...")
            ax.plot([plot_x_max - theo_line_length, plot_x_max], 
                    [late_normalized, late_normalized], 
                    color=param_colors[color_key], linestyle=':', linewidth=2.0)
        
        print(f"Completed processing 3D thin simulation {i+1}/{len(simulation_3d_thin)}")
    
    # Process 3D simulations
    print("\nProcessing 3D simulations...")
    for i, sim in enumerate(simulation_3d):
        print(f"Processing 3D simulation {i+1}/{len(simulation_3d)}: {sim}")
        base_path = determine_base_path(sim)
        print(f"Base path: {base_path}")
        
        summary_file = f"{base_path}/summary0.dat"
        print(f"Reading parameters from {summary_file}")
        parameters = read_parameters(summary_file)
        
        # Extract alpha and beta for coloring
        if 'ss0' in sim:
            alpha = 0
            beta = 1  # Added this to fix the bug
            color_key = "a0_b1"
            label = r"$\alpha=0, \beta=1$ (3D)"
        elif 'ss15' in sim and 'fim05' in sim:
            alpha = 1.5
            beta = 2
            color_key = "a15_b2"
            label = r"$\alpha=3/2, \beta=2$ (3D)"
        else:  # ss15 and fi05
            alpha = 1.5
            beta = 0
            color_key = "a15_b0"
            label = r"$\alpha=3/2, \beta=0$ (3D)"
        
        print(f"Simulation parameters: alpha={alpha}, beta={beta}")
        
        # Read torque data
        torque_file = f"{base_path}/monitor/gas/torq_planet_0.dat"
        tqwk_file = f"{base_path}/tqwk0.dat"
        
        print(f"Checking for torque files...")
        if os.path.exists(torque_file):
            print(f"Reading torque data from {torque_file}")
            date_torque, torque = read_torque_data(torque_file)
            print(f"Read {len(torque)} data points")
        elif os.path.exists(tqwk_file):
            print(f"Reading torque data from {tqwk_file}")
            date_torque, torque, orbit_numbers = read_alternative_torque(tqwk_file)
            print(f"Read {len(torque)} data points")
        else:
            print(f"Torque file not found for simulation {sim}")
            continue
        
        # Extract planet mass and compute normalization
        print("Extracting planet mass and computing normalization...")
        qp, _ = extract_planet_mass_and_migration(summary_file)
        print(f"Planet mass ratio q = {qp}")
        
        # Get gamma value for scaling
        gamma = parameters['GAMMA']
        print(f"Gamma = {gamma}")
        
        # Compute theoretical torques for each parameter set
        # For early time: Use default isothermal torque (eq 49)
        print("Computing theoretical torques for early time (unsaturated)...")
        _, predicted_torque_early, GAM0 = patched_compute_theoretical_torques(parameters, qp, eq_label=None)
        
        # For late time: Use Paardekooper2/Equation14 (saturated, Lindblad only)
        # Only compute if not in short mode
        if not short_mode:
            print("Computing theoretical torques for late time (saturated)...")
            predicted_torque_late, _, _ = patched_compute_theoretical_torques(parameters, qp, eq_label="Paardekooper2")
            print(f"Theoretical torques: early (unsaturated, iso)={predicted_torque_early}, late (saturated)={predicted_torque_late}")
        else:
            print(f"Theoretical torque (unsaturated, iso)={predicted_torque_early}")
        
        print(f"Normalization factor GAM0={GAM0}")
        
        # Convert to orbits and smooth
        print("Converting to orbits and applying smoothing...")
        time_in_orbits = date_torque / (2 * np.pi)
        time_averaged_torque = dynamic_smoothing(torque, rolling_window_size)
        
        # Apply time limit for short mode
        if short_mode:
            # Create a mask for data within the time range
            time_mask = time_in_orbits <= plot_x_max
            plot_time = time_in_orbits[time_mask]
            plot_torque = time_averaged_torque[time_mask]
        else:
            plot_time = time_in_orbits
            plot_torque = time_averaged_torque
        
        # Scale and plot
        print("Scaling and plotting the data...")
        scaled_torque = qp * plot_torque
        normalized_torque = (scaled_torque / GAM0) * gamma
        ax.plot(plot_time, normalized_torque, 
                label=label, color=param_colors[color_key], linestyle=linestyles["3D"])
        
        # Store min and max values for determining plot range, but only consider data after min_orbit_for_y_range
        # and before plot_x_max (for short mode)
        time_mask = (time_in_orbits > min_orbit_for_y_range)
        if short_mode:
            time_mask = time_mask & (time_in_orbits <= plot_x_max)
            
        if np.any(time_mask):  # Make sure there's data after our time cutoff
            filtered_torque = (qp * time_averaged_torque[time_mask] / GAM0) * gamma
            all_data_min.append(np.min(filtered_torque))
            all_data_max.append(np.max(filtered_torque))
            print(f"Y-range data for filtered time range: min={np.min(filtered_torque)}, max={np.max(filtered_torque)}")
        
        # Calculate normalized theoretical torques (with gamma)
        early_normalized = (predicted_torque_early/GAM0) * gamma
        
        # Store these for y-axis range calculation
        all_data_min.append(early_normalized)
        all_data_max.append(early_normalized)
        
        # Plot theoretical predictions (early time - unsaturated)
        # Include gamma multiplication for the theoretical torques!
        print("Plotting theoretical predictions for early time (unsaturated)...")
        ax.plot([0, theo_line_length], 
                [early_normalized, early_normalized], 
                color=param_colors[color_key], linestyle=':', linewidth=2.0)
        
        # Plot theoretical predictions (late time - saturated) - only if not in short mode
        if not short_mode:
            late_normalized = (predicted_torque_late/GAM0) * gamma
            all_data_min.append(late_normalized)
            all_data_max.append(late_normalized)
            
            print("Plotting theoretical predictions for late time (saturated)...")
            ax.plot([plot_x_max - theo_line_length, plot_x_max], 
                    [late_normalized, late_normalized], 
                    color=param_colors[color_key], linestyle=':', linewidth=2.0)
        
        print(f"Completed processing 3D simulation {i+1}/{len(simulation_3d)}")
    
    # Add text labels for theoretical lines
    print("Adding text labels for theoretical lines...")
    
    # Add appropriate theoretical line labels based on mode
    if short_mode:
        ax.text(theo_line_length/2, min(all_data_min) - 0.5, r"$\Gamma_{\mathrm{unsaturated}}$", fontsize=12)
    else:
        ax.text(theo_line_length/2, min(all_data_min) - 0.5, r"$\Gamma_{\mathrm{unsaturated}}$", fontsize=12)
        ax.text(plot_x_max - theo_line_length/2, min(all_data_min) - 0.5, r"$\Gamma_{\mathrm{saturated}}$", fontsize=12, horizontalalignment='center')
    
    # Set up the plot
    print("Setting up plot labels and formatting...")
    ax.set_xlabel('t (orbits)', fontsize=14)
    ax.set_ylabel(r'$\gamma\Gamma/\Gamma_0$', fontsize=14)
    ax.set_xlim(0, plot_x_max)
    
    # Dynamically set y-axis limits with some padding
    data_min = min(all_data_min)
    data_max = max(all_data_max)
    y_padding = 0.1 * (data_max - data_min)  # 10% padding
    
    # Make sure we have at least some minimum range
    if data_max - data_min < 2:
        y_padding = 1.0
        
    print(f"Setting y-range: [{data_min - y_padding}, {data_max + y_padding}]")
    ax.set_ylim(data_min - y_padding, data_max + y_padding)
    
    ax.grid(True, alpha=0.3)
    
    # Add a single legend entry for the theoretical torque
    print("Adding legend...")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='black', linestyle=':', linewidth=2.0, label='Theoretical Torque'))
    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=11)
    
    # Create title based on mode
    if short_mode:
        plot_title = f'Total torque on planet in isothermal simulations - Early evolution (Smoothing: {smoothing_time_orbits:.1f} orbits)'
    else:
        plot_title = f'Total torque on planet in isothermal simulations (Smoothing: {smoothing_time_orbits:.1f} orbits)'
    
    plt.title(plot_title, fontsize=14)
    
    # Add text explaining the simulation parameters
    print("Adding simulation parameters text...")
    plt.figtext(0.5, 0.01, r'$q = 1.26 \times 10^{-5}, h = 0.05, \gamma \approx 1, b/h = 0.4$', 
                fontsize=12, ha='center')
    
    # Create appropriate filename based on mode
    if short_mode:
        output_filename = f"{output_path}/torque_3D_isothermal_comparison_early.pdf"
    else:
        output_filename = f"{output_path}/torque_3D_isothermal_comparison.pdf"
    
    # Save and transfer the figure
    print(f"Saving figure to {output_filename}...")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Torque comparison plot saved to {output_filename}")
    
    # Transfer the file
    print("Transferring file...")
    local_directory = "/Users/mariuslehmann/Downloads/Profiles/planet_evolution/"
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Main function to handle command line arguments
def main():
    parser = argparse.ArgumentParser(description="Generate isothermal torque comparison plots")
    parser.add_argument("--short", action="store_true", help="Plot only first 40 orbits with early torque predictions")
    parser.add_argument("--output", "-o", default="profiles", help="Output directory for plots")
    
    args = parser.parse_args()
    
    plot_isothermal_torque_comparison(output_path=args.output, short_mode=args.short)

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script completed")
