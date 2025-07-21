import numpy as np
import matplotlib.pyplot as plt
import os
from data_storage import determine_base_path, scp_transfer  # Correct import

def load_data(npz_file):
    """Load data from an npz file."""
    print("Loading data from: {}".format(npz_file))
    data = np.load(npz_file)
    return data['pressure_dev'], data['time_array'], data['xgrid']

def load_metallicity_data(npz_file):
    """Load metallicity data from an npz file, or return zero metallicity if the file is missing."""
    if os.path.exists(npz_file):
        print("Loading metallicity data from: {}".format(npz_file))
        data = np.load(npz_file)
        return data['metallicity']
    else:
        print("Metallicity data not found; setting metallicity to zero.")
        return None  # Use None to signify zero metallicity

def extract_metallicity_label(sim_dir):
    """Extract metallicity Z in decimal format from the simulation directory name."""
    metallicity_str = [part for part in sim_dir.split('_') if part.startswith('Z')]
    if metallicity_str:
        metallicity_str = metallicity_str[0]
        if 'dm' in metallicity_str:
            metallicity = metallicity_str.replace('Z', '').replace('dm', 'e-')
            metallicity = "{:.3f}".format(float(metallicity))
        else:
            metallicity = metallicity_str.replace("Z", "")  # For cases like Z0.1
    else:
        metallicity = "0"  # Default to Z=0 if no metallicity information is present
    return r"$Z = " + metallicity + "$"

# Set the number of simulations to plot (2, 3, or 4)
num_simulations = 4  # Change to 2, 3, or 4 as needed

# Define the list of simulation directories based on the number of simulations
if num_simulations == 2:
    simulations = [
        "cos_bet1d0_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1d0_St1dm1_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap"
    ]
elif num_simulations == 3:
    simulations = [
        "cos_bet1d0_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1d0_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1d0_St1dm1_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap"
    ]
elif num_simulations == 4:
    simulations = [
        "cos_bet1d0_nodust_r6H_z08H_LR_stnew",  # Z=0
        "cos_bet1d0_St1dm1_Z1dm3_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1d0_St1dm1_Z1dm2_r6H_z08H_LR_PRESETx10_stnew",
        "cos_bet1d0_St1dm1_Z5dm2_r6H_z08H_LR_PRESETx10_stnew_tap"
    ]

# Load pressure deviation and metallicity data for each simulation
pressure_data = []
metallicity_data = []
metallicity_labels = []

for sim_dir in simulations:
    subdir_path = determine_base_path(sim_dir)
    
    # Construct paths to pressure deviation and metallicity npz files
    pressure_npz = os.path.join(subdir_path, "{}_pressure_deviation_data.npz".format(os.path.basename(sim_dir)))
    metallicity_npz = os.path.join(subdir_path, "{}_metallicity_data.npz".format(os.path.basename(sim_dir)))
    
    # Load data
    pressure, time_array, xgrid = load_data(pressure_npz)
    metallicity = load_metallicity_data(metallicity_npz)
    
    # Handle zero metallicity case
    if metallicity is None:
        # Set metallicity to zero with a tiny noise to avoid issues in imshow
        metallicity = np.zeros_like(pressure) + np.random.normal(0, 1e-10, pressure.shape)
    
    # Store data and metallicity label
    pressure_data.append(pressure)
    metallicity_data.append(metallicity)
    metallicity_labels.append(extract_metallicity_label(sim_dir))

# Create the 2xN plot (2 rows and N columns where N is the number of simulations)
fig, axs = plt.subplots(2, num_simulations, figsize=(5 * num_simulations, 10))

# Plot each simulation in its respective column
for i in range(num_simulations):
    # Pressure deviation (top row)
    im1 = axs[0, i].imshow(pressure_data[i], extent=[xgrid.min(), xgrid.max(), time_array.min(), time_array.max()],
                           aspect='auto', origin='lower', cmap='seismic', vmin=pressure_data[i].min(), vmax=pressure_data[i].max())
    axs[0, i].set_ylabel("Time (orbits)")
    axs[0, i].tick_params(labelbottom=False)  # Remove x-axis labels from top row
    cbar1 = fig.colorbar(im1, ax=axs[0, i], orientation='vertical')
    cbar1.set_label('Scaled Pressure Deviation (Sim {})'.format(i+1), labelpad=10)
    axs[0, i].text(0.05, 0.95, metallicity_labels[i], transform=axs[0, i].transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    # Square root of metallicity (bottom row)
    im2 = axs[1, i].imshow(np.sqrt(metallicity_data[i]), extent=[xgrid.min(), xgrid.max(), time_array.min(), time_array.max()],
                           aspect='auto', origin='lower', cmap='viridis', vmin=np.sqrt(metallicity_data[i]).min(), vmax=np.sqrt(metallicity_data[i]).max())
    axs[1, i].set_xlabel("Disk Radius")
    axs[1, i].set_ylabel("Time (orbits)")
    cbar2 = fig.colorbar(im2, ax=axs[1, i], orientation='vertical')
    cbar2.set_label(r'$\sqrt{{\text{{Metallicity}}}}$ (Sim {})'.format(i+1), labelpad=10)
    axs[1, i].text(0.05, 0.95, metallicity_labels[i], transform=axs[1, i].transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

# Adjust layout and save the plot
plt.tight_layout()
output_filename = "comparison_space_time_plot_{}_simulations.pdf".format(num_simulations)
plt.savefig(output_filename)
plt.close()
print("{}x2 comparison plot saved to {}".format(num_simulations, output_filename))

# Define the local directory on your laptop for SCP transfer
local_directory = "/Users/mariuslehmann/Downloads/Contours"
# Transfer the file
scp_transfer(output_filename, local_directory, "mariuslehmann")
print("Time evolution plot transferred to {}".format(local_directory))
