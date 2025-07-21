#!/usr/bin/env python3
"""
Script to display time-averaged density deviation with:
1. Adjusted color scale (capped at 5e-5)
2. Multiple streamlines in different regions, including the horseshoe region
3. Improved high-resolution grid and streamline diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.integrate import solve_ivp
from matplotlib.colors import Normalize
from data_storage import determine_base_path, scp_transfer
from data_reader import read_parameters, determine_nt, read_single_snapshot
from planet_data import extract_planet_mass_and_migration
from scipy.interpolate import RegularGridInterpolator  # Added import for high-res grid

def shift_array_azimuth(arr):
    """Shift the array to put the planet at phi=PI"""
    return np.roll(arr, shift=arr.shape[0] // 2, axis=0)

def plot_multiple_region_streamlines(output_path, simulation_name, qp, snapshot_number=None, n_avg=20, vertical_mode='midplane'):

    """
    Creates a plot of the time-averaged density deviation with:
    1. Adjusted color scale (capped at 5e-5)
    2. Multiple streamlines in different regions, including the horseshoe region
    
    Parameters:
    -----------
    output_path : str
        Path to the simulation output directory.
    simulation_name : str
        Name of the simulation.
    qp : float
        Planet-to-star mass ratio.
    snapshot_number : int, optional
        Center snapshot number for time averaging. If None, the last snapshot is used.
    n_avg : int, optional
        Number of snapshots to average (centered on snapshot_number).
    """
    print(f"Starting multiple region streamlines plot ({n_avg} snapshots)...")
    print(f"Vertical mode: {vertical_mode}")

    
    # Determine the total number of snapshots
    nt = determine_nt(output_path)
    
    # Use the specified snapshot or the last one if not specified or too large
    if snapshot_number is None or snapshot_number >= nt:
        snapshot_number = nt - 1
    
    # Calculate Hill radius
    r_h = (qp/3.0)**(1.0/3.0)
    print(f"Hill radius: {r_h}")
    
    # Read initial density for reference
    print("Reading initial gas density...")
    initial_data, xgrid, ygrid, zgrid, parameters = read_single_snapshot(
        output_path, 0, read_gasdens=True
    )
    initial_gas_density = initial_data['gasdens']
    
    # Handle dimensions for initial density
    if len(initial_gas_density.shape) == 3:
        initial_gas_density = initial_gas_density[:, :, initial_gas_density.shape[2]//2]
    
    # Apply π shift to initial density
    initial_gas_density = shift_array_azimuth(initial_gas_density)
    
    # Initialize arrays for accumulating density deviation and velocities
    density_deviation_sum = None
    vx_sum = None
    vy_sum = None
    
    # Determine the snapshots to average
    start_snapshot = max(0, snapshot_number - n_avg // 2)
    end_snapshot = min(nt, start_snapshot + n_avg)
    actual_n_avg = end_snapshot - start_snapshot
    
    print(f"Time-averaging snapshots {start_snapshot} to {end_snapshot-1} ({actual_n_avg} snapshots)")
    
    # Process each snapshot
    for snap in range(start_snapshot, end_snapshot):
        print(f"Processing snapshot {snap}...")
        data_arrays, xgrid, ygrid, zgrid, snap_params = read_single_snapshot(
            output_path, snap, read_gasdens=True, read_gasvx=True, read_gasvy=True
        )
        
        gasdens = data_arrays['gasdens']
        gasvx = data_arrays['gasvx']
        gasvy = data_arrays['gasvy']
        
        # Now handle the vertical dimension for 3D data
        if len(gasdens.shape) == 3:
            if vertical_mode == 'midplane':
                # Extract midplane (middle z-index)
                midplane_idx = gasdens.shape[2] // 2
                print(f"Using midplane slice (z-index: {midplane_idx})")
                gasdens = gasdens[:, :, midplane_idx]
                gasvx = gasvx[:, :, midplane_idx]
                gasvy = gasvy[:, :, midplane_idx]
            elif vertical_mode == 'average':
                # Average over vertical dimension (z-axis)
                print(f"Averaging over {gasdens.shape[2]} vertical layers")
                gasdens = np.mean(gasdens, axis=2)
                gasvx = np.mean(gasvx, axis=2)
                gasvy = np.mean(gasvy, axis=2)
            else:
                raise ValueError(f"Unknown vertical_mode: {vertical_mode}")
        
        # Apply π shift in azimuth to all arrays
        gasdens = shift_array_azimuth(gasdens)
        gasvx = shift_array_azimuth(gasvx)
        gasvy = shift_array_azimuth(gasvy)
        
        # Calculate density deviation
        deviation = (gasdens - initial_gas_density)
        
        # Initialize sum arrays if first iteration
        if density_deviation_sum is None:
            density_deviation_sum = np.zeros_like(deviation)
            vx_sum = np.zeros_like(gasvx)
            vy_sum = np.zeros_like(gasvy)
        
        # Add to running sums
        density_deviation_sum += deviation
        vx_sum += gasvx
        vy_sum += gasvy
    
    # Calculate averages
    density_deviation_avg = density_deviation_sum / actual_n_avg
    vx_avg = vx_sum / actual_n_avg
    vy_avg = vy_sum / actual_n_avg
    
    print(f"Time-averaging complete. Array shapes: density={density_deviation_avg.shape}, vx={vx_avg.shape}, vy={vy_avg.shape}")
    
    # Define the zoomed domain around the planet
    r_min = 0.94
    r_max = 1.06
    phi_min = np.pi - 0.3
    phi_max = np.pi + 0.3
    
    print(f"Domain: r = [{r_min:.4f}, {r_max:.4f}], φ = [{phi_min:.4f}, {phi_max:.4f}]")
    
    # Find indices for the zoomed domain
    r_indices = np.where((xgrid >= r_min) & (xgrid <= r_max))[0]
    phi_indices = np.where((ygrid >= phi_min) & (ygrid <= phi_max))[0]
    
    if len(r_indices) == 0:
        print(f"Warning: No radial points found in the range [{r_min}, {r_max}]")
        print(f"Using full radial range instead: [{xgrid.min()}, {xgrid.max()}]")
        zoomed_xgrid = xgrid
    else:
        print(f"Found {len(r_indices)} points in radial range [{r_min}, {r_max}]")
        zoomed_xgrid = xgrid[r_indices]
    
    if len(phi_indices) == 0:
        print(f"Warning: No azimuthal points found in the range [{phi_min}, {phi_max}]")
        print(f"Using full azimuthal range instead: [{ygrid.min()}, {ygrid.max()}]")
        zoomed_ygrid = ygrid
    else:
        print(f"Found {len(phi_indices)} points in azimuthal range [{phi_min}, {phi_max}]")
        zoomed_ygrid = ygrid[phi_indices]
    
    # Extract the zoomed portions of the arrays
    if len(r_indices) > 0 and len(phi_indices) > 0:
        zoomed_density = density_deviation_avg[phi_indices][:, r_indices]
        zoomed_vx = vx_avg[phi_indices][:, r_indices]
        zoomed_vy = vy_avg[phi_indices][:, r_indices]
    elif len(r_indices) > 0:
        zoomed_density = density_deviation_avg[:, r_indices]
        zoomed_vx = vx_avg[:, r_indices]
        zoomed_vy = vy_avg[:, r_indices]
    elif len(phi_indices) > 0:
        zoomed_density = density_deviation_avg[phi_indices]
        zoomed_vx = vx_avg[phi_indices]
        zoomed_vy = vy_avg[phi_indices]
    else:
        zoomed_density = density_deviation_avg
        zoomed_vx = vx_avg
        zoomed_vy = vy_avg
    
    # === BEGIN NEW CODE: High-resolution grid interpolation ===
    print("Creating high-resolution grid for visualization...")
    # First, ensure we're using the correct portions of the original grids
    if len(r_indices) > 0 and len(phi_indices) > 0:
        r_grid_for_interp = xgrid[r_indices]
        phi_grid_for_interp = ygrid[phi_indices]
    else:
        r_grid_for_interp = zoomed_xgrid
        phi_grid_for_interp = zoomed_ygrid

    # To avoid out-of-bounds errors, ensure the high-res grid stays within original bounds
    # Add a small buffer to avoid floating point precision issues
    epsilon = 1e-10
    r_min_safe = r_grid_for_interp.min() + epsilon
    r_max_safe = r_grid_for_interp.max() - epsilon
    phi_min_safe = phi_grid_for_interp.min() + epsilon
    phi_max_safe = phi_grid_for_interp.max() - epsilon

    print(f"Safe interpolation bounds: r=[{r_min_safe}, {r_max_safe}], phi=[{phi_min_safe}, {phi_max_safe}]")

    # Create higher resolution grid for visualization that stays within bounds
    num_r_points = 300
    num_phi_points = 300
    r_highres = np.linspace(r_min_safe, r_max_safe, num_r_points)
    phi_highres = np.linspace(phi_min_safe, phi_max_safe, num_phi_points)

    # Create interpolation functions for the fields
    print("Creating interpolation functions...")
    density_interp = RegularGridInterpolator((phi_grid_for_interp, r_grid_for_interp), zoomed_density)
    vx_interp = RegularGridInterpolator((phi_grid_for_interp, r_grid_for_interp), zoomed_vx)
    vy_interp = RegularGridInterpolator((phi_grid_for_interp, r_grid_for_interp), zoomed_vy)

    # Create high-resolution meshgrid
    r_mesh, phi_mesh = np.meshgrid(r_highres, phi_highres)

    # Prepare points for interpolation - use a safer approach with explicit point generation
    print("Preparing interpolation points...")
    interp_points = []
    for i in range(phi_mesh.shape[0]):
        for j in range(phi_mesh.shape[1]):
            interp_points.append([phi_mesh[i,j], r_mesh[i,j]])
    interp_points = np.array(interp_points)

    # Interpolate fields to high-resolution grid
    print("Applying interpolation...")
    try:
        zoomed_density_values = density_interp(interp_points)
        zoomed_vx_values = vx_interp(interp_points)
        zoomed_vy_values = vy_interp(interp_points)
        
        # Reshape the results back to grid form
        zoomed_density_highres = zoomed_density_values.reshape(phi_mesh.shape)
        zoomed_vx_highres = zoomed_vx_values.reshape(phi_mesh.shape)
        zoomed_vy_highres = zoomed_vy_values.reshape(phi_mesh.shape)
        
        # Use high-resolution grid for further operations
        zoomed_xgrid = r_highres
        zoomed_ygrid = phi_highres
        zoomed_density = zoomed_density_highres
        zoomed_vx = zoomed_vx_highres
        zoomed_vy = zoomed_vy_highres
        print("High-resolution grid created and fields interpolated successfully.")
    except Exception as e:
        print(f"Interpolation failed with error: {e}")
        print("Continuing with original resolution data...")
        # If interpolation fails, just continue with the original data
        r_mesh, phi_mesh = np.meshgrid(zoomed_xgrid, zoomed_ygrid)
    # === END NEW CODE: High-resolution grid interpolation ===

# === ADD STAGNATION POINT DETECTION ===
    def find_stagnation_points(xgrid, ygrid, vx, vy):
        """Find stagnation points in the velocity field"""
        print("Searching for stagnation points...")
        
        # Calculate velocity magnitude
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Convert to co-rotating frame velocities if needed
        # Note: This assumes your velocity fields are already in the co-rotating frame
        # If they are not, you would need to adjust them first
        
        # Find the midplane index (closest to phi = pi)
        midplane_index = np.argmin(np.abs(ygrid - np.pi))
        print(f"Midplane index: {midplane_index}, phi value: {ygrid[midplane_index]}")
        
        # Extract midplane velocity magnitudes
        midplane_v_mag = v_mag[midplane_index, :]
        
        # Create a mask for points near the planet's orbit
        r_mask = np.abs(xgrid - 1.0) < 0.1
        
        # Apply the mask to limit search area
        masked_v_mag = np.copy(midplane_v_mag)
        masked_v_mag[~r_mask] = np.max(midplane_v_mag) * 10  # Set values outside mask to high value
        
        # Find local minima
        from scipy.signal import find_peaks
        try:
            minima_indices, _ = find_peaks(-masked_v_mag)
            print(f"Found {len(minima_indices)} potential minima in velocity field")
            
            # Process minima to find stagnation points
            stagnation_points = []
            for idx in minima_indices:
                r_stag = xgrid[idx]
                phi_stag = ygrid[midplane_index]
                v_mag_at_point = midplane_v_mag[idx]
                
                print(f"  Potential point at r={r_stag:.4f}, v_mag={v_mag_at_point:.8f}")
                
                # Only consider points with low velocity and near the planet orbit
                if v_mag_at_point < 1e-4 and abs(r_stag - 1.0) < 0.2:
                    stagnation_points.append((r_stag, phi_stag))
                    print(f"  ACCEPTED stagnation point at r={r_stag:.4f}, phi={phi_stag:.4f}, v_mag={v_mag_at_point:.8f}")
        except Exception as e:
            print(f"Error in stagnation point detection: {e}")
            minima_indices = []
            stagnation_points = []
        
        # If automatic detection finds less than 2 points, use theoretical approximation
        if len(stagnation_points) < 2:
            print("Insufficient stagnation points found. Using theoretical approximation...")
            stagnation_points = [
                (1.0 - 0.7*r_h, np.pi),  # Inner point
                (1.0 + 0.7*r_h, np.pi)   # Outer point
            ]
            print(f"Theoretical stagnation points at r = {1.0 - 0.7*r_h:.4f} and {1.0 + 0.7*r_h:.4f}")
        
        return stagnation_points
    
    # Find stagnation points in the high-resolution velocity field
    stagnation_points = find_stagnation_points(zoomed_xgrid, zoomed_ygrid, zoomed_vx, zoomed_vy)
    print(f"Final stagnation points: {stagnation_points}")
    
    # === BEGIN NEW CODE: Enhanced streamline integration function ===
    def integrate_streamline(r0, phi0, forward=True):
        """Integrate a streamline starting at (r0, phi0) with improved diagnostics"""
        
        def streamline_ode(t, state):
            """ODE function for streamline integration"""
            r, phi = state
            
            # Find the nearest grid points
            r_idx = np.searchsorted(zoomed_xgrid, r) - 1
            phi_idx = np.searchsorted(zoomed_ygrid, phi) - 1
            
            # Check if we're out of bounds
            if (r_idx < 0 or r_idx >= len(zoomed_xgrid) - 1 or 
                phi_idx < 0 or phi_idx >= len(zoomed_ygrid) - 1):
                # Print diagnostic when out of bounds
                print(f"Warning: Point ({r:.4f}, {phi:.4f}) is out of bounds!")
                # Return zero velocities to signal termination
                return [0, 0]
            
            # Calculate interpolation weights
            r_weight = (r - zoomed_xgrid[r_idx]) / (zoomed_xgrid[r_idx + 1] - zoomed_xgrid[r_idx])
            phi_weight = (phi - zoomed_ygrid[phi_idx]) / (zoomed_ygrid[phi_idx + 1] - zoomed_ygrid[phi_idx])
            
            # Bilinear interpolation for velocities
            vr = (1-r_weight)*(1-phi_weight)*zoomed_vy[phi_idx, r_idx] + \
                 r_weight*(1-phi_weight)*zoomed_vy[phi_idx, r_idx+1] + \
                 (1-r_weight)*phi_weight*zoomed_vy[phi_idx+1, r_idx] + \
                 r_weight*phi_weight*zoomed_vy[phi_idx+1, r_idx+1]
                 
            vphi = (1-r_weight)*(1-phi_weight)*zoomed_vx[phi_idx, r_idx] + \
                  r_weight*(1-phi_weight)*zoomed_vx[phi_idx, r_idx+1] + \
                  (1-r_weight)*phi_weight*zoomed_vx[phi_idx+1, r_idx] + \
                  r_weight*phi_weight*zoomed_vx[phi_idx+1, r_idx+1]
            
            # Convert azimuthal velocity to angular velocity
            vphi_angular = vphi / r
            
            # Check for very small velocities
            if abs(vr) < 1e-10 and abs(vphi_angular) < 1e-10:
                print(f"Warning: Near-zero velocity at ({r:.4f}, {phi:.4f})")
            
            # Reverse direction if integrating backward
            if not forward:
                return [-vr, -vphi_angular]
            else:
                return [vr, vphi_angular]
        
        # Add event function to detect near-zero velocities or out-of-bounds
        def termination_event(t, state):
            r, phi = state
            # Out of bounds check
            if (r < r_min or r > r_max or phi < phi_min or phi > phi_max):
                return 0
            # Check for velocity field
            try:
                dr_dt, dphi_dt = streamline_ode(t, state)
                # Terminate if velocity is very close to zero
                if abs(dr_dt) < 1e-10 and abs(dphi_dt) < 1e-10:
                    return 0
                return 1
            except:
                return 0
        
        termination_event.terminal = True
        
        # Integration parameters
        max_time = 300  # Increased from 100
        t_span = [0, max_time]
        
        # Perform the integration with diagnostic output
        print(f"Starting integration from ({r0:.4f}, {phi0:.4f}), forward={forward}")
        
        sol = solve_ivp(
            streamline_ode, 
            t_span, 
            [r0, phi0], 
            method='RK45',
            rtol=1e-9,  # Increased precision
            atol=1e-9,  # Increased precision
            max_step=0.02,  # Smaller step for more detail
            events=termination_event
        )
        
        # Diagnostic output
        if sol.t[-1] < max_time:
            print(f"Integration terminated early at t={sol.t[-1]:.2f}")
            r_final, phi_final = sol.y[0][-1], sol.y[1][-1]
            print(f"Final position: ({r_final:.4f}, {phi_final:.4f})")
        
        return sol.y[0], sol.y[1]  # Return r and phi trajectories
    # === END NEW CODE: Enhanced streamline integration function ===


    # Add this after the integrate_streamline function (around line 460)
    # === END NEW CODE: Enhanced streamline integration function ===

    def add_direction_arrows(ax, r_traj, phi_traj, color, n_arrows=3, arrow_size=15, zorder=8):
        """
        Add direction arrows along a streamline with improved aspect ratio.
        """
        # Skip if trajectory is too short
        if len(r_traj) < 10:
            return
        
        # Calculate total length of trajectory to space arrows evenly
        points = np.array(list(zip(r_traj, phi_traj)))
        segments = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_length = np.concatenate(([0], np.cumsum(segments)))
        total_length = cumulative_length[-1]
        
        # Place arrows at evenly spaced points, skipping the beginning and end
        arrow_positions = np.linspace(0.2 * total_length, 0.8 * total_length, n_arrows)
        
        # Get figure dimensions for proper scaling
        fig_width, fig_height = ax.figure.get_size_inches()
        data_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        data_height = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        # Calculate aspect ratio correction
        aspect_correction = (data_height/fig_height) / (data_width/fig_width)
        
        # Scale factor for arrow size
        scale = arrow_size / 1000
        
        for pos in arrow_positions:
            # Find the closest point on the trajectory
            idx = np.argmin(np.abs(cumulative_length - pos))
            
            # Don't place arrows at the very beginning or end
            if idx < 5 or idx >= len(r_traj) - 5:
                continue
            
            # Get position and calculate direction
            r, phi = r_traj[idx], phi_traj[idx]
            
            # Calculate direction vector (using points a few steps ahead for smoother direction)
            dr = r_traj[idx + 3] - r
            dphi = phi_traj[idx + 3] - phi
            
            # Normalize the direction vector
            length = np.sqrt(dr**2 + dphi**2)
            if length > 0:
                dr /= length
                dphi /= length
                
                # Calculate arrow dimensions with correct aspect
                # Make head width proportionally smaller compared to length
                arrow_length = scale *0.5
                head_width = scale * 0.01 * aspect_correction  # Narrower head width
                head_length = scale * 0.4  # Shorter head length
                
                # Plot the arrow
                ax.arrow(r, phi, dr * arrow_length, dphi * arrow_length, 
                        head_width=head_width, 
                        head_length=head_length,
                        fc=color, ec=color, 
                        alpha=0.9, zorder=zorder,
                        length_includes_head=True)    
    # === BEGIN UPDATED STREAMLINE CONFIGURATION ===
    # Updated streamline points with consistent coloring and more planet-bound candidates
    streamline_points = [
        # Outer disk flow (right side) - green in paper - ADD MORE STREAMLINES
        (1.0 + 1.2*r_h, np.pi, 'green', 'Outer Disk'),
        (1.0 + 1.5*r_h, np.pi, 'green', 'Outer Disk'),
        (1.0 + 1.8*r_h, np.pi, 'green', 'Outer Disk'),
        
        # Outer disk flow (left side) - yellow in paper - ADD MORE STREAMLINES
        (1.0 - 1.2*r_h, np.pi, 'gold', 'Outer Disk'),
        (1.0 - 1.5*r_h, np.pi, 'gold', 'Outer Disk'),
        (1.0 - 1.8*r_h, np.pi, 'gold', 'Outer Disk'),
        
        # Horseshoe region (upper) - blue in paper
        (1.0, np.pi - 0.2, 'blue', 'Horseshoe'),
        (1.0 - 0.3*r_h, np.pi - 0.18, 'blue', 'Horseshoe'),
        (1.0 + 0.3*r_h, np.pi - 0.18, 'blue', 'Horseshoe'),
        
        # Horseshoe region (lower) - red in paper
        (1.0, np.pi + 0.2, 'red', 'Horseshoe'),
        (1.0 - 0.3*r_h, np.pi + 0.18, 'red', 'Horseshoe'),
        (1.0 + 0.3*r_h, np.pi + 0.18, 'red', 'Horseshoe'),
        
        # Planet-bound region - magenta in paper (smaller orbits closer to planet)
        (1.0 + 0.2*r_h, np.pi, 'magenta', 'Planet Bound'),
        (1.0 - 0.2*r_h, np.pi, 'magenta', 'Planet Bound'),
        (1.0, np.pi + 0.1*r_h, 'magenta', 'Planet Bound'),
        (1.0, np.pi - 0.1*r_h, 'magenta', 'Planet Bound'),
        (1.0 + 0.05*r_h, np.pi, 'magenta', 'Planet Bound'),
        (1.0 - 0.05*r_h, np.pi, 'magenta', 'Planet Bound'),
    
    # NEW ADDITION: Planet-bound region (upward) - purple color
    # Points just slightly above the planet in the upper half
    (1.0, np.pi - 0.005, 'purple', 'Planet Bound - Upper'),
    (1.0 + 0.05*r_h, np.pi - 0.008, 'purple', 'Planet Bound - Upper'),
    (1.0 - 0.05*r_h, np.pi - 0.008, 'purple', 'Planet Bound - Upper'),
    
    # Points in upper quadrants of Hill sphere
    (1.0 + 0.3*r_h, np.pi - 0.01, 'purple', 'Planet Bound - Upper'),
    (1.0 - 0.3*r_h, np.pi - 0.01, 'purple', 'Planet Bound - Upper'),
    
    # Points near stagnation regions (upper side)
    (1.0 + 0.6*r_h, np.pi - 0.005, 'purple', 'Planet Bound - Upper'),
    (1.0 - 0.6*r_h, np.pi - 0.005, 'purple', 'Planet Bound - Upper')
] 
    # === END UPDATED STREAMLINE CONFIGURATION ===

    # === BEGIN ADDITIONAL HORSESHOE STREAMLINES (REMOVE TO CLEAN UP HS REGION) ===
    r_offsets = [0.0, -0.2*r_h, 0.2*r_h]  # Radial offsets around r = 1.0
    phi_offsets = np.linspace(0.05, 0.15, 4)  # Closer φ distances from the planet

    for sign, color in [(-1, 'blue'), (1, 'red')]:  # Upper (blue) and lower (red)
        for dr in r_offsets:
            for dp in phi_offsets:
                r0 = 1.0 + dr
                phi0 = np.pi + sign * dp
                streamline_points.append((r0, phi0, color, 'Horseshoe - Refined'))
    # === END ADDITIONAL HORSESHOE STREAMLINES (REMOVE TO CLEAN UP HS REGION)  ===
    
    # Create a figure for the density deviation
    fig, ax = plt.subplots(figsize=(12, 10))  # Use fig and ax for more control
    plt.rcParams.update({'font.size': 14})
    
    # Create a custom color normalization that caps at 5e-5 but keeps the min value
    vmin = np.min(zoomed_density)
    vmax = min(5e-5, np.max(zoomed_density))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the density deviation with the custom normalization
    im = ax.pcolormesh(r_mesh, phi_mesh, zoomed_density, 
                   shading='auto', cmap='RdBu_r', norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density Deviation (capped at 5e-5)', fontsize=16)
    
    # Group streamlines by type for legend
    legend_entries = {}
    
    # Process each streamline
    for i, (r0, phi0, color, label) in enumerate(streamline_points):
        print(f"\nTracing streamline {i+1}: {label}...")
        
        # Integrate forward
        r_forward, phi_forward = integrate_streamline(r0, phi0, forward=True)
        
        # Integrate backward
        r_backward, phi_backward = integrate_streamline(r0, phi0, forward=False)
        
        # Combine the trajectories
        r_traj = np.concatenate((r_backward[::-1], r_forward))
        phi_traj = np.concatenate((phi_backward[::-1], phi_forward))
        
        # Check if we found a horseshoe orbit (crosses both sides of planet's orbit)
        if "Horseshoe" in label:
            if (np.min(r_traj) < 1.0 and np.max(r_traj) > 1.0):
                print(f"Potential horseshoe orbit found for {label}!")
        
        # Check if we found a planet-bound orbit (forms a closed loop)
        if "Planet Bound" in label:
            # Calculate distance from start point to end point
            r_end, phi_end = r_traj[-1], phi_traj[-1]
            dist = np.sqrt((r_end - r0)**2 + (phi_end - phi0)**2)
            if dist < 0.01:  # If start and end points are close
                print(f"Potential closed planet-bound orbit found for {label}!")
        
        # Extract base label for legend grouping
        base_label = label.split('(')[0].strip()
        if base_label not in legend_entries:
            legend_entries[base_label] = {"color": color, "label": base_label}
            plot_label = base_label  # Use the base label for the first instance
        else:
            plot_label = None  # No label for subsequent instances of same type
        
        # Plot the streamline
        ax.plot(r_traj, phi_traj, color=color, linewidth=2, label=plot_label)

        # Add direction arrows to the streamline
        # Adjust n_arrows based on streamline type
        if "Horseshoe" in label:
            n_arrows = 4  # More arrows for longer horseshoe orbits
        elif "Outer Disk" in label:
            n_arrows = 3  # Several arrows for disk flows
        else:
            n_arrows = 2  # Fewer arrows for smaller orbits
    
        if "Planet Bound" not in label:  # Only add arrows for non-planet-bound streamlines
            add_direction_arrows(ax, r_traj, phi_traj, color, n_arrows=n_arrows)

        
        # Mark the starting point
        plt.scatter([r0], [phi0], color=color, s=100, edgecolors='white', zorder=6)
    
    # Add vertical line at planet position (r=1)
    ax.axvline(x=1.0, color='white', linestyle='--', linewidth=1.5)
    
    # Add horizontal line at planet position (φ=π)
    ax.axhline(y=np.pi, color='white', linestyle='--', linewidth=1.5)

    # Mark planet position (at r=1, φ=π)
    ax.scatter([1.0], [np.pi], color='red', s=150, edgecolors='white', zorder=5, label='Planet')
   
    # Add circle at 1 Hill radius from planet
    circle = plt.Circle((1.0, np.pi), r_h, fill=False, linestyle='-', 
                      color='white', alpha=0.7, linewidth=1.5)
    plt.gca().add_artist(circle)
    
    # Add stagnation points to the density plot
    for r_stag, phi_stag in stagnation_points:
        plt.scatter([r_stag], [phi_stag], marker='x', color='black', s=100, linewidth=2, zorder=7)
        print(f"Added stagnation point at r={r_stag:.4f}, phi={phi_stag:.4f}")
    
    # Set plot properties
    plt.xlabel('Radius (r)', fontsize=16)
    plt.ylabel('Azimuth (φ)', fontsize=16)
    plt.title(f'Time-Averaged Density Deviation (n={actual_n_avg})', fontsize=18)
    plt.legend(fontsize=14, loc='upper right')
    
    # Set axis limits to show the reduced radial domain
    ax.set_xlim(0.97, 1.03)  # Reduced domain as requested
    ax.set_ylim(phi_min, phi_max)

    # Set axis ticks to create a sparse grid
    r_major_ticks = np.arange(0.97, 1.031, 0.0025)  # 4 ticks per 0.01 radius
    phi_major_ticks = np.arange(phi_min, phi_max + 1e-6, 0.025)  # 4 ticks per 0.1 azimuth

    # Sparse gridlines (still every 0.0025 and 0.025)
    ax.set_xticks(r_major_ticks, minor=True)
    ax.set_yticks(phi_major_ticks, minor=True)

    # Major ticks only every 0.01 in radius, 0.1 in azimuth
    ax.set_xticks(np.arange(0.97, 1.031, 0.01))
    ax.set_yticks(np.arange(phi_min, phi_max + 1e-6, 0.1))

    # Enable grid along those ticks
    ax.grid(True, which='major', linestyle=':', color='gray', linewidth=0.7, alpha=0.5)

    # Optional: avoid raster grid artifacts in the colormesh (especially in PDFs)
    im.set_rasterized(True)
    
    # Add annotation with key parameters
    plt.figtext(0.02, 0.02, 
               f"Hill Radius = {r_h:.5f}\n"
               f"Snapshots: {start_snapshot}-{end_snapshot-1}",
               fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.5", fc="white"))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_filename = os.path.join(
        output_path, f"{simulation_name}_multiple_streamlines_n{actual_n_avg}.pdf"
    )
    plt.savefig(output_filename)
    print(f"Multiple streamlines plot saved to {output_filename}")
    
    # Transfer the plot
    local_directory = "/Users/mariuslehmann/Downloads/Contours/planet_evolution"
    print(f"Transferring plot to {local_directory}...")
    scp_transfer(output_filename, local_directory, "mariuslehmann")
    
    plt.close()
    
    # Create a reference-style plot in Hill radius units
    fig_ref, ax_ref = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})
    
    # Convert to Hill radius units
    r_scaled = (r_mesh - 1.0) / r_h
    phi_scaled = (phi_mesh - np.pi) / r_h
    
    # Plot density deviation in scaled coordinates
# Plot density deviation in scaled coordinates
    plt.pcolormesh(r_scaled, phi_scaled, zoomed_density, 
                 shading='auto', cmap='RdBu_r', norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('Density Deviation', fontsize=14)
    
    # Reset legend entries for the reference plot
    legend_entries = {}
    
    # Plot each streamline in scaled coordinates with grouped legend
    for i, (r0, phi0, color, label) in enumerate(streamline_points):
        # Get the saved trajectory
        r_forward, phi_forward = integrate_streamline(r0, phi0, forward=True)
        r_backward, phi_backward = integrate_streamline(r0, phi0, forward=False)
        r_traj = np.concatenate((r_backward[::-1], r_forward))
        phi_traj = np.concatenate((phi_backward[::-1], phi_forward))
        
        # Convert to scaled coordinates
        r_scaled_traj = (r_traj - 1.0) / r_h
        phi_scaled_traj = (phi_traj - np.pi) / r_h
        
        # Extract base label for legend grouping
        base_label = label.split('(')[0].strip()
        if base_label not in legend_entries:
            legend_entries[base_label] = {"color": color, "label": base_label}
            plot_label = base_label  # Use the base label for the first instance
        else:
            plot_label = None  # No label for subsequent instances of same type
        
        # Plot the streamline
        plt.plot(r_scaled_traj, phi_scaled_traj, color=color, linewidth=2, label=plot_label)
        
        # Mark the starting point
        plt.scatter([(r0 - 1.0) / r_h], [(phi0 - np.pi) / r_h], color=color, s=80, edgecolors='white', zorder=6)
    
    # Mark planet at origin
    plt.scatter([0], [0], color='red', s=100, edgecolors='white', 
               zorder=5, label='Planet')
    
    # Add coordinate axes
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add reference circles
    circle1 = plt.Circle((0, 0), 1.0, fill=False, linestyle='-', 
                       color='black', linewidth=1)
    plt.gca().add_artist(circle1)
    
    circle2 = plt.Circle((0, 0), 2.0, fill=False, linestyle=':', 
                       color='black', linewidth=1)
    plt.gca().add_artist(circle2)
    
    # Mark the stagnation points on the reference plot (convert to Hill radius units)
    print("Marking stagnation points on reference plot...")
    for r_stag, phi_stag in stagnation_points:
        # Convert to Hill radius units
        x_stag = (r_stag - 1.0) / r_h
        y_stag = (phi_stag - np.pi) / r_h
        plt.scatter([x_stag], [y_stag], marker='x', color='black', s=100, linewidth=2, zorder=7)
        print(f"Stagnation point marked at ({x_stag:.2f}, {y_stag:.2f}) Hill radii")
    
    # Set plot properties
    plt.axis('equal')
    plt.xlim(-2, 2)  # Match the paper's x-range
    plt.ylim(-2, 2)  # Match the paper's y-range
    plt.xlabel(r'(r-a)/r$_H$', fontsize=14)
    plt.ylabel(r'a($\phi$-$\phi_p$)/r$_H$', fontsize=14)
    plt.title('Streamlines in Co-rotating Frame', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the reference-style plot
    ref_filename = os.path.join(
        output_path, f"{simulation_name}_reference_multiple_n{actual_n_avg}.pdf"
    )
    plt.savefig(ref_filename)
    print(f"Reference-style plot saved to {ref_filename}")
    
    # Transfer the reference-style plot
    print(f"Transferring reference-style plot to {local_directory}...")
    scp_transfer(ref_filename, local_directory, "mariuslehmann")
    
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Plot density with multiple streamlines.")
    parser.add_argument("simulation_name", help="Name of the simulation")
    parser.add_argument("snapshot_number", nargs='?', type=int, default=None, 
                        help="Center snapshot number for time averaging (default: last snapshot)")
    parser.add_argument("--n-avg", type=int, default=20, 
                        help="Number of snapshots to average (default: 20)")
    parser.add_argument("--vertical-mode", type=str, choices=['midplane', 'average'], default='midplane',
                        help="How to handle vertical dimension in 3D data (default: midplane)")
    # Parse arguments
    args = parser.parse_args()
    
    # Start execution
    try:
        # Get basic parameters
        simulation_name = args.simulation_name
        snapshot_number = args.snapshot_number
        n_avg = args.n_avg
        
        # Determine the base path and output path
        base_path = determine_base_path(simulation_name)
        output_path = base_path
        
        # Read summary file for parameters
        summary_file = os.path.join(output_path, "summary0.dat")
        
        if not os.path.exists(summary_file):
            print(f"Summary file not found: {summary_file}")
            sys.exit(1)
        
        # Extract planet mass and migration data
        qp, migration = extract_planet_mass_and_migration(summary_file)
        
        # Plot density with multiple streamlines - MOVED INSIDE THE TRY BLOCK
        plot_multiple_region_streamlines(
            output_path, 
            simulation_name, 
            qp, 
            snapshot_number,
            args.n_avg,
            args.vertical_mode
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("Done!")
