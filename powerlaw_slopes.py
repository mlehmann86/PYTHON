#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import warnings

# ==============================================================================
# Assumed Imports from Helper Modules
# ==============================================================================
try:
    from data_storage import determine_base_path, scp_transfer
    from data_reader  import read_parameters, determine_nt, reconstruct_grid
    from planet_data  import extract_planet_mass_and_migration
except ImportError as e:
    print(f"Error importing required functions: {e}")
    sys.exit(1)

# ==============================================================================
# read_field_file: user’s original routine
# ==============================================================================
def read_field_file(path, field_name, snapshot, nx, ny, nz):
    file = os.path.join(path, f"{field_name}{snapshot}.dat")
    if not os.path.exists(file):
        return None
    try:
        data = np.fromfile(file, dtype=np.float64)
        expected = nx * ny * nz
        if data.size != expected:
            print(f"ERROR: {file} size mismatch: expected {expected}, got {data.size}")
            return None
        return data.reshape((nz, nx, ny)).transpose(2, 1, 0)
    except Exception as e:
        print(f"ERROR reading {file}: {e}")
        return None

# ==============================================================================
# get_param_value: robust parameter extractor
# ==============================================================================
def get_param_value(p_dict, key, expected_type, default=None):
    val = p_dict.get(key)
    if val is None:
        if default is not None:
            print(f"INFO: '{key}' not found, using default {default}")
            return default
        raise ValueError(f"Missing required '{key}'")
    if isinstance(val, (list, tuple)):
        if not val:
            raise ValueError(f"Empty list for '{key}'")
        val = val[0]
    if expected_type == int and isinstance(val, (int, float)):
        f = float(val)
        if f.is_integer():
            return int(f)
        raise ValueError(f"'{key}' value {val} not integer")
    if expected_type == float and isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.match(r"^\s*([+-]?\d+(\.\d*)?([eE][+-]?\d+)?)", val)
        if m:
            return expected_type(m.group(1))
        if expected_type == str:
            return val
    if isinstance(val, expected_type):
        return val
    raise TypeError(f"Cannot convert '{key}'={val} to {expected_type}")

# ==============================================================================
# find_all_snapshot_numbers
# ==============================================================================
def find_all_snapshot_numbers(path, nt_max):
    snaps = set()
    pat = re.compile(r'gasdens(\d+)\.dat')
    try:
        for fn in os.listdir(path):
            m = pat.match(fn)
            if m:
                idx = int(m.group(1))
                if nt_max is None or idx <= nt_max:
                    snaps.add(idx)
    except Exception as e:
        print(f"ERROR scanning {path}: {e}")
        return []
    if not snaps:
        print(f"ERROR: No gasdens*.dat in {path}")
        return []
    lst = sorted(snaps)
    if len(lst) > 10:
        print(f"Found {len(lst)} snapshots: {lst[:5]}...{lst[-5:]}")
    else:
        print(f"Found snapshots: {lst}")
    return lst

# ==============================================================================
# Main: compute & plot
# ==============================================================================
def compute_and_plot_slope_evolution(simulation_name):
    output_path = "plots_output"
    slice_az = np.pi
    slice_r  = 1.0
    r_center = 1.0
    radial_width_H_outer = 6.0

    print(f"\n=== Processing {simulation_name} ===")
    sim_path = determine_base_path(simulation_name)
    if not sim_path or not os.path.isdir(sim_path):
        print(f"ERROR: Invalid path '{sim_path}'"); return

    summary_file = os.path.join(sim_path, "summary0.dat")
    if not os.path.exists(summary_file):
        print(f"ERROR: summary0.dat not found at '{summary_file}'"); return

    # --- Read parameters & grid ---
    try:
        params = read_parameters(summary_file)
        xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(params)
        print(f"Grid: nx={nx}, ny={ny}, nz={nz}")

        gamma        = get_param_value(params, 'GAMMA', float)
        h            = get_param_value(params, 'ASPECTRATIO', float)
        dt           = get_param_value(params, 'DT', float)
        ninterm      = get_param_value(params, 'NINTERM', int)
        thick_smooth = get_param_value(params, 'THICKNESSSMOOTHING', float, default=0.4)

        qp, _        = extract_planet_mass_and_migration(summary_file)
        qp = float(qp)

        sigmaslope_param   = params.get('SIGMASLOPE', None)
        flaringindex_param = params.get('FLARINGINDEX', '0.0')

        orbit_period = 2.0 * np.pi
        output_interval_orbits = ninterm * dt / orbit_period

        print(f"Params: GAMMA={gamma}, H/R={h}, q={qp:.2e}")
        print(f"Dt={dt}, NINTERM={ninterm}, ThickSmooth={thick_smooth}")
        print(f"Output every {output_interval_orbits:.3f} orbits")

        is_3d = (nz > 1)
        if is_3d:
            print(f"Note: 3D data detected (nz={nz}); will vertically average into 2D.")
            # compute volume‐density slope p = σ + η + 1
            ss = get_param_value(params, 'SIGMASLOPE', float)
            fi = get_param_value(params, 'FLARINGINDEX', float, default=0.0)
            expected_density_slope = ss + fi + 1.0
        else:
            expected_density_slope = get_param_value(params, 'SIGMASLOPE', float)

    except Exception as e:
        print(f"ERROR reading params/grid: {e}")
        return

    # --- Snapshots ---
    try:
        nt_max = determine_nt(sim_path)
    except:
        nt_max = None
    snapshot_indices = find_all_snapshot_numbers(sim_path, nt_max)
    if not snapshot_indices:
        return
    times = np.array(snapshot_indices) * ninterm * dt / orbit_period

    # --- Slice indices ---
    idx_az = np.argmin(np.abs(ygrid - slice_az))
    idx_r  = np.argmin(np.abs(xgrid - slice_r))

    # --- Regions ---
    log_r = np.log(xgrid)
    # Inner horseshoe
    term_smooth = (thick_smooth/0.4)**0.25 if thick_smooth>0 else 1.0
    term_gamma  = (1.0/gamma)**0.25
    term_q_h    = np.sqrt(qp/h)
    x_s = 1.1 * r_center * term_smooth * term_gamma * term_q_h
    half = x_s/2.0
    inner_idx = np.where((xgrid>=r_center-half)&(xgrid<=r_center+half))[0]
    # Outer
    wide   = np.where((xgrid>=r_center-radial_width_H_outer*h)&(xgrid<=r_center+radial_width_H_outer*h))[0]
    if inner_idx.size>0:
        outer_idx = np.array(sorted(set(wide)-set(inner_idx)),int)
    else:
        outer_idx = wide

    # --- Storage ---
    alpha_i=[]; beta_i=[]
    alpha_o=[]; beta_o=[]
    proc=[]

    rho_rad_init=temp_rad_init=rho_rad_fin=temp_rad_fin=None
    rho_azi_init=temp_azi_init=rho_azi_fin=temp_azi_fin=None
    first=True

    # --- Loop snapshots ---
    for i, snap in enumerate(snapshot_indices):
        print(f"Snapshot {snap} ({i+1}/{len(snapshot_indices)})")
        gd3 = read_field_file(sim_path,'gasdens',snap,nx,ny,nz)
        ge3 = read_field_file(sim_path,'gasenergy',snap,nx,ny,nz)
        if gd3 is None or ge3 is None:
            continue

        # vertical average or reshape
        if nz>1:
            gd2 = np.mean(gd3,axis=2)
            ge2 = np.mean(ge3,axis=2)
        else:
            gd2 = gd3.reshape(ny,nx)
            ge2 = ge3.reshape(ny,nx)
        if gd2.shape!=(ny,nx):
            print(f"  ERROR shape {gd2.shape}, expected ({ny},{nx})"); continue

        gasdens = np.maximum(gd2,1e-30)
        gasene  = np.maximum(ge2,1e-30)

        temp_proxy = gasene/gasdens

        # slices
        rs = gasdens[idx_az,:].copy(); ts = temp_proxy[idx_az,:].copy()
        zs = gasdens[:,idx_r].copy(); tz = temp_proxy[:,idx_r].copy()

        if first:
            rho_rad_init  = rs; temp_rad_init  = ts
            rho_azi_init  = zs; temp_azi_init  = tz
        rho_rad_fin, temp_rad_fin = rs, ts
        rho_azi_fin, temp_azi_fin = zs, tz

        # slopes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            slog_rho = np.gradient(np.log(np.mean(gasdens,axis=0)), log_r)
            slog_T   = np.gradient(np.log(np.mean(temp_proxy,axis=0)), log_r)

        ai = np.nan; bi=np.nan; ao=np.nan; bo=np.nan
        if inner_idx.size>=2:
            ai = -np.mean(slog_rho[inner_idx])
            bi = -np.mean(slog_T[inner_idx])
        if outer_idx.size>=2:
            ao = -np.mean(slog_rho[outer_idx])
            bo = -np.mean(slog_T[outer_idx])
        if np.isfinite(ai) or np.isfinite(ao):
            alpha_i.append(ai); beta_i.append(bi)
            alpha_o.append(ao); beta_o.append(bo)
            proc.append(i)
            if first: first=False

    if not proc:
        print("ERROR: No data processed"); return

    proc = np.array(proc)
    tproc= times[proc]
    alpha_i=np.array(alpha_i); beta_i=np.array(beta_i)
    alpha_o=np.array(alpha_o); beta_o=np.array(beta_o)

    os.makedirs(output_path, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # === Plot 1: Slope Evolution ===
    fig1, ax1 = plt.subplots(2,1,figsize=(12,9),sharex=True)
    ax1[0].plot(tproc, alpha_i, 'b-', label='Inner α')
    ax1[0].plot(tproc, alpha_o, 'g--',label='Outer α')
    ax1[0].axhline(expected_density_slope, color='grey', linestyle=':', linewidth=1.5,
                   label=(f'p={expected_density_slope:.2f}' if is_3d else f'σ={expected_density_slope:.2f}'))
    ax1[0].set_ylabel('α'); ax1[0].legend(); ax1[0].grid(True)

    ax1[1].plot(tproc, beta_i, 'r-', label='Inner β_T')
    ax1[1].plot(tproc, beta_o, 'm--',label='Outer β_T')
    eb = 1.0 - 2.0 * get_param_value(params,'FLARINGINDEX',float,default=0.0)
    ax1[1].axhline(eb, color='grey', linestyle=':', linewidth=1.5, label=f'β_T={eb:.2f}')
    ax1[1].set_ylabel('β_T'); ax1[1].set_xlabel('Orbits')
    ax1[1].legend(); ax1[1].grid(True)

    f1 = os.path.join(output_path, f"{simulation_name}_slope_evolution_regions.pdf")
    fig1.savefig(f1, dpi=300, bbox_inches='tight'); plt.close(fig1)
    print(f"Saved {f1}")

    # === Plot 2: Radial Slices ===
    if rho_rad_init is not None:
        fig2, ax2 = plt.subplots(2,1,figsize=(10,8),sharex=True)
        t0, tf = tproc[0], tproc[-1]
        ax2[0].semilogy(xgrid, rho_rad_init,'k--', label=f'Init t={t0:.1f}')
        ax2[0].semilogy(xgrid, rho_rad_fin, 'b-', label=f'Final t={tf:.1f}')
        rho1 = np.interp(r_center, xgrid, rho_rad_fin)
        ax2[0].plot(xgrid, rho1*(xgrid/r_center)**(-expected_density_slope),
                   color='grey', linestyle=':', label=(f'p={expected_density_slope:.2f}' if is_3d else f'σ={expected_density_slope:.2f}'))
        ax2[0].set_ylabel('Σ'); ax2[0].legend(); ax2[0].grid(True)

        ax2[1].semilogy(xgrid, temp_rad_init,'k--', label=f'Init t={t0:.1f}')
        ax2[1].semilogy(xgrid, temp_rad_fin, 'r-', label=f'Final t={tf:.1f}')
        T1 = np.interp(r_center, xgrid, temp_rad_fin)
        ax2[1].plot(xgrid, T1*(xgrid/r_center)**(-(1.0-2.0*get_param_value(params,'FLARINGINDEX',float,default=0.0))),
                   color='grey', linestyle=':', label=f'β_T={eb:.2f}')
        ax2[1].set_xlabel('r/r_p'); ax2[1].set_ylabel('T'); ax2[1].legend(); ax2[1].grid(True)

        f2 = os.path.join(output_path, f"{simulation_name}_radial_slices.pdf")
        fig2.savefig(f2, dpi=300, bbox_inches='tight'); plt.close(fig2)
        print(f"Saved {f2}")

    # === Plot 3: Azimuthal Slices ===
    if rho_azi_init is not None:
        fig3, ax3 = plt.subplots(2,1,figsize=(10,8),sharex=True)
        t0, tf = tproc[0], tproc[-1]
        ax3[0].plot(ygrid, rho_azi_init,'k--', label=f'Init t={t0:.1f}')
        ax3[0].plot(ygrid, rho_azi_fin, 'b-', label=f'Final t={tf:.1f}')
        ax3[0].set_ylabel('Σ'); ax3[0].legend(); ax3[0].grid(True)

        ax3[1].plot(ygrid, temp_azi_init,'k--', label=f'Init t={t0:.1f}')
        ax3[1].plot(ygrid, temp_azi_fin, 'r-', label=f'Final t={tf:.1f}')
        ax3[1].set_xlabel('φ'); ax3[1].set_ylabel('T'); ax3[1].legend(); ax3[1].grid(True)

        f3 = os.path.join(output_path, f"{simulation_name}_azimuthal_slices.pdf")
        fig3.savefig(f3, dpi=300, bbox_inches='tight'); plt.close(fig3)
        print(f"Saved {f3}")

    # === SCP transfer ===
    local_dir = "/Users/mariuslehmann/Downloads/Profiles/"
    user      = "mariuslehmann"
    for f in (f1, locals().get('f2'), locals().get('f3')):
        if f and os.path.exists(f):
            scp_transfer(f, local_dir, user)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("simulation_name", help="Simulation directory name")
    args = p.parse_args()
    compute_and_plot_slope_evolution(args.simulation_name)
    print("\n=== Finished ===")
