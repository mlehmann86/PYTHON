import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from data_reader import read_parameters, reconstruct_grid, read_idefix_vtk

# === CONFIGURATION ===
subdir_path = "/tiara/home/mlehmann/data/idefix-mkl/outputs/PlanetDisk2D/test_noniso"
summary_file = os.path.join(subdir_path, "idefix.0.log")
vtk_file = os.path.join(subdir_path, "data.0000.vtk")

# === Load parameters and reconstruct grid ===
params = read_parameters(summary_file, IDEFIX=True)
xgrid, ygrid, zgrid, ny, nx, nz = reconstruct_grid(params, IDEFIX=True)
print(f"\n✅ Grid sizes: nx={nx}, ny={ny}, nz={nz}")
print(f"✅ Radial range: [{xgrid[0]}, {xgrid[-1]}]")

# === VTK file inspection ===
print("\n🔍 Inspecting VTK file header and fields:")
reader = vtk.vtkStructuredGridReader()
reader.SetFileName(vtk_file)
reader.ReadAllScalarsOn()
reader.ReadAllVectorsOn()
reader.Update()

output = reader.GetOutput()
cell_data = output.GetCellData()
n_arrays = cell_data.GetNumberOfArrays()
n_cells = output.GetNumberOfCells()

print(f"  → Number of cells: {n_cells}")
print(f"  → Number of cell data arrays: {n_arrays}")
print("  → Field names:")
for i in range(n_arrays):
    name = cell_data.GetArrayName(i)
    print(f"     • {i}: {name}")

# === Read snapshot using your reader ===
data_arrays, data_types, *_ = read_idefix_vtk(
    subdir_path, gas=True, dust=False, noniso=False, itstart=0, itend=1, nsteps=1
)
gasdens = data_arrays['gasdens']  # shape: [ny, nx, nz, nt]

# === Extract midplane and azimuthal slice ===
iy = ny // 2   # mid azimuth
iz = nz // 2   # midplane
r_vals = xgrid
rho_vals = gasdens[iy, :, iz, 0]

# === Plot ===
plt.figure(figsize=(6, 4))
plt.plot(r_vals, rho_vals, label='gasdens from VTK')

# Optional: overlay power-law
sigma0 = params.get('sigma0', 1.0)
p = params.get('sigmaSlope', 1.5)
plt.plot(r_vals, sigma0 * r_vals**(-p), 'k--', label=r'$\Sigma \propto r^{-%.1f}$' % p)

plt.xlabel('Radius r')
plt.ylabel('Gas density')
plt.title('Radial profile of gasdens (t=0)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
