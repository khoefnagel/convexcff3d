import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import runpy

plt.rcParams.update({'font.size': 12})

cells_per_radius = runpy.run_path('./resolution_definition.py')['cells_per_radius']
Re = runpy.run_path('../Reynolds_definition.py')['Re']
case = runpy.run_path('../../case_definition.py')['Case'](cells_per_radius=cells_per_radius, Re=Re)

D = case.D
Nx = case.Nx
Ny = case.Ny
Nz = case.Nz
Nt = case.Nt
sphere_center = case.x_c

for boundary_treatment_folder in glob.glob('./output/*/'):
    boundary_treatment = os.path.basename(os.path.normpath(boundary_treatment_folder))
    center_profile = pv.read(f'{boundary_treatment_folder}/center_profile_{Nt:05d}.vti')
    ux = center_profile['u'][:,0] if 'u' in center_profile.array_names else center_profile['velocity'][:,0]

    cell_centers = center_profile.cell_centers().points

    zero_crossings = np.where((ux[1:] >= 0) & (ux[:-1] < 0))[0]
    if len(zero_crossings) == 0:
        x_cross = sphere_center[0] + 0.5*D
        x_s = 0
    else:
        crossing_idx = np.where((ux[1:] >= 0) & (ux[:-1] < 0))[0][-1]

        # Interpolate to find more accurate crossing point
        x1, x2 = cell_centers[crossing_idx, 0], cell_centers[crossing_idx+1, 0]
        ux1, ux2 = ux[crossing_idx], ux[crossing_idx+1]
        x_cross = x1 + (x2 - x1) * (-ux1) / (ux2 - ux1)

        # Calculate separation length
        x_s = (x_cross - sphere_center[0] - 0.5*D)/ D

    if not os.path.exists(f'plots/{boundary_treatment}'):
        os.makedirs(f'plots/{boundary_treatment}')

    plt.figure()
    plt.plot(cell_centers[:,0], ux)
    plt.axvline(x_cross, color='red', linestyle='--', label=f'Separation length (x_s={x_s:.3f}D)')
    plt.xlabel('x')
    plt.ylabel('u_x')
    plt.title('Centerline Velocity Profile')
    legend = plt.legend()
    legend.set_zorder(100)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./plots/{boundary_treatment}/centerline_velocity_profile.png', dpi=300)

    # Save as csv file
    with open(f'{boundary_treatment_folder}/separation_length.csv', 'w') as f:
        f.write('x_s\n')
        f.write(f'{x_s}\n')

    # Surface forces are not implemented for Bouzidi, so we skip this part for that case
    if not boundary_treatment == 'Bouzidi':
        surface_force = pv.read(f'{boundary_treatment_folder}/fsurf_{Nt:05d}.vtp')
        normals = surface_force.face_normals
        fsurf = surface_force['force']

        e_u = np.array([1, 0, 0]) # Freestream velocity direction

        # Meridional tangential direction (how the fluid would move over the sphere surface)
        t =  e_u - normals * np.sum(e_u * normals, axis=-1)[:, None]
        t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-15) # Normalize and add small epsilon to avoid division by zero

        # Angle wrt freestream direction
        theta = np.arccos(np.sum(-normals * e_u, axis=-1))

        # Meridional tangential force magnitude
        F_t = np.sum(fsurf * t, axis=-1)

        # Bin theta into 1 degree bins and average F_t in each bin
        theta_bins = np.deg2rad(np.arange(0, 181, 1))  # 0 to 180 degrees in radians
        F_t_binned = np.zeros(len(theta_bins)-1)
        for i in range(len(theta_bins)-1):
            bin_mask = (theta >= theta_bins[i]) & (theta < theta_bins[i+1])
            if np.sum(bin_mask) > 0:
                F_t_binned[i] = np.mean(F_t[bin_mask])

        # Find crossing index
        crossing_idx = np.where((F_t_binned[:-1] > 0) & (F_t_binned[1:] < 0) & (theta_bins[:-2] > np.deg2rad(60)))[0][0].item()

        # Interpolate to find more accurate crossing angle
        theta1, theta2 = theta_bins[crossing_idx] + np.deg2rad(0.5), theta_bins[crossing_idx+1] + np.deg2rad(0.5)  # Use bin centers for interpolation
        F1, F2 = F_t_binned[crossing_idx], F_t_binned[crossing_idx+1]
        theta_crossing = theta1 - F1 * (theta2 - theta1) / (F2 - F1)

        plt.figure()
        plt.scatter(np.rad2deg(theta), F_t, s=5, zorder=10, c='b')
        plt.plot(np.rad2deg(theta_bins[:-1] + np.diff(theta_bins)/2), F_t_binned, 'r-', label='Binned Average', zorder=20)
        plt.axvline(np.rad2deg(theta_crossing), color='g', linestyle='--', label=f'Crossing Angle: {np.rad2deg(theta_crossing):.2f}°', zorder=30)
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Tangential Force Magnitude')
        legend = plt.legend()
        legend.set_zorder(100)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'./plots/{boundary_treatment}/tangential_force_magnitude.png', dpi=300)

        # Save as .csv file
        with open(f'{boundary_treatment_folder}/separation_angle.csv', 'w') as f:
            f.write('separation_angle\n')
            f.write(f'{np.rad2deg(theta_crossing)}\n')



