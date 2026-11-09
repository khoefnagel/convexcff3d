import numpy as np
import sys
import pyvista as pv
import matplotlib.pyplot as plt
import runpy

plt.rcParams.update({'font.size': 14})

h = runpy.run_path('./resolution_definition.py')['h']
theta_deg = runpy.run_path('../theta_definition.py')['theta_deg']
case = runpy.run_path('../../case_definition.py')['Case'](h=h, theta_deg=theta_deg)

Nt = case.Nt


boundary_treatments = [
    'CFF', 
    'Chen',
    'Bouzidi', 
]




# Error norms
L1_err = lambda u, u_ref: np.sum(np.abs(u - u_ref)) / np.sum(np.abs(u_ref))
L2_err = lambda u, u_ref: np.sqrt(np.sum((u - u_ref)**2) / np.sum(u_ref**2))
Linf_err = lambda u, u_ref: np.max(np.abs(u - u_ref))

colors = ['r', 'b', 'g']
markers = ['s', 'o', '^']
for j, boundary_treatment in enumerate(boundary_treatments):
    data = pv.read(f'./output/{boundary_treatment}/full_field_{Nt:05d}.vti')

    u = np.linalg.norm(data['u'], axis=1)
    V = data['V']

    fluid_inds = np.where(V > 1e-6)[0]
    full_cell_inds = np.where(V > 1-1e-6)[0]

    # Transform to -0.5 < x/h < 0.5
    wall_dist = data['wall_dist']
    x_rel = case.x_rel(wall_dist)
    u_ref = case.analytical_solution(x_rel)

    # Save error norms to CSV files
    for err_name, err_func in zip(['L1', 'L2', 'Linf'], [L1_err, L2_err, Linf_err]):
        err = err_func(u[full_cell_inds] / case.u_max, u_ref[full_cell_inds] / case.u_max)
        with open(f'./output/{boundary_treatment}/{err_name}_error.csv', 'w') as f:
            f.write(f'{err_name} velocity error\n')
            f.write(f'{err}\n')


    # Plot the velocity profile
    plot_inds = full_cell_inds if boundary_treatment == 'Bouzidi' else fluid_inds
    plt.plot(x_rel[plot_inds], u[plot_inds] / case.u_max, label=boundary_treatment, color=colors[j], marker=markers[j])
    plt.plot(-x_rel[plot_inds], u[plot_inds] / case.u_max, color=colors[j], marker=markers[j]) # Symmetric part

    if j == len(boundary_treatments) - 1:
        x_analytical = np.linspace(-0.5, 0.5, 100) # Fine grid for analytical solution
        plt.plot(x_analytical, case.analytical_solution(x_analytical) / case.u_max, label='Analytical', color='k', linestyle='--')

    
plt.xlabel(r'$x$ / $h$')
plt.ylabel(r'$u_y$ / $u_{y,\max}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./plots/velocity_profiles.png', dpi=300)
