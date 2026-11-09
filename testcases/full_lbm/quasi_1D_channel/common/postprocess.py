import numpy as np
import runpy
import pyvista as pv
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

pfluid = runpy.run_path('../pfluid_definition.py')['pfluid']
Nx = runpy.run_path('./resolution_definition.py')['Nx']
case = runpy.run_path('../../case_definition.py')['Case'](pfluid=pfluid, Nx=Nx)
Nt = case.Nt


boundary_treatments = [
    'CFF', 
    'Chen',
    'Bouzidi', 
]


# Error norms
L1_err = lambda u, u_ref: np.sum(np.abs(u - u_ref)) / len(u_ref)
L2_err = lambda u, u_ref: np.sqrt(np.sum((u - u_ref)**2) / len(u_ref))
Linf_err = lambda u, u_ref: np.max(np.abs(u - u_ref))

colors = ['r', 'b', 'g']
markers = ['s', 'o', '^']
for j, boundary_treatment in enumerate(boundary_treatments):
    data = pv.read(f'./output/{boundary_treatment}/full_field_{Nt:07d}.vti')

    x_glob = data['c'][:, 0]
    u = data['u'][:, 1]
    V = data['V']

    fluid_inds = np.where(V > 1e-6)[0]
    full_cell_inds = np.where(V > 1-1e-6)[0]

    # Transform to -0.5 < x/h < 0.5
    x_rel = case.x_rel(x_glob)
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

    if j == 0:
        x_analytical = np.linspace(-0.5, 0.5, 100) # Fine grid for analytical solution
        plt.plot(x_analytical, case.analytical_solution(x_analytical) / case.u_max, label='Analytical', color='k', linestyle='--', zorder=100)

    
plt.xlabel(r'$x$ / $h$')
plt.ylabel(r'$u_y$ / $u_{y,\max}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./plots/velocity_profiles.pdf')