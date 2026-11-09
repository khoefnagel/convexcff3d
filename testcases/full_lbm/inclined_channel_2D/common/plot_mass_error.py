import glob
import runpy
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

plt.rcParams.update({'font.size' : 14})

colors = {8: 'r', 16: 'b', 24: 'g', 32: 'y', 48: 'c', 64: 'm'}
markers = {'CFF' : 's', 'Bouzidi' : 'o'}

legend_lines = []
legend_labels = []

for NFolder in sorted(glob.glob('./N*')):

    # theta_deg = runpy.run_path('./theta_definition.py')['theta_deg']
    h = runpy.run_path(f'{NFolder}/resolution_definition.py')['h']
    # case = runpy.run_path(f'../case_definition.py')['Case'](theta_deg=theta_deg, h=h)

    delta = 1/h #Cell spacing for channel height=1

    legend_lines.append(plt.Line2D([0], [0], color=colors[h], label=f'h={h}'))
    legend_labels.append(f'h={h}')

    for boundary_treatment_folder in glob.glob(f'{NFolder}/output/*/'):
        boundary_treatment = boundary_treatment_folder.split('/')[-2]

        if boundary_treatment not in markers.keys():
            continue

        mass_though_time = []
        times = []
        for file in glob.glob(f'{boundary_treatment_folder}/velocity_*.vti'):
            time = float(file.split('/')[-1].split('_')[-1].split('.vti')[0])
            data = pv.read(file)
            rho = data.cell_data['rho']
            V = data.cell_data['V']
            if boundary_treatment == 'Bouzidi': # Bouzidi doesn't use cut cells
                mass = np.sum(rho[np.where(V > 1 - 1e-6)])
            else:
                mass = np.sum(rho*V)
            mass_though_time.append(mass)
            times.append(time)

        sort_inds = np.argsort(times)
        mass_though_time = np.array(mass_though_time)[sort_inds]
        
        percentage_error = (mass_though_time - mass_though_time[0]) / mass_though_time[0] * 100

        plt.plot(np.linspace(0, 1, len(np.abs(percentage_error)))[1:], np.abs(percentage_error)[1:], label=f'{boundary_treatment}, h={h}', 
                 c=colors[h], marker=markers[boundary_treatment], markevery=5)

legend_lines.append(plt.Line2D([0], [0], color='k', label='CFF', marker=markers['CFF']))
legend_labels.append('CFF')
legend_lines.append(plt.Line2D([0], [0], color='k', label='Bouzidi', marker=markers['Bouzidi']))
legend_labels.append('Bouzidi')

plt.yscale('log')
plt.xlabel(r'$t/N_t$ [-]')
plt.ylabel(r'$|\Delta M/M_0|$ [%]')
plt.grid()
plt.legend(handles=legend_lines, labels=legend_labels, loc='best', ncol=3)
plt.tight_layout()
plt.savefig('mass_error.pdf')
plt.show()
        