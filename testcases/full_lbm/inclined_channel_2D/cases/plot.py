"""
Script to plot the error norms for all cases in the quasi-1D channel testcase.
Also computes and prints the observed order of convergence (OOC) for all cases.
The figures are saved in the Figures/ folder.

The orders of convergence are written to orderOfConvergence.txt

Requires matplotlib, numpy, glob, json, os, sys
"""

import matplotlib.pyplot as plt
import glob
import sys
import numpy as np
import os
import runpy
from importlib.util import spec_from_file_location, module_from_spec



plt.rcParams.update({'font.size' : 14})

# Cases to plot, with plotting details
plotCases = [
    #[Case, theta, color, marker, label, zorder]
    ['CFF', '10', 'r', '+', r'CFF, $theta=10^\circ$', 100],
    ['CFF', '20', 'b', '1', r'CFF, $theta=20^\circ$', 100],
    ['CFF', '30', 'm', 'd', r'CFF, $theta=30^\circ$', 50],
    ['CFF', '40', 'deeppink', 'D', r'CFF, $theta=40^\circ$', 20],
    ['CFF', '45', 'c', '^', r'CFF, $theta=45^\circ$', 30],
    #['Chen', '10', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    ['Chen', '20', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    #['Chen', '30', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    #['Chen', '40', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    ['Bouzidi', '10', 'g', 'o', r'Bouzidi, $theta=10^\circ$', 10],
    ['Bouzidi', '20', 'g', '+', r'Bouzidi, $theta=20^\circ$', 10],
    ['Bouzidi', '30', 'g', 'd', r'Bouzidi, $theta=30^\circ$', 10],
    ['Bouzidi', '40', 'g', 'D', r'Bouzidi, $theta=40^\circ$', 10],
    ['Bouzidi', '45', 'g', '^', r'Bouzidi, $theta=45^\circ$', 10]
]

# Cases to compute order of convergence for
OOCCases = [
    ['Chen', '10'],
    ['Chen', '20'],
    ['Chen', '30'],
    ['Chen', '40'],
    ['Chen', '45'],
    ['CFF', '10'],
    ['CFF', '20'],
    ['CFF', '30'],
    ['CFF', '40'],
    ['CFF', '45'],
    ['Bouzidi', '10'],
    ['Bouzidi', '20'],
    ['Bouzidi', '30'],
    ['Bouzidi', '40'],
    ['Bouzidi', '45']
]  



# Read in all errors into allErrDict. Structure is:
# all_err_dict = {'err_type' : {'theta' : {'boundary_treatment' : [[delta], [error]]}}}
err_types = ['L1', 'L2', 'Linf']
all_err_dict = {err: {} for err in err_types}
for theta_folder in glob.glob('./theta*'):

    theta_str = theta_folder.split('theta')[-1]
    for err_type in err_types:
        all_err_dict[err_type][theta_str] = {}

    theta_deg = runpy.run_path(f'{theta_folder}/theta_definition.py')['theta_deg']

    for N_folder in glob.glob(f'{theta_folder}/N*'):

        h = runpy.run_path(f'{N_folder}/resolution_definition.py')['h']
        for boundary_treament_folder in glob.glob(f'{N_folder}/output/*/'):
            boundary_treatment = boundary_treament_folder.split('/')[-2]
            for err_type in err_types:
                if not os.path.exists(f'{boundary_treament_folder}/{err_type}_error.csv'):
                    continue

                if boundary_treatment not in all_err_dict[err_type][theta_str]:
                    all_err_dict[err_type][theta_str][boundary_treatment] = [[], []]

                case = runpy.run_path('./case_definition.py')['Case'](h=h, theta_deg=theta_deg)
                delta = 1 / case.h

                err = np.loadtxt(f'{boundary_treament_folder}/{err_type}_error.csv', delimiter=',', skiprows=1)

                all_err_dict[err_type][theta_str][boundary_treatment][0].append(delta)
                all_err_dict[err_type][theta_str][boundary_treatment][1].append(err)


# Plot all cases
errLabels = {'L1' : r'$L_1$', 'L2' : r'$L_2$', 'Linf' : r'$L_\infty$'}
for ei, errType in enumerate(all_err_dict.keys()):
    plt.figure(num=ei+1)
    for j, (case, pfluidStr, c, marker, label, zorder) in enumerate(plotCases):
        Delta, E = all_err_dict[errType][pfluidStr][case]
        sortInds = np.argsort(Delta)[::-1]
        plt.loglog(np.array(Delta)[sortInds], np.array(E)[sortInds], label=label,
                   marker=marker, c=c, zorder=zorder)

    plt.xlabel(r'Cell size: $\Delta$')
    plt.ylabel(errLabels[errType] + ' error norm of velocity magnitude')
    plt.legend(ncol=2, loc=(0.,0.95), fancybox=True, shadow=True)
    plt.grid(which='both')
    plt.xlim([0.01, 0.2])
    plt.ylim([0.0002, 0.7])
    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
    plt.gca().tick_params(axis='x', which='minor', labelbottom=False)
    plt.yticks([0.001, 0.01, 0.1], ['0.001', '0.01', '0.1'])
    plt.tight_layout()
    plt.savefig(f'./Figures/allCasesInclinedChannel{errType}.pdf')
plt.show()


# Compute and print OOC for all cases in OOCCases
with open('./orderOfConvergence.txt', 'w') as f:
    # Pipe printed output to file
    sys.stdout = f
    for errType in all_err_dict.keys():
        print(f'Observed Order of Convergence of the {errType}-norm for all cases:')
        for case, theta_str in OOCCases:
            delta = all_err_dict[errType][theta_str][case][0]
            L1Err = all_err_dict[errType][theta_str][case][1]
            OOC, _ = np.polyfit(np.log(delta), np.log(L1Err), 1)
            print(f'{case}, theta {theta_str} deg: {OOC:.02f}')
        print()
sys.stdout = sys.__stdout__
print('Orders of convergence written to orderOfConvergence.txt')

