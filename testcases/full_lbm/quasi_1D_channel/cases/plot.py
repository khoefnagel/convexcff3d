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



plt.rcParams.update({'font.size' : 14})

# Cases to plot, with plotting details
plotCases = [
    #[Case, pfluid, color, marker, label, zorder]
    ['CFF', '1/9', 'r', '+', r'CFF, $p_{fluid}=$1/9', 100],
    ['CFF', '1/3', 'g', '1', r'CFF, $p_{fluid}=$1/3', 200],
    ['CFF', '1/2', 'm', 'd', r'CFF, $p_{fluid}=$1/2', 50],
    ['CFF', '2/3', 'deeppink', 'D', r'CFF, $p_{fluid}=$2/3', 30],
    ['CFF', '8/9', 'c', '^', r'CFF, $p_{fluid}=$8/9', 60],
    ['CFF', '1/1', 'k', 'o', r'CFF, $p_{fluid}=$1/1', 40],
    ['Chen', '1/3', 'b', 's', r'Chen, $p_{fluid}=$1/3', 20],
    ['Bouzidi', '1/3', 'orange', 'v', r'Bouzidi, $p_{fluid}=$1/3', 10],
    # ['Bouzidi', '2/3', 'purple', 'D', r'Bouzidi, $p_{fluid}=$2/3', 15],
    # ['Bouzidi', '1/9', 'purple', '+', r'Bouzidi, $p_{fluid}=$1/9', 15],
    ['Bouzidi', '8/9', 'purple', 'p', r'Bouzidi, $p_{fluid}=$8/9', 15],
    # ['Bouzidi', '1/1', 'purple', 'o', r'Bouzidi, $p_{fluid}=$1/1', 15],
]

# Cases to compute order of convergence for
OOCCases = [
    ['Chen', '1/9'],
    ['Chen', '1/3'],
    ['Chen', '1/2'],
    ['Chen', '2/3'],
    ['Chen', '8/9'],
    ['Chen', '1/1'],
    ['CFF', '1/9'],
    ['CFF', '1/3'],
    ['CFF', '1/2'],
    ['CFF', '2/3'],
    ['CFF', '8/9'],
    ['CFF', '1/1'],
    ['Bouzidi', '1/9'],
    ['Bouzidi', '1/3'],
    ['Bouzidi', '1/2'],
    ['Bouzidi', '2/3'],
    ['Bouzidi', '8/9'],
    ['Bouzidi', '1/1']
]  



# Read in all errors into allErrDict. Structure is:
# all_err_dict = {'err_type' : {'pfluid' : {'boundary_treatment' : [[delta], [error]]}}}
err_types = ['L1', 'L2', 'Linf']
all_err_dict = {err: {} for err in err_types}
for pfluid_folder in glob.glob('./pfluid*'):

    pfluid_str = pfluid_folder[-3:].replace('-', '/')
    for err_type in err_types:
        all_err_dict[err_type][pfluid_str] = {}

    pfluid = runpy.run_path(f'{pfluid_folder}/pfluid_definition.py')['pfluid']

    for N_folder in glob.glob(f'{pfluid_folder}/N*'):

        Nx = runpy.run_path(f'{N_folder}/resolution_definition.py')['Nx']
        for boundary_treament_folder in glob.glob(f'{N_folder}/output/*/'):
            boundary_treatment = boundary_treament_folder.split('/')[-2]
            for err_type in err_types:
                if not os.path.exists(f'{boundary_treament_folder}/{err_type}_error.csv'):
                    continue

                if boundary_treatment not in all_err_dict[err_type][pfluid_str]:
                    all_err_dict[err_type][pfluid_str][boundary_treatment] = [[], []]

                case = runpy.run_path('./case_definition.py')['Case'](Nx=Nx, pfluid=pfluid)
                delta = 1 / case.h

                err = np.loadtxt(f'{boundary_treament_folder}/{err_type}_error.csv', delimiter=',', skiprows=1)

                all_err_dict[err_type][pfluid_str][boundary_treatment][0].append(delta)
                all_err_dict[err_type][pfluid_str][boundary_treatment][1].append(err)


# Plot all cases
errLabels = {'L1' : r'$L_1$', 'L2' : r'$L_2$', 'Linf' : r'$L_\infty$'}
for ei, err_type in enumerate(all_err_dict.keys()):
    plt.figure(num=ei+1)
    plt.clf()
    for j, (case, pfluid_str, c, marker, label, zorder) in enumerate(plotCases):

        Delta, E = all_err_dict[err_type][pfluid_str][case]
        sortInds = np.argsort(Delta)[::-1]
        plt.loglog(np.array(Delta)[sortInds], np.array(E)[sortInds], label=label,
                   marker=marker, c=c, zorder=zorder)

    plt.xlabel(r'Cell size: $\Delta$')
    plt.ylabel(errLabels[err_type] + ' error norm of y-velocity')
    plt.legend(ncol=2, loc=(0.02,0.95), fancybox=True, shadow=True)
    plt.grid(which='both')
    plt.xlim([0.01, 0.2])
    plt.ylim([0.0001, 0.4])
    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
    plt.gca().tick_params(axis='x', which='minor', labelbottom=False)
    plt.yticks([0.0001, 0.001, 0.01, 0.1], ['0.0001', '0.001', '0.01', '0.1'])
    plt.tight_layout()
    plt.savefig(f'./Figures/allCasesGravityChannel{err_type}.pdf')


# Compute and print OOC for all cases in OOCCases
with open('./orderOfConvergence.txt', 'w') as f:
    # Pipe printed output to file
    sys.stdout = f
    for errType in all_err_dict.keys():
        print(f'Observed Order of Convergence of the {errType}-norm for all cases:')
        for case, pfluidStr in OOCCases:
            delta = all_err_dict[errType][pfluidStr][case][0]
            L1Err = all_err_dict[errType][pfluidStr][case][1]
            OOC, _ = np.polyfit(np.log(delta), np.log(L1Err), 1)
            print(f'{case}, pfluid {pfluidStr}: {OOC:.02f}')
        print()
sys.stdout = sys.__stdout__
print('Orders of convergence written to orderOfConvergence.txt')

