"""
Run different boundary treatments for a quasi-1D channel flow driven
by a body force. The following case parameters should be defined in
the file ../caseDef.json:
- Nx : Number of fluid nodes in the x direction
- pfluid : Fluid fraction of the cut cells
- g : Body force in the y direction
- tau : Relaxation time
- Nt : Number of timesteps to run

The resulting velocity magnitude fields are saved in files U_<case>.npy
and the time convergence of the maximum velocity is plotted and saved
as ../timeConvergence/<case>_timeConvergence.pdf

Requires: numpy, matplotlib, json, sys
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pyvista as pv
import sys
import shapely
import glob
import runpy

sys.path.insert(0, '../../../../../../src/')
from LBM_Simulator import simulate

sys.path.insert(0, '../../../../../../../numbacff')
from CFF import CFF2D
from Bouzidi.Bouzidi import Bouzidi2D
from Chen.Chen import Chen2D


use_cuda = True

pfluid = runpy.run_path('../pfluid_definition.py')['pfluid']
Nx = runpy.run_path('./resolution_definition.py')['Nx']
case = runpy.run_path('../../case_definition.py')['Case'](pfluid=pfluid, Nx=Nx)

pfluid = case.pfluid
Nx = case.Nx
Nx_domain = case.Nx_domain
pad = case.pad
tau = case.tau
Nt = case.Nt
g = case.g
precision = case.precision

stencil = 'd2q9'

boundary_treatments = [
    'CFF', 
    'Bouzidi',
    'Chen'
]


# Define geometry. Note that 5 cells are added at each side in x to
# prevent taking cells on the other side of the domain as neighbors.
Ny = 1
walls = [np.array([[0, 0], [case.pad + 1 - pfluid, 0],
                   [case.pad + 1 - pfluid, Ny], [0, Ny], [0, 0]]),
         np.array([[Nx_domain - case.pad - 1 + pfluid, 0], [Nx_domain, 0],
                   [Nx_domain, Ny], [Nx_domain - case.pad - 1 + pfluid, Ny],
                   [Nx_domain - case.pad - 1 + pfluid, 0]])]
walls = [shapely.geometry.Polygon(wall) for wall in walls]


# Body force in lattice units
gVec = np.array([0, g])

boundary_treatment_def = {   
    'CFF' : lambda : CFF2D('D2Q9', (Nx_domain, Ny), walls,
                                     forceDens=gVec, concaveCorrectionMethod=None),
    'Bouzidi' : lambda : Bouzidi2D('D2Q9', (Nx_domain, Ny), walls,
                                     forceDens=gVec, concaveCorrectionMethod=None),
    'Chen' : lambda : Chen2D('D2Q9', (Nx_domain, Ny), walls,
                               forceDens=gVec, concaveCorrectionMethod=None)
}




for boundary_treatment in boundary_treatments:

    write_velocity = {
        'write_interval' : case.write_interval,
        'fields' : ['u'],
        'filename_template' : f'output/{boundary_treatment}' + '/velocity_{:07d}',
        'start_time' : Nt - case.write_interval * (Nt // case.write_interval) # Ensure that the last timestep is written
    }

    write_full_field = {
        'write_interval' : case.Nt,
        'fields' : ['rho', 'u', 'V', 'c'],
        'filename_template' : f'output/{boundary_treatment}' + '/full_field_{:07d}',
    }

    writers = [write_velocity, write_full_field]

    S = boundary_treatment_def[boundary_treatment]()

    CSM = S.CSM.copy()
    cell_volumes = S.getCellVolumes()[:, :, :, None, :]
    cell_centroids = S.getCentroidsGlob()[None, :, :, None, :]

    if not os.path.exists(f'output/{boundary_treatment}'):
        os.makedirs(f'output/{boundary_treatment}')
    if not os.path.exists(f'plots/{boundary_treatment}'):
        os.makedirs(f'plots/{boundary_treatment}')


    simulate(stencil=stencil, nx=Nx_domain, ny=1, nz=1, Nt=Nt, tau=tau, body_force=gVec, CSM=CSM, use_cuda=use_cuda,
             writers=writers, precision=precision, cell_volumes=cell_volumes, cell_centroids=cell_centroids)


    # Plot time convergence
    times, u_max = [], []
    for file in glob.glob(f'output/{boundary_treatment}/velocity_*.vti'):
        times.append(int(file.split('_')[-1].split('.')[0]))
        u_max.append(pv.read(file).cell_data['u'][pad:-pad, 1].max())

    sort_inds = np.argsort(times)
    plt.plot(np.asarray(times)[sort_inds], np.asarray(u_max)[sort_inds])
    plt.xlabel('Time step')
    plt.ylabel('Maximum velocity')
    plt.savefig(f'plots/{boundary_treatment}/timeConvergence.pdf')
    plt.clf()


    




