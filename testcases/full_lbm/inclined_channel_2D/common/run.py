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

h = runpy.run_path('./resolution_definition.py')['h']
theta_deg = runpy.run_path('../theta_definition.py')['theta_deg']
case = runpy.run_path('../../case_definition.py')['Case'](h=h, theta_deg=theta_deg)

use_cuda = True


height = case.height
width = case.width
theta_deg_actual = case.theta_deg_actual
h = case.h
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


# Define geometry
theta = theta_deg_actual*np.pi/180

b = h//2 + 0.1 # Vertical solid portion on bottom left
a = height - b - h/np.cos(theta) # Vertical solid portion on top left

# Locations where the walls intersect the domain boundaries
x1, x2, x3, x4 = 0, a/np.tan(theta), width-b/np.tan(theta), width
y1, y2, y3, y4 = 0, b, height-a, height

# Define the three polygons that make up the geometry
lowerTriangle = np.array([[x3, y1], [x4, y1], [x4, y2], [x3, y1]])
center = np.array([[x1, y1], [x2, y1], [x4, y3], [x4, y4],
                   [x3, y4], [x1, y2], [x1, y1]],)
upperTriangle = np.array([[x1, y3], [x2, y4], [x1, y4], [x1, y3]])
geom = [shapely.Polygon(lowerTriangle), shapely.Polygon(center), shapely.Polygon(upperTriangle)]

# Body force in lattice units
gVec = np.array([-np.cos(theta)*g, -np.sin(theta)*g])

boundary_treatment_def = {   
    'CFF' : lambda : CFF2D('D2Q9', (width, height), geom,
                                     forceDens=gVec, concaveCorrectionMethod=None),
    'Bouzidi' : lambda : Bouzidi2D('D2Q9', (width, height), geom,
                                     forceDens=gVec, concaveCorrectionMethod=None),
    'Chen' : lambda : Chen2D('D2Q9', (width, height), geom,
                               forceDens=gVec, concaveCorrectionMethod=None)
}




for boundary_treatment in boundary_treatments:

    write_velocity = {
        'write_interval' : case.write_interval,
        'fields' : ['rho', 'u', 'V'],
        'filename_template' : f'output/{boundary_treatment}' + '/velocity_{:05d}',
        'start_time' : Nt - case.write_interval * (Nt // case.write_interval) # Ensure that the last timestep is written
    }

    write_full_field = {
        'write_interval' : case.Nt,
        'fields' : ['rho', 'u', 'V', 'c', 'wall_dist'],
        'filename_template' : f'output/{boundary_treatment}' + '/full_field_{:05d}',
    }

    writers = [write_velocity, write_full_field]

    S = boundary_treatment_def[boundary_treatment]()

    CSM = S.CSM.copy()
    cell_volumes = S.getCellVolumes()[:, :, :, None, :]
    cell_centroids = S.getCentroidsGlob()[None, :, :, None, :]
    wall_dists = S.getWallDists()[:, :, :, None, :]

    if not os.path.exists(f'output/{boundary_treatment}'):
        os.makedirs(f'output/{boundary_treatment}')
    if not os.path.exists(f'plots/{boundary_treatment}'):
        os.makedirs(f'plots/{boundary_treatment}')


    simulate(stencil=stencil, nx=width, ny=height, nz=1, Nt=Nt, tau=tau, body_force=gVec, CSM=CSM, use_cuda=use_cuda,
             writers=writers, precision=case.precision, cell_volumes=cell_volumes, cell_centroids=cell_centroids, wall_dists=wall_dists)


    # Plot time convergence
    times, u_max = [], []
    for file in glob.glob(f'output/{boundary_treatment}/velocity_*.vti'):
        times.append(int(file.split('_')[-1].split('.')[0]))
        data = pv.read(file)
        u, V = data.cell_data['u'], data.cell_data['V']
        fluid_inds = np.where(V > 0)[0]
        u_max.append(np.linalg.norm(u[fluid_inds], axis=1).max())

    sort_inds = np.argsort(times)
    plt.plot(np.asarray(times)[sort_inds], np.asarray(u_max)[sort_inds])
    plt.xlabel('Time step')
    plt.ylabel('Maximum velocity')
    plt.savefig(f'plots/{boundary_treatment}/timeConvergence.pdf')
    plt.clf()


    




