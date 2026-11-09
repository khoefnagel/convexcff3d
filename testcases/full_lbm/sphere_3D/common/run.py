import numpy as np
import matplotlib.pyplot as plt
import trimesh
import time
import os
import runpy


import jax
jax.config.update("jax_platforms", "cpu")


import sys
sys.path.append('../../../../../../../numbacff/')
from CFF import CFF2D, CFF3D
from Bouzidi.Bouzidi import Bouzidi3D
from Chen.Chen import Chen3D


sys.path.append('../../../../../../src/')
from LBM_Simulator import simulate

cells_per_radius = runpy.run_path('./resolution_definition.py')['cells_per_radius']
Re = runpy.run_path('../Reynolds_definition.py')['Re']
case = runpy.run_path('../../case_definition.py')['Case'](cells_per_radius=cells_per_radius, Re=Re)

use_cuda = True

D = case.D
Nx = case.Nx
Ny = case.Ny
Nz = case.Nz
t_flow_through = case.t_flow_through
Nt = case.Nt
ux_inf = case.ux_inf
tau = case.tau
sphere_center = case.x_c
precision = case.precision

stencil = 'd3q19'


# Create sphere in trimesh
sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.5*D)
# Translate the sphere to the center
sphere_center = np.array([0.2*Nx, 0.5*Ny, 0.5*Nz])
sphere.apply_translation(sphere_center)

if Re == 100:
    boundary_treatments = [
        'CFF', 
        'Bouzidi',
        'Chen'
    ]
else:
    boundary_treatments = ['CFF']


boundary_treatment_def = {
    'CFF' : lambda : CFF3D('D3Q19', (Nx, Ny, Nz), [sphere], concaveCorrectionMethod=None),
    'Bouzidi' : lambda : Bouzidi3D('D3Q19', (Nx, Ny, Nz), [sphere], concaveCorrectionMethod=None),
    'Chen' : lambda : Chen3D('D3Q19', (Nx, Ny, Nz), [sphere], concaveCorrectionMethod=None),
}


for boundary_treatment in boundary_treatments:
    S = boundary_treatment_def[boundary_treatment]()
    
    CSM = S.CSM.copy()
    Fi2Ftot = S.Fi2Ftot.copy()

    # Surface forces are not implemented for Bouzidi, so we skip this part for that case
    if not boundary_treatment == 'Bouzidi':
        Fi2fsurfInput = S.Fi2fsurfInput.copy()
        vertices = S.inputFacetsRaw.reshape(-1, 3).copy()
        num_triangles = S.inputFacetsRaw.shape[0]
        cells_meta = np.full((num_triangles, 1), 3, dtype=np.int32)
        connectivity = np.arange(num_triangles * 3, dtype=np.int32).reshape(num_triangles, 3)
        faces = (np.hstack((cells_meta, connectivity)).ravel()).copy()
    else:
        Fi2fsurfInput = None
        vertices = None
        faces = None

    # Delete the CFF object to free memory
    del S
    import gc
    gc.collect()

    if not os.path.exists(f'output/{boundary_treatment}'):
        os.makedirs(f'output/{boundary_treatment}')
    if not os.path.exists(f'plots/{boundary_treatment}'):
        os.makedirs(f'plots/{boundary_treatment}')

    write_z_slice = {
        'slices_xyz' : (slice(None), slice(None), int(Nz//2)), 
        'write_interval' : case.z_slice_write_interval,
        'fields' : ['rho', 'u'],
        'filename_template' : f'output/{boundary_treatment}' + '/slice_z_{:05d}'
    }

    write_center_profile = {
        'slices_xyz' : (slice(None), int(Ny//2), int(Nz//2)),
        'write_interval' : case.center_profile_write_interval,
        'fields' : ['rho', 'u'],
        'filename_template' : f'output/{boundary_treatment}' + '/center_profile_{:05d}'
    }

    write_Ftot = {
        'write_interval' : case.Ftot_write_interval,
        'filename' : f'output/{boundary_treatment}/Ftot',
        'fields' : ['Ftot'],
        'type' : 'total_force'
    }

    # Surface forces are not implemented for Bouzidi, so we skip this part for that case
    if not boundary_treatment == 'Bouzidi':
        write_fsurf = {
            'filename_template' : f'output/{boundary_treatment}' + '/fsurf_{:05d}',
            'write_interval' : case.fsurf_write_interval,
            'fields' : ['fsurfInput'],
            'type' : 'surface_force',
            'vertices' : vertices,
            'faces' : faces
        }
        writers = [write_z_slice, write_center_profile, write_Ftot, write_fsurf]

    else:
        writers = [write_z_slice, write_center_profile, write_Ftot]



    rho0 = 1.0
    u0 = np.zeros((1, Nx, Ny, Nz, 3))
    u0[..., 0] += ux_inf

    inflow = {'type' : 'equilibrium', 'rho' : 1.0, 'u' : (ux_inf, 0.0, 0.0)}
    outflow = {'type' : 'zero_gradient'}
    domain_boundaries = {
        (1, 0, 0) : inflow,
        (-1, 0, 0): outflow
    }

    t1 = time.time()
    lbm = simulate(nx=Nx, ny=Ny, nz=Nz, stencil=stencil, tau=tau, Nt=Nt, rho0=rho0, u0=u0, CSM=CSM, Fi2Ftot=Fi2Ftot, 
                    domain_boundaries=domain_boundaries, writers=writers, Fi2fsurfInput=Fi2fsurfInput, faces=faces, vertices=vertices,
                precision=precision, 
                )
    t2 = time.time()
    print(f"Time taken for {boundary_treatment}: {t2-t1:.2f} seconds")

    t, C_D, _, _ = np.genfromtxt(f'output/{boundary_treatment}/Ftot.csv', delimiter=',', skip_header=1).T

    plt.plot(t, C_D, zorder=20)
    plt.xlabel('Time step')
    plt.ylabel('Drag Coefficient')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./plots/{boundary_treatment}/C_D_timeConvergence.pdf')


