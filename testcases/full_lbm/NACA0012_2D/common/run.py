"""
Run flow around a NACA0012 airfoil at a specified angle of attack and resolution using the LBM_Simulator.

Expects the file 'resolution_definition.py' to be present in the same directory, which should define the variable 'cells_per_chord'.
Expects the file 'alpha_definition.py' to be present in the parent directory, which should define the variable 'alpha_deg'.
Expects the file 'case_definition.py' to be present in the grandparent directory.
"""

import numpy as np
import os
import shapely
import sys
import matplotlib.pyplot as plt
import runpy
import pyvista as pv

sys.path.append('../../../../../../src/')
from CFF import CFF2D
from Bouzidi.Bouzidi import Bouzidi2D
from Chen.Chen import Chen2D

from LBM_Simulator import simulate

cells_per_chord = runpy.run_path('./resolution_definition.py')['cells_per_chord']
alpha_deg = runpy.run_path('../alpha_definition.py')['alpha_deg']
case = runpy.run_path('../../case_definition.py')['Case'](cells_per_chord=cells_per_chord, alpha_deg=alpha_deg)

#================================================================================
# Definitions

use_cuda = True

Nx, Ny = case.Nx, case.Ny
t_flow_through = case.t_flow_through
Nt = case.Nt
tau = case.tau
alpha_deg = case.alpha_deg
chord = case.chord
x_c = case.x_c
ux_inf = case.ux_inf
precision = case.precision

stencil = 'd2q9'

# Initial conditions
rho0 = 1.0
u0 = (0.0, 0.0)


boundary_treatments = [
    'CFF', 
    'Bouzidi',
    'Chen'
]

boundary_treatment_def = {
    'CFF' : lambda : CFF2D('D2Q9', (Nx, Ny), objects, concaveCorrectionMethod=None),
    'Bouzidi' : lambda : Bouzidi2D('D2Q9', (Nx, Ny), objects, concaveCorrectionMethod=None),
    'Chen' : lambda : Chen2D('D2Q9', (Nx, Ny), objects, concaveCorrectionMethod=None, projection='PC', deg=0),
}

#================================================================================
# Reading and processing airfoil coordinates

# Airfoil coordinate file
airfoilFile = './NACA0012.dat'

# Read in and process airfoil coords to form closed shape
# MODIFY IF NECESSARY!!
airfoilRaw = np.genfromtxt(airfoilFile, skip_header=1)
airfoilCoords = np.vstack((airfoilRaw, airfoilRaw[0]))

# Scale, translate, rotate airfoil coords
alpha = np.deg2rad(alpha_deg)
rotationMatrix = np.array([[np.cos(alpha), np.sin(alpha)],
                           [-np.sin(alpha), np.cos(alpha)]])
airfoilCoordsScaled = (airfoilCoords + np.array([-0.25, 0]))*chord
airfoilCoordsRotated = np.einsum('ik,jk->ij', airfoilCoordsScaled, rotationMatrix)
airfoilCoordsTrans = airfoilCoordsRotated + x_c

# List of solid objects in the flow (in this case only the airfoil)    
objects = [shapely.Polygon(airfoilCoordsTrans)]

# Compute normal and tangent vectors of the airfoil panels
dxdyAirfoil = np.diff(airfoilCoordsTrans, axis=0)
lAirfoil = np.linalg.norm(dxdyAirfoil, axis=1, keepdims=True)
tangentsAirfoil = dxdyAirfoil/lAirfoil
normalsAirfoil = dxdyAirfoil[:,::-1]*np.array([-1, 1])/lAirfoil

# Local (untransformed coordinates)
airfoilCoordsLocal = (airfoilCoords[:-1] + airfoilCoords[1:])/2


#================================================================================
# Domain boundary conditions

u = lambda x, y, z, t : (ux_inf * t / Nt * 10, 0.0) if t < Nt / 10 else (ux_inf, 0.0)
# equilibrium = {'type' : 'equilibrium', 'rho' : 1.0, 'u' : u}
velocity = {'type' : 'velocity', 'u' : u}
zero_gradient = {'type' : 'zero_gradient'}
# pressure = {'type' : 'density', 'rho' : 1.0}
symmetry = {'type' : 'symmetry'}
domain_boundaries = {
    (0, 1) : symmetry,
    (0, -1) : symmetry,
    (1, 0) : velocity,
    (-1, 0): zero_gradient,
}


#================================================================================
# Run simulation for each boundary treatment

for boundary_treatment in boundary_treatments:
    S = boundary_treatment_def[boundary_treatment]()
    
    CSM = S.CSM.copy()
    Fi2Ftot = S.Fi2Ftot.copy()
    if not boundary_treatment == 'Bouzidi':
        Fi2fsurfInput = S.Fi2fsurfInput.copy()      

    # Clean up memory
    del S
    import gc
    gc.collect()


    # Create output directories if they don't exist    
    if not os.path.exists(f'output/{boundary_treatment}'):
        os.makedirs(f'output/{boundary_treatment}')
    if not os.path.exists(f'plots/{boundary_treatment}'):
        os.makedirs(f'plots/{boundary_treatment}')


    # Define output writers
    write_full_field = {
        'write_interval' : case.field_write_interval,
        'fields' : ['rho', 'u'],
        'filename_template' : f'output/{boundary_treatment}' + '/full_field_{:05d}',
    }
    write_Ftot = {
        'write_interval' : case.Ftot_write_interval,
        'filename' : f'output/{boundary_treatment}/Ftot',
        'fields' : ['Ftot'],
        'type' : 'total_force',
    }
    writers = [write_full_field, write_Ftot]


    lbm = simulate(nx=Nx, ny=Ny, stencil=stencil, tau=tau, Nt=Nt, rho0=rho0, u0=u0, CSM=CSM, Fi2Ftot=Fi2Ftot,
             domain_boundaries=domain_boundaries, writers=writers, precision=precision, use_cuda=use_cuda)

    t, D, L = np.genfromtxt(f'output/{boundary_treatment}/Ftot.csv', delimiter=',', skip_header=1).T

    # Compute the average density in the inflow at the final time step to use in the lift and drag coefficient calculations
    # This is because rho is not specified at the boundaries, so it will deviate slightly from 1.
    field = pv.read(f'output/{boundary_treatment}/full_field_{Nt:05d}.vti')
    rho_inf = np.average(field['rho'].reshape((Ny, Nx))[:,0])

    # Compute, plot, and save lift and drag coefficients
    C_l = L/(0.5*ux_inf**2*chord*rho_inf)
    C_d = D/(0.5*ux_inf**2*chord*rho_inf)


    plt.figure()
    plt.plot(t, C_l)
    plt.xlabel('Time step')
    plt.ylabel('Lift coefficient')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'plots/{boundary_treatment}/C_l_time_convergence.pdf')

    plt.figure()
    plt.plot(t, C_d)
    plt.xlabel('Time step')
    plt.ylabel('Drag coefficient')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'plots/{boundary_treatment}/C_d_time_convergence.pdf')

    # Save the final lift and drag coefficients to a .npy file
    np.save(f'output/{boundary_treatment}/C_l_C_d', (C_l[-1], C_d[-1]))


    # Compute and plot pressure and skin friction coefficients over the surface
    if not boundary_treatment == 'Bouzidi':
        lbm.collide_and_periodic_stream()
        fsurf = Fi2fsurfInput @ lbm.F.copy_to_host().ravel() if use_cuda else Fi2fsurfInput @ lbm.F.ravel()
        p = np.sum(fsurf.reshape(-1, 2)*normalsAirfoil, axis=1)
        Cp = (p - 1/3)/(1/2*ux_inf**2) #Pressure coefficient
        tau_w = np.sum(fsurf.reshape(-1, 2)*tangentsAirfoil, axis=1)
        Cf = tau_w/(1/2*ux_inf**2) #Skin friction coefficient

        plt.figure()
        plt.plot(airfoilCoordsLocal[:,0], Cp)
        plt.xlabel('X-coordinate')
        plt.ylabel('Pressure coefficient')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'plots/{boundary_treatment}/Cp_distribution.pdf')

        plt.figure()
        plt.plot(airfoilCoordsLocal[:,0], Cf)
        plt.xlabel('X-coordinate')
        plt.ylabel('Skin friction coefficient')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'plots/{boundary_treatment}/Cf_distribution.pdf')
