import numpy as np
import os
import shapely
import sys
import matplotlib.pyplot as plt
import runpy
import pyvista as pv

sys.path.append('../../../../../../../numbacff/')
from CFF import CFF2D
from Bouzidi.Bouzidi import Bouzidi2D
from Chen.Chen import Chen2D

sys.path.append('../../../../../../src/')
from LBM_Simulator import simulate, LBM_Simulator

cells_per_chord = runpy.run_path('./resolution_definition.py')['cells_per_chord']
alpha_deg = runpy.run_path('../alpha_definition.py')['alpha_deg']
case = runpy.run_path('../../case_definition.py')['Case'](cells_per_chord=cells_per_chord, alpha_deg=alpha_deg)

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

u = lambda x, y, z, t : (ux_inf * t / Nt * 10, 0.0) if t < Nt / 10 else (ux_inf, 0.0)
inflow = {'type' : 'equilibrium', 'rho' : 1.0, 'u' : u}
velocity = {'type' : 'velocity', 'u' : (ux_inf, 0.0)}
outflow = {'type' : 'zero_gradient'}
pressure = {'type' : 'density', 'rho' : 1.0}
symmetry = {'type' : 'symmetry'}
domain_boundaries = {
    # (0, 1) : symmetry,
    # (0, -1) : symmetry,
    # (1, 0) : inflow,
    # (-1, 0): outflow,
}





boundary_treatments = [
    'CFF', 
    # 'Bouzidi',
    # 'Chen'
]

boundary_treatment_def = {
    'CFF' : lambda : CFF2D('D2Q9', (Nx, Ny), objects, concaveCorrectionMethod=None),
    'Bouzidi' : lambda : Bouzidi2D('D2Q9', (Nx, Ny), objects, concaveCorrectionMethod=None),
    'Chen' : lambda : Chen2D('D2Q9', (Nx, Ny), objects, concaveCorrectionMethod=None, projection='PC', deg=0),
}

for boundary_treatment in boundary_treatments:
    S = boundary_treatment_def[boundary_treatment]()
    
    CSM = S.CSM.copy()
    Fi2Ftot = S.Fi2Ftot.copy()

    sys.path.append('/home/kaj/Documents/03_Projects/01_LBM_related/convexcff2d/src/')
    from ConvexCFF2D import ConvexCFF2D
    S2 = ConvexCFF2D('D2Q9', (Ny, Nx), [airfoilCoordsTrans], 1, weighing='volume')
    # Populations at the inflow: equilibrium condition
    fIn = S2.fEq(1, np.array([ux_inf, 0])) #Inflow f

    # # Initialize based on velocity and density
    # rho0 = np.ones((1, *S.N, 1))
    # U0 = np.ones((1, *S.N, 1))*np.array([ux_inf, 0])
    # S.initialize(rho0, U0)    

    # Initialize based on velocity and density
    rho0 = np.ones((1, *S2.N, 1))
    U0 = np.ones((1, *S2.N, 1))*np.array([ux_inf, 0])
    S2.initialize(rho0, U0)    


    # for it in range(Nt): 

    #     S2.collide(tau)
    #     S.collide(tau)

    #     F2 = S2.Fi2Ftot @ S2.F.flatten()
    #     temp = S.F.copy()
    #     S.streamPeriodic()
    #     F = S.Fi2Ftot @ S.F.flatten()
    #     S.F = temp
    #     print(F - F2)


    #     S2.F = (S2.SM @ S2.F.flatten()).reshape((S2.Q, *S2.N, 1))
    #     S.stream()          

    #     FF = np.transpose(S2.F, (0, 2, 1, 3))
    #     VV = np.transpose(S2.cellVolumes, (0, 2, 1, 3))
    #     diff = np.abs(FF - S.F)
    #     diff[:, (VV < 1e-10)[0,:,:,:]] = 0.0
    #     print(diff.max())



        # #------------------------------------------------------------------------
        # #Boundary conditions

        # F_pre = S.F.copy()
        

        # #Inlet: specified pressure and velocity
        # S.F[[1,5,8],:,0,None,:] = fIn[[1,5,8]]

        # #Outlet: zero gradient
        # S.F[[3,6,7],:,-1,None,:] = S.F[[3,6,7],:,-2,None,:]



    if not os.path.exists(f'output/{boundary_treatment}'):
        os.makedirs(f'output/{boundary_treatment}')
    if not os.path.exists(f'plots/{boundary_treatment}'):
        os.makedirs(f'plots/{boundary_treatment}')

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

    rho0 = 1.0
    u0 = (ux_inf, 0.0)

    # lbm = simulate(nx=Nx, ny=Ny, stencil=stencil, tau=tau, Nt=Nt, rho0=rho0, u0=u0, CSM=CSM, Fi2Ftot=Fi2Ftot,
    #          domain_boundaries=domain_boundaries, writers=writers, precision=precision, use_cuda=use_cuda)

    # Initialize based on velocity and density
    rho0 = np.ones((1, *S.N, 1))
    U0 = np.ones((1, *S.N, 1))*np.array([ux_inf, 0])
    S.initialize(rho0, U0)    

    # Initialize based on velocity and density
    rho0 = np.ones((1, *S2.N, 1))
    U0 = np.ones((1, *S2.N, 1))*np.array([ux_inf, 0])
    S2.initialize(rho0, U0)    

    rho0 = np.ones((1, *S.N, 1, 1))
    U0 = np.ones((1, *S.N, 1, 2))*np.array([ux_inf, 0])
    S3 = LBM_Simulator(nx=Nx, ny=Ny, stencil=stencil, tau=tau, Nt=Nt, CSM=CSM, Fi2Ftot=Fi2Ftot,
             domain_boundaries=domain_boundaries, writers=writers, precision=precision, use_cuda=use_cuda)
    LBM_Simulator.initialize_f(S3, rho0, U0)
    Nt = 4000
    for it in range(Nt):
        # S.collide(tau)
        # S.stream()
        # F1 = S.Fi2Ftot @ S.F.flatten()

        S2.collide(tau)
        F2 = S2.Fi2Ftot @ S2.F.flatten()
        S2.F = (S2.SM @ S2.F.flatten()).reshape((S2.Q, *S2.N, 1))
        
        S3.collide_and_periodic_stream()
        F3 = S3.Fi2Ftot @ S3.F.copy_to_host().flatten()
        S3.correct_boundaries()

        print(F2, F3)


    t, D, L = np.genfromtxt(f'output/{boundary_treatment}/Ftot.csv', delimiter=',', skip_header=1).T

    field = pv.read(f'output/{boundary_treatment}/full_field_{Nt:05d}.vti')
    rho_inf = np.average(field['rho'].reshape((Ny, Nx))[:,0])

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
