"""
LBMBase.py

Base class for LBM, can be used for fully periodic problems.

Requires: numpy
"""

import numpy as np
from scipy import sparse
import os

# Prevent JAX from preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

# Use double precision
jax.config.update("jax_enable_x64", True)

#===================================================================================
#Velset class

class VelSet:
    """
    Simple container class for velocity sets.

    Attributes:
    ----------
    D : scalar
        Dimension
    Q : scalar
        Number of populations
    Wi : array_like
        Weights, shape (Q, 1, ..., 1)
    Cij : array_like
        Population vectors, shape (Q, 1, ..., D)
    """

    def __init__(self, D, Q, Wi, Cij):

        self.D = D
        self.Q = Q
        self.Wi = np.array(Wi).reshape((Q, *[1]*(D+1)))
        self.Cij = np.array(Cij).reshape((Q, *[1]*D, D))

        return

#Initialize standard velocity sets
D1Q3 = VelSet(1, 3, [2/3, 1/6, 1/6], [[0], [1], [-1]])

D2Q9 = VelSet(2, 9, [4/9, *[1/9]*4, *[1/36]*4],
              [[0, 0],
               [1, 0], [0, 1], [-1, 0], [0, -1],
               [1, 1], [-1, 1], [-1, -1], [1, -1]])

D3Q19 = VelSet(3, 19, [1/3, *[1/18]*6, *[1/36]*12],
               [[0, 0, 0], #Still
                [1, 0, 0], [0, 1, 0], [0, 0, 1], #Mag 1
                [-1, 0, 0], [0, -1, 0], [0, 0, -1], #Mag 1
                [1, 1, 0], [1, 0, 1], [0, 1, 1], #Mag sqrt(2)
                [-1, 1, 0], [-1, 0, 1], [0, -1, 1], #Mag sqrt(2)
                [1, -1, 0], [1, 0, -1], [0, 1, -1], #Mag sqrt(2)
                [-1, -1, 0], [-1, 0, -1], [0, -1, -1] #Mag sqrt(2)
                ])

#===================================================================================
#SpeciesBase class

class LBMBase():
    """
    Base class for LBM, can be used for fully periodic problems.

    Attributes
    ----------
    D, Q, Wi, Cij : See velSet class
    N : array_like,
         Number of lattice nodes in each coordinate direction e.g. (Nz, Ny, Nx)
         for 3D, shape (D).
    F : array_like,
        Populations at the current step, shape (Q, *N, 1).
    ibar : array_like
        Population index opposite to population index i. For example, if i
        corresponds to C_i=[1, -1] then ibar corresponds to C_{ibar}=[-1, 1].
        Here, ibar is a list corresponding to range(Q) as input i.
    forceDens : array_like,
        Body force density, shape (1, *N, D). Default is None.
    bodyForce : bool,
        Whether there is a body force present. Default is False.

    Methods
    -------
    fEq(rho, U)
        Get the equilibrium distribution function for a given density and velocity.
    initialize(rho, U)
        Initialize the populations at equilibrium corresponding to specified density and velocity.
    getMoments()
        Get the moments (density and velocity) based on the current populations.
    collide(tau)
        Perform the collision step with the BGK collision model.
    streamPeriodic()
        Perform the streaming step using numpy roll, assuming periodic boundaries.   
    """


    def __init__(self, velSetName, N, forceDens=None):

        assert isinstance(velSetName, str), 'velSetName should be a string'
        assert velSetName in globals(), f'Velocity set {velSetName} not defined'
        assert len(N) == globals()[velSetName].D, f'Length of N should be {globals()[velSetName].D}'

        velSet = globals()[velSetName]

        self.D = velSet.D
        self.Q = velSet.Q
        self.Wi = velSet.Wi
        self.Cij = velSet.Cij

        self.N = np.array(N).flatten()
        # self.F = jnp.zeros((self.Q, *self.N, 1))
        
        self.Ibar = self._getIbar()

        if forceDens is None:
            self.bodyForce = False
        else:
            self.bodyForce = True
            self.forceDens = forceDens.reshape((*[1]*(self.D+1), self.D))

            # Precompute components of the force term to speed up collision step
            if self.D == 2:
                self.forceTermConst, self.forceTermUx, self.forceTermUy = self._precomputeForceTermComps2D()
            elif self.D == 3:
                self.forceTermConst, self.forceTermUx, self.forceTermUy, self.forceTermUz = self._precomputeForceTermComps3D()


        @jax.jit
        def _getMoments(F):
            """
            Get the moments (density and velocity) based on the current populations F.

            Parameters
            ----------
            F : array_like
                Populations, shape (Q, *N, 1).

            Returns
            -------
            rho : array_like
                Density, shape (1, *N, 1).
            U : array_like
                Velocity, shape (1, *N, D).
            """
            rho = jnp.sum(F, axis=0, keepdims=True)

            if not self.bodyForce:
                U = jnp.sum(F*self.Cij, axis=0, keepdims=True)/jnp.maximum(rho, 1e-12)
            else:
                U = (jnp.sum(F*self.Cij, axis=0, keepdims=True) +\
                    0.5*self.forceDens)/jnp.maximum(rho, 1e-12)

            return rho, U
        self._getMoments = _getMoments

        @jax.jit
        def _fEq(rho, U):
            """
            Get the equilibrium distribution function for a given density and velocity.

            Parameters
            ----------
            rho : array_like
                Density, shape (1, *N, 1).
            U : array_like
                Velocity, shape (1, *N, D).
            
            Returns
            -------
            F : array_like
                Equilibrium populations, shape (Q, *N, 1).
            """

            # Precompute U dot Cij and U dot U
            UdotCij = jnp.sum(U*self.Cij, axis=-1, keepdims=True)
            UdotU = jnp.sum(U*U, axis=-1, keepdims=True)

            # Return fEq, equation simplified using that Cs^2=1/3 in lattice units
            return jnp.array(self.Wi*rho*(1 + 3*UdotCij + 4.5*(UdotCij*UdotCij) - 1.5*UdotU))
        self._fEq = _fEq
        
        @jax.jit
        def _forceTerm(U):
            """Force term to correct the collision step, see Ch. 6 of the Kruger book.
            
            Parameters
            ----------
            U : array_like
                Velocity, shape (1, *N, D).
            
            Returns
            -------
            forceTerm : array_like
                Force term, shape (Q, *N, 1).
            """

            if self.D == 2:
                Ux = U[:,:,:,0,None]
                Uy = U[:,:,:,1,None]
                return self.forceTermConst + self.forceTermUx*Ux + self.forceTermUy*Uy
            elif self.D == 3:
                Ux = U[:,:,:,:,0,None]
                Uy = U[:,:,:,:,1,None]
                Uz = U[:,:,:,:,2,None]
                return self.forceTermConst + self.forceTermUx*Ux + self.forceTermUy*Uy + self.forceTermUz*Uz
        
        @jax.jit
        def _collide(F, tau):
            """
            Perform the collision step with the BGK collision model.

            Parameters
            ----------
            F : array_like
                Populations, shape (Q, *N, 1).
            tau : scalar
                Relaxation time in lattice units.

            Returns
            -------
            F : array_like
                Post-collision populations, shape (Q, *N, 1).
            """

            rho, U = _getMoments(F)

            fEq = _fEq(rho, U)

            if not self.bodyForce:
                F = F - (F - fEq) / tau
                return F
            else:
                forceTerm = _forceTerm(U)
                F = F - (F - fEq) / tau + (1 - 0.5/tau)*forceTerm
                return F
        self._collide = _collide

        @jax.jit
        def _streamPeriodic(F):
            """
            Perform the streaming step assuming periodic boundaries.

            Parameters
            ----------
            F : array_like
                Pre-stream populations, shape (Q, *N, 1).
            
            Returns
            -------
            F : array_like
                Periodically streamed populations, shape (Q, *N, 1).
            """
            return jax.vmap(lambda fi, c: jnp.roll(fi, shift=c, axis=tuple(range(self.D))), in_axes=(0,0))(F, jnp.squeeze(self.Cij))
        self._streamPeriodic = _streamPeriodic

        return
    
    def _getIbar(self):
        """
        Get the list of opposite population indices. I.e. Cij[ibar[i]]=-Cij[i].

        Returns
        -------
        ibar : array_like
            List of opposite population indices, shape (Q,).
        """

        #Find for each Cj the closest antiparallel vector; the corresponding i
        #is ibar.
        Ibar = np.where(np.min(np.isclose(self.Cij.reshape(self.Q,self.D),
                        -self.Cij.reshape(self.Q,1,self.D)), axis=-1))[1]
        return Ibar
    
    def fEq(self, rho, U):
        """Wrapper around jitted _fEq function."""
        return self._fEq(rho, U)


    def initialize(self, rho, U):
        '''Function to initialize the species at equilibrium corresponding to
        specified density and velocity.
        Inputs:
        rho : array_like
              Density, shape (1, *N, 1).
        U : array_like
            Velocity, shape (1, *N, D).'''

        if not self.bodyForce:
            self.F = self.fEq(rho, U)
        else:
            self.F = self.fEq(rho, U-0.5*self.forceDens/rho)   
        return

    def getMoments(self):
        """Wrapper around jitted _getMoments function."""
        return self._getMoments(self.F)

    def _precomputeForceTermComps2D(self):
        """
        Precompute 2D components of the force term to correct the collision step, see Ch. 6 of the Kruger book.
        This is done to speed up the computation of the force term during each collision step.

        Returns
        -------
        forceTermConst: array_like
            Constant term in the force term, shape (Q, *N, 1).
        forceTermUx: array_like
            Coefficient of Ux in the force term, shape (Q, *N, 1).
        forceTermUy: array_like
            Coefficient of Uy in the force term, shape (Q, *N, 1).
        """
        Cx = self.Cij[:,:,:,0,None]
        Cy = self.Cij[:,:,:,1,None]        
        Fx = self.forceDens[:,:,:,0]
        Fy = self.forceDens[:,:,:,1]

        forceTermConst = 3*self.Wi*(Cx*Fx + Cy*Fy)
        forceTermUx = 9*self.Wi*((Cx**2 - 1/3)*Fx + (Cx*Cy)*Fy)
        forceTermUy = 9*self.Wi*((Cx*Cy)*Fx + (Cy**2 - 1/3)*Fy)

        return forceTermConst, forceTermUx, forceTermUy

    def _precomputeForceTermComps3D(self):
        """
        Precompute 3D components of the force term to correct the collision step, see Ch. 6 of the Kruger book.
        This is done to speed up the computation of the force term during each collision step.

        Returns
        -------
        forceTermConst: array_like
            Constant term in the force term, shape (Q, *N, 1).
        forceTermUx: array_like
            Coefficient of Ux in the force term, shape (Q, *N, 1).
        forceTermUy: array_like
            Coefficient of Uy in the force term, shape (Q, *N, 1).
        forceTermUz: array_like
            Coefficient of Uz in the force term, shape (Q, *N, 1).
        """
        Cx = self.Cij[:,:,:,:,0,None]
        Cy = self.Cij[:,:,:,:,1,None]
        Cz = self.Cij[:,:,:,:,2,None]
        Fx = self.forceDens[:,:,:,:,0]
        Fy = self.forceDens[:,:,:,:,1]
        Fz = self.forceDens[:,:,:,:,2]

        forceTermConst = 3*self.Wi*(Cx*Fx + Cy*Fy + Cz*Fz)
        forceTermUx = 9*self.Wi*((Cx**2 - 1/3)*Fx + (Cx*Cy)*Fy + (Cx*Cz)*Fz)
        forceTermUy = 9*self.Wi*((Cx*Cy)*Fx + (Cy**2 - 1/3)*Fy + (Cy*Cz)*Fz)
        forceTermUz = 9*self.Wi*((Cx*Cz)*Fx + (Cy*Cz)*Fy + (Cz**2 - 1/3)*Fz)

        return forceTermConst, forceTermUx, forceTermUy, forceTermUz



    def collide(self, tau):
        """Wrapper around jitted _collide function."""
        self.F = self._collide(self.F, tau)
        return


    def streamPeriodic(self):
        """Wrapper around jitted _streamPeriodic function."""
        self.F = self._streamPeriodic(self.F)
        return

