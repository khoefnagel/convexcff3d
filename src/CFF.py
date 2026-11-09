"""
Module containing the CFF class, which implements a nD LBM solver with
the CFF boundary condition for convex, non-moving solids.

Requires: numpy, itertools, jax, pyevtk
"""

# Fix for Numba Dict __repr__ to avoid printing all entries in the dict, which slows down the debugger
from numba.typed import Dict
Dict.__repr__ = lambda self: f"<Numba Dict len={len(self)}>"

import numpy as np
import matplotlib.pyplot as plt
import pyevtk
import jax
import jax.numpy as jnp

from LBMBase import LBMBase
from Pgrams.Pgrams import Pgrams1D, Pgrams2D, Pgrams3D
from Concave.Concave import Concave2D, Concave3D
from Matrices.Matrices import Matrices
from cff_utils.tuple3D import tuple3D



class CFF(LBMBase, Matrices):
    """
    Represents a nD LBM solver which implements the CFF boundary condition for
    convex, non-moving solids.

    Attributes
    ----------
    velSetName : string
        Name of the velocity set used, e.g. "D2Q9".
    N : array_like
        Number of lattice nodes in each coordinate direction e.g. (Ny, Nx)
        for 2D, shape (D).
    solids : list of array_like
        List of the coordinates of the boundary of the solids, each entry
        should be an array of shape (Nv, D) where Nv is the number of
        boundary vertices.
    forceDens : None or array_like
        Body force density vector or None to not use a body force.
        The default is None.
    projection : string
        Type of projection used to compute the spline coefficients in the cut cells.
        Options are 'NIS' (nearest interpolation site), 'PC' (piecewise constant),
        and 'CBS' (conservative B-spline). Default is 'NIS'.
    deg : integer
        Degree used for the splines. Default is 1.
    degs : list of tuples
        List of degrees used in each coordinate direction for the splines.
    degsCells : list of tuples
        List of degrees used in each coordinate direction for the splines
        in the cut cells. This may be different from degs when using
        the 'CBS' projection since it requires higher degree splines to compute
        the gradient products.
    onlyPgramOverlapped : bool
        Whether to only consider pgram-overlapped cells in the spline construction
        when using the 'CBS' projection. Default is False.
    minDeriv : integer
        Minimum derivative order to consider when using the 'CBS' projection. Default is 1.
    shapef : tuple
        Shape of the populations array F, (Q, *N, 1).
    ravelf : function
        Function to compute the raveled index in the F array from multi-indices.
    unravelf : function
        Function to compute the multi-indices in the F array from a raveled index.
    shapes : tuple
        Shape of the spline coefficients S, (Q, *N, 1).
    ravels : function
        Function to compute the raveled index in the S array from multi-indices.
    unravels : function
        Function to compute the multi-indices in the S array from a raveled index.
    shapep : tuple
        Shape of the polynomial coefficients array PC, (Q, *N, *[deg+1]*D).
    ravelp : function
        Function to compute the raveled index in the PC array from multi-indices.
    unravelp : function
        Function to compute the multi-indices in the PC array from a raveled index.
    F : array_like
        Populations, shape (Q, Ny, Nx, 1).
    SM : sparse matrix
        Sparse streaming matrix including the effect of the CFF boundary condition,
        shape (Q*Ny*Nx*1, Q*Ny*Nx*1). Stream is performed as F = SM @ F.
    Fi2fsurf : sparse matrix
        Matrix to compute the force on each facet from the input populations,
        shape (2*N_facets, Q*Nx*Ny) with N_facets the number of facets.
    Fi2Ftot : sparse matrix
        Matrix to compute the total force on each solid from the input populations,
        shape (2*N_solids, Q*Nx*Ny) with N_solids the number of solids.
    """
    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='NIS', deg=1,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        
        LBMBase.__init__(self, velSetName, N, forceDens=forceDens)

        self.deg = deg

        self.projection = projection
        if projection == 'NIS':
            assert deg == 1, "NIS projection only implemented for degree 1 splines."
            assert not onlyPgramOverlapped, "onlyPgramOverlapped not applicable for NIS projection."
            assert minDeriv == 1, "minDeriv not applicable for NIS projection."

            if self.D == 1:
                self.degs = [[0], [1]]
            elif self.D == 2:
                self.degs = [(0,0), (1,0), (0,1)]
            elif self.D == 3:
                self.degs = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
            self.degsCells = self.degs # Degrees to compute in the cut cells
        
        elif projection == 'PC': # Piecewise constant
            assert deg == 0, "PC projection only implemented for degree 0 splines."

            self.degs = [[0]*self.D]

            # Use first order cell moments to be able to compute centroids
            if self.D == 1:
                self.degsCells = [[0], [1]]
            elif self.D == 2:
                self.degsCells = [(0,0), (1,0), (0,1)]
            elif self.D == 3:
                self.degsCells = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
        
        elif projection == 'CBS': # Conservative B-spline, minimizing gradient
            self.onlyPgramOverlapped = onlyPgramOverlapped # Whether to only consider pgram-overlapped cells in the spline construction
            self.minDeriv = minDeriv
            gradDegMax = 2*deg+1
            if self.D == 1:
                self.degs = [[i] for i in range(deg+1)]
                self.degsCells = [[i] for i in range(gradDegMax)]
            elif self.D == 2:
                self.degs = [(i,j) for i in range(deg+1) for j in range(deg+1)]
                self.degsCells = [(i,j) for i in range(gradDegMax) for j in range(gradDegMax)]
            elif self.D == 3:
                self.degs = [(i,j,k) for i in range(deg+1) for j in range(deg+1) for k in range(deg+1)]
                self.degsCells = [(i,j,k) for i in range(gradDegMax) for j in range(gradDegMax) for k in range(gradDegMax)]
        elif projection == 'MLBS': # Mass-lumped B-spline
            assert minDeriv == 1, "minDeriv not applicable for MLBS projection."

            self.onlyPgramOverlapped = onlyPgramOverlapped # Whether to only consider pgram-overlapped cells in the spline construction

            if self.D == 1:
                self.degs = [[i] for i in range(deg+1)]
            if self.D == 2:
                self.degs = [(i,j) for i in range(deg+1) for j in range(deg+1)]
            if self.D == 3:
                self.degs = [(i,j,k) for i in range(deg+1) for j in range(deg+1) for k in range(deg+1)]
            self.degsCells = self.degs
        else:
            raise NotImplementedError(f"Projection '{projection}' not implemented.")

        self.concaveCorrectionMethod = concaveCorrectionMethod

        self.shapef = (self.Q, *self.N, 1)
        self.ravelf = lambda *key : np.ravel_multi_index(key, self.shapef)
        self.unravelf = lambda key : np.unravel_index(key, self.shapef)

        self.shapes = (self.Q, *self.N, 1)
        self.ravels = lambda *key : np.ravel_multi_index(key, self.shapes)
        self.unravels = lambda key : np.unravel_index(key, self.shapes)

        self.shapep = (self.Q, *self.N, *[self.deg + 1]*self.D)
        self.ravelp = lambda *key : np.ravel_multi_index(key, self.shapep)
        self.unravelp = lambda key : np.unravel_index(key, self.shapep)

        @jax.jit
        def _stream(F, row, col, data):
            """Streaming step using the streaming correction matrix CSM.

            Parameters
            ----------
            F : array_like
                Populations, shape (Q, *N, 1).
            row : array_like
                Row indices of the non-zero entries of the streaming correction matrix CSM.
            col : array_like
                Column indices of the non-zero entries of the streaming correction matrix CSM.
            data : array_like
                Data entries of the non-zero entries of the streaming correction matrix CSM.

            Returns
            -------
            F : array_like
                Post-streaming populations, shape (Q, *N, 1).
            """
            
            
            # Perform periodic streaming
            F = self._streamPeriodic(F)

            # Compute corrections to streaming due to CFF boundary condition
            updates = data*F.reshape(-1)[col]

            # Add corrections to periodically streamed populations
            F_flat = F.reshape(-1)
            F_flat = F_flat.at[row].add(updates)

            return F_flat.reshape(F.shape)
        
        self._stream = _stream

    def setCSM(self):
        """Set the streaming correction matrix CSM and its component arrays
        row, col, data for use in the jitted _stream function.
        """
        self.CSM = self._getCSM()
        self.row = jnp.array(self.CSM.tocoo().row)
        self.col = jnp.array(self.CSM.tocoo().col)
        self.data = jnp.array(self.CSM.tocoo().data)

    def getSM(self):
        """Get the full streaming matrix SM including the effect of the CFF
        boundary condition."""
        return self.CSM @ self._getSPM() + self._getCSM()

    def getNi(self):
        """Get the number of particles in each cell. In full cells this is
        simply the populations (particle density), while in cut cells these
        are multiplied by the volume fraction.
        
        Returns
        -------
        Ni : array_like
            Number of particles in each cell, shape (Q, *N, 1).
        """
        Ni = np.copy(self.F)
        for ci, moments in self.cutCellMoments.items():
            Ni[:,*ci[:self.D],:] *= moments[0,0,0].item() # Volume fraction
        for ci in self.solidCellInds:
            Ni[:,*ci[:self.D],:] = 0.0
        return Ni
    
    def getCentroidsGlob(self):
        """Get centroids of all cells in global coordinates.

        Returns
        -------
        centroidsGlob : array_like
            Centroids of all cells in global coordinates, shape (*N, D).
        """
        centroidsGlob = np.stack(np.meshgrid(*[np.arange(n) for n in self.N], indexing='ij'), axis=-1) + 0.5
        for ci, centroidLoc in self.cutCellCentroidsLoc.items():
            centroidsGlob[ci[:self.D]] += centroidLoc - 0.5
        return centroidsGlob
    
    def getCellVolumes(self):
        """Get the volumes of all cells. In full cells this is 1.0, while in cut cells
        this is the volume fraction.

        Returns
        -------
        V : array_like
            Volumes of all cells, shape (1, *N, 1).
        """
        V = np.ones(self.N)
        for ci, moments in self.cutCellMoments.items():
            V[ci[:self.D]] = moments[0,0,0].item() # Volume fraction
        for ci in self.solidCellInds:
            V[ci[:self.D]] = 0.0
        return V.reshape((1, *self.N, 1))

    def getWallDists(self):
        """Get the distances from the cell centroids to the nearest wall in each cell.

        Returns
        -------
        wallDists : array_like
            Distances from the cell centroids to the nearest wall in each cell, shape (1, *N, 1).
        """
        centroids = self.getCentroidsGlob()
        if self.D == 1:
            raise NotImplementedError("getWallDists not implemented for 1D.")
        elif self.D == 2:
            import shapely
            points = shapely.points(centroids[...,0], centroids[...,1])
            dists_min = np.full(points.shape, np.inf)
            for Cj in np.squeeze(self.Cij):
                # Check also translated copies to account for periodicity
                boundary = shapely.affinity.translate(self.solidsUnion.boundary, xoff=self.N[0]*Cj[0], yoff=self.N[1]*Cj[1])
                distances = shapely.distance(points, boundary)
                dists_min = np.minimum(dists_min, distances)
            return dists_min.reshape((1, *self.N, 1))
        elif self.D == 3:
            import trimesh
            points = trimesh.points.PointCloud(centroids.reshape(-1, 3))
            dists_min = np.full(points.shape[0], np.inf)
            for Cj in np.squeeze(self.Cij):
                # Check also translated copies to account for periodicity
                boundary = trimesh.Trimesh(vertices=self.solidsUnion.vertices + np.array(self.N)*Cj, faces=self.solidsUnion.faces)
                distances = np.abs(trimesh.proximity.signed_distance(boundary, points))
                dists_min = np.minimum(dists_min, distances)
            return dists_min.reshape((1, *self.N, 1))

    def stream(self):
        """Wrapper around jitted _stream function."""
        self.F = self._stream(self.F, self.row, self.col, self.data)
        return



    def evaluateBasis(self, xi):
        """Evaluate the spline basis functions at given local coordinates xi.
        
        Parameters
        ----------
        xi : array_like
            Local coordinates where to evaluate the basis functions,
            shape (..., D).
        
        Returns
        -------
        fInterp : array_like
            Interpolated populations at xi, shape (Q, ..., 1).
        rhoInterp : array_like
            Interpolated density at xi, shape (1, ..., 1).
        UInterp : array_like
            Interpolated velocity at xi, shape (1, ..., D).
        """
        assert xi.shape[-1] == self.D, "xi must have shape (..., D)."

        # Get polynomial coefficients in each cell
        PC = (self.Fi2PC @ self.F.flatten()).reshape(self.shapep)

        # Get cell indices
        ci = np.floor(xi).astype(int)

        # Get xi in local cell coordinates
        xLoc = xi - ci

        # At the locations in the mesh where no spline projection is performed,
        # set populations to piecewise constant.
        zeroInds = np.where(np.sum(np.abs(PC), axis=tuple(list(range(-self.D,0)))) < 1e-14)
        PC[*zeroInds,*[0]*self.D] = self.F[*zeroInds, 0]

        coeffs = PC[:,*ci.reshape(-1,self.D).T]

        # Add contributions from each degree
        fInterp = np.zeros((self.Q, *xi.shape[:-1], 1))
        for deg in self.degs:
            fInterp += (coeffs[:,:, *deg] * np.prod(xLoc.reshape(-1, self.D)**deg, axis=1)).reshape(self.Q, *xi.shape[:-1], 1)
        

        rhoInterp = np.sum(fInterp, axis=0, keepdims=True)
        UInterp = np.sum((fInterp*self.Cij), axis=0, keepdims=True) / rhoInterp
        
        return fInterp, rhoInterp, UInterp


class CFF1D(CFF, Pgrams1D):
    """Specialization of the CFF class for 1D LBM solvers using Pgrams1D."""

    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='NIS', deg=1,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        CFF.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)
        Pgrams1D.__init__(self, N, solids, self.Cij, self.degs, self.degsCells, self.deg)
        CFF.setCSM(self)


class CFF2D(CFF, Concave2D):
    """Specialization of the CFF class for 2D LBM solvers using Pgrams2D."""

    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='NIS', deg=1,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        CFF.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)
        Concave2D.__init__(self, N, solids, self.Cij, self.degs, self.degsCells, self.deg)
        CFF.setCSM(self)

    def write(self, fn):
        rho, U = self.getMoments()

        # Convert from jax to numpy
        rho = np.array(rho)
        U = np.array(U)
        
        #V = self.getCellVolumes()
        X, Y = [np.arange(n+1).astype(np.float32) for n in self.N]
        Z = np.array([0., 1.], dtype=np.float32)
        # 'U': (U[0,:,:,0], U[0,:,:,1])})

        pyevtk.hl.gridToVTK(fn, X, Y, Z, cellData = {'rho': rho[0,:,:,None,0].astype(np.float32), 
                                                       'U' : (U[0,:,:,None,0].astype(np.float32), U[0,:,:,None,1].astype(np.float32), np.zeros_like(U[0,:,:,None,0].astype(np.float32))),
                                                       #'V' : V[0,:,:,0].astype(np.float32)
                                                       })


class CFF3D(CFF, Concave3D):
    """Specialization of the CFF class for 3D LBM solvers using Pgrams3D."""

    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='NIS', deg=1,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        CFF.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)
        Concave3D.__init__(self, N, solids, self.Cij, self.degs, self.degsCells, self.deg)
        CFF.setCSM(self)

    # Destructor
    def __del__(self):
        pass

    def write(self, fn):
        rho, U = self.getMoments()

        # Convert from jax to numpy
        rho = np.array(rho)
        U = np.array(U)
        
        #V = self.getCellVolumes()
        X, Y, Z = [np.arange(n+1).astype(np.float32) for n in self.N]
        # 'U': (U[0,:,:,:,0], U[0,:,:,:,1], U[0,:,:,:,2])})

        pyevtk.hl.gridToVTK(fn, X, Y, Z, cellData = {'rho': rho[0,:,:,:,0].astype(np.float32), 
                                                     'U' : (U[0,:,:,:,0].astype(np.float32), U[0,:,:,:,1].astype(np.float32), U[0,:,:,:,2].astype(np.float32)),
                                                     #'V' : V[0,:,:,:,0].astype(np.float32)
                                                     })