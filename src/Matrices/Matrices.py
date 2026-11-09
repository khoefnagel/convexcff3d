"""Module for constructing various matrices used in the CFF boundary condition"""


import scipy.sparse as sparse
import numpy as np
import time

from cff_utils.tuple3D import tuple3D

from .getRSSM import getRSSM
from .getDSCCM import getDSCCM
from .Projections.NIS import getFi2PC as NIS_getFi2PC
from .Projections.CBS import getFi2PC as CBS_getFi2PC
from .Projections.MLBS import getFi2PC as MLBS_getFi2PC
# from .getRI import getRI
from .getForces import getForces
from .matrixUtil import getActiveBsplineInds
from .convert_CSM_to_post import convert_CSM_to_post
from .compressed_coo_matmul import compressed_coo_matmul


class Matrices:
    """Class containing functions to construct various matrices used in the
       CFF boundary condition."""

    def stream1DInds(self, inds, reverse=False):
        """Get the streamed indices for given array of flat indices.
        
        Parameters
        ----------
        inds : array_like
            Flat indices to be streamed, shape (Ninds, ).
        reverse : bool, optional
            If True, perform reverse streaming, by default False.
            
        Returns
        -------
        streamedInds : array_like
            Flat indices after streaming, shape (Ninds, ).
        """

        sign = -1 if reverse else 1
        multiDInds = np.vstack(np.unravel_index(inds, (self.Q, *self.N))).T
        streamedMultiDInds = np.array(multiDInds)
        streamedMultiDInds[:,1:] = (streamedMultiDInds[:,1:] + sign*self.Cij[multiDInds[:,0],*[0]*self.D])%self.N
        return np.ravel_multi_index(streamedMultiDInds.T, (self.Q, *self.N))

    def _getCSM(self):
        """Construct the complete correct streaming matrix (CSM)."""

        # 1: Remove solid streaming matrix (RSSM)
        self.RSSM = getRSSM(self.Cij, self.N, self.cutCellMoments, self.solidCellInds, self.shapef)

        # 2: Direct stream from cut-cell matrix (DSCCM)
        self.DSCCM = getDSCCM(self.Cij, self.N, self.cutCellMoments, self.shapef)

        # 3: Fi2PC projection matrix from populations to spline coefficients
        if self.projection == 'NIS':
            self.Fi2PC = NIS_getFi2PC(self.Cij, self.N, self.pgramOverlappedCellInds, self.cutCellMoments,
                                    self.cutCellCentroidsLoc, self.solidCellInds, self.shapef, self.shapep,
                                        self.degs)
            
        elif self.projection == 'PC':
            col = np.unique(self.subzoneMatrix.col)
            self.Fi2PC = sparse.coo_matrix((np.ones(len(col)), (col, col)), shape=(np.prod(self.shapef), np.prod(self.shapef)))
            # self.Fi2PC = sparse.eye(np.prod(self.shapef)).asformat('coo')

        elif self.projection == 'CBS':

            # Find the indices of the splines that live in the pgram-overlapped cells
            self.activeBsplineInds = getActiveBsplineInds(self.pgramOverlappedCellInds, self.N, self.deg, self.Cij)
            (self.SC2PC, self.Fi2SC, self.H, self.C) =\
                 CBS_getFi2PC(self.Cij, self.N, self.pgramOverlappedCellInds, self.cutCellMoments,
                              self.cutCellCentroidsLoc, self.solidCellInds, 
                              self.shapef, self.shapes, self.shapep,
                              self.degs, self.deg, onlyPgramOverlapped=self.onlyPgramOverlapped,
                              minDeriv=self.minDeriv)
            self.Fi2PC = self.SC2PC @ self.Fi2SC
        elif self.projection == 'MLBS':
            self.Fi2SC, self.SC2PC = MLBS_getFi2PC(self.Cij, self.N, self.pgramOverlappedCellInds, self.cutCellMoments,
                                    self.solidCellInds, self.shapef, self.shapes,
                                    self.shapep, self.deg, onlyPgramOverlapped=self.onlyPgramOverlapped)
            self.Fi2PC = self.SC2PC @ self.Fi2SC
        else:
            raise ValueError("Projection not (yet) implemented")

        # Get the force matrices
        (
            self.PC2Ftot,
            self.PC2fsurfInput,
            self.PC2fsurfGrid
        ) = getForces(
            self.Cij,
            self.N,
            self.subzoneMoments,
            self.origCellInds,
            self.pgram2solid,
            self.pgram2inputFacet,
            self.pgram2gridCutFacet,
            self.pgram2subGridCutFacet,
            self.inputFacetAreas,
            self.gridCutFacetAreas,
            self.shapef,
            self.shapep,
            self.degs,
        )

        # Covert to coo format
        self.Fi2Ftot = compressed_coo_matmul(self.PC2Ftot, self.Fi2PC)
        self.Fi2fsurfInput = compressed_coo_matmul(self.PC2fsurfInput, self.Fi2PC)
        self.Fi2fsurfGrid = compressed_coo_matmul(self.PC2fsurfGrid, self.Fi2PC)

        # 5: Solid stream matrix (SSM)
        self.SSM = self._getSSM()

        # 6: Reflect and integrate (RI) matrix
        nDegs = np.prod(self.shapep[-self.D:])
        rows = self.subzoneMatrix.row // nDegs
        self.RI = sparse.coo_matrix((self.subzoneMatrix.data, (rows, self.subzoneMatrix.col)), shape=(np.prod(self.shapef), np.prod(self.shapep)))
        self.RI.sum_duplicates()


        CSM = self._correctVolume((self.RSSM + self.RI @ self.Fi2PC + self.DSCCM).tocoo()) - self.DSCCM + self.SSM     


        # Convert CSM to post-stream populations to post-stream corrections
        CSM_post = convert_CSM_to_post(CSM, self.N, self.Cij, self.shapef)

        self.Fi2Ftot = convert_CSM_to_post(self.Fi2Ftot, self.N, self.Cij, self.shapef)
        self.Fi2fsurfInput = convert_CSM_to_post(self.Fi2fsurfInput, self.N, self.Cij, self.shapef)
        self.Fi2fsurfGrid = convert_CSM_to_post(self.Fi2fsurfGrid, self.N, self.Cij, self.shapef)

        return CSM_post
    
    def _getSPM(self):
        """Construct the stream periodic matrix (SPM).
        
        Returns
        -------
        SPM : scipy.sparse.spmatrix
            Stream periodic matrix in CSR format, shape (Q*prod(N), Q*prod(N)).
        """
        colInds = np.arange(np.prod(self.shapef))
        rowInds = self.stream1DInds(colInds)
        vals = np.ones_like(colInds)
        SPM = sparse.coo_matrix((vals, (rowInds, colInds)), shape=(np.prod(self.shapef), np.prod(self.shapef)))
        return SPM

    def _getSSM(self):
        """Construct the solid stream matrix (SSM).
        
        Returns
        -------
        SSM : scipy.sparse.spmatrix
            Solid stream matrix in CSR format, shape (Q*prod(N), Q*prod(N)).
        """
        if len(self.solidCellInds) == 0:
            return sparse.coo_matrix((np.prod(self.shapef), np.prod(self.shapef)))
        inds = (np.ravel_multi_index(np.array(list(self.solidCellInds))[:,:self.D].T, self.N) +\
                 np.prod(self.N)*np.arange(self.Q)[:,None]).flatten()
        vals = np.ones_like(inds)
        SSM = sparse.coo_matrix((vals, (inds, inds.copy())), shape=(np.prod(self.shapef), np.prod(self.shapef)))
        return SSM

    def _correctVolume(self, matrix):
        """Correct the volume of the populations in a give matrix by dividing the rows
              corresponding to cut-cells by their volume.
        
        Parameters
        ----------
        matrix : scipy.sparse.spmatrix
            Matrix to be volume-corrected, shape (..., Q*prod(N)).
        Returns
        -------
        matrix : scipy.sparse.spmatrix
            Volume-corrected matrix, shape (..., Q*prod(N)).
        """

        assert matrix.shape[0] == np.prod(self.shapef)

        if len(self.cutCellMoments) == 0:
            return matrix


        for j in range(len(matrix.row)):
            rowInd = matrix.row[j]
            row_multi_index = np.unravel_index(rowInd, self.shapef)
            cellInd = row_multi_index[1:-1]
            if tuple3D(cellInd) in self.cutCellMoments and self.cutCellMoments[tuple3D(cellInd)][0,0,0] > 1e-12:
                matrix.data[j] /= self.cutCellMoments[tuple3D(cellInd)][0,0,0]


        return matrix
