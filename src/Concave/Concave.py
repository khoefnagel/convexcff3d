import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from numba import njit, types
from numba.typed import Dict, List
from scipy import sparse

import shapely
import trimesh


from .getSubzoneMatrixNaive import getSubzoneMatrixNaive
from .getSubzoneMatrixNaiveChen import getSubzoneMatrixNaiveChen
from .computeCutCellMoments import computeCutCellMoments, computeCellMoments
from .addDirectStream import addDirectStream
from .getSubzoneMatrixRowsColsValues import getSubzoneMatrixRowsColsValues
from .getCellMoments import getCellMoments
from .computePgramMoments import computePgramMoments
from .getCutCellMoments import getCutCellMoments
from .find_uncut_cell_index import find_uncut_cell_index

from .getPgramOverlappedCellInds import getPgramOverlappedCellInds

from Solids.getCutCellCentroidsLoc import getCutCellCentroidsLoc

from Pgrams.Pgrams import Pgrams1D, Pgrams2D, Pgrams3D
from cff_utils.tuple3D import tuple3D
from Matrices.shapesRavels import ravelf, ravelp

class Concave:

    def __init__(self):

        # 1. Find the index of any uncut cell in the grid
        uncut_cell_index = find_uncut_cell_index(self.cellHashMap, self.N)

        # 2. Check if the uncut cell is fully fluid or fully solid
        uncut_cell_is_solid = False
        for solidObj in self.solidsObj:
            # Check which candidate cells are contained in the solid. Note that this is a vectorized operation
            if self.D == 1:
                 shapely.contains_xy(solidObj, uncut_cell_index[0] + 0.5, 0.5) | uncut_cell_is_solid
            elif self.D == 2:
                uncut_cell_is_solid = shapely.contains_xy(solidObj, uncut_cell_index[0] + 0.5, uncut_cell_index[1] + 0.5) | uncut_cell_is_solid
            elif self.D == 3:
                uncut_cell_is_solid = solidObj.contains(np.asarray([uncut_cell_index]) + 0.5) | uncut_cell_is_solid


        # 3. Find subzone matrix
        if hasattr(self, 'Chen') and self.Chen:
            print('Using Chen method for subzone matrix construction.')
            self.subzoneMatrixNaive = getSubzoneMatrixNaiveChen(self.subzoneMoments, self.origCellInds, self.destCellInds,
                                                    self.shapep, self.Cij, self.Ibar, self.degs)

            # Use the CFF subzone matrix to compute cell moments such that centroids can also be computed.
            # Since these require first order moments.
            shapep_ = np.array(self.shapep)
            for d in range(self.D):
                shapep_[-self.D+d] = 2
            subzoneMatrixNaive_compute_cell_moments = getSubzoneMatrixNaive(self.subzoneMoments, self.origCellInds, self.destCellInds,
                                                    shapep_, self.Cij, self.Ibar, self.degsCells)
        else:
            self.subzoneMatrixNaive = getSubzoneMatrixNaive(self.subzoneMoments, self.origCellInds, self.destCellInds,
                                                    self.shapep, self.Cij, self.Ibar, self.degs)
            shapep_ = np.array(self.shapep)
            subzoneMatrixNaive_compute_cell_moments = self.subzoneMatrixNaive


        # 4. Find the moments of each pgram (sum of the columns of the subzone matrix)
        self.pgramMoments = computePgramMoments(subzoneMatrixNaive_compute_cell_moments)

        # 5. Compute the moments of the cut cells
        cellMoments = computeCellMoments(self.pgramMoments, self.cellHashMap, uncut_cell_index, uncut_cell_is_solid, self.Cij,
                                              self.N, self.Ibar, self.degsCells, np.max(self.degsCells), shapep_)



        # 6. Find the solid cell indices
        key_type = types.UniTuple(types.int64, 3)
        value_type = types.bool_
        @njit(cache=True)
        def getDict(arr):
            '''Function to convert an array of cell indices to a numba typed dictionary with tuple keys.'''
            d = Dict.empty(key_type, value_type)
            for a in arr:
                d[tuple3D(a)] = True
            return d
        self.solidCellInds = getDict(np.vstack(np.where(cellMoments[:,:,:,0,0,0] < 1e-12)).T)

        # 7. Find the cell indices overlapped by any pgram
        self.pgramOverlappedCellInds = getPgramOverlappedCellInds(self.origCellInds, self.solidCellInds, self.Cij)

        # 8. Get a dictionary with the moments of each cut cell, indexed by their cell indices.
        self.cutCellMoments = getCutCellMoments(cellMoments, self.cellHashMap)

        # 9. Get a dictionary with the centroids of each cut cell, indexed by their cell indices.
        self.cutCellCentroidsLoc = getCutCellCentroidsLoc(self.cutCellMoments, self.D)

        self.cellMoments = cellMoments


        ## TODO: Concave treatment

        
        subzoneMatrix = self.subzoneMatrixNaive

        # Add the direct stream to the subzone matrix
        subzoneMatrix = addDirectStream(subzoneMatrix, self.pgramOverlappedCellInds, self.solidCellInds, self.Cij, 
                                             self.N, self.degs, self.shapep)

        if True: # Remove large intermediate matrices to save memory
            del self.subzoneMatrixNaive, self.pgramMoments

        # Obtain the row indices, column indices, and values of the subzone matrix in COO format
        row_indices, col_indices, values = getSubzoneMatrixRowsColsValues(subzoneMatrix)
        self.subzoneMatrix = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(np.prod(self.shapep), np.prod(self.shapep)))

        return


class Concave1D(Concave, Pgrams1D):

    def __init__(self, N, solids : list, Cij, degs, degsCells, deg):
        Pgrams1D.__init__(self, N, solids, Cij, degs, degsCells, deg)
        Concave.__init__(self)

class Concave2D(Concave, Pgrams2D):

    def __init__(self, N, solids : list, Cij, degs, degsCells, deg):
        Pgrams2D.__init__(self, N, solids, Cij, degs, degsCells, deg)
        Concave.__init__(self)

class Concave3D(Concave, Pgrams3D):

    def __init__(self, N, solids : list, Cij, degs, degsCells, deg):
        Pgrams3D.__init__(self, N, solids, Cij, degs, degsCells, deg)
        Concave.__init__(self)





















