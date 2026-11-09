import numpy as np
from numba import njit

from cff_utils.tuple3D import tuple3D
from Matrices.shapesRavels import ravelp

@njit(cache=True)
def addDirectStream(subzoneMatrix, pgramOverlappedCellInds, solidCellInds, Cij, N, degs, shapep):

    D = Cij.shape[-1]
    for i, Cj in enumerate(Cij.reshape(-1, D)):
        CjTuple = tuple3D(Cj)
        for cellInd in pgramOverlappedCellInds[CjTuple]:
            if tuple3D(cellInd) in solidCellInds:
                continue
            for degTuple in degs:
                colInd = ravelp(i, np.asarray(cellInd)[:D], degTuple, shapep)

                Pa = 0.0
                if colInd not in subzoneMatrix:
                    continue
                for val in subzoneMatrix[colInd].values():
                    Pa += val

                directDestInd = tuple3D((np.asarray(cellInd)[:D] + Cj) % N)
                rowInd = ravelp(i, np.asarray(directDestInd)[:D], degTuple, shapep)

                if rowInd not in subzoneMatrix[colInd]:
                    subzoneMatrix[colInd][rowInd] = 0.0
                subzoneMatrix[colInd][rowInd] += -Pa
    
    return subzoneMatrix