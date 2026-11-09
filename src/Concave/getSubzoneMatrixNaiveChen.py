import numpy as np
from numba import njit, types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D
from Matrices.shapesRavels import ravelp

zeros = np.array([0, 0, 0], dtype=np.int64)

value_type1 = types.DictType(types.int64, types.float64)
# @njit(cache=True)
def getSubzoneMatrixNaiveChen(subzoneMoments, origCellInds, destCellInds, shapep, Cij, Ibar, degs):

    D = Cij.shape[-1]

    # Create naive subzone matrix as nested numba typed dictionary.
    # Note: first entry is column index, second entry is row index!
    subzoneMatrixNaive = Dict.empty(key_type=types.int64, value_type=value_type1)
    for i, Cj in enumerate(Cij.reshape(-1, D)):
        CjTuple = tuple3D(Cj)
        for pgramInd in range(len(subzoneMoments[CjTuple])):

            # 1) Find pgram volume
            pgramVolume = 0.
            for subzoneInd in range(len(subzoneMoments[CjTuple][pgramInd])):
                pgramVolume += subzoneMoments[CjTuple][pgramInd][subzoneInd][0, 0, 0].item()

            if pgramVolume < 1e-14:
                # Skip this pgram if its volume is too small
                continue

            for subzoneInInd in range(len(subzoneMoments[CjTuple][pgramInd])):
                origInd = origCellInds[CjTuple][pgramInd][subzoneInInd]
                volIn = subzoneMoments[CjTuple][pgramInd][subzoneInInd][0, 0, 0].item()

                for subzoneOutInd in range(len(subzoneMoments[CjTuple][pgramInd])):
                    destInd = destCellInds[CjTuple][pgramInd][subzoneOutInd] # This is for Chen since we loop over both in and out subzones
                    volOut = subzoneMoments[CjTuple][pgramInd][subzoneOutInd][0, 0, 0].item()

                    colInd = ravelp(i, origInd[:D], zeros[:D], shapep)
                    rowInd = ravelp(Ibar[i], destInd[:D], zeros[:D], shapep)

                    if colInd not in subzoneMatrixNaive:
                        subzoneMatrixNaive[colInd] = Dict.empty(key_type=types.int64, value_type=types.float64)
                    if rowInd not in subzoneMatrixNaive[colInd]:
                        subzoneMatrixNaive[colInd][rowInd] = 0.0
                    subzoneMatrixNaive[colInd][rowInd] += volIn * volOut / pgramVolume

    return subzoneMatrixNaive