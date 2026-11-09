import numpy as np
from numba import njit, types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D
from Matrices.shapesRavels import ravelp

value_type1 = types.DictType(types.int64, types.float64)
@njit(cache=True)
def getSubzoneMatrixNaive(subzoneMoments, origCellInds, destCellInds, shapep, Cij, Ibar, degs):

    D = Cij.shape[-1]

    # Create naive subzone matrix as nested numba typed dictionary.
    # Note: first entry is column index, second entry is row index!
    subzoneMatrixNaive = Dict.empty(key_type=types.int64, value_type=value_type1)
    for i, Cj in enumerate(Cij.reshape(-1, D)):
        CjTuple = tuple3D(Cj)
        for pgramInd in range(len(subzoneMoments[CjTuple])):
            for subzoneInd in range(len(subzoneMoments[CjTuple][pgramInd])):
                origInd = origCellInds[CjTuple][pgramInd][subzoneInd]
                reflectDestInd = destCellInds[CjTuple][pgramInd][subzoneInd]
                for degTuple in degs:

                    moment = subzoneMoments[CjTuple][pgramInd][subzoneInd][*tuple3D(degTuple)].item()

                    colInd = ravelp(i, origInd[:D], degTuple, shapep)
                    reflectRowInd = ravelp(Ibar[i], reflectDestInd[:D], degTuple, shapep)

                    if colInd not in subzoneMatrixNaive:
                        subzoneMatrixNaive[colInd] = Dict.empty(key_type=types.int64, value_type=types.float64)
                    if reflectRowInd not in subzoneMatrixNaive[colInd]:
                        subzoneMatrixNaive[colInd][reflectRowInd] = 0.0

                    subzoneMatrixNaive[colInd][reflectRowInd] += moment

    return subzoneMatrixNaive