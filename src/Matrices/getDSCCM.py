"""Module to obtain the direct stream cut-cell matrix (DSCCM) for given."""

import numpy as np
from numba import njit, types
from numba.typed import Dict
from scipy import sparse

from .shapesRavels import ravelf
from .matrixUtil import dictToSparseMat


key_type = types.UniTuple(types.int64, 2)
value_type = types.float64
@njit(cache=True)
def getDSCCMDict(Cij, N, cutCellMoments, shapef):
    """Function to obtain the DSCCM as a numba typed dictionary.
    
    Parameters
    ----------
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    cutCellMoments : dict
        Dictionary mapping cell indices to cut-cell moments arrays.
    shapef : array_like
        Shape of the population array, shape (Q, *N, 1).
    
    Returns
    -------
    DSCCMDict : numba.typed.Dict
        Dictionary representation of the DSCCM, where keys are (row, col) tuples
        and values are the matrix entries.
    """
    D = Cij.shape[-1]
    shapef = np.array(shapef)

    DSCCMDict = Dict.empty(key_type, value_type)
    
    for i, Cj in enumerate(Cij.reshape(-1, D)):
        for ci in cutCellMoments.keys():
            ci = np.array(ci)

            # Cell index from which you stream
            cj = (ci[:D] - Cj) % N

            rowInd = ravelf(i, ci[:D], shapef)
            colInd = ravelf(i, cj[:D], shapef)

            DSCCMDict[(rowInd, colInd)] = 1

    return DSCCMDict

def getDSCCM(Cij, N, cutCellMoments, shapef):
    """Wrapper around getDSCCMDict, converts the numba typed dictionary to a sparse matrix."""
    if len(cutCellMoments) == 0:
        cutCellMoments = Dict.empty(types.UniTuple(types.int64, 3), types.float64[:, :, :])
    DSCCMDict = getDSCCMDict(Cij, N, cutCellMoments, shapef)
    return dictToSparseMat(DSCCMDict, (np.prod(shapef), np.prod(shapef)))

