"""Module to obtain the remove solid stream matrix (RSSM)."""

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from .shapesRavels import ravelf
from .matrixUtil import dictToSparseMat



key_type = types.UniTuple(types.int64, 2)
value_type = types.float64
@njit(cache=True)
def getRSSMDict(Cij, N, cutCellMoments, solidCellInds, shapef):
    """Function to obtain the RSSM as a numba typed dictionary.
    
    Parameters
    ----------
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    cutCellMoments : dict
        Dictionary mapping cell indices to cut-cell moments arrays.
    solidCellInds : list
        List of solid cell indices.
    shapef : array_like
        Shape of the population array, shape (Q, *N, 1).
        
    Returns
    -------
    RSSMDict : numba.typed.Dict
        Dictionary representation of the RSSM, where keys are (row, col) tuples
        and values are the matrix entries."""

    D = Cij.shape[-1]
    shapef = np.array(shapef)

    RSSMDict = Dict.empty(key_type, value_type)
    
    for i, Cj in enumerate(Cij.reshape(-1, D)):

        # 1: Remove solid streaming from cut-cells
        for ci, M in cutCellMoments.items():
            ci = np.array(ci)
            V = M[0, 0, 0]  # Cut cell volume

            # Cell index to which you stream
            cj = (ci[:D] + Cj) % N

            colInd = ravelf(i, ci[:D], shapef)
            rowInd = ravelf(i, cj[:D], shapef)

            # Remove the stream of the solid volume (1 - V)
            RSSMDict[(rowInd, colInd)] = -(1 - V)
        
        # 2: Remove solid streaming from fully solid cells
        for ci in solidCellInds:
            ci = np.array(ci)
            cj = (ci[:D] + Cj) % N

            colInd = ravelf(i, ci[:D], shapef)
            rowInd = ravelf(i, cj[:D], shapef)

            # Remove the stream of the solid volume (1 for solid cells)
            RSSMDict[(rowInd, colInd)] = -1

    return RSSMDict

def getRSSM(Cij, N, cutCellMoments, solidCellInds, shapef):
    """Wrapper around getRSSMDict, converts the numba typed dictionary to a sparse matrix."""
    
    if len(solidCellInds) == 0:
        solidCellInds = List.empty_list(types.UniTuple(types.int64, 3))
    if len(cutCellMoments) == 0:
        cutCellMoments = Dict.empty(types.UniTuple(types.int64, 3), types.float64[:, :, :])
    RSSMDict = getRSSMDict(Cij, N, cutCellMoments, solidCellInds, shapef)
    return dictToSparseMat(RSSMDict, (np.prod(shapef), np.prod(shapef)))