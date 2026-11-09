import numpy as np
from numba import njit
from numba import types
from numba.typed import List, Dict

from cff_utils.tuple3D import tuple3D

key_type = types.UniTuple(types.int64, 3)
value_type_bool = types.boolean

@njit(cache=True)
def getPgramOverlappedCellIndsCj(origCellInds, solidCellInds):
    """Function to obtain the pgram-overlapped cell indices for a single Cj.

    Parameters
    ----------
    origCellInds : array_like
        Array of origin cell indices for the given Cj, shape (nPgrams, 3).
    
    Returns
    -------
    pgramOverlappedCellInds : numba.typed.Dict
        Dictionary with the pgram-overlapped cell indices as keys.
    """

    D = origCellInds.shape[-1]

    pgramOverlappedCellInds = Dict.empty(key_type, value_type_bool)
    
    for ocis in origCellInds:
        for i in range(ocis.shape[0]):
            row = tuple3D(ocis[i])
            if row not in pgramOverlappedCellInds and row not in solidCellInds:
                pgramOverlappedCellInds[row] = True
    return pgramOverlappedCellInds

value_type_dict_bool = types.DictType(key_type, value_type_bool)
@njit(cache=True)
def getPgramOverlappedCellInds(origCellInds, solidCellInds, Cij):
    """Function to obtain the pgram-overlapped cell indices

    Parameters
    ----------
    origCellInds : numba.typed.Dict
        Dictionary mapping Cj tuples to origin cell indices arrays.
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    
    Returns
    -------
    pgramOverlappedCellInds : numba.typed.Dict
        Dictionary mapping Cj tuples to dictionaries of pgram-overlapped cell indices.
    """

    D = Cij.shape[-1]

    pgramOverlappedCellInds = Dict.empty(key_type, value_type_dict_bool)
    for Cj in Cij.reshape(-1, D):

        CjTuple = tuple3D(Cj)

        if len(origCellInds[CjTuple]) == 0: # Still populations
            pgramOverlappedCellInds[CjTuple] = Dict.empty(key_type, value_type_bool)
        else:
            pgramOverlappedCellInds[CjTuple] = getPgramOverlappedCellIndsCj(origCellInds[CjTuple], solidCellInds)
    return pgramOverlappedCellInds

