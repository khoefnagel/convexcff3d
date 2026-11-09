"""Module containing utility functions for the Matrices package.
These functions are numba compiled for performance."""

import numpy as np
from scipy import sparse
import itertools as it
from numba import njit, types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D
from cff_utils.it_product import it_product

@njit(cache=True)
def dictToArrays(Dict):
    """Convert a numba typed dictionary to three separate arrays of
    the row indices, column indices, and values.
    
    Parameters
    ----------
    Dict : numba.typed.Dict
        Dictionary with (row, col) tuple keys and float values.
    
    Returns
    -------
    rows : np.ndarray
        Array of row indices.
    cols : np.ndarray
        Array of column indices.
    vals : np.ndarray
        Array of values.
    """
    rows = np.empty(len(Dict), dtype=np.int64)
    cols = np.empty(len(Dict), dtype=np.int64)
    vals = np.empty(len(Dict), dtype=np.float64)
    for i, ((row, col), val) in enumerate(Dict.items()):
        rows[i] = row
        cols[i] = col
        vals[i] = val
    return rows, cols, vals

# Not jittable
def dictToSparseMat(Dict, shape):
    """Convert a numba typed dictionary to a scipy sparse matrix.

    Parameters
    ----------
    Dict : numba.typed.Dict
        Dictionary with (row, col) tuple keys and float values.
    shape : tuple
        Shape of the resulting sparse matrix.
    
    Returns
    -------
    Mat : scipy.sparse.csr_matrix
        Sparse matrix representation of the dictionary.
    """
    if len(Dict) == 0:
        return sparse.csr_matrix(shape)
    rows, cols, vals = dictToArrays(Dict)
    Mat = sparse.coo_matrix((vals, (rows, cols)), shape=shape)
    return Mat#.tocsr()


key_type = types.UniTuple(types.int64, 3)
value_type_splineInds = types.DictType(types.UniTuple(types.int64, 3), types.boolean)
value_type_bool = types.boolean
@njit(cache=True)
def getActiveBsplineInds(pgramOverlappedCellsInds, N, deg, Cij):
    """Determine the indices of the B-splines that overlap with any pgram-overlapped cell.
    
    Parameters
    ----------
    pgramOverlappedCellsInds : numba.typed.Dict
        Dictionary mapping velocity tuples to lists of pgram-overlapped cell indices.
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    deg : int
        Degree of the B-splines.
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
        
    Returns
    -------
    validSplineInds : numba.typed.Dict
        Dictionary with the indices of splines that overlap any pgram-overlapped cell as keys
    """
    
    D = Cij.shape[-1]

    validSplineInds = Dict.empty(key_type, value_type_splineInds)
    for i, Cj in enumerate(Cij.reshape(-1, D)): # Population index
        CjTuple = tuple3D(Cj)

        validSplineInds[CjTuple] = Dict.empty(key_type, value_type_bool)
        for poc in pgramOverlappedCellsInds[CjTuple]:

            for di in it_product(np.arange(-deg, 1), D): # Delta index within the spliness
                si = (di + np.array(poc)[:D]) % N # Spline index that overlaps poc
                siTuple = tuple3D(si)

                validSplineInds[CjTuple][siTuple] = True
    return validSplineInds