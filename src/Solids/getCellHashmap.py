'''Create a hash map (dictionary) of grid cut facets indexed by their cell indices.
This is later used to efficiently access grid cut facets for each cell.'''

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from cff_utils.tuple3D import tuple3D


key_type = types.UniTuple(types.int64, 3)
value_type = types.ListType(types.int64)
@njit(cache=True)
def getCellHashmap(cutCellInds):
    """Create a hash map (dictionary) of grid cut facets indexed by their cell indices.
    This is later used to efficiently access grid cut facets for each cell.
    
    Parameters
    ----------
    cellInds : np.ndarray
        Array of shape (N, 3) containing the cell indices of each grid cut facet.
        If D < 3, the missing dimensions are padded with zeros.
    
    Returns
    -------
    cellHashmap : Dict
        A dictionary where keys are tuples of cell indices (i, j, k) and values are
        lists of grid cut facet indices in the corresponding cells.
    """

    cellHashmap = Dict.empty(key_type, value_type)
    
    for gcfi, cellInd in enumerate(cutCellInds):
        cellIndTuple = tuple3D(cellInd)
        if cellIndTuple in cellHashmap:
            cellHashmap[cellIndTuple].append(gcfi)
        else:
            cellHashmap[cellIndTuple] = List([gcfi])
    return cellHashmap    

