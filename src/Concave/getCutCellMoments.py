from numba import njit, types
from numba.typed import Dict


key_type = types.UniTuple(types.int64, 3)
value_type = types.float64[:,:,:]
@njit(cache=True)   
def getCutCellMoments(cellMoments, cellHashMap):
    '''Function to get the moments of the cut cells.
    
    Parameters
    ----------
    cellMoments : Dict
        A dictionary where keys are tuples representing the cell indices (ix, iy, iz)
        and values are 3D numpy arrays of shape (degMax+1, degMax+1, degMax+1) containing the computed
        fluid moments for each cut cell.
    D : int
        The number of spatial dimensions (1, 2, or 3).
        
    Returns
    -------
    cutCellMoments : Dict
        A dictionary where keys are tuples representing the cell indices (ix, iy, iz)
        and values are 3D numpy arrays containing the moments of each cut cell.
    '''

    cutCellMoments = Dict.empty(key_type=key_type, value_type=value_type)

    for ci in cellHashMap.keys():
        cutCellMoments[ci] = cellMoments[ci]

    return cutCellMoments



