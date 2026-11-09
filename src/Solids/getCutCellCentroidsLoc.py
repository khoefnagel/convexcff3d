import numpy as np
from numba import njit, types
from numba.typed import Dict

key_type = types.UniTuple(types.int64, 3)
value_type = types.float64[:]
@njit(cache=True)
def getCutCellCentroidsLoc(cutCellMoments, D):
    '''Function to the centroids of the cut cells in local coordinates.
    
    Parameters
    ----------
    cutCellMoments : Dict
        A dictionary where keys are tuples representing the cell indices (ix, iy, iz)
        and values are 3D numpy arrays of shape (degMax+1, degMax+1, degMax+1) containing the computed
        fluid moments for each cut cell.
    D : int
        The number of spatial dimensions (1, 2, or 3).
        
    Returns
    -------
    cutCellCentroidsLoc : Dict
        A dictionary where keys are tuples representing the cell indices (ix, iy, iz)
        and values are 1D numpy arrays containing the local coordinates of the centroids
        of each cut cell.
    '''
    
    cutCellCentroidsLoc = Dict.empty(key_type, value_type)

    for ci in cutCellMoments.keys():
        M = cutCellMoments[ci]
        V = M[0,0,0]
        if V > 1e-12:
            divV = 1/V
            if D == 1:
                centroid = np.array([M[1,0,0]*divV])
            elif D == 2:
                centroid = np.array([M[1,0,0]*divV, M[0,1,0]*divV])
            elif D == 3:
                centroid = np.array([M[1,0,0]*divV, M[0,1,0]*divV, M[0,0,1]*divV])
            cutCellCentroidsLoc[ci] = centroid
        else:
            cutCellCentroidsLoc[ci] = np.zeros(3)[:D] + 0.5

    return cutCellCentroidsLoc