'''Get the grid cell indices of the cut cells, given the cut facets.
Note that the indices are always 3D, padding with zeros in case of D < 3.'''

import numpy as np
from numba import njit

@njit(cache=True)
def getCutCellInds(gridCutFacets, D):
    '''Function to get the grid cell indices of the cut cells.
    
    Parameters
    ----------
    gridCutFacets : List of np.ndarray
        List of grid cut facets, each of shape (numPointsPerFacet, D).
    D : int
        Dimension of the space (1, 2, or 3).

    Returns
    -------
    cutCellInds : np.ndarray
        Array of shape (numGridCutFacets, 3) containing the grid cell indices.
        In case of D < 3, the last indices are padded with zeros.    
    '''

    # Pad with zeros in case of D < 3
    cutCellInds = np.zeros((len(gridCutFacets), 3), dtype=np.int64)
    for i, facet in enumerate(gridCutFacets):
        # Determine in which cell the mean of the facet lies
        ci = np.floor(np.sum(facet, axis=0) / facet.shape[0]).astype(np.int64)
        cutCellInds[i,:D] = ci
    return cutCellInds