'''Script to filter out grid cut facets that are either completely outside
 or on the edge of the domain.'''

import numpy as np
from numba import njit
from numba.typed import List
from numba import types

@njit(cache=True)
def filterGridCutFacets(gridCutFacets, N):
    '''Function to filter out grid cut facets that are either completely outside
    or on the edge of the domain.
    
    Parameters
    ----------
    gridCutFacets : List of float64[:,:]
        List of grid cut facets, where each facet is represented by an (numVertices, D)
        array of its vertices.
    N : array_like, shape (D,)
        Number of grid points in each dimension.

    Returns
    -------
    filterMask : ndarray, shape (numGridCutFacets,)
        Boolean mask indicating which grid cut facets are kept (True) or filtered out (False).
    filteredGridCutFacets : List of float64[:,:]
        List of filtered grid cut facets that are inside the domain.    
    '''

    filteredGridCutFacets = List.empty_list(types.float64[:,:])
    N = np.asarray(N)
    filterMask = np.zeros(len(gridCutFacets), dtype=np.bool_)

    for i, facet in enumerate(gridCutFacets):
        
        # Compute the mean of the facet's vertices
        mean = np.zeros(facet.shape[-1])
        for vert in facet:
            mean += vert
        mean /= len(facet)

        # Check if the mean is strictly inside the domain
        if ((1e-12 < mean) & (mean < N - 1e-12)).all(): 
            filterMask[i] = True
            filteredGridCutFacets.append(facet)

    return filterMask, filteredGridCutFacets