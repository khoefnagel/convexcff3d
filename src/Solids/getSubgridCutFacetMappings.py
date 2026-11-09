'''Get variables related to sub-grid cut facets from the associated grid cut facets.
Specifically, the mapping to the solid index, input facet index, normals, and cell index.'''

import numpy as np
from numba import njit, types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D

# These must be defined outside the function!
key_type = types.UniTuple(types.int64, 3)
value_type_int1D = types.int64[:]
value_type_int2D = types.int64[:,:]
value_type_float2D = types.float64[:,:]

@njit(cache=True)
def getSubgridCutFacetMappings(
    subGridCutFacet2gridCutFacet, 
    gridCutFacet2inputFacet, 
    inputFacet2solid, 
    inputFacetNormals, 
    gridCutFacetCellInds, 
    Cij):
    '''Function to get variables related to sub-grid cut facets from the associated grid cut facets.
    Specifically, the mapping to the solid index, input facet index, normals, and cell index.
    
    Parameters
    ----------
    subGridCutFacet2gridCutFacet : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        which in turn maps each sub-grid cut facet to its corresponding grid cut facet index.
    gridCutFacet2inputFacet : np.ndarray
        Array of shape (numGridCutFacets,) mapping each grid cut facet to its corresponding 
        input facet index.
    inputFacet2solid : np.ndarray
        Array of shape (numInputFacets,) mapping each input facet to its corresponding solid index.
    inputFacetNormals : np.ndarray
        Array of shape (numFacets, D) containing the unit normals of the input facets.
    gridCutFacetCellInds : np.ndarray
        Array of shape (numGridCutFacets, 3) containing the index of the cell in which each
        grid cut facet resides.
    Cij : np.ndarray
        Array of shape (Q, ..., D) containing the discrete velocity vectors Cj.

    Returns
    -------
    subGridCutFacet2solid : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        which in turn maps each sub-grid cut facet to its corresponding solid index.
    subGridCutFacet2inputFacet : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        which in turn maps each sub-grid cut facet to its corresponding input facet index.
    subGridCutFacetNormals : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        which in turn stores the unit normal vectors of the sub-grid cut facets.
    subGridCutFacetCellInds : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        which in turn stores the index of the cell in which each sub-grid cut facet resides.
    '''


    D = Cij.shape[-1] # Dimension

    subGridCutFacet2solid = Dict.empty(key_type, value_type_int1D)
    subGridCutFacet2inputFacet = Dict.empty(key_type, value_type_int1D)
    subGridCutFacetNormals = Dict.empty(key_type, value_type_float2D)
    subGridCutFacetCellInds = Dict.empty(key_type, value_type_int2D)
    

    for Cj in Cij.reshape(-1, D):

        # Pad Cj to 3D with zeros if D < 3
        CjTuple = tuple3D(Cj)
    
        subGridCutFacet2inputFacet[CjTuple] = gridCutFacet2inputFacet[subGridCutFacet2gridCutFacet[CjTuple]]
        subGridCutFacetNormals[CjTuple] = inputFacetNormals[subGridCutFacet2inputFacet[CjTuple]]
        subGridCutFacetCellInds[CjTuple] = gridCutFacetCellInds[subGridCutFacet2gridCutFacet[CjTuple]]
        subGridCutFacet2solid[CjTuple] = inputFacet2solid[subGridCutFacet2inputFacet[CjTuple]]

    return (
        subGridCutFacet2solid,
        subGridCutFacet2inputFacet,
        subGridCutFacetNormals, 
        subGridCutFacetCellInds
    )

