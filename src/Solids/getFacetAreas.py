''' Get the areas of facets in various dimensions.
In 1D the area is always 1.0.'''

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from cff_utils.tuple3D import tuple3D

@njit(cache=True)
def computeFacetArea1D(facet):
    ''' Function to compute the area of a 1D facet, 
    since it is just a point, we return area 1.0.

    Parameters
    ----------
    facet : np.ndarray
        A 1D facet represented by a single point, shape (1, 1).
    
    Returns
    -------
    area : float
        The area of the 1D facet, which is always 1.0.
    '''
    return 1.0

@njit(cache=True)
def computeFacetArea2D(facet):
    ''' Compute the length of a line segment in 2D 
    
    Parameters
    ----------
    facet : np.ndarray
        A 2D facet represented by two points, shape (2, 2).
    
    Returns
    -------
    area : float
        The area of the 2D facet, which is the length of the line segment.
    '''
    return np.sqrt((facet[0][0] - facet[1][0])**2 + (facet[0][1] - facet[1][1])**2)

@njit(cache=True)
def computeFacetArea3D(facet):
    ''' Compute the area of a planar polygon in 3D using the shoelace formula 
    
    Parameters
    ----------
    facet : np.ndarray
        A 3D facet which is a planar polygon, represented by multiple points, shape (numPoints, 3).
    
    Returns
    -------
    area : float
        The area of the 3D facet, which is computed using the shoelace formula.
    '''
    area = np.zeros(3, dtype=np.float64)
    for i in range(len(facet)):
        j = (i + 1) % len(facet)
        area += np.cross(facet[i], facet[j])
    return 0.5 * np.linalg.norm(area)


@njit(cache=True)
def getFacetAreas(facets, D):
    ''' Function to compute the areas of an array of facets. Works for 1D, 2D, and 3D facets.

    Parameters
    ----------
    facets : np.ndarray
        An array of facets, where each facet is represented by its vertices,
        shape (numFacets, numPointsPerFacet, D).
    D : int
        The dimension of the facets (1, 2, or 3).

    Returns
    -------
    facetAreas : np.ndarray
        An array of the computed areas for each facet.
    '''

    facetAreas = np.empty(len(facets), dtype=np.float64)

    for i, facet in enumerate(facets):
        if D == 1:
            facetArea = computeFacetArea1D(facet)
        elif D == 2:
            facetArea = computeFacetArea2D(facet)
        elif D == 3:
            facetArea = computeFacetArea3D(facet)
        else:
            raise ValueError("Dimension not supported")

        facetAreas[i] = facetArea

    return facetAreas


key_type = types.UniTuple(types.int64, 3)
value_type = types.float64[:]
@njit(cache=True)
def getSubGridCutFacetAreas(subGridCutFacetsRaw, Cij):
    ''' Function to compute the areas of sub-grid cut facets.
    
    Parameters
    ----------
    subGridCutFacetsRaw : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a List of np.ndarray (value),
        each of shape (numPointsPerFacet, D), containing the sub-grid cut facets for that Cj.
    Cij : np.ndarray
        Array of shape (Q, ..., D) containing the discrete velocity vectors.
        
    Returns
    -------
    subGridCutFacetAreas : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        containing the areas of the sub-grid cut facets for that Cj.
    '''

    D = Cij.shape[-1]

    subGridCutFacetAreas = Dict.empty(key_type, value_type)

    for Cj in Cij.reshape(-1, D):        
        CjTuple = tuple3D(Cj)
        subGridCutFacetAreas[CjTuple] = getFacetAreas(subGridCutFacetsRaw[CjTuple], D)
    
    return subGridCutFacetAreas



