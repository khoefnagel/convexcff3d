'''Script to compute normals of facets in 1D, 2D and 3D.
Returns array of shape (numFacets, D) with unit normals.'''

import numpy as np
from numba import njit

@njit(cache=True)
def getNormals1D(facets):
    '''Function to compute normals of 1D facets (points).
    It is assumed that two consecutive facets form a solid (line segment).
    The normal of the first facet points away from the second facet and vice versa.
    
    Parameters
    ----------
    facets : np.ndarray
        Array of shape (numFacets, 1, 1) containing the coordinates of the facets.
    
    Returns
    -------
    normals : np.ndarray
        Array of shape (numFacets, 1) containing the unit normals of the facets.
    '''

    assert facets.shape[-2] == facets.shape[-1] == 1, "1D facets must have shape (numFacets, 1, 1)"
    
    normals = np.zeros((len(facets), 1))
    for j in range(len(facets)//2):
        if facets[2*j] > facets[2*j+1]:
            normals[2*j] = 1.0
            normals[2*j+1] = -1.0
        else:
            normals[2*j] = -1.0
            normals[2*j+1] = 1.0
    return normals


@njit(cache=True)
def getNormals2D(facets):
    '''Function to compute normals of 2D facets (line segments).
    The normal is computed using the right-hand rule: if the segment is traversed from 
    the first point to the second point, the normal points to the right.
    
    Parameters
    ----------
    facets : np.ndarray
        Array of shape (numFacets, 2, 2) containing the coordinates of the facets.
    
    Returns
    -------
    normals : np.ndarray
        Array of shape (numFacets, 2) containing the unit normals of the facets
    '''

    normals = np.zeros((len(facets), 2))
    for i, facet in enumerate(facets):
        #Angle of segment: centered on self.start and measured ccw w.r.t. x-axis
        angle = np.arctan2(facet[1][1] - facet[0][1], facet[1][0] - facet[0][0])
        normals[i] = (np.sin(angle), -np.cos(angle))    
    return normals


@njit(cache=True)
def getNormals3D(facets):
    '''Function to compute normals of 3D facets (triangles).
    The normal is computed using the right-hand rule: if the triangle is traversed from
    the first point to the second point to the third point, the normal points outwards.

    Parameters
    ----------
    facets : np.ndarray
        Array of shape (numFacets, 3, 3) containing the coordinates of the facets.
    
    Returns
    -------
    normals : np.ndarray
        Array of shape (numFacets, 3) containing the unit normals of the facets
    '''

    normals = np.zeros((len(facets), 3))

    for i, facet in enumerate(facets):
        # Edge vectors
        v1 = facet[1] - facet[0]
        v2 = facet[2] - facet[0]

        # Cross product
        normal = np.array([
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        ])

        # Normalize
        norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if norm > 0.0:
            normal /= norm
        normals[i] = normal
    
    return normals


def getNormals(facets, D):
    '''Function to compute normals of facets in 1D, 2D or 3D.
    
    Parameters
    ----------
    facets : np.ndarray
        Array of shape (numFacets, numPointsPerFacet, D) containing the coordinates of the facets.
    D : int
        Dimension of the space (1, 2 or 3).
    
    Returns
    -------
    normals : np.ndarray
        Array of shape (numFacets, D) containing the unit normals of the facets.
    '''
    
    if D == 1:
        return getNormals1D(facets)
    elif D == 2:
        return getNormals2D(facets)
    elif D == 3:
        return getNormals3D(facets)
    else:
        raise ValueError("D must be 1,2 or 3")