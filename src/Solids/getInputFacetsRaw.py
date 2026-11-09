'''Script to obtain the raw input facets from shapely/trimesh objects.
The raw input facets are numpy arrays of shape (numFacets, numPointsPerFacet, D),
where numPointsPerFacet=1 for D=1, 2 for D=2 and >=3 for D=3.'''

import numpy as np
import trimesh

def getInputFacetsRaw1D(solidsObj):
    '''Get raw input facets for 1D solids.
    
    Parameters
    ----------
    solidsObj : list of shapely LineString objects
        List of 1D solid objects.
    
    Returns
    -------
    inputFacetsRaw : np.ndarray, shape (N_facets, 1, 1)
        Array of raw input facets.
    inputFacet2solid : np.ndarray, shape (N_facets,)
        Array mapping each input facet to its solid index.'''

    inputFacetsRaw = []
    inputFacet2solid = []
    for solidInd, solidObj in enumerate(solidsObj):
        inputFacetsRaw.append(np.array(solidObj.coords)[:, 0].reshape(-1, 1, 1))
        inputFacet2solid.append(np.ones(len(solidObj.coords), dtype=np.int64) * solidInd)
    return np.concatenate(inputFacetsRaw, axis=0).astype(np.float64), np.concatenate(inputFacet2solid)

def getInputFacetsRaw2D(solidsObj):
    '''Get raw input facets for 2D solids.

    Parameters
    ----------
    solidsObj : list of shapely Polygon objects
        List of 2D solid objects.

    Returns
    -------
    inputFacetsRaw : np.ndarray, shape (N_facets, 2, 2)
        Array of raw input facets.
    inputFacet2solid : np.ndarray, shape (N_facets,)
        Array mapping each input facet to its solid index.'''

    inputFacetsRaw = []
    inputFacet2solid = []
    for solidInd, solidObj in enumerate(solidsObj):

        boundary = solidObj.boundary
        if boundary.geom_type == 'LineString':
            solidVerts = [np.array(boundary.coords)]
        elif boundary.geom_type == 'MultiLineString':
            solidVerts = [np.array(geom.coords) for geom in boundary.geoms]
        else: raise ValueError('Unsupported geometry type for solid boundary.')

        for verts in solidVerts:
            inputFacetsRaw.append(np.stack([verts[:-1], verts[1:]], axis=1))
            inputFacet2solid.append(np.ones(len(verts) - 1, dtype=np.int64) * solidInd)
    return np.concatenate(inputFacetsRaw, axis=0), np.concatenate(inputFacet2solid)

def getInputFacetsRaw3D(solidsObj):
    '''Get raw input facets for 3D solids.
    
    Parameters
    ----------
    solidsObj : list of trimesh Trimesh objects
        List of 3D solid objects.
    
    Returns
    -------
    inputFacetsRaw : np.ndarray, shape (N_facets, 3, 3)
        Array of raw input facets.
    inputFacet2solid : np.ndarray, shape (N_facets,)
        Array mapping each input facet to its solid index.'''
    
    triangles = np.array(trimesh.boolean.union(solidsObj).triangles)
    return triangles, np.zeros(len(triangles), dtype=np.int64)

def getInputFacetsRaw(solidsObj, D):
    '''Get raw input facets for solids in 1D, 2D or 3D.

    Parameters
    ----------
    solidsObj : list of shapely LineString/Polygon or trimesh Trimesh objects
        List of solid objects.
    D : int
        Dimension of the solids (1, 2 or 3).
    
    Returns
    -------
    inputFacetsRaw : np.ndarray, shape (N_facets, numPointsPerFacet, D)
        Array of raw input facets.
    inputFacet2solid : np.ndarray, shape (N_facets,)
        Array mapping each input facet to its solid index.'''
    
    if D == 1:
        return getInputFacetsRaw1D(solidsObj)
    elif D == 2:
        return getInputFacetsRaw2D(solidsObj)
    elif D == 3:
        return getInputFacetsRaw3D(solidsObj)
    else:
        raise ValueError("D must be 1, 2 or 3")


