import numpy as np
from numba import njit, types
from numba.typed import List, Dict

from cff_utils.tuple3D import tuple3D

@njit(cache=True, inline='always')
def getGridIntersections(vertices, Cj):
    """Function to obtain the grid intersections of a single pgram.
    
    Parameters
    ----------
    vertices : array_like
        An array of shape (N_vertices, D) representing the vertices of the pgram.
    Cj : array_like
        A 1D array of shape (D,) representing the lattice velocity vector.
    
    Returns
    -------
    intersections : array_like
        An array of shape (N_vertices, N_intersections, D) representing the grid intersections
        of the pgram along the direction of Cj.
    """
    
    # Initialize with nans
    nIntersections = np.count_nonzero(Cj) # One grid intersection for straight, two for diagonal
    intersections = np.empty((vertices.shape[0], nIntersections, vertices.shape[1]))
    intersections.fill(np.nan)

    for ai, axis in enumerate(np.nonzero(Cj)[0]):
        rint = np.rint(vertices[:, axis])
        onGrid = np.isclose(vertices[:, axis], rint, rtol=0., atol=1e-12)
        if np.all(onGrid):
            if np.all(rint == rint[0]):
                # All vertices are on the same grid plane, no intersections
                intersections[:, ai, :] = vertices
                continue
            else:
                # All vertices are on grid planes, but not the same one
                cutLoc = max(rint) if Cj[axis] < 0 else min(rint)
        
        # Not all vertices are on grid planes: use the first off-grid one to determine cut location
        else:
            ind = np.where(~onGrid)[0][0]
            cutLoc = np.ceil(vertices[ind, axis]) if Cj[axis] < 0 else np.floor(vertices[ind, axis])
        
        # Distances to the cutting plane along the axis
        dists = np.abs(vertices[:,axis] - cutLoc)

        # Compute the intersections
        intersections[:, ai, :] = vertices - dists[:,None]*Cj

    return intersections


@njit(cache=True, inline='always')
def getSingleSubzoneBounds(vertices, Cj, addReflected=True):
    """Function to obtain the subzone bounds for a single subgrid cut facet.
    
    Parameters
    ----------
    vertices : array_like
        An array of shape (N_vertices, D) representing the vertices of the subgrid cut
        facet.
    Cj : array_like
        A 1D array of shape (D,) representing the lattice velocity vector.
    addReflected : bool, optional
        Whether to add the reflected grid intersections, needed for CFF but not for
        Chen implementation (used for concave). Default is True.
    
    Returns
    -------
    subzoneBounds : array_like
        An array of shape (N_vertices, N_subzones+1, D) representing the bounds of each subzone.
    """

    #Grid intersections, shape (N_vertices, N_intersections, D)
    intersectionsUnsorted = getGridIntersections(vertices, Cj)

    nIntersections = intersectionsUnsorted.shape[1]

    if addReflected:
        nSubzones = nIntersections*2 + 1
        nInnerBounds = nIntersections*2
    else:
        nSubzones = nIntersections + 1
        nInnerBounds = nIntersections

    # Stack intersectionsUnsorted on first axis
    innerBoundsUnsorted = np.empty((vertices.shape[0], nInnerBounds, vertices.shape[1]))
    innerBoundsUnsorted[:, :nIntersections, :] = intersectionsUnsorted

    #Reflected grid intersections, shape (N_vertices, N_intersections, D)
    #Stack also on first axis
    if addReflected:
        crefl = 2*vertices[:,None,:] - Cj - intersectionsUnsorted
        innerBoundsUnsorted[:, nIntersections:, :] = crefl


    # Sort based on distance to facet along Cj
    dists = np.zeros(nInnerBounds)
    for i in range(nInnerBounds):
        for j in range(vertices.shape[0]):
            dists[i] += np.sum((vertices[j] - innerBoundsUnsorted[j, i]) * Cj)
    sortInds = np.argsort(dists)

    # Add the outer bounds (vertices and vertices - Cj)
    subzoneBounds = np.empty((vertices.shape[0], nSubzones+1, vertices.shape[1]))
    subzoneBounds[:,0,:] = vertices
    for i in range(len(sortInds)):
        subzoneBounds[:,i+1,:] = innerBoundsUnsorted[:,sortInds[i],:]
    subzoneBounds[:,-1,:] = vertices - Cj

    return subzoneBounds

@njit(cache=True)
def getSubzoneBoundsCj(subGridCutFacetsCj, subGridCutFacetNormalsCj, Cj, addReflected=True):
    """Function to obtain the subzone bounds for all subgrid cut facets for a single Cj.

    Parameters
    ----------
    subGridCutFacetsCj : list
        A list of arrays, each of shape (N_vertices, D), representing the vertices of the
        subgrid cut facets for a single Cj.
    subGridCutFacetNormalsCj : list
        An array of shape (N_facets, D), representing the normals of the subgrid cut facets
        for a single Cj.
    Cj : array_like
        A 1D array of shape (D,) representing the lattice velocity vector.
    addReflected : bool, optional
        Whether to add the reflected grid intersections, needed for CFF but not for
        Chen implementation (used for concave). Default is True.

    Returns
    -------
    subzoneBoundsCj : list
        A list of arrays, each of shape (N_vertices, N_subzones+1, D), representing the bounds of each subzone for a single Cj.
    pgram2subGridCutFacetCj : list
        A list of integers representing the indices of the subgrid cut facets corresponding to each pgram.
    subGridCutFacet2pgramCj : list
        A list of integers representing the indices of the pgrams corresponding to each subgrid cut facet.
        If a facet does not generate a pgram, its entry is -1.
    """

    subzoneBoundsCj = List.empty_list(types.float64[:,:,:])
    pgram2subGridCutFacetCj = List.empty_list(types.int64)
    subGridCutFacet2pgramCj = List.empty_list(types.int64)

    if tuple3D(Cj) == (0,0,0): # Skip still populations
        return subzoneBoundsCj, pgram2subGridCutFacetCj, subGridCutFacet2pgramCj

    pgramCount = 0
    for i, (facet, normal) in enumerate(zip(subGridCutFacetsCj, subGridCutFacetNormalsCj)):

        # Only construct pgrams that point out of the solid
        if np.sum(normal*Cj) > -1e-12:
            subGridCutFacet2pgramCj.append(-1)
            continue

        intersections = getSingleSubzoneBounds(facet, Cj, addReflected=addReflected)

        subzoneBoundsCj.append(intersections)
        pgram2subGridCutFacetCj.append(i)
        subGridCutFacet2pgramCj.append(pgramCount)
        pgramCount += 1


    return subzoneBoundsCj, pgram2subGridCutFacetCj, subGridCutFacet2pgramCj

key_type = types.UniTuple(types.int64, 3)
value_type = types.ListType(types.float64[:,:,:])
value_type2 = types.ListType(types.int64)
@njit(cache=True)
def getSubzoneBounds(subGridCutFacets, subGridCutFacetNormals, Cij, addReflected=True):
    """Function to obtain the subzone bounds for all subgrid cut facets
    
    Parameters
    ----------
    subGridCutFacets : Dict
        A dictionary mapping Cj tuples to lists of subgrid cut facet vertices arrays.
    subGridCutFacetNormals : Dict
        A dictionary mapping Cj tuples to array of facet normals.
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    addReflected : bool, optional
        Whether to add the reflected grid intersections, needed for CFF but not for
        Chen implementation (used for concave). Default is True.
        
    Returns
    -------
    subzoneBounds : Dict
        A dictionary mapping Cj tuples to lists of subzone bounds arrays.
    pgram2subGridCutFacet : Dict
        A dictionary mapping Cj tuples to lists of subgrid cut facet indices corresponding to each pgram.
    subGridCutFacet2pgram : Dict
        A dictionary mapping Cj tuples to lists of pgram indices corresponding to each subgrid cut facet.
        If a facet does not generate a pgram, its entry is -1.
    """

    D = Cij.shape[-1]
    
    subzoneBounds = Dict.empty(key_type, value_type)
    pgram2subGridCutFacet = Dict.empty(key_type, value_type2)
    subGridCutFacet2pgram = Dict.empty(key_type, value_type2)

    for Cj in Cij.reshape(-1, D):
        CjTuple = tuple3D(Cj)

        subzoneBoundsCj, pgram2subGridCutFacetCj, subGridCutFacet2pgramCj = getSubzoneBoundsCj(subGridCutFacets[CjTuple],
                                                                      subGridCutFacetNormals[CjTuple], 
                                                                      Cj, addReflected=addReflected)

        subzoneBounds[CjTuple] = subzoneBoundsCj
        pgram2subGridCutFacet[CjTuple] = pgram2subGridCutFacetCj
        subGridCutFacet2pgram[CjTuple] = subGridCutFacet2pgramCj

    return subzoneBounds, pgram2subGridCutFacet, subGridCutFacet2pgram

