'''Cut the grid cut facets on the sub-grid to obtain sub-grid cut facets.
Note that the sub-grid depends on the population velocity Cj, so the cut is done for each Cj.'''

import numpy as np
from numba import njit
from numba.typed import List, Dict
from numba import types

from .splitOnPlane import splitOnPlane
from cff_utils.tuple3D import tuple3D


value_type_float2D = types.float64[:, :]
@njit(cache=True)
def clipSubGrid(gridCutFacets, cellInds, planeNormals):
    '''Function to cut grid cut facets on the sub-grid of their respective cell
    for one specific Cj.
    
    Parameters
    ----------
    gridCutFacets : List of np.ndarray
        List of grid cut facets, each of shape (numPointsPerFacet, D).
    cellInds : np.ndarray
        Array of shape (numGridCutFacets, 3) containing the cell indices of each grid cut facet.
    planeNormals : np.ndarray
        Array of shape (numPlanes, D) containing the normals of the cutting planes
        corresponding to the Cj.
        Note that each plane goes through the center of the respective cell.
        
    Returns
    -------
    subGridCutFacets : List of np.ndarray
        List of sub-grid cut facets, each of shape (numPointsPerFacet, D).
    subGridCutFacet2gridCutFacet : np.ndarray
        Array mapping each sub-grid cut facet to its corresponding grid cut facet index.
    '''

    # If there are no grid cut facets, return empty results
    if len(gridCutFacets) == 0:
        return gridCutFacets, np.empty(0, dtype=np.int64)

    D = gridCutFacets[0].shape[1] # Dimension

    subGridCutFacets = List.empty_list(value_type_float2D)
    subGridCutFacet2gridCutFacet = List()
    for fi, (facet, ci) in enumerate(zip(gridCutFacets, cellInds)): # Loop over grid cut facets

        # Initialize with the uncut grid cut facet
        split_polygons = List([facet])

        for normal in planeNormals: # Loop over cutting planes
            
            # The plane always goes through the center of the cell.
            # Note that ci is padded with zeros if D < 3, so slice it to D
            planePoint = ci[:D] + 0.5

            # Create a new list to store the new split polygons resulting from cutting
            split_polygons_new = List()

            for polygon in split_polygons: # Loop over existing split polygons and cut

                # Note: splitOnPlane pads the output polygons, they need to be sliced to
                # the correct number of points based on c1 and c2.
                poly1, c1, poly2, c2, _ = splitOnPlane(polygon, planePoint, normal)

                if c1 >= D: # Skip degenerate facets
                    split_polygons_new.append(poly1[:c1])
                if c2 >= D: # Skip degenerate facets
                    split_polygons_new.append(poly2[:c2])

            # Update the list of split polygons for the next cutting plane
            split_polygons = split_polygons_new

        # Now the input facet has been cut by cutting planes.
        # Add the resulting split polygons to the final list of sub-grid cut facets
        subGridCutFacets.extend(split_polygons)
        subGridCutFacet2gridCutFacet.extend([fi] * len(split_polygons))

    # Convert subGridCutFacet2gridCutFacet to a numpy array
    subGridCutFacet2gridCutFacetArr = np.empty(len(subGridCutFacets), dtype=np.int64)
    for i in range(len(subGridCutFacet2gridCutFacet)):
        subGridCutFacet2gridCutFacetArr[i] = subGridCutFacet2gridCutFacet[i]

    return subGridCutFacets, subGridCutFacet2gridCutFacetArr


# These must be declared outside the function!
key_type = types.UniTuple(types.int64, 3)
value_type_float2D = types.float64[:, :]
value_type_int1D = types.int64[:]
value_type_list_float2D = types.ListType(types.float64[:,:])

@njit(cache=True)
def getSubGridCutFacets(gridCutFacetsRaw, cellInds, Cij):
    '''Function to cut grid cut facets on the sub-grid of their respective cell for each Cj.

    Parameters
    ----------
    gridCutFacetsRaw : List of np.ndarray
        List of grid cut facets, each of shape (numPointsPerFacet, D).
    cellInds : np.ndarray
        Array of shape (numGridCutFacets, 3) containing the cell indices of each grid cut facet.
    Cij : np.ndarray
        Array of shape (Q, ..., D) containing the discrete velocity vectors.
    
    Returns
    -------
    subGridCutFacetsRaw : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a List of np.ndarray (value),
        each of shape (numPointsPerFacet, D), containing the sub-grid cut facets for that Cj.
    subGridCutFacet2gridCutFacet : Dict
        Dictionary mapping each Cj (key, tuple of 3 ints) to a np.ndarray (value),
        which in turn maps each sub-grid cut facet to its corresponding grid cut facet index.
    '''

    D = Cij.shape[-1] # Dimension


    # Manually create the plane normals for each Cj
    # Note that these are the same for any two Cj with the same abs(Cj).
    # Thus, we only compute the normals for each unique abs(Cj).
    planeNormalDict = Dict.empty(key_type, value_type_float2D)

    # 2D
    if D == 2:
        planeNormalDict[(1, 0, 0)] = np.array([[1., 0.]]) #x
        planeNormalDict[(0, 1, 0)] = np.array([[0., 1.]]) #y
        planeNormalDict[(1, 1, 0)] = np.array([[1., 0.], [0., 1.], [-1., 1.], [1., 1.]]) #xy

    # 3D
    elif D == 3:
        planeNormalDict[(1, 0, 0)] = np.array([[1., 0., 0.]]) #x
        planeNormalDict[(0, 1, 0)] = np.array([[0., 1., 0.]]) #y
        planeNormalDict[(0, 0, 1)] = np.array([[0., 0., 1.]]) #z
        planeNormalDict[(1, 1, 0)] = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 1., 0.], [1., 1., 0.]]) #xy
        planeNormalDict[(1, 0, 1)] = np.array([[1., 0., 0.], [0., 0., 1.], [-1., 0., 1.], [1., 0., 1.]]) #xz
        planeNormalDict[(0, 1, 1)] = np.array([[0., 1., 0.], [0., 0., 1.], [0., -1., 1.], [0., 1., 1.]]) #yz


    subGridCutFacetsRaw = Dict.empty(key_type, value_type_list_float2D)
    subGridCutFacet2gridCutFacet = Dict.empty(key_type, value_type_int1D)

    for Cj in Cij.reshape(-1, D):

        # Note that the sub-grid cut facets are the same for any two Cj with the same abs(Cj).
        # Thus, we only compute the cut for each unique abs(Cj) and store the results.

        CjAbs = np.abs(Cj)
        CjTuple = tuple3D(Cj)
        CjAbsTuple = tuple3D(CjAbs)

        if CjAbs.sum() == 0 or D == 1: #Still stream or 1D: sub-grid is the same as the grid.     
            subGridCutFacetsRaw[CjTuple] = gridCutFacetsRaw
            subGridCutFacet2gridCutFacet[CjTuple] = np.arange(len(gridCutFacetsRaw)).astype(np.int64)
            continue

        if CjAbsTuple not in subGridCutFacetsRaw: #Cutting not yet done for the absolute of this Cj
            planeNormals = planeNormalDict[CjAbsTuple]
            (
                subGridCutFacetsRaw[CjAbsTuple],
                subGridCutFacet2gridCutFacet[CjAbsTuple]
            ) = clipSubGrid(gridCutFacetsRaw, cellInds, planeNormals)

        # Copy the results for the absolute Cj to the actual Cj
        subGridCutFacetsRaw[CjTuple] = subGridCutFacetsRaw[CjAbsTuple]
        subGridCutFacet2gridCutFacet[CjTuple] = subGridCutFacet2gridCutFacet[CjAbsTuple]
        
    return subGridCutFacetsRaw, subGridCutFacet2gridCutFacet




