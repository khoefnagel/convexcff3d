'''Script to compute the grid cut facets by cutting the input facets on the 
Cartesian grid planes.'''


import numpy as np
from numba import njit
from numba import types
from numba.typed import List

from .splitOnPlane import splitOnPlane


@njit(cache=True)
def getGridCutFacets(inputFacets, D):
    '''Function to compute the grid cut facets by cutting the input facets on the 
    Cartesian grid planes.
    
    Parameters
    ----------
    inputFacets : np.ndarray
        Array of shape (numFacets, numPointsPerFacet, D) containing the input facets.
    D : int
        Dimension of the space (1, 2, or 3).
    
    Returns
    -------
    gridCutFacets : List of np.ndarray
        List of grid cut facets, each of shape (numPointsPerFacet, D).
    gridCutFacet2inputFacet : np.ndarray
        Array mapping each grid cut facet to its corresponding input facet index.
    '''

    # In 1D, the grid cut facets are the same as the input facets
    if D == 1:
        return List([inputFacet for inputFacet in inputFacets]), np.arange(len(inputFacets)).astype(np.int64)

    gridCutFacets = List.empty_list(types.float64[:,:])
    gridCutFacet2inputFacet = List.empty_list(types.int64)
    for fi, facet in enumerate(inputFacets): # Loop over input facets

        # Initialize list of split facets with the uncut input facet
        split_polygons = List([facet])
        for axis in range(D): # Loop over each axis (x, y, z in 3D)

            # Normal of the cutting plane, e.g. (1,0,0) for cutting on the x-axis
            plane_normal = np.zeros(D)
            plane_normal[axis] = 1

            # Create a new list to store the new split polygons resulting from cutting
            split_polygons_new = List.empty_list(types.float64[:,:])
            for polygon in split_polygons: # Loop over existing split polygons and cut

                # Determine the grid planes to cut on based on the min and max coordinates of the polygon
                axis_coords = np.arange(np.ceil(polygon[:,axis].min() + 1e-12),
                                        np.ceil(polygon[:,axis].max() - 1e-12))
                
                # Points to define the cutting planes (one point for each grid plane)
                plane_points = axis_coords[:,None]*plane_normal

                # Loop over each cutting plane in increasing coordinate. The negative side of the cut
                # is saved in split_polygons_new, the positive side is further cut by the next planes.
                remaining_polygon = polygon
                for plane_point in plane_points:

                    # Note: splitOnPlane pads the output polygons, they need to be sliced to
                    # the correct number of points based on c1 and c2.
                    poly1, c1, poly2, c2, _ = splitOnPlane(remaining_polygon, plane_point, plane_normal)
                    split_polygon = poly1[:c1]
                    remaining_polygon = poly2[:c2]
                    if split_polygon.shape[0] >= D: # Skip degenerate facets
                        split_polygons_new.append(split_polygon)

                # Also add the remaining part after all cuts
                if remaining_polygon.shape[0] >= D: # Skip degenerate facets
                    split_polygons_new.append(remaining_polygon)
            
            # Update the list of split polygons for the next axis
            split_polygons = split_polygons_new

        # Now the input facet has been cut by all grid planes in all directions
        # Add the resulting split polygons to the final list of grid cut facets
        gridCutFacets.extend(split_polygons)
        gridCutFacet2inputFacet.extend([fi] * len(split_polygons))

    # Convert gridCutFacet2inputFacet to a numpy array
    gridCutFacet2inputFacetArr = np.empty(len(gridCutFacet2inputFacet), dtype=np.int64)
    for i in range(len(gridCutFacet2inputFacet)):
        gridCutFacet2inputFacetArr[i] = gridCutFacet2inputFacet[i]

    return gridCutFacets, gridCutFacet2inputFacetArr
