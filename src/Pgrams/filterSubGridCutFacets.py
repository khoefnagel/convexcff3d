import numpy as np
from numba import njit, types
from numba.typed import Dict


key_type = types.UniTuple(types.int64, 3)
value_type = types.int64[:]
value_type2 = types.float64[:]

@njit(cache=True)
def filterSubGridCutFacets(
        pgram2subGridCutFacet,
        subGridCutFacet2solid, 
        subGridCutFacet2inputFacet,
        subGridCutFacet2gridCutFacet, 
        subGridCutFacetAreas
    ):
    """Function to obtain index mappings from pgrams to solid, input facet, and grid cut facet,
    as well as the facet areas.
    
    Parameters
    ----------
    pgram2subGridCutFacet : dict
        A dictionary mapping Cj tuples to lists of subgrid cut facet indices corresponding to each pgram.
    subGridCutFacet2solid : dict
        A dictionary mapping Cj tuples to arrays of solid indices corresponding to each subgrid cut facet.
    subGridCutFacet2inputFacet : dict
        A dictionary mapping Cj tuples to arrays of input facet indices corresponding to each subgrid cut facet.
    subGridCutFacet2gridCutFacet : dict
        A dictionary mapping Cj tuples to arrays of grid cut facet indices corresponding to each subgrid cut facet.
    subGridCutFacetAreas : dict
        A dictionary mapping Cj tuples to arrays of subgrid cut facet areas.
    
    Returns
    -------
    pgram2solid : dict
        A dictionary mapping Cj tuples to arrays of solid indices corresponding to each pgram.
    pgram2inputFacet : dict
        A dictionary mapping Cj tuples to arrays of input facet indices corresponding to each pgram.
    pgram2gridCutFacet : dict
        A dictionary mapping Cj tuples to arrays of grid cut facet indices corresponding to each pgram.
    pgramFacetAreas : dict
        A dictionary mapping Cj tuples to arrays of facet areas corresponding to each pgram.
    """

    pgram2solid = Dict.empty(key_type, value_type)
    pgram2inputFacet = Dict.empty(key_type, value_type)
    pgram2gridCutFacet = Dict.empty(key_type, value_type)
    
    pgramFacetAreas = Dict.empty(key_type, value_type2)

    # Loop over Cj
    for key in pgram2subGridCutFacet.keys():
        pgram2solid[key] = np.empty(len(pgram2subGridCutFacet[key]), dtype=np.int64)
        pgram2inputFacet[key] = np.empty(len(pgram2subGridCutFacet[key]), dtype=np.int64)
        pgram2gridCutFacet[key] = np.empty(len(pgram2subGridCutFacet[key]), dtype=np.int64)
        

        pgramFacetAreas[key] = np.empty(len(pgram2subGridCutFacet[key]), dtype=np.float64)
        for j, subGridCutFacetInd in enumerate(pgram2subGridCutFacet[key]):
            pgram2solid[key][j] = subGridCutFacet2solid[key][subGridCutFacetInd]
            pgram2inputFacet[key][j] = subGridCutFacet2inputFacet[key][subGridCutFacetInd]
            pgram2gridCutFacet[key][j] = subGridCutFacet2gridCutFacet[key][subGridCutFacetInd]
            
            pgramFacetAreas[key][j] = subGridCutFacetAreas[key][subGridCutFacetInd]

    return (
        pgram2solid,
        pgram2inputFacet,
        pgram2gridCutFacet,
        pgramFacetAreas
    )






