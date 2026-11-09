

import time

t1 = time.perf_counter()
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import shapely
import trimesh
from numba.typed import List
print(f"Time to import external packages: {time.perf_counter()-t1:.2f} s")

t1 = time.perf_counter()

from .getGridCutFacets import getGridCutFacets
from .getSubGridCutFacets import getSubGridCutFacets
from .getCutCellInds import getCutCellInds
from .getNormals import getNormals
from .filterGridCutFacets import filterGridCutFacets
from .getCellHashmap import getCellHashmap
from .getSubgridCutFacetMappings import getSubgridCutFacetMappings
from .getFacetAreas import getFacetAreas, getSubGridCutFacetAreas
from .getInputFacetsRaw import getInputFacetsRaw
from .SolidsPlotting import SolidsPlotting, SolidsPlotting1D, SolidsPlotting2D, SolidsPlotting3D
print(f"Time to import internal modules: {time.perf_counter()-t1:.2f} s")

class Solids(SolidsPlotting):

    def __init__(self, N, solidsObj, Cij, degs, degsCells, DEBUG=False):

        self.DEBUG = DEBUG  
        
        self.N = np.array(N).astype(np.int64)
        self.Cij = np.array(Cij).astype(np.int64)
        self.D = Cij.shape[-1]
        self.degs = np.array(degs).astype(np.int64)
        self.degsCells = np.array(degsCells).astype(np.int64)

        # 1. Get raw input facets and their mappings to solids, merging any overlapping geometries
        self.inputFacetsRaw, self.inputFacet2solid = getInputFacetsRaw(solidsObj, self.D)
        assert self.inputFacetsRaw.ndim == 3, "inputFacetsRaw must be a 3D array of shape (numFacets, numPointsPerFacet, D)"

        # 2. Get normals for input facets
        self.inputFacetNormals = getNormals(self.inputFacetsRaw, self.D)
        assert self.inputFacetNormals.shape == (len(self.inputFacetsRaw), self.D), "Normals shape mismatch"

        # 3. Cut the input facets on the grid lines to obtain grid cut facets.
        # Also obtain their mapping to the input facets.
        gridCutFacetsRaw, gridCutFacet2inputFacet = getGridCutFacets(self.inputFacetsRaw, self.D)

        # 4. Filter grid cut facets outside the domain
        filterMask, filteredGridCutFacets = filterGridCutFacets(gridCutFacetsRaw, N)
        self.gridCutFacetsRaw = filteredGridCutFacets
        self.gridCutFacet2inputFacet = gridCutFacet2inputFacet[filterMask]
        self.gridCutFacet2solid = self.inputFacet2solid[self.gridCutFacet2inputFacet]
        self.gridCutFacetNormals = self.inputFacetNormals[self.gridCutFacet2inputFacet]

        # 5. Get a mapping between each grid cut facet and the corresponding cell index.
        self.gridCutFacetCellInds = getCutCellInds(self.gridCutFacetsRaw, self.D)

        # 6. Cut the grid cut facets on the subgrid lines to obtain subgrid cut facets.
        # Also obtain their mapping to the grid cut facets.
        (
            self.subGridCutFacetsRaw, 
            self.subGridCutFacet2gridCutFacet
        ) = getSubGridCutFacets(self.gridCutFacetsRaw, self.gridCutFacetCellInds, self.Cij)

        # 7. Get mappings between the subgrid cut facets and the input facets, solids, normals, and cell indices.
        (
            self.subGridCutFacet2solid,
            self.subGridCutFacet2inputFacet, 
            self.subGridCutFacetNormals, 
            self.subGridCutFacetCellInds
        ) = getSubgridCutFacetMappings(
            self.subGridCutFacet2gridCutFacet, 
            self.gridCutFacet2inputFacet, 
            self.inputFacet2solid, 
            self.inputFacetNormals, 
            self.gridCutFacetCellInds, 
            self.Cij
        )

        # 8. Compute areas of input facets, grid cut facets, and subgrid cut facets.
        self.inputFacetAreas = getFacetAreas(self.inputFacetsRaw, self.D)
        self.gridCutFacetAreas = getFacetAreas(self.gridCutFacetsRaw, self.D)
        self.subGridCutFacetAreas = getSubGridCutFacetAreas(self.subGridCutFacetsRaw, self.Cij)

        # 9. Create a hash map (dictionary) of grid cut facets indexed by their cell indices.
        self.cellHashMap = getCellHashmap(self.gridCutFacetCellInds)



class Solids1D(Solids, SolidsPlotting1D):
    def __init__(self, N, solidsObj : list, Cij, degs, degsCells):
        
        # Check that solidsObj are line segments
        for solidObj in solidsObj:
            assert solidObj.geom_type == 'LineString', "All solidsObj must be LineString geometries."
            assert len(solidObj.coords) == 2, "Each LineString must have exactly two endpoints."
        self.solidsObj = solidsObj
        self.solidsUnion = shapely.ops.unary_union(self.solidsObj)


        super().__init__(N, self.solidsObj, Cij, degs, degsCells)



class Solids2D(Solids, SolidsPlotting2D):

    def __init__(self, N, solidsObj : list, Cij, degs, degsCells):
        #Check that solidsObj are shapely polygons
        for solid in solidsObj:
            assert isinstance(solid, shapely.geometry.Polygon), "All solidsObj must be shapely Polygons."

        #Ensure all polygons have same orientation (counter-clockwise)
        self.solidsObj = [shapely.geometry.polygon.orient(solidObj) for solidObj in solidsObj]
        self.solidsUnion = shapely.ops.unary_union(self.solidsObj)
        

        Solids.__init__(self, N, self.solidsObj, Cij, degs, degsCells)


        
class Solids3D(Solids, SolidsPlotting3D):

    def __init__(self, N, solidsObj : list, Cij, degs, degsCells):
        # Check that solidsObj are trimesh objects
        for solid in solidsObj:
            assert isinstance(solid, trimesh.Trimesh), "All solidsObj must be trimesh objects."
        
        # Ensure all trimesh objects have the same vertex format
        [solidObj.fix_normals() for solidObj in solidsObj]
        self.solidsObj = solidsObj
        self.solidsUnion = trimesh.boolean.union(self.solidsObj, engine=None)



        Solids.__init__(self, N, self.solidsObj, Cij, degs, degsCells)






