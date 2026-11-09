import numpy as np
import pyvista as pv
import time
import matplotlib.pyplot as plt

from Solids.Solids import Solids1D, Solids2D, Solids3D

from .getSubzoneBounds import getSubzoneBounds
from .getSubzoneMoments import getSubzoneMoments
from .filterSubGridCutFacets import filterSubGridCutFacets

class Pgrams:

    def __init__(self, deg):
        
        # 1. Obtain the bounds of the subzones within each pgram.
        # Also obtain the mapping between each subgrid cut facet and the corresponding pgram.
        subzoneBounds, self.pgram2subGridCutFacet, self.subGridCutFacet2pgram = getSubzoneBounds(self.subGridCutFacetsRaw, self.subGridCutFacetNormals, self.Cij)
        self.subzoneBounds = subzoneBounds


        # 2. Obtain mappings from pgrams to solids, input facets, grid cut facets, and subgrid cut facet areas.
        (
            self.pgram2solid,
            self.pgram2inputFacet,
            self.pgram2gridCutFacet, 
            self.pgramFacetAreas
        ) = filterSubGridCutFacets(
            self.pgram2subGridCutFacet,
            self.subGridCutFacet2solid, 
            self.subGridCutFacet2inputFacet,
            self.subGridCutFacet2gridCutFacet, 
            self.subGridCutFacetAreas
        )

        # 3. Compute the moments of the subzones within each pgram.
        # Also find the cell index where each subzone is located and
        # the cell index of the corresponding subzone that it reflects and streams to.
        (
            self.subzoneMoments,
            self.origCellInds, 
            self.destCellInds
        ) = getSubzoneMoments(subzoneBounds, self.Cij, self.N, self.degsCells, np.max(self.degsCells))



    def plotPgrams(self, Cj, subzones=False, overlapped=False, plotter=pv.Plotter()):
        if subzones:
            plotter = self._plotSubzones(Cj, plotter=plotter)
        if overlapped:
            plotter = self._plotPgramOverlappedCells(Cj, plotter=plotter)
        return self._plotPgrams(Cj, plotter=plotter)

class Pgrams1D(Pgrams, Solids1D):
    def __init__(self, N, solids : list, Cij, degs, degsCells, deg):
        Solids1D.__init__(self, N, solids, Cij, degs, degsCells)
        Pgrams.__init__(self, deg)
    
    def _plotPgrams(self, Cj, plotter=None):
        self.plotSubGridCutFacets(Cj)

        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3

        # Cj aligned lines
        for pgram in self.subzoneBounds[CjTuple]:
            plt.plot([pgram[0,0,0], pgram[0,-1,0], pgram[0,-1,0], pgram[0,0,0]],
                     [-0.4, -0.4, 0.4, 0.4], 'b', linestyle='dashed')
        plt.ylim(-0.5, 0.5)
    
    def _plotSubzones(self, Cj, plotter=None):
        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3

        for pgram in self.subzoneBounds[CjTuple]:
            for i in range(1, pgram.shape[1]):
                plt.fill(pgram[[0,0,0,0],[i-1, i, i, i-1],0].flatten(),
                         [-0.4, -0.4, 0.4, 0.4], 
                         color='cyan', alpha=0.5, edgecolor='black')        

    def _plotPgramOverlappedCells(self, Cj, plotter=None):
        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3

        pgramOverlappedCellInds = self.pgramOverlappedCellInds[CjTuple]
        if len(pgramOverlappedCellInds) == 0:
            return

        for poci in pgramOverlappedCellInds:
            square = np.array([[poci[0], -0.5],
                               [poci[0]+1, -0.5],
                               [poci[0]+1, 0.5],
                               [poci[0], 0.5]])
            plt.fill(square[:,0], square[:,1], color='red', alpha=0.3, edgecolor='black')

        return

class Pgrams2D(Pgrams, Solids2D):

    def __init__(self, N, solids : list, Cij, degs, degsCells, deg):
        Solids2D.__init__(self, N, solids, Cij, degs, degsCells)
        Pgrams.__init__(self, deg)

    def _plotPgrams(self, Cj, plotter=None):

        self.plotSubGridCutFacets(Cj)

        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3

        # Cj aligned lines
        for pgram in self.subzoneBounds[CjTuple]:
            plt.plot(pgram[:,[0,-1],0].T, pgram[:,[0,-1],1].T, 'b', linestyle='dashed')
            plt.plot(pgram[:,-1,0], pgram[:,-1,1], 'b', linestyle='dashed')
        plt.show()

    def _plotSubzones(self, Cj, plotter=None):
        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3

        for pgram in self.subzoneBounds[CjTuple]:
            for i in range(1, pgram.shape[1]):
                plt.fill(pgram[[0,0,1,1],[i-1, i, i, i-1],0].flatten(),
                         pgram[[0,0,1,1],[i-1, i, i, i-1],1].flatten(), 
                         color='cyan', alpha=0.5, edgecolor='black')

        return
    
    def _plotPgramOverlappedCells(self, Cj, plotter=None):
        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3

        pgramOverlappedCellInds = self.pgramOverlappedCellInds[CjTuple]
        if len(pgramOverlappedCellInds) == 0:
            return

        for poci in pgramOverlappedCellInds:
            square = np.array([[poci[0], poci[1]],
                               [poci[0]+1, poci[1]],
                               [poci[0]+1, poci[1]+1],
                               [poci[0], poci[1]+1]])
            plt.fill(square[:,0], square[:,1], color='red', alpha=0.3, edgecolor='black')          

        return

class Pgrams3D(Pgrams, Solids3D):

    def __init__(self, N, solids : list, Cij, degs, degsCells, deg):
        Solids3D.__init__(self, N, solids, Cij, degs, degsCells)
        Pgrams.__init__(self, deg)


    def _plotPgrams(self, Cj, plotter=pv.Plotter()):

        plotter = self.plotSubGridCutFacets(Cj, plotter=plotter)
    
        # Cj aligned lines
        verts = []
        for pgram in self.subzoneBounds[tuple(Cj)]:
            verts.append(pgram[:,[0,-1],:].reshape(-1,3))
        plotter.add_lines(np.vstack(verts), color="blue", width=1)

        # Lines at the end of the pgram
        verts2 = []
        for pgram in self.subzoneBounds[tuple(Cj)]:
            n_vertices = pgram.shape[0]
            for j in range(n_vertices):
                verts2.append(pgram[j, -1, :])
                verts2.append(pgram[(j+1)%n_vertices, -1, :])
        plotter.add_lines(np.vstack(verts2), color="green", width=1)

        return plotter

    def _plotSubzones(self, Cj, plotter=pv.Plotter()):

        # Collect all points and face connectivity
        all_points = []
        faces = []
        offset = 0

        for pgram in self.subzoneBounds[tuple(Cj)][::10]:
            # Each pgram is a list of polygons (2D faces in 3D space)
            polygons = [pgram[:,i,:] for i in range(1,pgram.shape[1])]
            for poly in polygons:
                n = len(poly)
                all_points.append(poly)
                # Connectivity: [n, offset, offset+1, ..., offset+n-1]
                faces.append(np.hstack([[n], np.arange(offset, offset + n)]))
                offset += n

        # Stack into arrays
        all_points = np.vstack(all_points)
        faces = np.hstack(faces)

        # Create a single PolyData with multiple polygons
        mesh = pv.PolyData(all_points, faces)

        # Plot
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="cyan", opacity=0.5, show_edges=True)

        return plotter