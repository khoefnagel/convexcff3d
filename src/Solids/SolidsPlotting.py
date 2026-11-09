import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os

# if os.environ.get("DISPLAY") is not None:
#     pv.Plotter = lambda : None

class SolidsPlotting:

    def plotInputFacets(self, normals=False, cutCells=False, centroids=False, plotter=pv.Plotter(), color='lightgray', opacity=1.0):
        if normals:
            plotter = self._plotNormals(self.inputFacetsRaw, self.inputFacetNormals, plotter=plotter)
        if cutCells:
            plotter = self._plotCutCells(plotter=plotter)
        if centroids:
            plotter = self._plotCutCellCentroids(plotter=plotter)
        return self._plotFacets(self.inputFacetsRaw, plotter=plotter, color=color, opacity=opacity)

    def plotGridCutFacets(self, normals=False, cutCells=False, centroids=False, plotter=pv.Plotter(), color='lightgray', opacity=1.0):
        if normals:
            plotter = self._plotNormals(self.gridCutFacetsRaw, self.gridCutFacetNormals, plotter=plotter)
        if cutCells:
            plotter = self._plotCutCells(plotter=plotter)
        if centroids:
            plotter = self._plotCutCellCentroids(plotter=plotter)
        return self._plotFacets(self.gridCutFacetsRaw, plotter=plotter, color=color, opacity=opacity)

    def plotSubGridCutFacets(self, Cj, normals=False, cutCells=False, centroids=False, plotter=pv.Plotter(), color='lightgray', opacity=1.0):
        CjTuple = (*Cj, *[0]*(3 - len(Cj)))  # Pad Cj to length 3
        if normals:
            plotter = self._plotNormals(self.subGridCutFacetsRaw[CjTuple], self.subGridCutFacetNormals[CjTuple], plotter=plotter)
        if cutCells:
            plotter = self._plotCutCells(plotter=plotter)
        if centroids:
            plotter = self._plotCutCellCentroids(plotter=plotter)
        return self._plotFacets(self.subGridCutFacetsRaw[CjTuple], plotter=plotter, color=color, opacity=opacity)
    
    def plotCutCells(self, geoms=False, plotter=pv.Plotter()):
        if geoms:
            plotter = self._plotCutCellGeoms(plotter=plotter)
        return self._plotCutCells(plotter=plotter)
    
    def plotCellInds(self, cellInds, plotter=pv.Plotter()):
        return self._plotCellInds(cellInds, plotter=plotter)
    

class SolidsPlotting1D:
    def _plotFacets(self, facets, plotter=None, color='random', opacity=1.0):
        for i, facet in enumerate(facets):
            color_ = np.random.rand(3) if color == 'random' else color
            plt.axvline(facet.item(), lw=2, color=color_)
        plt.ylim([-0.5, 0.5])

    def _plotNormals(self, facets, normals, plotter=None):
        for facet, normal in zip(facets, normals):
            plt.arrow(facet.item(), 0, normal.item()*0.4, 0, head_width=0.03, head_length=0.04, fc='red', ec='red')
        plt.ylim([-0.5, 0.5])
    
    def _plotCutCells(self, plotter=None):
        cutCellInds = np.array(list(self.cutCellMoments.keys()))
        for cutCellInd in cutCellInds:
            square = np.array([[cutCellInd[0], -0.5],
                               [cutCellInd[0]+1, -0.5],
                               [cutCellInd[0]+1, 0.5],
                               [cutCellInd[0], 0.5]])
            plt.fill(square[:,0], square[:,1], color='lightblue', alpha=0.5, zorder=10, edgecolor='black')
        plt.ylim([-0.5, 0.5])
    
    def _plotCutCellCentroids(self, plotter=None):
        centroidsLoc = np.array(list(self.cutCellCentroidsLoc.values()))
        cellInds = np.array(list(self.cutCellCentroidsLoc.keys()))
        centroidsGlob = centroidsLoc + cellInds[:,:1] + 0.5
        plt.scatter(centroidsGlob[:,0], np.zeros(centroidsGlob.shape[0]), color='green', s=5, zorder=20)
        plt.ylim([-0.5, 0.5])

    def _plotCutCellGeoms(self, plotter=None):
        for cutCellGeom in self.cutCellGeoms.values():
            for lineCoords in cutCellGeom:
                rect = np.array([[lineCoords[0], -0.5],
                                 [lineCoords[1], -0.5],
                                 [lineCoords[1], 0.5],
                                 [lineCoords[0], 0.5]])
                plt.fill(rect[:,0], rect[:,1], color='orange', alpha=0.5, zorder=15, edgecolor='black')
        plt.ylim([-0.5, 0.5])

class SolidsPlotting2D:
    def _plotFacets(self, facets, plotter=None, color='random', opacity=1.0):

        for i, facet in enumerate(facets):
            color_ = np.random.rand(3) if color == 'random' else color

            plt.plot(facet[:,0], facet[:,1], color=color_)
        plt.gca().set_aspect('equal', adjustable='box')

    def _plotNormals(self, facets, normals, plotter=None):
        # Plot normals as arrows originating from the centroid of each facet
        centroids = np.empty((len(facets), facets[0].shape[1]))
        for i, facet in enumerate(facets):
            centroids[i] = facet.mean(axis=0)

        plt.quiver(centroids[:,0], centroids[:,1], normals[:,0], normals[:,1], color='red', scale=10)

    def _plotCutCells(self, plotter=None):
        cutCellInds = np.array(list(self.cutCellMoments.keys()))
        for cutCellInd in cutCellInds:
            square = np.array([[cutCellInd[0], cutCellInd[1]],
                               [cutCellInd[0]+1, cutCellInd[1]],
                               [cutCellInd[0]+1, cutCellInd[1]+1],
                               [cutCellInd[0], cutCellInd[1]+1]])
            plt.fill(square[:,0], square[:,1], color='lightblue', alpha=0.5, zorder=10, edgecolor='black')

    def _plotCutCellCentroids(self, plotter=None):
        centroidsLoc = np.array(list(self.cutCellCentroidsLoc.values()))
        cellInds = np.array(list(self.cutCellCentroidsLoc.keys()))
        centroidsGlob = centroidsLoc + cellInds[:,:2]
        plt.scatter(centroidsGlob[:,0], centroidsGlob[:,1], color='green', s=5, zorder=20)

    def _plotCutCellGeoms(self, plotter=None):
        for cutCellGeom in self.cutCellGeoms.values():
            for polyCoords in cutCellGeom:
                plt.fill(polyCoords[:,0], polyCoords[:,1], color='orange', alpha=0.5, zorder=15, edgecolor='black')

    def _plotCellInds(self, cellInds, plotter=None):
        for cellInd in cellInds:
            square = np.array([[cellInd[0], cellInd[1]],
                               [cellInd[0]+1, cellInd[1]],
                               [cellInd[0]+1, cellInd[1]+1],
                               [cellInd[0], cellInd[1]+1]])
            plt.fill(square[:,0], square[:,1], color='orange', alpha=0.5, zorder=15, edgecolor='black')


class SolidsPlotting3D:
    def _plotFacets(self, facets, plotter=pv.Plotter(), opacity=1.0, color='lightgray'):

        # Flatten the list of facets into a single mesh
        all_faces = []
        vertices = []
        vertex_offset = 0

        for facet in facets:
            vertices.extend(facet)
            # VTK expects faces in the format: [n_points, v0, v1, v2, ...]
            all_faces.append([facet.shape[0]] + list(range(vertex_offset, vertex_offset + facet.shape[0])))
            vertex_offset += facet.shape[0]

        vertices = np.array(vertices)
        faces = np.hstack(all_faces)

        # Create the mesh
        mesh = pv.PolyData(vertices, faces)

        # Create an interactive plotter
        plotter.add_mesh(mesh, color=color, show_edges=True, opacity=opacity)
        plotter.show_grid()

        return plotter
    
    def _plotNormals(self, facets, normals, plotter=pv.Plotter()):
        # Plot normals as arrows originating from the centroid of each facet
        centroids = np.empty((len(facets), facets[0].shape[1]))
        for i, facet in enumerate(facets):
            centroids[i] = facet.mean(axis=0)

        pdata = pv.PolyData(centroids)
        pdata["vectors"] = normals
        plotter.add_arrows(centroids, normals, mag=0.2, color="red")
        return plotter
    
    def _plotCutCells(self, plotter=pv.Plotter()):
        cube = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
        centers = np.array(list(self.cutCellMoments.keys())) + 0.5
        points = pv.PolyData(centers)
        glyphs = points.glyph(scale=False, geom=cube)
        plotter.add_mesh(glyphs, color="lightblue", opacity=0.5, show_edges=True)
        return plotter
    
    def _plotCellInds(self, cellInds, plotter=pv.Plotter()):
        cube = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
        centers = np.array(cellInds) + 0.5
        points = pv.PolyData(centers)
        glyphs = points.glyph(scale=False, geom=cube)
        plotter.add_mesh(glyphs, color="orange", opacity=0.5, show_edges=True)
        return plotter




