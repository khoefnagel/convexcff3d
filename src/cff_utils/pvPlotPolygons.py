import numpy as np

from .pvPlotTriangles import pvPlotTriangles


def pvPlotPolygons(polygons):
    '''Plot a list of convex polygons using pyvista.'''

    triangles = []
    for poly in polygons:
        # Triangulate polygon
        for j in range(1, len(poly)-1):
            triangles.append(np.array([poly[0], poly[j], poly[j+1]]))
    triangles = np.array(triangles).reshape((-1,3,3))

    return pvPlotTriangles(triangles)
    
