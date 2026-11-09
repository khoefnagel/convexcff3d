import pyvista as pv
import numpy as np


def pvPlotTriangles(triangles):
    assert isinstance(triangles, np.ndarray), "Input must be a numpy array"
    assert triangles.ndim == 3, "Input must be a 3D array of triangles"
    assert triangles.shape[1] == 3 and triangles.shape[2] == 3, "Input must be an array of triangles with shape (n, 3, 3)"

    # triangles: (N, 3, 3)
    N = triangles.shape[0]

    # 1. Flatten points to (N*3, 3)
    points = triangles.reshape(-1, 3)

    # 2. Build face array
    # Each face is: [3, i0, i1, i2]
    faces = np.hstack([
        np.full((N, 1), 3),
        np.arange(N * 3).reshape(N, 3)
    ]).astype(np.int64)

    # 3. Create PolyData
    mesh = pv.PolyData(points, faces)

    # Add mesh to plotter as: plotter.add_mesh(mesh)
    return mesh