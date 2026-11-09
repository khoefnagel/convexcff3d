import pyvista as pv
import numpy as np

from ..derived.get_surface_force import get_surface_force_factory

def write_surface_forces_factory(ctx, filename_template, vertices, faces):
    """Create a function to write surface forces to a .vtp file at specified timestep."""

    D = ctx.D
    use_cuda = ctx.use_cuda

    # Function to compute the surface force
    get_surface_force = get_surface_force_factory(ctx)

    # Create pyvista PolyData object for the surface mesh
    surface_mesh = pv.PolyData(vertices, faces)
    surface_mesh.cell_data['force'] = np.zeros((surface_mesh.n_faces, D))

    def write_surface_forces(F, t):
        """Write the surface forces to a .vtp file at the given timestep.

        Args:
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            t: Current timestep.
        """

        # Generate the filename for the current timestep using the provided template.
        filename = filename_template.format(t)

        # Compute the surface forces
        surface_mesh.cell_data['force'][:] = get_surface_force(F).copy_to_host().reshape(-1, D) if use_cuda else get_surface_force(F).reshape(-1, D)

        # Save the surface mesh with the computed forces to a .vtp file
        surface_mesh.save(filename + '.vtp')

    return write_surface_forces


