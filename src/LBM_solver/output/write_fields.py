import numpy as np
import pyvista as pv

from ..solver_util.get_slice_inds import get_slice_inds

def write_fields_factory(ctx, filename_template, fields, slices_ijk, write_cells=True):
    """Create a function that writes the specified fields to .vti files.
    
    Args:
        ctx: Context object containing simulation parameters and models.
        filename_template: Template for the output filename, should include a placeholder for the timestep.
        fields: List of field names to write.
        slices_ijk: Tuple of slices or integers for each dimension (x, y, z) specifying the region of the domain to write.
        write_cells: Boolean indicating whether to write cell data or point data, default is True.
    """

    float_t = ctx.float_t
    D = ctx.D
    Q = ctx.Q
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    # Get the size of the sliced sub-domain and its origin.
    n_, origin = get_slice_inds(ctx, slices_ijk)

    # Create a PyVista grid for the sliced sub-domain.
    origin = origin if write_cells else np.array(origin) + 0.5
    dimensions = np.array(n_) + 1 if write_cells else n_
    grid = pv.ImageData(
        dimensions=dimensions,
        spacing=(1.0, 1.0, 1.0),
        origin=origin,
    )
    data = grid.cell_data if write_cells else grid.point_data

    # Dictionary to hold the functions that compute the specified field, with the field name as the key.
    get_field_functions = {}

    # Dictionary to hold the device data for each field, with the field name as the key. Only used if use_cuda is True.
    device_data = {}

    # Dictionary to hold the dimension of each field, with the field name as the key.
    dims = {}
    for field in fields:

        match field:
            case "rho":
                from ..derived.get_rho import get_rho_factory
                data["rho"] = np.empty(np.prod(n_), dtype=float_t)
                get_field_functions["rho"] = get_rho_factory(ctx, slices_ijk)
                dims["rho"] = 1
                if use_cuda:
                    device_data["rho"] = cuda.to_device(data["rho"].reshape((1, *n_, 1)))
            case "u":
                from ..derived.get_u import get_u_factory
                data["u"] = np.empty((np.prod(n_), D), dtype=float_t)
                get_field_functions["u"] = get_u_factory(ctx, slices_ijk)
                dims["u"] = D
                if use_cuda:
                    device_data["u"] = cuda.to_device(data["u"].reshape((1, *n_, D)))
            case "F":
                from ..derived.get_F_q import get_F_q_factory
                for q in range(Q):
                    data[f"F_{q}"] = np.empty(np.prod(n_), dtype=float_t)
                    get_field_functions[f"F_{q}"] = get_F_q_factory(ctx, q, slices_ijk)
                    dims[f"F_{q}"] = 1
                    if use_cuda:
                        device_data[f"F_{q}"] = cuda.to_device(data[f"F_{q}"].reshape((1, *n_, 1)))
            case "V": # Cell volume
                from ..derived.get_volume import get_volume_factory
                data["V"] = np.empty(np.prod(n_), dtype=float_t)
                get_field_functions["V"] = get_volume_factory(ctx, slices_ijk)
                dims["V"] = 1
                if use_cuda:
                    device_data["V"] = cuda.to_device(data["V"].reshape((1, *n_, 1)))
            case "c": # Cell centroid
                from ..derived.get_centroid import get_centroid_factory
                data["c"] = np.empty((np.prod(n_), D), dtype=float_t)
                get_field_functions["c"] = get_centroid_factory(ctx, slices_ijk)
                dims["c"] = D
                if use_cuda:
                    device_data["c"] = cuda.to_device(data["c"].reshape((1, *n_, D)))
            case "wall_dist": # Distance to wall
                from ..derived.get_wall_dist import get_wall_dist_factory
                data["wall_dist"] = np.empty((np.prod(n_), 1), dtype=float_t)
                get_field_functions["wall_dist"] = get_wall_dist_factory(ctx, slices_ijk)
                dims["wall_dist"] = 1
                if use_cuda:
                    device_data["wall_dist"] = cuda.to_device(data["wall_dist"].reshape((1, *n_, 1)))

    def write_fields(F, t):
        """Write the specified fields to a .vti file at the given timestep.
        
        Args:
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            t: Current timestep.
        """

        # Generate the filename for the current timestep using the provided template.
        filename = filename_template.format(t)

        for field in get_field_functions.keys():

            # GPU code
            if use_cuda:
                # Compute the field on the device
                get_field_functions[field](F, device_data[field], t)

                # Copy the computed field to the host and reshape
                data[field][:] = device_data[field].copy_to_host().reshape(data[field].shape, order="F")

            # CPU code
            else:
                # Compute the field on the host and reshape
                get_field_functions[field](F, data[field].reshape((1, *n_, dims[field]), order="F"), t)

        # Write the grid and computed fields to a .vti file.
        grid.save(f"{filename}.vti")

    return write_fields
        
