import numpy as np

from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory
from .default_apply_domain_boundary_updates import default_apply_domain_boundaries_updates_function_factory as apply_domain_boundary_updates_factory

def get_domain_boundary_updates_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, side_normal, normal_index):
    """Create a function that computes the updates for the unknown populations at the specified boundary based
    on the zero gradient boundary condition. The unknown populations are set to the values of the corresponding
    known populations at the adjacent cell in the direction of the boundary normal.
    
    Args:
        ctx: Context object containing simulation parameters and models.
        domain_boundary: Dictionary containing the domain boundary condition parameters.
        unknown_populations: Indices of the unknown populations that need to be reconstructed by the boundary condition.
        boundary_offset: Tuple of length 3 representing the offset of the boundary in each spatial dimension (x, y, z).
        slice_shape: Tuple of length 3 representing the shape of the slice of the domain boundary in each spatial dimension (x, y, z).
        side_normal: Tuple of length 3 representing the normal direction of the domain boundary, inwards positive.
        normal_index: Index of the normal direction in the spatial dimensions."""

    int_t = ctx.int_t

    side_normal_int = np.array(side_normal, dtype=int_t)

    @ctx.device_decorator
    def get_domain_boundary_updates_kernel(ijk, F, buffer, t):
        """Compute the updates for the unknown populations at the specified boundary based on the
        zero gradient boundary condition. The unknown populations are set to the values of the corresponding
        known populations at the adjacent cell in the direction of the boundary normal.

        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            buffer: Output buffer to store the updates for the unknown populations at the boundary.
            t: Current timestep.
        """


        i, j, k = ijk
        i2 = i + boundary_offset[0] + side_normal_int[0]
        j2 = j + boundary_offset[1] + side_normal_int[1]
        k2 = k + boundary_offset[2] + side_normal_int[2]

        for q in unknown_populations:
            buffer[q, i, j, k, 0] = F[q, i2, j2, k2, 0]

    # Apply the kernel over the specified slice of the domain boundary.
    return apply_kernel_over_domain_factory(ctx, get_domain_boundary_updates_kernel, slice_shape)
