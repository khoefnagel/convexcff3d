
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def default_apply_domain_boundaries_updates_function_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, side_normal, normal_index):
    """Create a function that applies the updates for the unknown populations at the specified boundary.
    
    Args:
        ctx: Context object containing simulation parameters and models.
        domain_boundary: Dictionary containing the domain boundary condition parameters.
        unknown_populations: Indices of the unknown populations that need to be reconstructed by the 
                             boundary condition.
        boundary_offset: Tuple of length 3 representing the offset of the boundary in each spatial dimension.
        slice_shape: Shape of the slice over which to apply the boundary updates.
        side_normal: Normal vector of the boundary side.
        normal_index: Index of the normal direction."""
    
    @ctx.device_decorator
    def apply_domain_boundaries_updates_kernel(ijk, F, buffer, t):
        """Apply the updates for the unknown populations at the specified boundary.

        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1), updated in place.
            buffer: Buffer containing the updates for the unknown populations at the boundary.
            t: Current timestep.
        """
        i, j, k = ijk
        
        i2 = i + boundary_offset[0]
        j2 = j + boundary_offset[1]
        k2 = k + boundary_offset[2]

        for q in unknown_populations:
            F[q, i2, j2, k2, 0] = buffer[q, i, j, k, 0]

    # Apply the kernel over the specified slice of the domain boundary.
    return apply_kernel_over_domain_factory(ctx, apply_domain_boundaries_updates_kernel, slice_shape)
