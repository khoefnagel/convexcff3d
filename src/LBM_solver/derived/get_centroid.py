
from ..solver_util.get_slice_inds import get_slice_inds
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def get_centroid_factory(ctx, slices_ijk=(slice(None), slice(None), slice(None))):
    """Create a function that computes the centroid of each cell in the specified slice of the domain.
    
    Args:
        ctx: Context object containing simulation parameters and models.
        slices_ijk: Tuple of slices or integers for each dimension (x, y, z)
         specifying the region of the domain to compute centroids for. Default is the entire domain."""

    D = ctx.D
    cell_centroids = ctx.cell_centroids

    assert cell_centroids is not None, "cell_centroids must be provided in the context to use get_centroid_factory."

    # Get the size of the sliced sub-domain and its origin.
    (nx_, ny_, nz_), (x0, y0, z0) = get_slice_inds(ctx, slices_ijk)

    @ctx.device_decorator
    def get_centroid_kernel(ijk, F, centroid, t):
        """Compute the centroid of the specified cell.

        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            centroid: Output array to store the centroid coordinates for each cell in the specified slice.
            t: Current timestep.
        """

        x_offset, y_offset, z_offset = ijk
        for d in range(D):
            centroid[0, x_offset, y_offset, z_offset, d] = cell_centroids[0, x0+x_offset, y0+y_offset, z0+z_offset, d]

        return centroid

    # Apply the kernel over the specified slice of the domain.
    return apply_kernel_over_domain_factory(ctx, get_centroid_kernel, (nx_, ny_, nz_))