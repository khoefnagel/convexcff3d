
from ..solver_util.get_slice_inds import get_slice_inds
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory


def get_F_q_factory(ctx, q, slices_ijk=(slice(None), slice(None), slice(None))):
    """Create a function that computes the q-th population of the distribution function F in the specified
    slice of the domain.

    Args:
        ctx: Context object containing simulation parameters and models.
        q: Index of the population to extract from the distribution function F.
        slices_ijk: Tuple of slices or integers for each dimension (x, y, z)
         specifying the region of the domain to compute F_q for. Default is the entire domain."""

    # Get the size of the sliced sub-domain and its origin.
    (nx_, ny_, nz_), (x0, y0, z0) = get_slice_inds(ctx, slices_ijk)

    @ctx.device_decorator
    def get_F_q(ijk, F, F_q, t):
        """Obtain the q-th population of the distribution function F in the specified cell.
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            F_q: Output array to store the q-th population of the distribution function F in the specified slice.
            t: Current timestep.
        """

        x_offset, y_offset, z_offset = ijk
        F_q[0, x_offset, y_offset, z_offset, 0] = F[q, x0+x_offset, y0+y_offset, z0+z_offset, 0]

    # Apply the kernel over the specified slice of the domain.
    return apply_kernel_over_domain_factory(ctx, get_F_q, (nx_, ny_, nz_))