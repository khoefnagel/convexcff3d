import numpy as np

from ..solver_util.get_slice_inds import get_slice_inds
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def get_rho_factory(ctx, slices_ijk=(slice(None), slice(None), slice(None))):
    """Create a function that computes the density rho in the specified slice of the domain.

    Args:
        ctx: Context object containing simulation parameters and models.
        slices_ijk: Tuple of slices or integers for each dimension (x, y, z)
         specifying the region of the domain to compute rho for. Default is the entire domain."""

    Q = ctx.Q
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    macroscopic = ctx.macroscopic_factory(ctx)

    if use_cuda:
        from numba import cuda

    # Get the size of the sliced sub-domain and its origin.
    (nx_, ny_, nz_), (x0, y0, z0) = get_slice_inds(ctx, slices_ijk)

    @ctx.device_decorator
    def get_rho(ijk, F, rho, t):
        """Compute the density rho in the specified cell.

        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            rho: Output array to store the density rho in the specified slice.
            t: Current timestep.        
        """

        # Copy the distribution function for the current cell into local (register) memory
        x_offset, y_offset, z_offset = ijk
        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        for q in range(Q):
            f_local[q] = F[q, x0+x_offset, y0+y_offset, z0+z_offset, 0]

        # Compute the macroscopic density and velocity from the local distribution function
        rho_local, u_local = macroscopic(f_local)

        # Copy into the output array
        rho[0, x_offset, y_offset, z_offset, 0] = rho_local

    # Apply the kernel over the specified slice of the domain.
    return apply_kernel_over_domain_factory(ctx, get_rho, (nx_, ny_, nz_))

