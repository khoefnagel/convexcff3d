import numpy as np

from .default_apply_domain_boundary_updates import default_apply_domain_boundaries_updates_function_factory as apply_domain_boundary_updates_factory
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def get_domain_boundary_updates_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, side_normal, normal_index):
    """Create a function that computes the updates for the unknown populations at the specified boundary based
    on the symmetry/slip boundary condition.
    
    Args:
        ctx: Context object containing simulation parameters and models.
        domain_boundary: Dictionary containing the domain boundary condition parameters.
        unknown_populations: Indices of the unknown populations that need to be reconstructed by the boundary condition.
        boundary_offset: Tuple of length 3 representing the offset of the boundary in each spatial dimension (x, y, z).
        slice_shape: Tuple of length 3 representing the shape of the slice of the domain boundary in each spatial dimension (x, y, z).
        side_normal: Tuple of length 3 representing the normal direction of the domain boundary, inwards positive.
        normal_index: Index of the normal direction in the spatial dimensions."""
    
    D = ctx.D
    int_t = ctx.int_t
    Cij_int = ctx.Cij_int
    n = ctx.n
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    # Indices of the populations that are the origin of the specularly reflected unknown populations.
    # E.g. the unkown population (1, -1) at the top boundary is the specular reflection of population (1, 1).
    i_reflects = np.empty(len(unknown_populations), dtype=int_t)
    for m, q in enumerate(unknown_populations):
        Ci = Cij_int[q].ravel()
        Ci_reflect = np.copy(Ci)
        Ci_reflect[normal_index] *= -1
        i_reflects[m] = np.where(np.all(Cij_int == Ci_reflect, axis=-1))[0][0]

    
    @ctx.device_decorator
    def get_domain_boundary_updates_kernel(ijk, F, buffer, t):
        """Compute the updates for the unknown populations at the specified boundary based on the
        specular reflection of the known populations.
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            buffer: Output buffer to store the updates for the unknown populations at the boundary.
            t: Current timestep.
        """

        # Find indices of the cell from which to get the pre-specular reflection population.
        # Note that we first do a periodic stream and then correct it, which is why the 
        ijk_orig = cuda.local.array(3, dtype=int_t) if use_cuda else np.empty(3, dtype=int_t)
        for d in range(3):
            ijk_orig[d] = ijk[d] + boundary_offset[d]

        ijk_orig[normal_index] = n[normal_index] - int_t(1) if side_normal[normal_index] > int_t(0) else int_t(0)
        

        for d, q in enumerate(unknown_populations):
            buffer[q, ijk[0], ijk[1], ijk[2], 0] = F[i_reflects[d], ijk_orig[0], ijk_orig[1], ijk_orig[2], 0]

    # Apply the kernel over the specified slice of the domain boundary.
    return apply_kernel_over_domain_factory(ctx, get_domain_boundary_updates_kernel, slice_shape)