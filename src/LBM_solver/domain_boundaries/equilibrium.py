'''Specify u and rho at a boundary, all populations at that boundary are then set to the corresponding
equilibrium values.'''

import numpy as np

from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory
from ..solver_util.validate_user_function import validate_user_function

def get_domain_boundary_updates_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, 
                                        side_normal, normal_index):
    """Create a function that computes the updates for all populations at the specified boundary.
    The populations are set to the corresponding equilibrium values based on the specified rho and u at the boundary.

    Args:
        ctx: Context object containing simulation parameters and models.
        domain_boundary: Dictionary containing the domain boundary condition parameters.
        unknown_populations: Indices of the unknown populations that need to be reconstructed by the boundary condition.
        boundary_offset: Tuple of length 3 representing the offset of the boundary in each spatial dimension (x, y, z).
        slice_shape: Tuple of length 3 representing the shape of the slice of the domain boundary in each 
                     spatial dimension (x, y, z).
        side_normal: Tuple of length 3 representing the normal direction of the domain boundary, inwards positive.
        normal_index: Index of the normal direction (0 for x, 1 for y, 2 for z).
    """

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    equilibrium = ctx.equilibrium_factory(ctx)

    if use_cuda:
        from numba import cuda

    assert 'rho' in domain_boundary, "Domain boundary must specify 'rho' value"
    assert 'u' in domain_boundary, "Domain boundary must specify 'u' value"
    rho_in = domain_boundary['rho']
    u_in = domain_boundary['u']

    # Check if rho is a function, if so, ensure it has the correct signature and returns a cuda device
    #  array of the correct type. If not, convert it to a function that returns a cuda device array.
    rho_func = validate_user_function(rho_in, ctx, expected_signature={'x', 'y', 'z', 't'})
    
    # Check if u is a function, if so, ensure it has the correct signature and returns a cuda device
    #  array of the correct type. If not, convert it to a function that returns a cuda device array.
    u_funcs = tuple([validate_user_function(u_in[d], ctx, expected_signature={'x', 'y', 'z', 't'}) for d in range(D)])



    @ctx.device_decorator
    def get_domain_boundary_updates_kernel(ijk, F, buffer, t):
        """Compute the updates for all populations at the specified boundary based on the
        equilibrium values corresponding to the specified rho and u.
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            buffer: Output buffer to store the updates for the unknown populations at the boundary.
            rho: Density at the boundary, either a (1, 1, 1, 1, 1) array or array of shape (1, nx, ny, nz, 1).
            u: Velocity at the boundary, either a (1, 1, 1, 1, D) array or array of shape (1, nx, ny, nz, D).
        """

        i, j, k = ijk
        x, y, z = i + boundary_offset[0] + 0.5, j + boundary_offset[1] + 0.5, k + boundary_offset[2] + 0.5


        rho_local = rho_func(x, y, z, t)
        u_local = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
        u_local[0] = u_funcs[0](x, y, z, t)
        if D > 1:
            u_local[1] = u_funcs[1](x, y, z, t)
        if D > 2:
            u_local[2] = u_funcs[2](x, y, z, t)

            
        # Compute the equilibrium distribution based on the density and velocity
        f_eq = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        equilibrium(rho_local, u_local, f_eq)

        # Write the equilibrium values to the output buffer for all populations at the boundary
        for q in range(Q):
            buffer[q, i, j, k, 0] = f_eq[q]

    # Apply the kernel over the specified slice of the domain boundary.
    get_domain_boundary_updates_ = apply_kernel_over_domain_factory(ctx, get_domain_boundary_updates_kernel, slice_shape)

    def get_domain_boundary_updates(F, buffer, t=0):
        """Compute the updates for all populations at the specified boundary based on the
        equilibrium values corresponding to the specified rho and u at the current timestep.
        
        Args:
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            buffer: Output buffer to store the updates for the unknown populations at the boundary.
            t: Current timestep, used to evaluate rho and u if they are functions of time.
        """


        get_domain_boundary_updates_(F, buffer, t)        

    return get_domain_boundary_updates



def apply_domain_boundary_updates_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, side_normal, normal_index):
    """Create a function that applies the updates for all populations at the specified boundary."""

    Q = ctx.Q

    @ctx.device_decorator
    def apply_domain_boundaries_updates_kernel(ijk, F, buffer, t):
        """Apply the updates for all populations at the specified boundary.
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            buffer: Input buffer containing the updates for the unknown populations at the boundary.
            t: Current timestep, used to evaluate rho and u if they are functions of time.
        """

        i, j, k = ijk

        i2 = i + boundary_offset[0]
        j2 = j + boundary_offset[1]
        k2 = k + boundary_offset[2]

        for q in unknown_populations:
            F[q, i2, j2, k2, 0] = buffer[q, i, j, k, 0]

    # Apply the kernel over the specified slice of the domain boundary.
    return apply_kernel_over_domain_factory(ctx, apply_domain_boundaries_updates_kernel, slice_shape)
