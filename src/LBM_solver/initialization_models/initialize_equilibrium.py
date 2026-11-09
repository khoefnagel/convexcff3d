import numpy as np

from ..forcing.get_body_force_local import get_body_force_local_factory
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def initialize_factory(ctx):
    """Create the function to initialize the distribution function to equilibrium based on specified
    initial macroscopic density and velocity fields."""
    D = ctx.D
    Q = ctx.Q
    nx, ny, nz = ctx.n
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda
    zero_body_force = ctx.zero_body_force

    if use_cuda:
        from numba import cuda

    get_body_force_local = get_body_force_local_factory(ctx)
    force_shift_velocity = ctx.force_shift_velocity_factory(ctx)
    equilibrium = ctx.equilibrium_factory(ctx)
    
    @ctx.device_decorator
    def initialize_equilibrium_kernel(ijk, F, rho0, u0, t):
        """Initialize the distribution function to equilibrium based on specified initial macroscopic
        density and velocity fields, corrected for body force.
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array to be initialized of shape (Q, nx, ny, nz, 1).
            rho0: Initial macroscopic density field array of shape (1, nx, ny, nz, 1).
            u0: Initial macroscopic velocity field array of shape (1, nx, ny, nz, D).
            t: Current timestep.
        """

        # Get the local cell indices
        i, j, k = ijk

        # Initialize density and velocity in local (register) memory
        rho0_local = rho0[0, i, j, k, 0]
        u0_local = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
        for d in range(D):
            u0_local[d] = u0[0, i, j, k, d]


        if not zero_body_force:
            # Apply force shift to the initial velocity to account for body force
            # A negative shift is used such that the evaluated equilibrium distribution function
            # corresponds to the desired initial macroscopic velocity after accounting for the body force.
            body_force_local = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
            get_body_force_local(ijk, t, body_force_local)
            shift = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
            force_shift_velocity(rho0_local, u0_local, body_force_local, shift)
            for d in range(D):
                u0_local[d] -= shift[d]
        
        # Compute equilibrium distribution function
        f_eq = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        equilibrium(rho0_local, u0_local, f_eq)

        # Set F to the computed equilibrium distribution function
        for q in range(Q):
            F[q, i, j, k, 0] = f_eq[q]
    
    # Apply initialization kernel over the entire domain
    initialize_equilibrium_domain = apply_kernel_over_domain_factory(ctx, initialize_equilibrium_kernel, (nx, ny, nz))

    def initialize_equilibrium(rho0, u0, F0=None, t=0):
        """Initialize the distribution function to equilibrium based on specified initial macroscopic
        density and velocity fields.

        Args:
            rho0: Initial macroscopic density field, either a scalar or an array of shape (nx, ny, nz).
            u0: Initial macroscopic velocity field, either a 1D array of length D or an array of shape (nx, ny, nz, D).
            t: Current timestep.
        
        Returns:
            F: Initialized distribution function array of shape (Q, nx, ny, nz, 1).
        """

        rho0_full_ = np.ascontiguousarray(np.broadcast_to(rho0, (1, nx, ny, nz, 1)), dtype=float_t)
        u0_full_ = np.ascontiguousarray(np.broadcast_to(np.array(u0).reshape(1, 1, 1, 1, -1)[..., :D], (1, nx, ny, nz, D)), dtype=float_t)

        rho0_full = cuda.to_device(rho0_full_) if use_cuda else rho0_full_
        u0_full = cuda.to_device(u0_full_) if use_cuda else u0_full_
        print(u0_full.shape)
        print(u0_full[0, 0, 0, 0, :])

        F = cuda.device_array((Q, nx, ny, nz, 1), dtype=float_t, order="C") if use_cuda\
              else np.zeros((Q, nx, ny, nz, 1), dtype=float_t, order="C")
        
        initialize_equilibrium_domain(F, rho0_full, u0_full, float_t(t))
        return F

    return initialize_equilibrium
