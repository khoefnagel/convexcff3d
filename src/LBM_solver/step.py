import numpy as np

from .solver_util.load_f import load_f_factory
from .solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

from .forcing.get_body_force_local import get_body_force_local_factory
from .streaming.stream_periodic import stream_periodic_factory


def lbm_step_factory(ctx):
    """Create the timestep kernel that performs fused collision and streaming."""

    D = ctx.D
    Q = ctx.Q
    nx, ny, nz = ctx.n
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda
    
    load_f = load_f_factory(ctx)
    get_body_force_local = get_body_force_local_factory(ctx)
    macroscopic = ctx.macroscopic_factory(ctx)
    force_shift_velocity = ctx.force_shift_velocity_factory(ctx)
    equilibrium = ctx.equilibrium_factory(ctx)
    collide = ctx.collide_factory(ctx)
    add_source_term = ctx.add_source_term_factory(ctx)
    stream_periodic = stream_periodic_factory(ctx)
    zero_body_force = ctx.zero_body_force

    @ctx.device_decorator
    def lbm_step_kernel(ijk, F, F_new, t):
        """Perform fused collision and streaming
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            F_new: Output distribution function array of shape (Q, nx, ny, nz, 1).
            t: Current timestep.
        """

        # 1. Copy distributions from global to local (register) memory, shape (Q,)
        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        load_f(F, f_local, ijk)

        if not zero_body_force:
            # 2. Get the body force density vector, shape (D,)
            body_force_local = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
            get_body_force_local(ijk, t, body_force_local)
    
        # 3. Compute macroscopic density and velocity
        rho, u = macroscopic(f_local)

        if not zero_body_force:
            # 4. Correct the velocity for the body force
            shift = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
            force_shift_velocity(rho, u, body_force_local, shift)
            for d in range(D):
                u[d] += shift[d]

        # 5. Compute equilibrium distribution, shape (Q,)
        f_eq = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        equilibrium(rho, u, f_eq)

        # 6. Perform collision step
        collide(f_local, f_eq)

        if not zero_body_force:
            # 7. Correct post-collision distributions for the body force
            add_source_term(f_local, rho, u, body_force_local)

        # 8. Periodically stream the post-collision distributions to neighboring cells and write to global memory
        # Note: This step ignores boundary conditions, which are handled in a separate kernel.
        stream_periodic(f_local, F_new, ijk)

    # Apply kernel over entire domain.
    return apply_kernel_over_domain_factory(ctx, lbm_step_kernel, (nx, ny, nz))

        
    










