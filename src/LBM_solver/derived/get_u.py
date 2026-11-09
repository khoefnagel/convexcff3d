import numpy as np

from ..solver_util.get_slice_inds import get_slice_inds
from ..forcing.get_body_force_local import get_body_force_local_factory
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def get_u_factory(ctx, slices_ijk=(slice(None), slice(None), slice(None))):

    Q = ctx.Q
    D = ctx.D
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda
    zero_body_force = ctx.zero_body_force


    macroscopic = ctx.macroscopic_factory(ctx)
    get_body_force_local = get_body_force_local_factory(ctx)
    force_shift_velocity = ctx.force_shift_velocity_factory(ctx)


    if use_cuda:
        from numba import cuda

    (nx_, ny_, nz_), (x0, y0, z0) = get_slice_inds(ctx, slices_ijk)

    @ctx.device_decorator
    def get_u(ijk, F, u, t):

        x_offset, y_offset, z_offset = ijk
        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        for q in range(Q):
            f_local[q] = F[q, x0+x_offset, y0+y_offset, z0+z_offset, 0]


        rho_local, u_local = macroscopic(f_local)

        if not zero_body_force:
            body_force_local = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
            get_body_force_local(ijk, t, body_force_local)
            shift = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
            force_shift_velocity(rho_local, u_local, body_force_local, shift)
            for d in range(D):
                u_local[d] += shift[d]
        
        for d in range(D):
            u[0, x_offset, y_offset, z_offset, d] = u_local[d]

    return apply_kernel_over_domain_factory(ctx, get_u, (nx_, ny_, nz_))

