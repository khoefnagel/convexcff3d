import numpy as np

from .solver_util.load_f import load_f_factory
from .solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

from .forcing.get_body_force_local import get_body_force_local_factory
from .streaming.stream_periodic import stream_periodic_factory



def lbm_stream_only_factory(ctx):

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


    @ctx.device_decorator
    def lbm_stream_only_kernel(ijk, F, F_new, t):

        # Copy distributions to local (register) memory
        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        load_f(F, f_local, ijk)

        # 4. PERIODIC STREAMING
        stream_periodic(f_local, F_new, ijk)

        return

    return apply_kernel_over_domain_factory(ctx, lbm_stream_only_kernel, (nx, ny, nz))

        
    










