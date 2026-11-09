import numpy as np


from .solver_util.load_f import load_f_factory
from .solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def collide_only_factory(ctx):

    Q = ctx.Q
    nx, ny, nz = ctx.n
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda


    if use_cuda:
        from numba import cuda

    
    load_f = load_f_factory(ctx)
    

    macroscopic = ctx.macroscopic_factory(ctx)
    equilibrium = ctx.equilibrium_factory(ctx)
    collide = ctx.collide_factory(ctx)


    @ctx.device_decorator
    def lbm_collide_only_kernel(ijk, F, F_new):

        # Copy distributions to local (register) memory
        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        load_f(F, f_local, ijk)

        # 1. MACROSCOPIC VARIABLES
        rho, u = macroscopic(f_local)

        # 2. COMPUTE EQUILIBRIUM
        f_eq = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        equilibrium(rho, u, f_eq)

        # 3. COLLISION
        collide(f_local, f_eq)

        # 4. STORE BACK TO GLOBAL MEMORY
        for q in range(Q):
            F_new[q, ijk[0], ijk[1], ijk[2], 0] = f_local[q]

        return

    
    return apply_kernel_over_domain_factory(ctx, lbm_collide_only_kernel, (nx, ny, nz))

        