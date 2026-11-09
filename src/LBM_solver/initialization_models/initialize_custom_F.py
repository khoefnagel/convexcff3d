import numpy as np

def initialize_factory(ctx):

    Q = ctx.Q
    nx, ny, nz = ctx.n
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    def initialize(rho0, u0, F0=None, t=0):
        assert F0 is not None, "F0 must be provided for initialize_custom_F"

        if F0.ndim == 1:
            F0 = F0[:, None, None, None, None]  # Reshape to (Q, 1, 1, 1, 1)
        F0_full_ = np.ascontiguousarray(np.broadcast_to(F0, (Q, nx, ny, nz, 1)), dtype=float_t).copy()

        return cuda.to_device(F0_full_) if use_cuda else F0_full_

    return initialize
