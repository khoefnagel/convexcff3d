import numpy as np

def macroscopic_factory(ctx):
    """Create the kernel to compute macroscopic density and velocity from the distribution function."""

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    Cij_float = ctx.Cij_float
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda
    
    @ctx.device_decorator
    def macroscopic(f_local):
        """Compute macroscopic density and velocity from the distribution function.
        
        Args:
            f_local: Local distribution function array of shape (Q,).
        
        Returns:
            rho: Macroscopic density.
            u: Macroscopic velocity vector of shape (D,).
        """
                
        rho = float_t(0.0)
        u = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
        for d in range(D):
            u[d] = float_t(0.0)

        # TODO: Compute macroscopic density rho and velocity u from the local distribution function f_local.
        raise NotImplementedError("macroscopic model not implemented yet")

        return rho, u

    return macroscopic