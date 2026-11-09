import numpy as np


def force_shift_velocity_factory(ctx):
    """Create the kernel that computes the velocity shift due to the body force for no body force.
            i.e. delta_u = 0.0"""

    D = ctx.D
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    @ctx.device_decorator
    def force_shift_velocity(rho, u, body_force_local, velocity_shift):
        """Compute the velocity shift due to the body force for no body force.
                i.e. delta_u = 0.0

        Args:
            rho: Macroscopic density.
            u: Macroscopic velocity vector of shape (D,).
            body_force_local: Local body force vector of shape (D,).
            velocity_shift: Array to store the velocity shift vector, shape (D,).
        """

        # No body force: return 0 shift vector
        velocity_shift = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
        for d in range(D):
            velocity_shift[d] = float_t(0.0)

    return force_shift_velocity

def add_source_term_factory(ctx):
    """Create the kernel that adds the source term to the post-collision distribution function
    for no body force.
    In this case, the source term is zero, so the function does nothing.
    """

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    Cij_float = ctx.Cij_float
    Wi_float = ctx.Wi_float
    
    div_Cs2 = float_t(1 / 3)  # Assuming Cs^2 = 1/3 for LBM
    div_Cs4 = float_t(1 / 9)  # Assuming Cs^4 = 1/9 for LBM

    @ctx.device_decorator
    def add_source_term(f_local, rho, u, body_force_local):
        """Add the source term to the post-collision distribution function for no body force.
                In this case, the source term is zero, so the function does nothing."""
        return  # No source term is added in this model.        

    return add_source_term













