import numpy as np


def force_shift_velocity_factory(ctx):
    """Create the kernel that computes the velocity shift due to the body force for the <name> forcing model."""

    D = ctx.D
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    @ctx.device_decorator
    def force_shift_velocity(rho, u, body_force_local):
        """Compute the velocity shift due to the body force for the <name> forcing model."""
        velocity_shift = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
        for d in range(D):
            raise NotImplementedError("This function is not yet implemented.")
        return velocity_shift

    return force_shift_velocity

def add_source_term_factory(ctx):
    """Create the kernel that adds the source term to the post-collision distribution function for the <name> forcing model."""

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    Cij_float = ctx.Cij_float
    Wi_float = ctx.Wi_float
    
    div_Cs2 = float_t(3)  # Assuming Cs^2 = 1/3 for LBM
    div_Cs4 = float_t(9)  # Assuming Cs^4 = 1/9 for LBM

    @ctx.device_decorator
    def add_source_term(f_local, rho, u, body_force_local):
        """Add the source term to the post-collision distribution function for the <name> forcing model."""

        for q in range(Q):
            raise NotImplementedError("This function is not yet implemented.")

    return add_source_term













