import numpy as np


def force_shift_velocity_factory(ctx):
    """Create the kernel that computes the velocity shift due to the body force for the Guo forcing model.
            delta_u = 0.5 * F / rho"""

    D = ctx.D
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    @ctx.device_decorator
    def force_shift_velocity(rho, u, body_force_local, velocity_shift):
        """Compute the velocity shift due to the body force for the Guo forcing model.
                delta_u = 0.5 * F / rho
        
        Args: 
            rho: Macroscopic density.
            u: Macroscopic velocity vector of shape (D,).
            body_force_local: Local body force vector of shape (D,).
            velocity_shift: Array to store the velocity shift vector, shape (D,).
        """

        
        for d in range(D):
            velocity_shift[d] = float_t(0.5) * body_force_local[d] / rho
        return velocity_shift

    return force_shift_velocity

def add_source_term_factory(ctx):
    """Create the kernel that adds the source term to the post-collision distribution function 
    for the Guo forcing model.
    
            source_term = (1 - 1/(2*omega)) * S_qd
            where S_qd = (C_{q,d} - u_d) * (F_d / rho) + (C_{q,d} * u) * (C_{q,d} * F_d / rho)
    """

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    Cij_float = ctx.Cij_float
    Wi_float = ctx.Wi_float
    
    div_Cs2 = float_t(3)  # Assuming Cs^2 = 1/3 for LBM
    div_Cs4 = float_t(9)  # Assuming Cs^4 = 1/9 for LBM

    # Pre-factor for the source term: 1 - 1/(2*omega)
    pre_factor = float_t(1 - 0.5 * ctx.omega)  

    @ctx.device_decorator
    def add_source_term(f_local, rho, u, body_force_local):
        """Add the source term to the post-collision distribution function for the Guo forcing model.

                source_term = (1 - 1/(2*omega)) * S_qd
                where S_qd = (C_{q,d} - u_d) * (F_d / rho) + (C_{q,d} * u) * (C_{q,d} * F_d / rho)

        Args:
            f_local: Local distribution function vector of shape (Q,).
            rho: Macroscopic density.
            u: Macroscopic velocity vector of shape (D,).
            body_force_local: Local body force vector of shape (D,).

        Returns:
            f_local: Updated local distribution function vector of shape (Q,).
        """

        for q in range(Q):

            # Compute Ci . u
            ci_dot_u = float_t(0.0)
            for d in range(D):
                ci_dot_u += Cij_float[q, 0, 0, 0, d] * u[d]

            for d in range(D):

                # Compute source term for direction q, dimension d
                S_qd = (
                    (Cij_float[q, 0, 0, 0, d] - u[d])*div_Cs2 +\
                    (ci_dot_u * Cij_float[q, 0, 0, 0, d]) * div_Cs4
                ) * Wi_float[q, 0, 0, 0, 0] * body_force_local[d]

                # Add 1 - 1/(2*omega) * S_qd to the distribution function f_local[q]
                f_local[q] += pre_factor * S_qd

    return add_source_term













