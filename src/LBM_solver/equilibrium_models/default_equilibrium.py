
def equilibrium_factory(ctx):
    """Create the function to compute the equilibrium distribution function from the macroscopic
    density and velocity."""

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    Wi_float = ctx.Wi_float
    Cij_float = ctx.Cij_float

    @ctx.device_decorator
    def equilibrium(rho, u, f_eq):
        """Compute the equilibrium distribution function from the macroscopic density and velocity.
        
        Args:
            rho: Macroscopic density.
            u: Macroscopic velocity vector of shape (D,).
            f_eq: Output equilibrium distribution function array of shape (Q,).
        
        Returns:
            f_eq: Equilibrium distribution function array of shape (Q,).
        """

        # Compute |u|^2
        u_sq = float_t(0.0)
        for d in range(D):
            u_sq += u[d] * u[d]

        for q in range(Q):

            # Compute Ci . u
            dot_u = float_t(0.0)
            for d in range(D):
                dot_u += Cij_float[q, 0, 0, 0, d] * u[d]

            # Compute equilibrium distribution function
            # f^{eq}_{q} = w_{q} * rho * (1 + 3 * (Ci . u) + 4.5 * (Ci . u)^2 - 1.5 * |u|^2) 
            f_eq[q] = Wi_float[q,0,0,0,0] * rho * (float_t(1.0) + float_t(3.0)*dot_u + float_t(4.5)*dot_u*dot_u - float_t(1.5)*u_sq)


    return equilibrium