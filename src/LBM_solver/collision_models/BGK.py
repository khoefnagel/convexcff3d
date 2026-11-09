

def collide_factory(ctx):
    """Create the kernel that performs the collision step for the BGK collision model."""

    omega = ctx.omega # 1 / tau
    Q = ctx.Q

    @ctx.device_decorator
    def collide(f_local, feq):
        """Perform the collision step for the BGK collision model.

        Args:
            f_local: Local distribution function array of shape (Q,), updated in place.
            feq: Equilibrium distribution function array of shape (Q,).
        """
        for q in range(Q):
            f_local[q] += (feq[q] - f_local[q]) * omega
    
    return collide