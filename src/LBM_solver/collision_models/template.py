
def collide_factory(ctx):
    """Create the kernel that performs the collision step for the <name> collision model."""

    omega = ctx.omega
    Q = ctx.Q

    @ctx.device_decorator
    def collide(f_local, feq):
        """Perform the collision step for the <name> collision model.

        Args:
            f_local: Local distribution function array of shape (Q,), updated in place.
            feq: Equilibrium distribution function array of shape (Q,).
        """
        # TODO: Implement the collision step to update the local distribution function f_local based on the equilibrium distribution feq.
        raise NotImplementedError("collision model not implemented yet")
    
    return collide