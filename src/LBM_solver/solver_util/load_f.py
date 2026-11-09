
def load_f_factory(ctx):
    """Create the kernel to load distribution function values from global memory into local (register) memory."""
    Q = ctx.Q

    @ctx.device_decorator
    def load_f(F, f_local, ijk):
        """Load the distribution function values from global memory into local (register) memory."""
        for q in range(Q):
            f_local[q] = F[q, ijk[0], ijk[1], ijk[2], 0]
    
    return load_f
