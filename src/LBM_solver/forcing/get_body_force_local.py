import numpy as np


def get_body_force_local_factory(ctx):
    """Create the kernel to get the body force density vector at the given cell indices and timestep."""

    D = ctx.D
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    # Get the function for each component of the body force
    body_force_x = ctx.body_force[0]
    body_force_y = ctx.body_force[1] if ctx.D > 1 else None
    body_force_z = ctx.body_force[2] if ctx.D > 2 else None


    @ctx.device_decorator
    def get_body_force_local(ijk, t, body_force_local):
        """Get the body force density vector at the cell center for the given cell indices and timestep.
        
        Args:
            ijk: Tuple of (i, j, k) cell indices for the current thread.
            t: Current timestep.
            body_force_local: Array to store the body force density vector at the cell center, shape (D,).
        """

        # Add 0.5 to evaluate the body force at the center of the cell
        x = float_t(ijk[0]) + float_t(0.5)
        y = float_t(ijk[1]) + float_t(0.5) if D > 1 else float_t(0.0)
        z = float_t(ijk[2]) + float_t(0.5) if D > 2 else float_t(0.0)
    
        body_force_local[0] = body_force_x(x, y, z, t)
        if D > 1:
            body_force_local[1] = body_force_y(x, y, z, t)
        if D > 2:
            body_force_local[2] = body_force_z(x, y, z, t)


    return get_body_force_local