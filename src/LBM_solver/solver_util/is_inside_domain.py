

def is_inside_domain_factory(ctx, n):
    """Create a kernel that checks if a given 3D cell index is inside the simulation domain."""

    # Shape of the simulation domain
    nx, ny, nz = n

    @ctx.device_decorator
    def is_inside_domain(ijk):
        """Check if the given 3D cell index is inside the simulation domain."""
        return (ijk[0] < nx and ijk[1] < ny and ijk[2] < nz)
    
    return is_inside_domain
