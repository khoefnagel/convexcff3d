
def unravel_index_factory(ctx, n):
    """Create a kernel that converts a flat index to a multi-dimensional index based on the shape n.
    
    Args:
        ctx: Context object containing simulation parameters and settings.
        n: Tuple representing the shape of the multi-dimensional array (nx, ny, nz).
    """    
    ny, nz = n[1], n[2]

    @ctx.device_decorator
    def unravel_index(ind):
        """Convert a flat index to a multi-dimensional index based on the shape n.

        Args:
            ind: Flat index (integer).
        
        Returns:
            A tuple (x, y, z) representing the multi-dimensional index corresponding to the flat index.
        """
        x = ind // (ny * nz)
        rem = ind % (ny * nz)
        y = rem // nz
        z = rem % nz
        return x, y, z

    return unravel_index
