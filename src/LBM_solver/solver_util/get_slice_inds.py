import numpy as np

def get_slice_inds(ctx, slices_ijk):
    """
    Get the shape and starting indices of a sliced domain based on the provided slices
    and the simulation domain shape (nx, ny, nz).

    Args:
        ctx: The simulation context.
        slices_ijk: A tuple of slice objects for each dimension.

    Returns:
        n_: The shape of the sliced domain, as a tuple (nx_, ny_, nz_).
        start_inds: The starting indices of the sliced domain, as a tuple (i_start, j_start, k_start).
    """

    # Shape of the sliced domain
    n_ = np.empty(ctx.n)[slices_ijk].shape

    # Starting indices of the sliced domain
    start_inds = tuple([s.indices(n)[0] for s, n in zip(slices_ijk, ctx.n)])

    return n_, start_inds