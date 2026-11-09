import numpy as np

from ..solver_util.set_vector_zero import set_vector_zero_factory
from ..solver_util.sparse_matrix_vector_product_update import sparse_matrix_vector_product_update_factory


def get_surface_force_factory(ctx):
    """Create a function that computes the surface force from the distribution function F.
    The surface force is computed as the product of a sparse matrix (Fi2fsurfInput) and the distribution function F.
    The sparse matrix should be a (3*Nsurf, Q*Nx*Ny*Nz) sparse matrix that takes the distribution function F and 
    outputs the surface force vector of shape (3*Nsurf,). 
    Fi2fsurfInput should be stored as a scipy.sparse.coo_matrix in the simulation context object ctx."""

    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    # Function to set a vector to zero
    set_vector_zero = set_vector_zero_factory(ctx)

    # Function to perform the sparse matrix-vector product update
    sparse_matrix_vector_product_update = sparse_matrix_vector_product_update_factory(ctx, ctx.Fi2fsurfInput.row, ctx.Fi2fsurfInput.col, ctx.Fi2fsurfInput.data)

    # Initialize the output surface force array
    Fsurf_shape = (ctx.Fi2fsurfInput.shape[0],)
    Fsurf = cuda.device_array(Fsurf_shape, dtype=ctx.float_t) if use_cuda else np.empty(Fsurf_shape, dtype=ctx.float_t)

    def get_surface_force(F):
        set_vector_zero(Fsurf)
        sparse_matrix_vector_product_update(F.ravel(), Fsurf)
        return Fsurf

    return get_surface_force