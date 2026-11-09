
from ..solver_util.sparse_matrix_vector_product_update import sparse_matrix_vector_product_update_factory

def apply_CSM_factory(ctx):
    """Create a function that applies the correct streaming matrix (CSM) to the distribution function F.
    The CSM matrix should be a (Q*Nx*Ny*Nz, Q*Nx*Ny*Nz) sparse matrix that takes the post-periodic-stream
    distribution function F and outputs corrections to be added to it to account for boundary conditions.
    
    The rows, cols, and values of CSM should be stored in the simulation context object ctx."""

    if ctx.rows is None or ctx.rows.shape[0] == 0:
        return lambda F: None  # No corrections needed

    # Function to update the distribution function F using the sparse matrix-vector product
    sparse_matrix_vector_product_update = sparse_matrix_vector_product_update_factory(ctx, ctx.rows, ctx.cols, ctx.values)

    return lambda F: sparse_matrix_vector_product_update(F.ravel(), F.ravel())


