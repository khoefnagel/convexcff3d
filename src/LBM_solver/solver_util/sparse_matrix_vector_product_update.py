import numpy as np


def sparse_matrix_vector_product_update_factory(ctx, rows, cols, values):
    """Create a kernel that performs a sparse matrix-vector product update.
    
    Args:
        ctx: Context object containing simulation parameters and settings.
        rows: Array of row indices for the sparse matrix.
        cols: Array of column indices for the sparse matrix.
        values: Array of non-zero values for the sparse matrix.
    """

    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    # Send the sparse matrix data to the device if using CUDA
    rows = cuda.to_device(rows) if use_cuda else rows
    cols = cuda.to_device(cols) if use_cuda else cols
    values = cuda.to_device(values) if use_cuda else values
    updates = cuda.device_array(rows.shape[0], dtype=ctx.float_t) if use_cuda else np.empty(rows.shape[0], dtype=ctx.float_t)

    # ===================================================================================================
    # GPU code
    if use_cuda:

        from .get_blocks_threads import get_blocks_threads
        blocks, threads = get_blocks_threads(rows.shape[0], 1, 1)

        @ctx.kernel_decorator
        def compute_updates_kernel(f_in, cols, values, updates):
            """Compute the updates for the sparse matrix-vector product.
            
            Args:
                f_in: Input vector array of shape (N,).
                cols: Array of column indices for the sparse matrix, shape (M,).
                values: Array of non-zero values for the sparse matrix, shape (M,).
                updates: Output array to store the computed updates, shape (M,).
            """
            ind = cuda.grid(3)[0]
            if ind >= cols.shape[0]:
                return

            updates[ind] = f_in[cols[ind]] * values[ind]

        @ctx.kernel_decorator
        def apply_updates_kernel(f_out, rows, updates):
            """Add the computed updates to the output vector.

            Args:
                f_out: Output vector array of shape (N,), the updates from the matrix-vector product
                        are added to this vector in place.
                rows: Array of row indices for the sparse matrix, shape (M,).
                updates: Array of computed updates, shape (M,).
            """
            ind = cuda.grid(3)[0]
            if ind >= rows.shape[0]:
                return

            i = rows[ind]
            cuda.atomic.add(f_out, i, updates[ind])

        def sparse_matrix_vector_product_update(f_in, f_out):
            """Perform the sparse matrix-vector product update.

            Args:
                f_in: Input vector array of shape (N,).
                f_out: Output vector array of shape (N,), the updates from the matrix-vector product 
                        are added to this vector in place.
            """

            # Compute the updates for the sparse matrix-vector product
            compute_updates_kernel[blocks, threads](f_in, cols, values, updates)

            # Ensure that the compute_updates_kernel has completed before applying updates
            cuda.synchronize() 

            # Add the computed updates to the output vector
            apply_updates_kernel[blocks, threads](f_out, rows, updates)

        return sparse_matrix_vector_product_update

    # ==================================================================================================
    # CPU code
    else:
        from numba import prange

        @ctx.kernel_decorator
        def compute_updates_kernel(f_in, cols, values, updates):
            """Compute the updates for the sparse matrix-vector product.

            Args:
                f_in: Input vector array of shape (N,).
                cols: Array of column indices for the sparse matrix, shape (M,).
                values: Array of non-zero values for the sparse matrix, shape (M,).
                updates: Output array to store the computed updates, shape (M,).
            """
            for ind in prange(cols.shape[0]):
                updates[ind] = f_in[cols[ind]] * values[ind]
        
        @ctx.kernel_decorator
        def apply_updates_kernel(f_out, rows, updates):
            """Add the computed updates to the output vector.

            Args:
                f_out: Output vector array of shape (N,), the updates from the matrix-vector product
                        are added to this vector in place.
                rows: Array of row indices for the sparse matrix, shape (M,).
                updates: Array of computed updates, shape (M,).
            """
            for ind in prange(rows.shape[0]):
                f_out[rows[ind]] += updates[ind]

        def sparse_matrix_vector_product_update(f_in, f_out):

            # Compute the updates for the sparse matrix-vector product
            compute_updates_kernel(f_in, cols, values, updates)

            # Add the computed updates to the output vector
            apply_updates_kernel(f_out, rows, updates)

        return sparse_matrix_vector_product_update