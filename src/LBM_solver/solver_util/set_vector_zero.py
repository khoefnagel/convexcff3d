
def set_vector_zero_factory(ctx):
    """Create a kernel that sets all elements of a vector to zero."""

    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    # ==================================================================================================
    # GPU code
    if use_cuda:
        from numba import cuda
        from .get_blocks_threads import get_blocks_threads

        @ctx.kernel_decorator
        def set_vector_zero(vector):
            """Set all elements of the input vector to zero.
            
            Args:
                vector: Input vector array of shape (N,), updated in place.
            """
            ind = cuda.grid(3)[0]
            if ind >= vector.shape[0]:
                return
            
            vector[ind] = float_t(0.0)

        def wrapper(vector):   
            """Wrapper function to launch the GPU kernel that sets all elements of the input vector to zero.
            
            Args:
                vector: Input vector array of shape (N,), updated in place.
            """
            blocks, threads = get_blocks_threads(vector.shape[0], 1, 1)
            set_vector_zero[blocks, threads](vector)

        return wrapper

    # ==================================================================================================
    # CPU code
    else:
        from numba import prange

        @ctx.kernel_decorator
        def set_vector_zero(vector):
            """Set all elements of the input vector to zero.

            Args:
                vector: Input vector array of shape (N,), updated in place.
            """
            for ind in prange(vector.shape[0]):
                vector[ind] = float_t(0.0)

        return set_vector_zero