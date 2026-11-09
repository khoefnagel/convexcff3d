import numpy as np

def stream_periodic_factory(ctx):
    """Create a function that performs periodic streaming of the distribution function F."""

    D = ctx.D
    Q = ctx.Q
    int_t = ctx.int_t
    Cij_int = ctx.Cij_int
    n = ctx.n
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    @ctx.device_decorator
    def stream_periodic_kernel(f_local, F_new, ijk):
        """Perform periodic streaming of the distribution function F.

        Args:
            f_local: Local copy of the distribution function for the current cell, shape (Q,).
            F_new: Global distribution function array of shape (Q, nx, ny, nz, 1),
                    to be updated with the post-periodic-streamed distribution function.
            ijk: Tuple of (i, j, k) cell indices for the current thread.
        """

        # Compute the periodically streamed indices for the current cell,
        # taking into account the periodic boundary conditions.
        ijk_new = cuda.local.array(3, dtype=int_t) if use_cuda else np.empty(3, dtype=int_t)
        for q in range(Q):
            for d in range(D):
                x = ijk[d]
                shift = Cij_int[q, 0, 0, 0, d]
                x_shifted = x + shift
                if x_shifted == int_t(-1):
                    x_shifted = n[d] - int_t(1)
                elif x_shifted == n[d]:
                    x_shifted = int_t(0)
                ijk_new[d] = x_shifted

            # If D < 3, copy the remaining indices from ijk to ijk_new
            for k in range(D, 3):
                ijk_new[k] = ijk[k]

            F_new[q, ijk_new[0], ijk_new[1], ijk_new[2], 0] = f_local[q]

    return stream_periodic_kernel