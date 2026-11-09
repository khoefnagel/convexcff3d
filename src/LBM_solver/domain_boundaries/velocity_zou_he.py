import numpy as np

from ..solver_util.validate_user_function import validate_user_function
from ..solver_util.load_f import load_f_factory


def get_domain_boundary_updates_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, 
                                        side_normal, normal_index):

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    int_t = ctx.int_t
    Cij_int = ctx.Cij_int
    Wi_float = ctx.Wi_float
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda


    # Indices of the populations that are opposite unknown populations.
    # E.g. the unkown population (1, -1) at the top boundary has opposite (-1, 1).
    i_opp = np.empty(len(unknown_populations), dtype=int_t)
    for m, q in enumerate(unknown_populations):
        Ci = Cij_int[q].ravel()
        Ci_opp = -np.copy(Ci)
        i_opp[m] = np.where(np.all(Cij_int == Ci_opp, axis=-1))[0][0]

    know_populations = np.array(list(set(range(Q)) - set(unknown_populations)))

    


    assert 'u' in domain_boundary, "Domain boundary must specify 'u' value"
    u_in = domain_boundary['u']

    # Check if u is a function, if so, ensure it has the correct signature and returns a cuda device
    #  array of the correct type. If not, convert it to a function that returns a cuda device array.
    u_func = validate_user_function(u_in, ctx, expected_signature={'x', 'y', 'z', 't'})

    load_f = load_f_factory(ctx)

    @ctx.device_decorator
    def get_domain_boundary_updates_kernel(ijk, F, buffer, t):

        i, j, k = ijk
        x, y, z = i + boundary_offset[0] + 0.5, j + boundary_offset[1] + 0.5, k + boundary_offset[2] + 0.5

        # 1. Copy distributions from global to local (register) memory, shape (Q,)
        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        load_f(F, f_local, i + boundary_offset[0], j + boundary_offset[1], k + boundary_offset[2])

        u_local = u_func(x, y, z, t)

        term1 = float_t(0.0) # Sum of known populations
        for q in know_populations:
            term1 += f_local[q]
            
        term2 = float_t(0.0) # Sum of opposites of unknown populations
        term3 = float_t(0.0) # Sum of Wi * ci_dot_u for unknown populations
        for q in unknown_populations:
            term2 += f_local[i_opp[q]]

            ci_dot_u = float_t(0.0)
            for d in range(D):
                ci_dot_u += Cij_int[q, d] * u_local[d]
            term3 += Wi_float[q] * ci_dot_u

        # rho = (sum of known populations + sum of opposites of unknown populations) / (1 - 6 * sum of Wi * ci_dot_u for unknown populations)
        rho = (term1 + term2) / (1 - 6 * term3)



        for q in unknown_populations:
            buffer[q, i, j, k, 0] = F[q, i + boundary_offset[0], j + boundary_offset[1], k + boundary_offset[2], 0]