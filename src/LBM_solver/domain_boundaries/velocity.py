import numpy as np

from ..solver_util.validate_user_function import validate_user_function
from ..solver_util.load_f import load_f_factory
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory
from .default_apply_domain_boundary_updates import default_apply_domain_boundaries_updates_function_factory as apply_domain_boundary_updates_factory


def get_domain_boundary_updates_factory(ctx, domain_boundary, unknown_populations, boundary_offset, slice_shape, 
                                        side_normal, normal_index):

    D = ctx.D
    Q = ctx.Q
    float_t = ctx.float_t
    int_t = ctx.int_t
    Cij_int = ctx.Cij_int
    Cij_float = ctx.Cij_float
    Wi_float = ctx.Wi_float
    n = ctx.n
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda


    # Indices of the populations that are opposite unknown populations.
    # E.g. the unkown population (1, -1) at the top boundary has opposite (-1, 1).
    i_opp = np.empty(Q, dtype=int_t)
    for q in range(Q):
        Ci = Cij_int[q].ravel()
        Ci_opp = -np.copy(Ci)
        i_opp[q] = np.where(np.all(Cij_int == Ci_opp, axis=-1))[0][0]


    assert 'u' in domain_boundary, "Domain boundary must specify 'u' value / function"
    u_in = domain_boundary['u']

    # Check if u is a function, if so, ensure it has the correct signature and returns a cuda device
    #  array of the correct type. If not, convert it to a function that returns a cuda device array.
    u_func = validate_user_function(u_in, ctx, expected_signature={'x', 'y', 'z', 't'})



    @ctx.device_decorator
    def get_domain_boundary_updates_kernel(ijk, F, buffer, t):

        i, j, k = ijk
        x, y, z = i + boundary_offset[0] + 0.5, j + boundary_offset[1] + 0.5, k + boundary_offset[2] + 0.5

        f_local = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        for q in range(Q):
            f_local[q] = F[
                q, 
                (i + boundary_offset[0] + Cij_int[q, 0, 0, 0, 0]) % n[0], 
                (j + boundary_offset[1] + Cij_int[q, 0, 0, 0, 1]) % n[1] if D > 1 else 0, 
                (k + boundary_offset[2] + Cij_int[q, 0, 0, 0, 2]) % n[2] if D > 2 else 0, 
                0
            ]

        rho_w = float_t(0.0) # Density at the wall
        for q in range(Q):
            rho_w += f_local[q]

        u_local = u_func(x, y, z, t)


        for q in unknown_populations:
            ci_dot_u = float_t(0.0)
            for d in range(D):
                ci_dot_u += Cij_float[q, 0, 0, 0, d] * u_local[d]
            buffer[q, i, j, k, 0] = f_local[i_opp[q]] + float_t(6.0) * Wi_float[q, 0, 0, 0, 0] * rho_w * ci_dot_u


    return apply_kernel_over_domain_factory(ctx, get_domain_boundary_updates_kernel, slice_shape)