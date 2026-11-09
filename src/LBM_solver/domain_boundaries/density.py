import numpy as np

from ..solver_util.validate_user_function import validate_user_function
from ..forcing.get_body_force_local import get_body_force_local_factory
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

    known_populations = np.array(list(set(range(Q)) - set(unknown_populations)))

    get_body_force_local = get_body_force_local_factory(ctx)
    macroscopic = ctx.macroscopic_factory(ctx)
    force_shift_velocity = ctx.force_shift_velocity_factory(ctx)
    equilibrium = ctx.equilibrium_factory(ctx)

    assert 'rho' in domain_boundary, "Domain boundary must specify 'rho' value / function"
    rho_in = domain_boundary['rho']

    # Check if rho is a function, if so, ensure it has the correct signature and returns a cuda device
    #  array of the correct type. If not, convert it to a function that returns a cuda device array.
    rho_func = validate_user_function(rho_in, ctx, expected_signature={'x', 'y', 'z', 't'})

    assert ctx.stencil == 'd2q9', "This function is only implemented for D2Q9 stencil. Please implement the appropriate kernel for your stencil."
    assert side_normal == (-1, 0), "This function is only implemented for the right boundary (side_normal=(-1, 0))."

    @ctx.device_decorator
    def get_domain_boundary_updates_kernel_D2Q9(ijk, F, buffer, t):

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

        f_fluid = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        for q in range(Q):
            f_fluid[q] = F[
                q, 
                (i + boundary_offset[0] + Cij_int[q, 0, 0, 0, 0] - side_normal[0]) % n[0], 
                (j + boundary_offset[1] + Cij_int[q, 0, 0, 0, 1] - side_normal[1]) % n[1] if D > 1 else 0, 
                (k + boundary_offset[2] + Cij_int[q, 0, 0, 0, 2] - side_normal[2]) % n[2] if D > 2 else 0, 
                0
            ]        

        rho_local = rho_func(x, y, z, t)

        # Find velocity one node inwards
        # 2. Get the body force density vector, shape (D,)
        body_force_local = cuda.local.array(D, dtype=float_t) if use_cuda else np.empty(D, dtype=float_t)
        get_body_force_local(ijk, t, body_force_local)
    
        # 3. Compute macroscopic density and velocity
        rho, u = macroscopic(f_fluid)

        # 4. Correct the velocity for the body force
        shift = force_shift_velocity(rho, u, body_force_local)
        for d in range(D):
            u[d] += shift[d]

        # 5. Compute equilibrium distribution, shape (Q,)
        f_eq_fluid = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        equilibrium(rho, u, f_eq_fluid)

        # 6. Compute non-equilibrium
        f_neq_fluid = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        for q in range(Q):
            f_neq_fluid[q] = f_fluid[q] - f_eq_fluid[q]

        f_eq_wall = cuda.local.array(Q, dtype=float_t) if use_cuda else np.empty(Q, dtype=float_t)
        equilibrium(rho_local, u, f_eq_wall)

        for q in unknown_populations:
            buffer[q, i, j, k, 0] = f_eq_wall[q] + f_neq_fluid[i_opp[q]]

        return

    return apply_kernel_over_domain_factory(ctx, get_domain_boundary_updates_kernel_D2Q9, slice_shape)