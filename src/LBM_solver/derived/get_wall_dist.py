
from ..solver_util.get_slice_inds import get_slice_inds
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def get_wall_dist_factory(ctx, slices_ijk=(slice(None), slice(None), slice(None))):

    D = ctx.D
    wall_dists = ctx.wall_dists

    assert wall_dists is not None, "wall_dists must be provided in the context to use get_wall_dist_factory."

    (nx_, ny_, nz_), (x0, y0, z0) = get_slice_inds(ctx, slices_ijk)

    @ctx.device_decorator
    def get_wall_dist(ijk, F, wall_dist, t):


        x_offset, y_offset, z_offset = ijk
        wall_dist[0, x_offset, y_offset, z_offset, 0] = wall_dists[0, x0+x_offset, y0+y_offset, z0+z_offset, 0]

        return wall_dist

    return apply_kernel_over_domain_factory(ctx, get_wall_dist, (nx_, ny_, nz_))