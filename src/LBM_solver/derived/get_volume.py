
from ..solver_util.get_slice_inds import get_slice_inds
from ..solver_util.apply_kernel_over_domain import apply_kernel_over_domain_factory

def get_volume_factory(ctx, slices_ijk=(slice(None), slice(None), slice(None))):

    use_cuda = ctx.use_cuda
    cell_volumes = ctx.cell_volumes

    assert cell_volumes is not None, "cell_volumes must be provided in the context to use get_volume_factory."

    (nx_, ny_, nz_), (x0, y0, z0) = get_slice_inds(ctx, slices_ijk)

    @ctx.device_decorator
    def get_volume(ijk, F, volume, t):


        x_offset, y_offset, z_offset = ijk
        volume[0, x_offset, y_offset, z_offset, 0] = cell_volumes[0, x0+x_offset, y0+y_offset, z0+z_offset, 0]

        return volume

    return apply_kernel_over_domain_factory(ctx, get_volume, (nx_, ny_, nz_))



        