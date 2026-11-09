import numpy as np

from ..solver_util.set_vector_zero import set_vector_zero_factory
from ..solver_util.sparse_matrix_vector_product_update import sparse_matrix_vector_product_update_factory


def get_total_force_factory(ctx):

    use_cuda = ctx.use_cuda
    float_t = ctx.float_t

    if use_cuda:
        from numba import cuda

    set_vector_zero = set_vector_zero_factory(ctx)
    sparse_matrix_vector_product_update = sparse_matrix_vector_product_update_factory(ctx, ctx.Fi2Ftot.row, ctx.Fi2Ftot.col, ctx.Fi2Ftot.data)

    Ftot_shape = (ctx.Fi2Ftot.shape[0],)
    Ftot = cuda.device_array(Ftot_shape, dtype=ctx.float_t) if use_cuda else np.empty(Ftot_shape, dtype=float_t)

    def get_total_force(F):
        set_vector_zero(Ftot)
        sparse_matrix_vector_product_update(F.ravel(), Ftot)
        return Ftot

    return get_total_force



