from numba import njit, types
from numba.typed import Dict




@njit(cache=True)
def computePgramMoments(subzoneMatrixNaive):

    pgramMoments = Dict.empty(key_type=types.int64, value_type=types.float64)
    for col_ind in subzoneMatrixNaive.keys():
        pgramMoments[col_ind] = 0.0
        for row_ind, entry in subzoneMatrixNaive[col_ind].items():
            pgramMoments[col_ind] += entry
    return pgramMoments