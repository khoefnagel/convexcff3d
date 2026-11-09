import numpy as np
from scipy import sparse
from numba import njit


from cff_utils.unravel_index import unravel_index
from Matrices.shapesRavels import ravelf


@njit(cache=True)
def get_streamed_col_inds(cols, Cij, N, shapef):

    D = Cij.shape[-1]
    Cij_squeezed = Cij.reshape(-1, D) # shape (Q, D)

    cols_new = np.empty_like(cols)
    for j, col_ind in enumerate(cols):
        col_ind_multi = unravel_index(col_ind, shapef)
        i = col_ind_multi[0]
        ci = col_ind_multi[1:D+1]
        ci_streamed = (ci + Cij_squeezed[i]) % np.asarray(N)
        col_ind_streamed = ravelf(i, ci_streamed, shapef)
        cols_new[j] = col_ind_streamed

    return cols_new

# @njit(cache=True)
def convert_CSM_to_post(CSM, N, Cij, shapef):
    """Convert the complete streaming matrix (CSM) from pre-stream populations to post-stream corrections.
    
    Parameters
    ----------
    CSM : scipy.sparse.spmatrix
        Complete streaming matrix in CSR format, shape (Q*prod(N), Q*prod(N)).
    
    Returns
    -------
    CSM_post : scipy.sparse.spmatrix
        Complete streaming matrix in CSR format, shape (Q*prod(N), Q*prod(N)), but from post-stream populations to post-stream corrections.
    """
    # The CSM is currently from pre-stream populations to post-stream corrections. We want to convert it to be from post-stream populations to post-stream corrections.

    CSM = CSM.tocoo() # shape (Q*prod(N), Q*prod(N))

    # Get the new column indices after streaming
    cols_new = get_streamed_col_inds(CSM.col, Cij, N, shapef)

    CSM = sparse.coo_matrix((CSM.data, (CSM.row, cols_new)), shape=CSM.shape)

    return CSM