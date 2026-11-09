import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def compressed_coo_matmul(A: coo_matrix, B: coo_matrix):
    """
    Safely and efficiently multiplies two massive, ultra-sparse COO matrices
    by compressing their active coordinate spaces.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    # 1. Extract unique active coordinates and get their compressed mappings
    # return_inverse=True gives us the compressed coordinates instantly (O(N))
    active_rows_A, A_row_comp = np.unique(A.row, return_inverse=True)
    active_cols_B, B_col_comp = np.unique(B.col, return_inverse=True)

    # 2. Fix the contract dimension bug: We only care about the INTERSECTION.
    # If a column in A doesn't have a matching row in B, its product is 0 anyway.
    active_k, A_k_idx, B_k_idx = np.intersect1d(A.col, B.row, return_indices=True)

    # Filter A and B to only keep elements that exist in the intersection
    # This completely avoids the searchsorted mapping bug
    mask_A = np.isin(A.col, active_k)
    mask_B = np.isin(B.row, active_k)
    
    if not np.any(mask_A) or not np.any(mask_B):
        # Quick exit if there is no overlap in the contracted dimension
        return coo_matrix((A.shape[0], B.shape[1]), dtype=A.dtype)

    # Re-filter and compress the contracted dimension safely using a map
    k_map = {orig_id: comp_id for comp_id, orig_id in enumerate(active_k)}
    A_col_comp = np.array([k_map[x] for x in A.col[mask_A]])
    B_row_comp = np.array([k_map[x] for x in B.row[mask_B]])

    # 3. Construct the compressed matrices
    # We must also apply the masks to the outer dimensions and data
    Ac = csr_matrix(
        (A.data[mask_A], (A_row_comp[mask_A], A_col_comp)),
        shape=(len(active_rows_A), len(active_k))
    )
    
    Bc = csr_matrix(
        (B.data[mask_B], (B_row_comp, B_col_comp[mask_B])),
        shape=(len(active_k), len(active_cols_B))
    )

    # 4. Multiply in compressed space
    Cc = (Ac @ Bc).tocoo()

    # 5. Map back to original massive coordinate space
    orig_rows = active_rows_A[Cc.row]
    orig_cols = active_cols_B[Cc.col]

    return coo_matrix(
        (Cc.data, (orig_rows, orig_cols)),
        shape=(A.shape[0], B.shape[1])
    )