import numpy as np
from numba import njit

@njit(cache=True)
def getSubzoneMatrixRowsColsValues(subzoneMatrix):
    # Return three numpy arrays with the row indices, column indices and values of the subzone matrix
    row_indices = np.empty(8*len(subzoneMatrix.keys()), dtype=np.int64)
    col_indices = np.empty(8*len(subzoneMatrix.keys()), dtype=np.int64)
    values = np.empty(8*len(subzoneMatrix.keys()), dtype=np.float64)

    count = 0
    for colInd in subzoneMatrix.keys():
        for rowInd in subzoneMatrix[colInd].keys():
            value = subzoneMatrix[colInd][rowInd]

            assert count < len(row_indices), "Error: preallocated arrays are too small."

            row_indices[count] = rowInd
            col_indices[count] = colInd
            values[count] = value
            count += 1
    return row_indices[:count], col_indices[:count], values[:count]