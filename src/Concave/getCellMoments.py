from numba import njit
import numpy as np

from .computeCutCellMoments import fullCellMoments1D, fullCellMoments2D, fullCellMoments3D


@njit(cache=True)
def getCellMoments1D(cutCellMoments, solidCellInds, shapep):
    cellMoments = np.zeros((1, shapep[1], shapep[2]), dtype=np.float64)
    cellMoments[0, :, :] = fullCellMoments1D[:shapep[-1],0,0]  # Full cell moments in 1D
    for ci, M in cutCellMoments.items():
        cellMoments[0, ci[0], :] = M[:,0,0]
    for ci in solidCellInds:
        cellMoments[0, ci[0], :] = 0.0
    return cellMoments.flatten()

@njit(cache=True)
def getCellMoments2D(cutCellMoments, solidCellInds, shapep):
    cellMoments = np.zeros((1, shapep[1], shapep[2], shapep[3], shapep[4]), dtype=np.float64)
    cellMoments[0, :, :, :, :] = fullCellMoments2D[:shapep[-2],:shapep[-1],0] # Full cell moments in 2D
    for ci, M in cutCellMoments.items():
        cellMoments[0, ci[0], ci[1], :, :] = M[:,:,0]
    for ci in solidCellInds:
        cellMoments[0, ci[0], ci[1], :, :] = 0.0  # Solid cell area
    return cellMoments.flatten()

@njit(cache=True)
def getCellMoments3D(cutCellMoments, solidCellInds, shapep):
    cellMoments = np.zeros((1, shapep[1], shapep[2], shapep[3], shapep[4], shapep[5], shapep[6]), dtype=np.float64)
    cellMoments[0, :, :, :, :, :, :] = fullCellMoments3D[:shapep[-3],:shapep[-2],:shapep[-1]] # Full cell moments in 3D
    for ci, M in cutCellMoments.items():
        cellMoments[0, ci[0], ci[1], ci[2], :, :, :] = M[:, :, :]
    for ci in solidCellInds:
        cellMoments[0, ci[0], ci[1], ci[2], :, :, :] = 0.0  # Solid cell volume
    return cellMoments.flatten()


def getCellMoments(cutCellMoments, solidCellInds, shapep, D):
    if D == 1:
        return getCellMoments1D(cutCellMoments, solidCellInds, shapep)
    elif D == 2:
        return getCellMoments2D(cutCellMoments, solidCellInds, shapep)
    elif D == 3:
        return getCellMoments3D(cutCellMoments, solidCellInds, shapep)
    else:
        raise ValueError("Invalid dimension D. Must be 1, 2, or 3.")

