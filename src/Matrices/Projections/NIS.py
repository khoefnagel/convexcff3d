"""Module to perform neighbour informed spline (NIS) projection 
from populations to spline coefficients."""


import numpy as np
import scipy.sparse as sparse
from numba import njit, types
from numba.typed import Dict, List
import itertools as it 

from ..shapesRavels import ravelf, ravelp
from ..matrixUtil import dictToSparseMat

from cff_utils.tuple3D import tuple3D

key_type = types.UniTuple(types.int64, 2)
value_type = types.float64
@njit(cache=True)
def getFi2PCDict(Cij, N, pgramOverlappedCellInds, cutCellMoments, cutCellCentroidsLoc, solidCellInds,
                shapef, shapep, degs):
    """Get the Fi2PC projection matrix to go from populations to spline coefficients
    using neighbour informed splines (NIS). It is returned as a numba typed dictionary,
    where the keys are tuples of (row, col) indices and the values are the matrix entries.
    
    Parameters
    ----------
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    pgramOverlappedCellInds : dict
        Dictionary mapping cell indices to list of pgram-overlapped cell indices.
    cutCellMoments : dict
        Dictionary mapping cell indices to cut-cell moments arrays.
    cutCellCentroidsLoc : dict
        Dictionary mapping cell indices to cut-cell centroid locations.
    solidCellInds : dict
        Dictionary with the solid cell indices as keys.
    shapef : array_like
        Shape of the population array, shape (Q, *N, 1).
    shapep : array_like
        Shape of the spline coefficient array, shape (Q, *N, *[deg+1]).
    degs : array_like"""

    Q = Cij.shape[0]
    D = Cij.shape[-1]

    Fi2PCDict = Dict.empty(key_type, value_type)

    CijSqueezed = Cij.reshape(Q, D)

    # Jit compiled ravel function requires arrays
    shapef, shapep = np.array(shapef), np.array(shapep)

    # Neighbours to consider
    deltaInds = CijSqueezed[1:] #Skip rest particle
    
    nNeighbours = len(deltaInds)

    # The idea of NIS is to solve a system of equations Ax=B for each cut cell.
    # Each row corresponds to enforcing conservation in one of the neighbour cells
    # (which is equivalent to passing through the centroid for first order splines).
    # Note that solid cells are skipped, so these rows are not included in A and B.
    # Each column of A corresponds to one degree of freedom in the spline representation.
    # However, one degree of freedom is taken out of A to ensure exact conservation
    # in the cell itself. Thus, A has shape (nValidNeighbours, D), where nValidNeighbours
    # is the number of neighbour cells that are not solid. 
    # The system of equations is then solved using volume weighted least squares.
    
    # Initialize with full shapes, slice later
    A = np.empty((nNeighbours, D), dtype=np.float64)
    B = np.concatenate((-np.ones((nNeighbours, 1)),
                            np.diag(np.ones(nNeighbours))), axis=1)   

    # Transpose of the A matrix time the weight matrix (diagonal of the weights)
    ATW = np.zeros((D, nNeighbours), dtype=np.float64)
    

    for i in range(Q):
        Cj = CijSqueezed[i]
        CjTuple = tuple3D(Cj)

        for ci in pgramOverlappedCellInds[CjTuple]:
            # Remove padding
            ci = np.array(ci)[:D]
            ciTuple = tuple3D(ci)

            # Indices of valid neighbours (not solid)
            validInds = np.zeros(nNeighbours, dtype=np.bool_)

            # Get centroid of cell ci
            if ciTuple in cutCellCentroidsLoc:
                xi = cutCellCentroidsLoc[ciTuple][:D]
            else:
                xi = 0.5*np.ones(D)

            # Loop over neighbours
            for j, ciDelta in enumerate(deltaInds):

                # Neighbour cell index
                cj = (ci + ciDelta)%N
                cjTuple = tuple3D(cj)
                
                # Skip solid cells
                if cjTuple in solidCellInds:
                    continue

                if cjTuple in cutCellMoments and cutCellMoments[cjTuple][0,0,0] < 1e-12:
                    continue

                validInds[j] = True

                # Get centroid of neighbour cell cj and fill A and ATW matrices
                if cjTuple in cutCellCentroidsLoc:
                    xj = cutCellCentroidsLoc[cjTuple][:D] + ciDelta
                    ATW[:, j] = (xj - xi) * cutCellMoments[cjTuple][0,0,0] #Volume
                else:
                    xj = ciDelta.astype(np.float64) + 0.5
                    ATW[:, j] = (xj - xi)
                A[j, :] = xj - xi

            # Slice A, ATW and B to only valid neighbours
            ATWcut = ATW[:, validInds]
            Acut = A[validInds, :]
            Bcut = B[validInds]

            # Handling of case where all neighbour cells are solid => assume piecewise constant
            if not np.any(validInds): 
                Fi2PCDict[(ravelp(i, ci, np.zeros(D, np.int64), shapep), ravelf(i, ci, shapef))] = 1.
                continue

            # Solve volume-weighted least squares system for spline coeffs C
            ATWAcut = ATWcut @ Acut
            if abs(np.linalg.det(ATWAcut)) < 1e-12:
                C = np.linalg.pinv(ATWAcut) @ ATWcut @ Bcut
            else:
                C = np.linalg.inv(ATWAcut) @ ATWcut @ Bcut

            # Fill Fi2PCDict using the solved coefficients
            for j, ciDelta in enumerate(CijSqueezed): 
                cj = (ci + ciDelta)%N
                
                cjTuple = tuple3D(cj)
                if cjTuple in solidCellInds:
                    continue

                coeffs = np.zeros(D+1)
                coeffs[1:] = C[:,j]

                # Constant coefficient follows from the others 
                coeffs[0] = -np.sum(coeffs[1:] * xi)

                # Ensure exact conservation in cell ci
                if np.all(ciDelta == 0):
                    coeffs[0] += 1
                
                # Add to Fi2PCDict
                for deg, coeff in zip(degs, coeffs):
                    key = (ravelp(i, ci, deg, shapep), ravelf(i, cj, shapef))
                    if key not in Fi2PCDict:
                        Fi2PCDict[key] = 0.0
                    Fi2PCDict[key] += coeff
    return Fi2PCDict

# Not jittable
def getFi2PC(Cij, N, pgramOverlappedCellInds, cutCellMoments, cutCellCentroidsLoc, solidCellInds, 
            shapef, shapep, degs):
    """Wrapper around jitted getFi2PCDict function. Converts the numba dict to scipy sparse matrix."""
    Fi2PCDict = getFi2PCDict(Cij, N, pgramOverlappedCellInds, cutCellMoments, cutCellCentroidsLoc, solidCellInds, 
                             shapef, shapep, degs)
    return dictToSparseMat(Fi2PCDict, (np.prod(shapep), np.prod(shapef)))


    


