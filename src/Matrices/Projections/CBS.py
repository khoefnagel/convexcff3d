"""Module to perform conservative B-spline (CBS) projection
from populations to spline coefficients."""

import itertools as it
import numpy as np
from scipy import sparse

from numba import njit, types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D

from .BSplines import CCBSplines1D, CCBSplines2D, CCBSplines3D
from .gradBSplines import (gradDotGradCCBSplines1D, gradDotGradCCBSplines2D, gradDotGradCCBSplines3D,
                            grad2DotGrad2CCBSplines1D, grad2DotGrad2CCBSplines2D, grad2DotGrad2CCBSplines3D)
from ..shapesRavels import ravelf, ravels, ravelp
from ..matrixUtil import dictToSparseMat, getActiveBsplineInds  

# Precomputed full cell moments for deg up to 9 in 1D-3D.
# The origin is at the center of the cell.
fullCellMoments1D = np.array([1., 0., 1/12., 0., 1/80., 0., 1/448., 0., 1/2304., 0.])
fullCellMoments2D = (fullCellMoments1D[:,None] * fullCellMoments1D[None,:])
fullCellMoments3D = fullCellMoments1D[:,None,None] * fullCellMoments1D[None,:,None] * fullCellMoments1D[None,None,:]


key_type = types.UniTuple(types.int64, 2)
value_type = types.float64
def CBS(Cij, N, pgramOverlappedCellInds, cutCellMoments, cutCellCentroidsLoc, solidCellInds,
                shapef, shapes, shapep, degs, deg, onlyPgramOverlapped=False, minDeriv=1): # Conservative B-spline
    """Get the Fi2PC projection matrix to go from populations to spline coefficients
    using conservative B-splines (CBS). It is returned as a numba typed dictionary,
    where the keys are tuples of (row, col) indices and the values are the matrix entries."""

    # Maximum degree for gradient calculations
    gradDegMax = 2*deg+1

    # Lagrange multiplier matrix
    CDict = Dict.empty(key_type, value_type)
    
    # Minimize gradient of the spline matrix
    HDict = Dict.empty(key_type, value_type)

    # Spline coefficient to polynomial coefficient matrix
    SC2PCDict = Dict.empty(key_type, value_type)

    D = Cij.shape[-1]

    # Select appropriate B-splines and their gradient products
    # CCBSpline: spline coefficients for each delta index within the spline
    # gradDotGradCCBSpline: integral of the dot product of the gradients of two B-splines
    # grad2DotGrad2CCBSpline: integral of the dot product of the second derivatives of two B-splines
    if D == 1:
        CCBSpline = CCBSplines1D[deg][:,:,None,None]
        gradDotGradCCBSpline = gradDotGradCCBSplines1D[deg][...,None,None]
        grad2DotGrad2CCBSpline = grad2DotGrad2CCBSplines1D[deg][...,None,None]
    elif D == 2:
        CCBSpline = CCBSplines2D[(deg,deg)][:,:,:,:,None]
        gradDotGradCCBSpline = gradDotGradCCBSplines2D[deg][...,None]
        grad2DotGrad2CCBSpline = grad2DotGrad2CCBSplines2D[deg][...,None]
    elif D == 3:
        CCBSpline = CCBSplines3D[(deg,deg,deg)]
        gradDotGradCCBSpline = gradDotGradCCBSplines3D[deg]
        grad2DotGrad2CCBSpline = grad2DotGrad2CCBSplines3D[deg]
    else:
        raise ValueError("Cij must have last dimension of size 1, 2 or 3")

    # Indices of the splines that live in any pgram-overlapped cell
    # Note: additional cells are in pgramOverlappedCellInds if onlyPgramOverlapped=False
    validSplineInds = getActiveBsplineInds(pgramOverlappedCellInds, N, deg, Cij)

    for i, Cj in enumerate(Cij.reshape(-1, D)): # Population index
        CjTuple = tuple3D(Cj)

        for si in validSplineInds[CjTuple]: # Spline index
            si = np.array(si)
            for di in it.product(range(deg+1), repeat=D): # Delta index within the spline
                ci = (si[:D] + np.array(di)) % N # Cell index
                ciTuple = tuple3D(ci)

                # Only consider cells in pgramOverlappedCells for the spline construction
                if ciTuple not in pgramOverlappedCellInds[CjTuple]:
                    continue

                if ciTuple in solidCellInds:
                    continue
                
                # Get the cell moments of cell ci
                if ciTuple in cutCellMoments:
                    moments = cutCellMoments[ciTuple]
                else:
                    if D == 1:
                        moments = fullCellMoments1D[:gradDegMax][:,None,None]
                    elif D == 2:
                        moments = fullCellMoments2D[:gradDegMax,:gradDegMax][:,:,None]
                    elif D == 3:
                        moments = fullCellMoments3D[:gradDegMax,:gradDegMax,:gradDegMax]                
                
                # Integrate the spline over the fluid portion of cell ci
                splineInt = np.sum(CCBSpline[*di]*moments[:deg+1,:deg+1,:deg+1])

                # Fill in the C matrix (conservation constraints)
                rowInd = ravelf(i, ci[:D], np.array(shapef))
                colInd = ravels(i, si[:D], np.array(shapes))
                if (rowInd, colInd) not in CDict:
                    CDict[(rowInd, colInd)] = 0.
                CDict[(rowInd, colInd)] += splineInt/moments.flatten()[0] # Normalize by volume of the cell

                # Fill in the spline coefficient to polynomial coefficient matrix (SC2PC)
                for pi in it.product(range(deg+1), repeat=D): # Polynomial coefficient index
                    rowInd = ravelp(i, ci[:D], pi, np.array(shapep))
                    colInd = ravels(i, si[:D], np.array(shapes))
                    if  (rowInd, colInd) not in SC2PCDict:
                        SC2PCDict[(rowInd, colInd)] = 0.
                    SC2PCDict[(rowInd, colInd)] += CCBSpline[*di][*pi].item()
                
                # Fill in the H matrix (minimize gradient or second derivative of the spline)
                # Loop over the second spline. di2 is again the delta index within the spline.
                for di2 in it.product(range(deg+1), repeat=D):

                    # Find the second spline index based on the cell index ci and delta index di2
                    # E.g. if ci = (6,9) and di2 = (3,2) then si2 = (3,7)
                    si2 = (ci - np.array(di2)) % N
                    si2Tuple = tuple3D(si2)

                    if si2Tuple not in validSplineInds[CjTuple]:
                        continue
                    
                    # Whether the minimize either the first or second derivative
                    if minDeriv == 1:
                        # Compute integral of dot product of gradients of the two splines
                        gradDotGradSplineInt = np.sum(gradDotGradCCBSpline[*di, *di2] * moments)
                    elif minDeriv == 2:
                        # Compute integral of dot product of second derivatives of the two splines
                        gradDotGradSplineInt = np.sum(grad2DotGrad2CCBSpline[*di, *di2] * moments)
                    else:
                        raise ValueError("minDeriv must be 1 or 2")

                    # Fill in the H matrix
                    rowInd2 = ravels(i, si2[:D], np.array(shapes))
                    colInd2 = ravels(i, si[:D], np.array(shapes))
                    if (rowInd2, colInd2) not in HDict:
                        HDict[(rowInd2, colInd2)] = 0.
                    HDict[(rowInd2, colInd2)] += gradDotGradSplineInt
                                
    return CDict, HDict, SC2PCDict


def getFi2PC(Cij, N, pgramOverlappedCellInds, cutCellMoments, cutCellCentroidsLoc, solidCellInds,
            shapef, shapes, shapep, degs, deg, onlyPgramOverlapped=True, minDeriv=1):
    """Get the Fi2PC projection matrix to go from populations to polynomial spline coefficients
    using conservative B-splines (CBS). It is returned as a scipy sparse matrix.
    
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
        Dictionary mapping cell indices to cut-cell centroid locations in local coordinates.
    solidCellInds : dict
        Dictionary with the solid cell indices as keys.
    shapef : array_like
        Shape of the population array, shape (Q, *N, 1).
    shapes : array_like
        Shape of the spline coefficient array, shape (Q, *N, 1).
    shapep : array_like
        Shape of the polynomial coefficient array, shape (Q, *N, *[deg+1]).
    degs : list of tuples
        List of degrees used in each coordinate direction for the splines.
    deg : integer
        Degree used for the splines.
    onlyPgramOverlapped : bool, optional
        If True, only consider splines that live in pgram-overlapped cells, by default True.
    minDeriv : int, optional
        Whether to minimize first (1) or second (2) derivative of the spline, by default 1.

    Returns
    -------
    Fi2PC : sparse matrix
        Fi2PC projection matrix as a scipy sparse matrix, shape (np.prod(shapep), np.prod(shapef)).
    """

    if minDeriv == 2 and deg < 2:
        raise ValueError("minDeriv=2 requires deg >= 2")

    D = Cij.shape[-1]

    # If onlyPgramOverlapped is False, expand the pgramOverlappedCellInds to include all cells
    # that are within 'deg' distance from the pgram-overlapped cells, since these
    # could influence the splines that live in pgram-overlapped cells.
    if not onlyPgramOverlapped:
        pgramOverlappedCellInds2 = Dict.empty(types.UniTuple(types.int64, 3), types.DictType(types.UniTuple(types.int64, 3), types.boolean))
        for i, Cj in enumerate(Cij.reshape(-1, D)):# Population index
            CjTuple = tuple3D(Cj)
            pgramOverlappedCellInds2[CjTuple] = Dict.empty(types.UniTuple(types.int64, 3), types.boolean)
            for ci in pgramOverlappedCellInds[CjTuple]:
                for di in it.product(range(-deg, deg+1), repeat=D): # Delta index within the spline
                    ci2 = (np.array(ci)[:D] + np.array(di)) % N
                    ci2Tuple = tuple3D(ci2)
                    if ci2Tuple in solidCellInds:
                        continue
                    if ci2Tuple not in pgramOverlappedCellInds2[CjTuple]:
                        pgramOverlappedCellInds2[CjTuple][ci2Tuple] = True
        pgramOverlappedCellInds = pgramOverlappedCellInds2


    # Get the CBS matrices as numba typed dictionaries and convert to scipy sparse matrices
    # The system to solve is:
    # | H   C^T | | Fi2SC | = | 0  |
    # | C    0  | |  Lagr |   | Fi |
    CDict, HDict, SC2PCDict = CBS(Cij, N, pgramOverlappedCellInds, cutCellMoments, cutCellCentroidsLoc, solidCellInds,
                shapef, shapes, shapep, degs, deg, onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv)
    C = dictToSparseMat(CDict, (np.prod(shapes), np.prod(shapef)))
    H = dictToSparseMat(HDict, (np.prod(shapes), np.prod(shapes)))
    SC2PC = dictToSparseMat(SC2PCDict, (np.prod(shapep), np.prod(shapes)))

    validSplineInds = getActiveBsplineInds(pgramOverlappedCellInds, N, deg, Cij)

    
    blocks = []
    # A sub-system is solved for each population direction since these are decoupled
    for i, Cj in enumerate(Cij.reshape(-1,D)):

        CjTuple = tuple3D(Cj)

        NN = np.prod(N)

        # C and H matrices for population i
        Cpopi = C[i*NN:(i+1)*NN, i*NN:(i+1)*NN]
        Hpopi = H[i*NN:(i+1)*NN, i*NN:(i+1)*NN]

        # Indices of cells and splines to consider for this population direction
        cellInds = np.array(list(pgramOverlappedCellInds[CjTuple].keys()))
        splineInds = np.array(list(validSplineInds[CjTuple].keys()))

        # If no cells or splines to consider, add empty block
        if len(cellInds) == 0 or len(splineInds) == 0:
            blocks.append(sparse.csc_matrix((np.prod(N), np.prod(N))))  
            continue

        # Ravel the cell and spline indices
        cellInds = np.sort(np.ravel_multi_index(cellInds[:,:D].T, N))
        splineInds = np.sort(np.ravel_multi_index(splineInds[:,:D].T, N))
        
        # Extract the sub-matrices for the current population direction where
        # the rows/columns correspond to only the relevant cells/splines.
        # Otherwise the system is singular.
        Hpopi2 = Hpopi[splineInds][:,splineInds]
        Cpopi2 = Cpopi[cellInds][:,splineInds]

        # Construct and invert the system matrix
        A = sparse.bmat([[Hpopi2, Cpopi2.T],
                         [Cpopi2, None]], format='csc')
        Ainv = np.linalg.pinv(A.toarray())
        Ainv = sparse.csc_matrix(Ainv)

        # Extract the Fi2SC matrix from the inverted system, this is the upper right block.
        Fi2SCpopi = sparse.dok_matrix((np.prod(N), np.prod(N)))
        AinvUpperRight = Ainv[:len(splineInds), -len(cellInds):]
        for r, c, v in zip(AinvUpperRight.tocoo().row, AinvUpperRight.tocoo().col, AinvUpperRight.tocoo().data):
            Fi2SCpopi[splineInds[r], cellInds[c]] = v

        blocks.append(Fi2SCpopi.tocsc())
    
    # Combine the population direction blocks into a block diagonal matrix
    # of full system.
    Fi2SC = sparse.block_diag(blocks, format='csc')
        
    return SC2PC, Fi2SC, H, C


