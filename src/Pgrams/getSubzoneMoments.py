import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D
from cff_utils.polygonMoment import polygonMoment
from cff_utils.tetravol import getTetraMoments



@njit(cache=True, inline='always')
def subzoneMoments1D(point1, point2, degs, deg):
    """Compute the moments of a single subzone in 1D defined by subzone bounds,
    represented as points.

    Parameters
    ----------
    point1 : np.ndarray
        A 1D array of shape (1,) representing the start and end points of the first point.
    point2 : np.ndarray
        A 1D array of shape (1,) representing the start and end points of the second point.
    degs : np.ndarray
        A 2D array of shape (numDegs, 1) representing the degrees of the moments to compute.
    deg : int
        The maximum degree of the moments to compute.


    Returns
    -------
    moments : np.ndarray
        A 3D array of shape (deg+1, 1, 1) containing the computed fluid moments for the subzone.
    origCellInd : np.ndarray
        A 1D array of shape (3,) representing the cell index where the subzone lives.
    """
    
    origCellInd = np.floor((point1 + point2) / 2)

    # Shift points to be relative to cell origin
    point1 -= origCellInd
    point2 -= origCellInd 

    moments = np.empty((deg+1, 1, 1), dtype=np.float64)
    for degx in degs:
        degx = degx.item()
        moments[degx, 0, 0] = (point2**(degx+1) - point1**(degx+1)) / (degx+1)

    # In 1D it is tricky to get the sign of the moment right
    # We assume the zeroth moment (volume) is always positive, and if not, flip all signs
    if moments[0,0,0] < 0:
        moments *= -1

    return moments, np.array([origCellInd, 0, 0], dtype=np.int64)

@njit(cache=True, inline='always')
def subzoneMoments2D(segment1, segment2, degs, deg):
    """Compute the moments of a single subzone in 2D defined by subzone bounds,
    represented as line segments.
    
    Parameters
    ----------
    segment1 : np.ndarray
        A 2D array of shape (2, 2) representing the start and end points of the first line segment.
    segment2 : np.ndarray
        A 2D array of shape (2, 2) representing the start and end points of the second line segment.
    degs : np.ndarray
        A 2D array of shape (numDegs, 2) representing the degrees of the moments to compute.
    deg : int
        The maximum degree of the moments to compute.
        
    Returns
    -------
    moments : np.ndarray
        A 3D array of shape (deg+1, deg+1, 1) containing the computed fluid moments for the subzone.
    origCellInd : np.ndarray
        A 1D array of shape (3,) representing the cell index where the subzone lives.
    """

    polygon = np.empty((5, 2))
    polygon[0] = segment1[0]
    polygon[1] = segment2[0]
    polygon[2] = segment2[1]
    polygon[3] = segment1[1]
    polygon[4] = segment1[0]

    # Compute cell index where the subzone lives
    origCellInd = np.floor(np.sum(polygon, axis=0) / polygon.shape[0])

    # Shift points to be relative to cell origin
    polygon -= origCellInd

    moments = np.empty((deg+1, deg+1, 1), dtype=np.float64)

    for degx, degy in degs:
        moments[degx, degy, 0] = polygonMoment(polygon, degx, degy)

    return moments, np.array([origCellInd[0], origCellInd[1], 0], dtype=np.int64)


@njit(cache=True, inline='always')
def subzoneMoments3D(facet1, facet2, degs, deg):
    """Compute the moments of a single subzone in 3D defined by subzone bounds,
    represented as facets.
    
    Parameters
    ----------
    facet1 : np.ndarray
        A 2D array of shape (nPoints, 3) representing the vertices of the first facet.
    facet2 : np.ndarray
        A 2D array of shape (nPoints, 3) representing the vertices of the second facet.
    degs : np.ndarray
        A 2D array of shape (numDegs, 3) representing the degrees of the moments to compute.
    deg : int
        The maximum degree of the moments to compute.
        
    Returns
    -------
    moments : np.ndarray
        A 3D array of shape (deg+1, deg+1, deg+1) containing the computed fluid moments for the subzone.
    origCellInd : np.ndarray
        A 1D array of shape (3,) representing the cell index where the subzone lives.
    """


    # Compute cell index where the subzone lives
    origCellInd = np.floor((np.sum(facet1, axis=0) + np.sum(facet2, axis=0)) / (facet1.shape[0] + facet2.shape[0]))

    # Shift points to be relative to cell origin
    facet1Loc = facet1 - origCellInd
    facet2Loc = facet2 - origCellInd

    moments = np.zeros((deg+1, deg+1, deg+1), dtype=np.float64)

    # Moments are computed by decomposing the subzone faces into tetrahedra with one vertex at the origin
    # and summing the moments of these tetrahedra.
    for j in range(1, facet1Loc.shape[0]-1):

        # Facet
        moments += getTetraMoments(np.vstack((facet1Loc[0], facet1Loc[j+1], facet1Loc[j])), degs, deg)

        # End of pgram. Reverse j+1 to j to get correct orientation
        moments += getTetraMoments(np.vstack((facet2Loc[0], facet2Loc[j], facet2Loc[j+1])), degs, deg)

    # Sides
    for j in range(facet1Loc.shape[0]):
        j2 = (j + 1) % facet1Loc.shape[0]
        moments += getTetraMoments(np.vstack((facet1Loc[j], facet1Loc[j2], facet2Loc[j2])), degs, deg)
        moments += getTetraMoments(np.vstack((facet1Loc[j], facet2Loc[j2], facet2Loc[j])), degs, deg)

    return moments, origCellInd.astype(np.int64)


@njit(cache=True)
def getSubzoneMomentsCj(subzoneBounds, Cj, N, D, degs, deg):
    """Function to obtain the subzone moments for a single Cj.
    
    Parameters
    ----------
    subzoneBounds : list
        List of subzone bounds arrays for a single Cj.
    Cj : array_like
        Lattice velocity vector, shape (D, ).
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    D : int
        Spatial dimension (1, 2 or 3).
    degs : array_like
        Array of degree tuples for which to compute moments.
    deg : int
        Maximum degree to compute.
    
    Returns
    -------
    subzoneMoments : array_like
        Array of subzone moments, shape (nSubzones, *shapeM).
    origCellInds : array_like
        Array of origin cell indices, shape (nSubzones, 3).
    destCellInds : array_like
        Array of destination cell indices, shape (nSubzones, 3).
    """
    nSubzones = subzoneBounds[0].shape[1] - 1

    shapeM = (deg+1, 1, 1) if D == 1 else (deg+1, deg+1, 1) if D == 2 else (deg+1, deg+1, deg+1)
    subzoneMoments = np.zeros((len(subzoneBounds), nSubzones, *shapeM), dtype=np.float64)*np.nan
    origCellInds = np.empty((len(subzoneBounds), nSubzones, 3), dtype=np.int64)
    destCellInds = np.empty((len(subzoneBounds), nSubzones, 3), dtype=np.int64)

    for pi, pgram in enumerate(subzoneBounds):
        for si in range(pgram.shape[1] - 1):
            polygon1 = pgram[:, si] 
            polygon2 = pgram[:, si+1]
            if D == 1:
                M, origCellInd = subzoneMoments1D(polygon1.item(), polygon2.item(), degs, deg)
            elif D == 2:
                M, origCellInd = subzoneMoments2D(polygon1, polygon2, degs, deg)
            elif D == 3:
                M, origCellInd = subzoneMoments3D(polygon1, polygon2, degs, deg)
            else:
                raise ValueError("Cij must have last dimension of size 1, 2 or 3")

            subzoneMoments[pi, si,:,:,:] = M[:,:,:]
            origCellInds[pi, si,:] = origCellInd[:]
            origCellInds[pi, si,:D] = origCellInd[:D] % N # Wrap around periodic domain

        destCellInds[pi, :, :] = origCellInds[pi, ::-1, :]


    return subzoneMoments, origCellInds, destCellInds


key_type = types.UniTuple(types.int64, 3)
value_type_float2D = types.float64[:,:]
value_type_float5D = types.float64[:,:,:,:,:]
value_type_int3D = types.int64[:,:,:]
@njit(cache=True)
def getSubzoneMoments(subzoneBounds, Cij, N, degs, deg):
    """Function to obtain the subzone moments of all subzones.
    
    Parameters
    ----------
    subzoneBounds : numba.typed.Dict
        Dictionary mapping Cj tuples to lists of subzone bounds arrays.
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    degs : array_like
        Array of degree tuples for which to compute moments.
    deg : int
        Maximum degree to compute.
        
    Returns
    -------
    subzoneMoments : numba.typed.Dict
        Dictionary mapping Cj tuples to subzone moments arrays.
    origCellInds : numba.typed.Dict
        Dictionary mapping Cj tuples to origin cell indices arrays.
    destCellInds : numba.typed.Dict
        Dictionary mapping Cj tuples to destination cell indices arrays.
    """

    D = Cij.shape[-1]


    subzoneMoments = Dict.empty(key_type, value_type_float5D)
    origCellInds = Dict.empty(key_type, value_type_int3D)
    destCellInds = Dict.empty(key_type, value_type_int3D)
    
    for Cj in Cij.reshape(-1, D):
        CjTuple = tuple3D(Cj)

        # Skip still populations (no subzones)
        if len(subzoneBounds[CjTuple]) == 0: 
            subzoneMoments[CjTuple] = np.empty((0, 0, 0, 0, 0), dtype=np.float64)
            origCellInds[CjTuple] = np.empty((0, 0, 3), dtype=np.int64)
            destCellInds[CjTuple] = np.empty((0, 0, 3), dtype=np.int64)
            continue

        # Compute subzone moments for this Cj
        M, oci, dci = getSubzoneMomentsCj(subzoneBounds[CjTuple], Cj, N, D, degs, deg)

        subzoneMoments[CjTuple] = M
        origCellInds[CjTuple] = oci
        destCellInds[CjTuple] = dci
    
    return subzoneMoments, origCellInds, destCellInds
