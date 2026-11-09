import numpy as np
from scipy import sparse
from numba import njit, types
from numba.typed import Dict
import itertools as it

from .BSplines import CCBSplines1D, CCBSplines2D, CCBSplines3D
from cff_utils.tuple3D import tuple3D
from ..shapesRavels import ravelf, ravels, ravelp
from ..matrixUtil import getActiveBsplineInds  

# Precomputed full cell moments for deg up to 9 in 1D-3D.
# The origin is at the center of the cell.
fullCellMoments1D = np.array([1., 0., 1/12., 0., 1/80., 0., 1/448., 0., 1/2304., 0.])
fullCellMoments2D = (fullCellMoments1D[:,None] * fullCellMoments1D[None,:])
fullCellMoments3D = fullCellMoments1D[:,None,None] * fullCellMoments1D[None,:,None] * fullCellMoments1D[None,None,:]


key_type = types.UniTuple(types.int64, 2)
value_type = types.float64
def getFi2PC(Cij, N, pgramOverlappedCellInds, cutCellMoments, solidCellInds, shapef, shapes, shapep, deg, onlyPgramOverlapped=False):

    D = Cij.shape[-1]

    Fi2SC = sparse.dok_matrix((np.prod(shapes), np.prod(shapef)), dtype=np.float64)
    SC2PC = sparse.dok_matrix((np.prod(shapep), np.prod(shapes)), dtype=np.float64)

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


    # Select appropriate B-splines
    # CCBSpline: spline coefficients for each delta index within the spline
    if D == 1:
        CCBSpline = CCBSplines1D[deg][:,:,None,None]
    elif D == 2:
        CCBSpline = CCBSplines2D[(deg,deg)][:,:,:,:,None]
    elif D == 3:
        CCBSpline = CCBSplines3D[(deg,deg,deg)]
    else:
        raise ValueError("Cij must have last dimension of size 1, 2 or 3")


    # Indices of the splines that live in any pgram-overlapped cell
    # Note: additional cells are in pgramOverlappedCellInds if onlyPgramOverlapped=False
    validSplineInds = getActiveBsplineInds(pgramOverlappedCellInds, N, deg, Cij)

    for i, Cj in enumerate(Cij.reshape(-1, D)): # Population index
        CjTuple = tuple3D(Cj)

        for si in validSplineInds[CjTuple]: # Spline index
            si = np.array(si)

            # Row index in F2SC
            rowInd = ravels(i, si[:D], shapes)

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
                        moments = fullCellMoments1D[:deg+1][:,None,None]
                    elif D == 2:
                        moments = fullCellMoments2D[:deg+1,:deg+1][:,:,None]
                    elif D == 3:
                        moments = fullCellMoments3D[:deg+1,:deg+1,:deg+1]                
                
                # Integrate the spline over the fluid portion of cell ci
                cellInt = np.sum(CCBSpline[*di]*moments[:deg+1,:deg+1,:deg+1])    

                # Column index in F2SC
                colInd = ravelf(i, ci, shapef)

                Fi2SC[rowInd, colInd] = cellInt

                # Add the polynomial coefficients of the spline to SC2PC
                for pi in it.product(range(deg+1), repeat=D): # Polynomial coefficient index
                    SC2PC[ravelp(i, ci[:D], pi[:D], shapep),
                          ravels(i, si[:D], shapes)] += CCBSpline[*di][*pi]

            # Normalize row to sum to 1
            rowSum = np.sum(Fi2SC[rowInd, :])
            if rowSum > 0:
                Fi2SC[rowInd, :] /= rowSum

    return Fi2SC, SC2PC




# key_type = types.UniTuple(types.int64, 3)
# value_type = types.DictType(types.UniTuple(types.int64, 3), types.boolean)
# def getActiveBsplineInds(pgramOverlappedCellsInds, N, deg, Cij):
#     """Determine the indices of the b-splines that overlap with any pgram-overlapped cell."""
    
#     D = Cij.shape[-1]

#     validSplineInds = Dict.empty(key_type, value_type)
#     for i, Cj in enumerate(Cij.reshape(-1, D)): # Population index
#         CjTuple = (Cj[0], Cj[1], 0) if D == 2 else (Cj[0], Cj[1], Cj[2])

#         for poc in pgramOverlappedCellsInds[CjTuple]:
            
#             for si in it.product(range(-deg, 1), repeat=D): #Spline index
#                 si = si % N
#                 siPadded = np.zeros(3, np.int64)
#                 siPadded[:D] = si
#                 validSplineInds[CjTuple][siPadded] = True
#     return validSplineInds

# key_type = types.UniTuple(types.int64, 2)
# value_type = types.float64
# def getDicts(pgramOverlappedCellInds, Cij, N, shapesRavels, deg=1):

#     F2SCDict = Dict.empty(key_type, value_type)
#     SC2PCDict = Dict.empty(key_type, value_type)

#     shapef, shapes, shapep = shapesRavels.shapef, shapesRavels.shapes, shapesRavels.shapep
#     ravelf, ravels, ravelp = shapesRavels.ravelf, shapesRavels.ravels, shapesRavels.ravelp

#     D = Cij.shape[-1]
#     CCBSplines = CCBSplines2D if D == 2 else CCBSplines3D

#     validSplineInds = getActiveBsplineInds(pgramOverlappedCellInds, N, deg, Cij)

#     for i, Cj in enumerate(Cij.reshape(-1, D)): # Population index
#         CjTuple = (Cj[0], Cj[1], 0) if D == 2 else (Cj[0], Cj[1], Cj[2])

#         for si in validSplineInds[CjTuple]: # Spline index
            
#             # Row index in F2SC
#             ri = ravels(i, si[:D], shapes)

#             for di in it.product(range(deg+1), repeat=D): # Delta index

#                 ci = (si + di)%N

#                 # Column index in F2SC
#                 ci = ravelf(i, ci, shapef)

#                 #Polynomial coefficients of the part of the spline in cell ci
#                 BSpline = CCBSplines[tuple([deg]*D)][*di]

#                 #Integrate the spline over the fluid portion of cell ci. 
#                 cellInt = np.sum(fluidMoments[0, *ci]*BSpline)  

#                 F2SC[ri, ci] = cellInt

#                 #Add the polynomial coefficients of the spline to SC2PC
#                 for pi in it.product(range(deg+1), repeat=D): # Polynomial coefficient index
#                     SC2PC[ravelp(i, *ci, *pi),
#                           ravels(i, *si, 0)] += BSpline[*pi]
            
#             F2SC[ri,:] /= np.sum(F2SC[ri,:]) #Normalize row to sum to 1

#     return F2SCDict, SC2PCDict

# def getS2PC(pgramOverlappedCellInds, Cij, N, shapesRavels, deg=1):

#     F2SCDict, SC2PCDict = MLBS(pgramOverlappedCellInds, Cij, N, shapesRavels, deg=deg)
#     F2SC = dictToSparseMat(F2SCDict, (np.prod(shapesRavels.shapes), np.prod(shapesRavels.shapef)))
#     SC2PC = dictToSparseMat(SC2PCDict, (np.prod(shapesRavels.shapep), np.prod(shapesRavels.shapes)))
#     S2PC = SC2PC @ F2SC
#     return S2PC