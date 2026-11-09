import numpy as np
from numba import njit, types
from numba.typed import Dict

from cff_utils.tuple3D import tuple3D
from Matrices.shapesRavels import ravelp

# Precomputed full cell moments for deg up to 9 in 1D-3D.
# The origin is at the center of the cell.
fullCellMoments1D = (1./np.arange(1, 20)).reshape((-1,1,1))
fullCellMoments2D = fullCellMoments1D * fullCellMoments1D[:,0]
fullCellMoments3D = fullCellMoments1D * fullCellMoments1D[:,0] * fullCellMoments1D[:,0,0]


@njit(cache=True)
def computeCellMoments_(pgramMoments, cellHashMap, uncut_cell_index, uncut_cell_is_solid, Cij, N, Ibar, degs, degMax, shapep, i1x, i2x, i1y, i2y, i1z, i2z):

    D = Cij.shape[-1]

    if uncut_cell_index in cellHashMap:
        raise NotImplementedError("Current implementation assumes either (0,0,0) or (nx,0,0) is not a cut cell.")
    


    # assert tuple3D(Cij[1].flatten()) == (1, 0, 0), "Cij[1] should be (1,0,0) for this implementation."
    # assert tuple3D(Cij[2].flatten()) == (0, 1, 0), "Cij[2] should be (0,1,0) for this implementation."
    # assert tuple3D(Cij[3].flatten()) == (0, 0, 1), "Cij[3] should be (0,0,1) for this implementation."
    # assert tuple3D(Cij[4].flatten()) == (-1, 0, 0), "Cij[4] should be (-1,0,0) for this implementation."
    # assert tuple3D(Cij[5].flatten()) == (0, -1, 0), "Cij[5] should be (0,-1,0) for this implementation."
    # assert tuple3D(Cij[6].flatten()) == (0, 0, -1), "Cij[6] should be (0,0,-1) for this implementation."
    
    if D == 1:
        moments = np.empty((N[0], 1, 1, degMax+1, 1, 1), dtype=np.float64)
        fullCellMoments = fullCellMoments1D
    elif D == 2:
        moments = np.empty((N[0], N[1], 1, degMax+1, degMax+1, 1), dtype=np.float64)
        fullCellMoments = fullCellMoments2D
    elif D == 3:
        moments = np.empty((N[0], N[1], N[2], degMax+1, degMax+1, degMax+1), dtype=np.float64)
        fullCellMoments = fullCellMoments3D



    for degTuple in degs:
        degTuple3D = tuple3D(degTuple)


        # Cell at (0, 0, 0)
        if uncut_cell_is_solid:
            moments[*uncut_cell_index, *degTuple3D] = 0.0
        else:
            moments[*uncut_cell_index, *degTuple3D] = fullCellMoments[*tuple3D(np.asarray(degTuple)[:D])]
        



        # 1D, +x direction
        i1 = i1x
        i2 = i2x
        for ix in range(uncut_cell_index[0]+1, N[0]):                
            colIndA = ravelp(i1, np.array([ix-1, *uncut_cell_index[1:]])[:D], degTuple, shapep)
            colIndB = ravelp(i2, np.array([ix, *uncut_cell_index[1:]])[:D], degTuple, shapep)
            Pa = 0.0 if colIndA not in pgramMoments else pgramMoments[colIndA]
            Pb = 0.0 if colIndB not in pgramMoments else pgramMoments[colIndB]
            moments[ix, *uncut_cell_index[1:], *degTuple3D] = moments[ix-1, *uncut_cell_index[1:], *degTuple3D] + Pb - Pa
        
        # 1D, -x direction
        for ix in range(uncut_cell_index[0]-1, -1, -1):                
            colIndA = ravelp(i2, np.array([ix+1, *uncut_cell_index[1:]])[:D], degTuple, shapep)
            colIndB = ravelp(i1, np.array([ix, *uncut_cell_index[1:]])[:D], degTuple, shapep)
            Pa = 0.0 if colIndA not in pgramMoments else pgramMoments[colIndA]
            Pb = 0.0 if colIndB not in pgramMoments else pgramMoments[colIndB]
            moments[ix, *uncut_cell_index[1:], *degTuple3D] = moments[ix+1, *uncut_cell_index[1:], *degTuple3D] + Pb - Pa

        if D == 1:
            continue
        
        # 2D
        i1 = i1y
        i2 = i2y
        for ix in range(N[0]):
            # +y direction
            for iy in range(uncut_cell_index[1]+1, N[1]):
                colIndA = ravelp(i1, np.array([ix, iy-1, *uncut_cell_index[2:]])[:D], degTuple, shapep)
                colIndB = ravelp(i2, np.array([ix, iy, *uncut_cell_index[2:]])[:D], degTuple, shapep)
                Pa = 0.0 if colIndA not in pgramMoments else pgramMoments[colIndA]
                Pb = 0.0 if colIndB not in pgramMoments else pgramMoments[colIndB]
                moments[ix, iy, *uncut_cell_index[2:], *degTuple3D] = moments[ix, iy-1, *uncut_cell_index[2:], *degTuple3D] + Pb - Pa

            # -y direction
            for iy in range(uncut_cell_index[1]-1, -1, -1):
                colIndA = ravelp(i2, np.array([ix, iy+1, *uncut_cell_index[2:]])[:D], degTuple, shapep)
                colIndB = ravelp(i1, np.array([ix, iy, *uncut_cell_index[2:]])[:D], degTuple, shapep)
                Pa = 0.0 if colIndA not in pgramMoments else pgramMoments[colIndA]
                Pb = 0.0 if colIndB not in pgramMoments else pgramMoments[colIndB]
                moments[ix, iy, *uncut_cell_index[2:], *degTuple3D] = moments[ix, iy+1, *uncut_cell_index[2:], *degTuple3D] + Pb - Pa


        if D == 2:
            continue

        # 3D
        i1 = i1z
        i2 = i2z
        for ix in range(N[0]):
            for iy in range(N[1]):
                # +z direction
                for iz in range(uncut_cell_index[2]+1, N[2]):
                    colIndA = ravelp(i1, np.array([ix, iy, iz-1])[:D], degTuple, shapep)
                    colIndB = ravelp(i2, np.array([ix, iy, iz])[:D], degTuple, shapep)
                    Pa = 0.0 if colIndA not in pgramMoments else pgramMoments[colIndA]
                    Pb = 0.0 if colIndB not in pgramMoments else pgramMoments[colIndB]
                    moments[ix, iy, iz, *degTuple3D] = moments[ix, iy, iz-1, *degTuple3D] + Pb - Pa

                # -z direction
                for iz in range(uncut_cell_index[2]-1, -1, -1):
                    colIndA = ravelp(i2, np.array([ix, iy, iz+1])[:D], degTuple, shapep)
                    colIndB = ravelp(i1, np.array([ix, iy, iz])[:D], degTuple, shapep)
                    Pa = 0.0 if colIndA not in pgramMoments else pgramMoments[colIndA]
                    Pb = 0.0 if colIndB not in pgramMoments else pgramMoments[colIndB]
                    moments[ix, iy, iz, *degTuple3D] = moments[ix, iy, iz+1, *degTuple3D] + Pb - Pa

    return moments



def computeCellMoments(pgramMoments, cellHashMap, uncut_cell_index, uncut_cell_is_solid, Cij, N, Ibar, degs, degMax, shapep):
    D = Cij.shape[-1]
    i1x = np.where(np.all(Cij == np.array([1, 0, 0])[:Cij.shape[-1]], axis=-1))[0][0]
    i2x = np.where(np.all(Cij == np.array([-1, 0, 0])[:Cij.shape[-1]], axis=-1))[0][0]
    i1y = np.where(np.all(Cij == np.array([0, 1, 0])[:Cij.shape[-1]], axis=-1))[0][0] if D > 1 else 0
    i2y = np.where(np.all(Cij == np.array([0, -1, 0])[:Cij.shape[-1]], axis=-1))[0][0] if D > 1 else 0
    i1z = np.where(np.all(Cij == np.array([0, 0, 1])[:Cij.shape[-1]], axis=-1))[0][0] if D > 2 else 0
    i2z = np.where(np.all(Cij == np.array([0, 0, -1])[:Cij.shape[-1]], axis=-1))[0][0] if D > 2 else 0
    return computeCellMoments_(pgramMoments, cellHashMap, uncut_cell_index, uncut_cell_is_solid, Cij, N, Ibar, degs, degMax, shapep,
                        i1x, i2x, i1y, i2y, i1z, i2z)


key_type = types.UniTuple(types.int64, 3)
value_type_moments = types.float64[:, :, :]
@njit(cache=True)
def computeCutCellMoments(subzoneMatrixNaive, cellHashMap, solidCellInds, Cij, N, Ibar, degs, degMax, shapep):

    D = Cij.shape[-1]

    if (0, 0, 0) in cellHashMap:
        raise NotImplementedError("Current implementation assumes (0,0,0) is not a cut cell.")
    
    cutCellMoments = Dict.empty(key_type, value_type_moments)
    
    # assert tuple3D(Cij[1].flatten()) == (1, 0, 0), "Cij[1] should be (1,0,0) for this implementation."
    # assert tuple3D(Cij[2].flatten()) == (0, 1, 0), "Cij[2] should be (0,1,0) for this implementation."
    # assert tuple3D(Cij[3].flatten()) == (0, 0, 1), "Cij[3] should be (0,0,1) for this implementation."
    # assert tuple3D(Cij[4].flatten()) == (-1, 0, 0), "Cij[4] should be (-1,0,0) for this implementation."
    # assert tuple3D(Cij[5].flatten()) == (0, -1, 0), "Cij[5] should be (0,-1,0) for this implementation."
    # assert tuple3D(Cij[6].flatten()) == (0, 0, -1), "Cij[6] should be (0,0,-1) for this implementation."
    
    if D == 1:
        moments = np.empty((N[0], 1, 1, degMax+1, 1, 1), dtype=np.float64)
        fullCellMoments = fullCellMoments1D
    elif D == 2:
        moments = np.empty((N[0], N[1], 1, degMax+1, degMax+1, 1), dtype=np.float64)
        fullCellMoments = fullCellMoments2D
    elif D == 3:
        moments = np.empty((N[0], N[1], N[2], degMax+1, degMax+1, degMax+1), dtype=np.float64)
        fullCellMoments = fullCellMoments3D

    for key in cellHashMap.keys():
        cutCellMoments[key] = np.empty(moments.shape[3:], dtype=np.float64)

    for degTuple in degs:
        degTuple3D = tuple3D(degTuple)

        # 1D
        for ix in range(1, N[0]):
            if (ix, 0, 0) in cellHashMap:
                
                Pa = 0.0
                colIndA = ravelp(1, np.array([ix-1, 0, 0])[:D], degTuple, shapep)
                if colIndA in subzoneMatrixNaive:
                    for val in subzoneMatrixNaive[colIndA].values():
                        Pa += val
                Pb = 0.0
                colIndB = ravelp(Ibar[1], np.array([ix, 0, 0])[:D], degTuple, shapep)
                if colIndB in subzoneMatrixNaive:
                    for val in subzoneMatrixNaive[colIndB].values():
                        Pb += val

                moments[ix, 0, 0, *degTuple3D] = moments[ix-1, 0, 0, *degTuple3D] + Pb - Pa
                cutCellMoments[(ix, 0, 0)][*degTuple3D] = moments[ix, 0, 0, *degTuple3D]
            elif (ix, 0, 0) in solidCellInds:
                moments[ix, 0, 0, *degTuple3D] = 0.0
            else:
                moments[ix, 0, 0, *degTuple3D] = fullCellMoments[*tuple3D(np.asarray(degTuple)[:D])]
        if D == 1:
            continue
        
        # 2D
        for ix in range(N[0]):
            for iy in range(1, N[1]):
                if (ix, iy, 0) in cellHashMap:

                    Pa = 0.0
                    colIndA = ravelp(2, np.array([ix, iy-1, 0])[:D], degTuple, shapep)
                    if colIndA in subzoneMatrixNaive:
                        for val in subzoneMatrixNaive[colIndA].values():
                            Pa += val

                    Pb = 0.0
                    colIndB = ravelp(Ibar[2], np.array([ix, iy, 0])[:D], degTuple, shapep)
                    if colIndB in subzoneMatrixNaive:
                        for val in subzoneMatrixNaive[colIndB].values():
                            Pb += val

                    moments[ix, iy, 0, *degTuple3D] = moments[ix, iy-1, 0, *degTuple3D] + Pb - Pa
                    cutCellMoments[(ix, iy, 0)][*degTuple3D] = moments[ix, iy, 0, *degTuple3D]
                elif (ix, iy, 0) in solidCellInds:
                    moments[ix, iy, 0, *degTuple3D] = 0.0
                else:
                    moments[ix, iy, 0, *degTuple3D] = fullCellMoments[*tuple3D(np.asarray(degTuple)[:D])]

        if D == 2:
            continue

        # 3D
        for ix in range(N[0]):
            for iy in range(N[1]):
                for iz in range(1, N[2]):
                    if (ix, iy, iz) in cellHashMap:
                        Pa = 0.0
                        colIndA = ravelp(3, np.array([ix, iy, iz-1])[:D], degTuple, shapep)
                        if colIndA in subzoneMatrixNaive:
                            for val in subzoneMatrixNaive[colIndA].values():
                                Pa += val

                        Pb = 0.0
                        colIndB = ravelp(Ibar[3], np.array([ix, iy, iz])[:D], degTuple, shapep)
                        if colIndB in subzoneMatrixNaive:
                            for val in subzoneMatrixNaive[colIndB].values():
                                Pb += val

                        moments[ix, iy, iz, *degTuple3D] = moments[ix, iy, iz-1, *degTuple3D] + Pb - Pa
                        cutCellMoments[(ix, iy, iz)][*degTuple3D] = moments[ix, iy, iz, *degTuple3D]
                    
                    elif (ix, iy, iz) in solidCellInds:
                        moments[ix, iy, iz, *degTuple3D] = 0.0
                    else:
                        moments[ix, iy, iz, *degTuple3D] = fullCellMoments[*tuple3D(np.asarray(degTuple)[:D])]

    return cutCellMoments
