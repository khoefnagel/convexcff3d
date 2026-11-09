"""Module to obtain the reflect and integrate (RI) matrix which performs the
CFF boundary condition given the polynomial spline coefficients in each
relevant cell. Also computes the matrices to obtain total and surface forces."""

import numpy as np
import scipy.sparse as sparse
from numba import njit, types
from numba.typed import Dict, List

from .shapesRavels import ravelf, ravelp
from .matrixUtil import dictToSparseMat
from cff_utils.tuple3D import tuple3D


key_type = types.UniTuple(types.int64, 2)
value_type = types.float64
# @njit(cache=True)
def getRIDict(
        Cij, 
        N, 
        Ibar, 
        subzoneMoments,
        origCellInds, 
        destCellInds, 
        pgram2solid, 
        pgram2inputFacet, 
        pgram2gridCutFacet,
        pgram2subGridCutFacet,
        inputFacetAreas,
        gridCutFacetAreas,
        shapef, 
        shapep, 
        degs
    ):
    """Function to obtain the RI matrix and force projection matrices as numba typed dictionaries.

    Parameters
    ----------
    Cij : array_like
        Lattice velocity set, shape (Q, ..., D).
    N : array_like
        Number of grid cells in each dimension, shape (D, ).
    Ibar : array_like
        Indices of the opposite velocity for each lattice velocity, shape (Q, ).
    subzoneMoments : dict
        Dictionary mapping Cj tuples to lists of subzone moments arrays.
    concavePgramMoments : dict
        Dictionary mapping Cj tuples to dictionaries of concave pgram moments.
    origCellInds : dict
        Dictionary mapping Cj tuples to lists of origin cell indices arrays.
    destCellInds : dict
        Dictionary mapping Cj tuples to lists of destination cell indices arrays.
    pgram2solid : dict
        Dictionary mapping Cj tuples to lists of solid indices corresponding to each pgram.
    pgram2inputFacet : dict
        Dictionary mapping Cj tuples to lists of input facet indices corresponding to each pgram.
    pgram2gridCutFacet : dict
        Dictionary mapping Cj tuples to lists of grid cut facet indices corresponding to each pgram.
    inputFacetAreas : array_like
        Array of input facet areas, shape (nInputFacets, ).
    gridCutFacetAreas : array_like
        Array of grid cut facet areas, shape (nGridCutFacets, ).
    shapef : array_like
        Shape of the population array, shape (Q, *N, 1).
    shapep : array_like
        Shape of the spline coefficient array, shape (Q, *N, *[deg+1]).
    degs : array_like
        Array of degree tuples for the splines.
    
    Returns
    -------
    RIDict : numba.typed.Dict
        Dictionary representation of the RI matrix, where keys are (row, col) tuples
        and values are the matrix entries.
    PC2FtotDict : numba.typed.Dict
        Dictionary representation of the PC2Ftot matrix from spline coefficients
        to total forces on solids, where keys are (row, col) tuples and values are
        the matrix entries.
    PC2fsurfInputDict : numba.typed.Dict
        Dictionary representation of the PC2fsurfInput matrix from spline coefficients
        to surface forces on input facets, where keys are (row, col) tuples and values are
        the matrix entries.
    PC2fsurfGridDict : numba.typed.Dict
        Dictionary representation of the PC2fsurfGrid matrix from spline coefficients
        to surface forces on grid cut facets, where keys are (row, col) tuples and values are
        the matrix entries.
    """


    Q = Cij.shape[0]
    D = Cij.shape[-1]


    # Jit compiled ravel function requires arrays
    shapef, shapep = np.array(shapef), np.array(shapep)


    RIDict = Dict.empty(key_type, value_type)

    # Maps spline coefficients to surface forces on grid cut facets
    PC2fsurfGridDict = Dict.empty(key_type, value_type)

    # Maps spline coefficients to surface forces on input facets
    PC2fsurfInputDict = Dict.empty(key_type, value_type)

    # Maps spline coefficients to total forces on solids
    PC2FtotDict = Dict.empty(key_type, value_type)


    for i, Cj in enumerate(Cij.reshape(Q, D)):
        print(f"Processing Cj={Cj}")
        CjTuple = tuple3D(Cj)       

        ibar = Ibar[i]
        assert len(subzoneMoments[CjTuple]) == len(origCellInds[CjTuple]) == len(destCellInds[CjTuple]) == len(pgram2gridCutFacet[CjTuple]), f"Lengths do not match for Cj={CjTuple}: {len(subzoneMoments[CjTuple])}, {len(origCellInds[CjTuple])}, {len(destCellInds[CjTuple])}, {len(pgram2gridCutFacet[CjTuple])}"
        
        # Loop over pgrams
        for (
            subzoneMomentsCj, 
            origCellIndsCj, 
            destCellIndsCj, 
            solidInd, 
            inputFacetInd, 
            gridCutFacetInd,
            subGridCutFacetInd
        ) in zip(
            subzoneMoments[CjTuple], 
            origCellInds[CjTuple], 
            destCellInds[CjTuple], 
            pgram2solid[CjTuple], 
            pgram2inputFacet[CjTuple], 
            pgram2gridCutFacet[CjTuple],
            pgram2subGridCutFacet[CjTuple]
        ):                
            inputFacetArea = inputFacetAreas[inputFacetInd]
            gridCutFacetArea = gridCutFacetAreas[gridCutFacetInd]

            if inputFacetArea < 1e-15 or gridCutFacetArea < 1e-15:
                continue


            # Loop over subzones within pgram
            for pgramMoments, origCellId, destCellId in zip(subzoneMomentsCj, origCellIndsCj, destCellIndsCj):

                # Loop over polynomial degrees
                for degTuple in degs:

                    #Get the polynomial moment over the subzone
                    subzoneMoment = pgramMoments[*tuple3D(degTuple)].item()

                    #Get the column index ci
                    origInd = origCellId[:D] % N
                    ci = ravelp(i, origInd, degTuple, shapep)

                    #Get rmri: the row index to remove the stream from
                    rmDestInd = (origInd + Cj) % N
                    rmri = ravelf(i, rmDestInd, shapef)

                    #Remove stream through boundary into solid
                    if (rmri, ci) not in RIDict:
                        RIDict[(rmri, ci)] = 0.0
                    RIDict[(rmri, ci)] -= subzoneMoment


                    #Get rflri: the row index where to add the reflection
                    rflDestInd = destCellId[:D] % N
                    rflri = ravelf(ibar, rflDestInd, shapef)

        

                    if (rflri, ci) not in RIDict:
                        RIDict[(rflri, ci)] = 0.0
                        
                    #Add reflection on boundary
                    RIDict[(rflri, ci)] += subzoneMoment


                ###Forces
                
                # Compute force on subGridCutFacet belonging to pgram
                FSurf = 2*Cj[:,None]*subzoneMoment

                for d in range(D):
                    rowIndSolid = solidInd*D + d
                    if (rowIndSolid, ci) not in PC2FtotDict:
                        PC2FtotDict[(rowIndSolid, ci)] = 0.0
                    PC2FtotDict[(rowIndSolid, ci)] += FSurf[d].item()

                    rowIndInput = inputFacetInd*D + d
                    if (rowIndInput, ci) not in PC2fsurfInputDict:
                        PC2fsurfInputDict[(rowIndInput, ci)] = 0.0
                    PC2fsurfInputDict[(rowIndInput, ci)] += FSurf[d].item()/inputFacetArea

                    rowIndGrid = gridCutFacetInd*D + d
                    if (rowIndGrid, ci) not in PC2fsurfGridDict:
                        PC2fsurfGridDict[(rowIndGrid, ci)] = 0.0
                    PC2fsurfGridDict[(rowIndGrid, ci)] += FSurf[d].item()/gridCutFacetArea

    return RIDict, PC2FtotDict, PC2fsurfInputDict, PC2fsurfGridDict

def getRI(
        Cij, 
        N, 
        Ibar, 
        subzoneMoments, 
        origCellInds, 
        destCellInds,
        pgram2solid,
        pgram2inputFacet,
        pgram2gridCutFacet,
        pgram2subGridCutFacet,
        inputFacetAreas,
        gridCutFacetAreas,
        shapef, 
        shapep, 
        degs
    ):
    """Wrapper around getRIDict, converts the numba typed dictionaries to sparse matrices."""

    (
        RIDict,
        PC2FtotDict,
        PC2fsurfInputDict,
        PC2fsurfGridDict        
    ) = getRIDict(
        Cij, 
        N, 
        Ibar, 
        subzoneMoments, 
        origCellInds, 
        destCellInds, 
        pgram2solid, 
        pgram2inputFacet, 
        pgram2gridCutFacet,
        pgram2subGridCutFacet,
        inputFacetAreas,
        gridCutFacetAreas,
        shapef, 
        shapep, 
        degs
    )
    nSolids = max(max(v) for v in pgram2solid.values() if len(v) > 0)+1
    nInputFacets = max(max(v) for v in pgram2inputFacet.values() if len(v) > 0)+1
    nGridCutFacets = max(max(v) for v in pgram2gridCutFacet.values() if len(v) > 0)+1

    D = Cij.shape[-1]
    return (
        dictToSparseMat(RIDict, (np.prod(shapef), np.prod(shapep))),
        dictToSparseMat(PC2FtotDict, (nSolids*D, np.prod(shapep))),
        dictToSparseMat(PC2fsurfInputDict, (nInputFacets*D, np.prod(shapep))),
        dictToSparseMat(PC2fsurfGridDict, (nGridCutFacets*D, np.prod(shapep))),
    )


