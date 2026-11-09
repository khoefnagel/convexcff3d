"""Module to compute the dot products of the first and second derivatives of
B-spline basis functions in cell-centered coordinates."""

import numpy as np
import sympy as sym


def computeGradDotGradBSplines():
    if __name__ == '__main__':
        from BSplines import CCBSplines1DExpr
    else:
        from .BSplines import CCBSplines1DExpr


    #==============================================================================
    # Obtain the expressions for the B-spline basis functions in 1D-3D as dictionaries.
    # Each key corresponds to a specific degree and contains an nD array of polynomial
    # expressions for the basis functions of that degree.
    #==============================================================================

    # 1D
    CCBSplines1DExprArr = {key : np.array(val) for key, val in CCBSplines1DExpr.items()}

    #Only compute up to degree 2 for to limit computation time
    CCBSplines1DExprArr.pop(3)
    CCBSplines1DExprArr.pop(4)

    x, y, z = sym.symbols('x y z')

    # 2D
    CCBSplines2DExprArr = {}
    for deg, exprs1D in CCBSplines1DExprArr.items():
        CCBSplines2DExprArr[deg] = np.empty((len(exprs1D), len(exprs1D)), dtype=object)
        for i, exprx in enumerate(exprs1D):
            for j, expry in enumerate(exprs1D):
                expry = expry.subs(x, y)
                expr2D = exprx * expry
                CCBSplines2DExprArr[deg][i, j] = expr2D

    # 3D
    CCBSplines3DExprArr = {}
    for deg, exprs1D in CCBSplines1DExprArr.items():
        CCBSplines3DExprArr[deg] = np.empty((len(exprs1D), len(exprs1D), len(exprs1D)), dtype=object)
        for i, exprx in enumerate(exprs1D):
            for j, expry in enumerate(exprs1D):
                expry = expry.subs(x, y)
                for k, exprz in enumerate(exprs1D):
                    exprz = exprz.subs(x, z)
                    expr3D = exprx * expry * exprz
                    CCBSplines3DExprArr[deg][i, j, k] = expr3D


    # =============================================================================
    # Compute the gradients and of the B-spline basis functions.
    # These are stored as dictionaries similar to the CCBSplinesNDExprArr dictionaries.
    # =============================================================================

    # 1D
    gradCCBSplines1DExprArr = {}
    for deg, exprs in CCBSplines1DExprArr.items():
        if deg == 0:
            continue
        gradCCBSplines1DExprArr[deg] = np.zeros((deg+1), dtype=object)
        for i, expr in enumerate(exprs):
            gradCCBSplines1DExprArr[deg][i] = sym.diff(expr, x)

    # 2D
    gradCCBSplines2DExprArr = {}
    for deg, exprs in CCBSplines2DExprArr.items():
        if deg == 0:
            continue
        gradCCBSplines2DExprArr[deg] = np.zeros((deg+1, deg+1, 2), dtype=object)
        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                expr = exprs[i, j]
                gradCCBSplines2DExprArr[deg][i, j, 0] = sym.diff(expr, x)
                gradCCBSplines2DExprArr[deg][i, j, 1] = sym.diff(expr, y)

    # 3D
    gradCCBSplines3DExprArr = {}
    for deg, exprs in CCBSplines3DExprArr.items():
        if deg == 0:
            continue
        gradCCBSplines3DExprArr[deg] = np.zeros((deg+1, deg+1, deg+1, 3), dtype=object)
        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                for k in range(exprs.shape[2]):
                    expr = exprs[i, j, k]
                    gradCCBSplines3DExprArr[deg][i, j, k, 0] = sym.diff(expr, x)
                    gradCCBSplines3DExprArr[deg][i, j, k, 1] = sym.diff(expr, y)
                    gradCCBSplines3DExprArr[deg][i, j, k, 2] = sym.diff(expr, z)

    # =============================================================================
    # Compute the second derivatives of the B-spline basis functions.
    # These are stored as dictionaries similar to the CCBSplinesNDExprArr dictionaries.
    # =============================================================================

    # 1D
    grad2CCBSplines1DExprArr = {}
    for deg, exprs in CCBSplines1DExprArr.items():
        if deg == 0:
            continue
        grad2CCBSplines1DExprArr[deg] = np.zeros((deg+1), dtype=object)
        for i, expr in enumerate(exprs):
            grad2CCBSplines1DExprArr[deg][i] = sym.diff(expr, x, 2)

    # 2D
    grad2CCBSplines2DExprArr = {}
    for deg, exprs in CCBSplines2DExprArr.items():
        if deg == 0:
            continue
        grad2CCBSplines2DExprArr[deg] = np.zeros((deg+1, deg+1, 2, 2), dtype=object)
        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                expr = exprs[i, j]
                grad2CCBSplines2DExprArr[deg][i, j, 0, 0] = sym.diff(expr, x, 2)
                grad2CCBSplines2DExprArr[deg][i, j, 0, 1] = sym.diff(expr, x, y)
                grad2CCBSplines2DExprArr[deg][i, j, 1, 0] = sym.diff(expr, y, x)
                grad2CCBSplines2DExprArr[deg][i, j, 1, 1] = sym.diff(expr, y, 2)

    # 3D
    grad2CCBSplines3DExprArr = {}
    for deg, exprs in CCBSplines3DExprArr.items():
        if deg == 0:
            continue
        grad2CCBSplines3DExprArr[deg] = np.zeros((deg+1, deg+1, deg+1, 3, 3), dtype=object)
        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                for k in range(exprs.shape[2]):
                    expr = exprs[i, j, k]
                    grad2CCBSplines3DExprArr[deg][i, j, k, 0, 0] = sym.diff(expr, x, 2)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 0, 1] = sym.diff(expr, x, y)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 0, 2] = sym.diff(expr, x, z)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 1, 0] = sym.diff(expr, y, x)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 1, 1] = sym.diff(expr, y, 2)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 1, 2] = sym.diff(expr, y, z)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 2, 0] = sym.diff(expr, z, x)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 2, 1] = sym.diff(expr, z, y)
                    grad2CCBSplines3DExprArr[deg][i, j, k, 2, 2] = sym.diff(expr, z, 2)

    # =============================================================================
    # Compute the dot products of the gradients in terms of polynomial coefficients.
    # These are stored as dictionaries where each key corresponds to a specific degree.
    # The first D axes correspond to the first basis function indices, the next D axes
    # to the second basis function indices, and the last D axes to the polynomial
    # coefficient indices of the resulting polynomial.
    # =============================================================================

    # 1D 
    gradDotGradCCBSplines1D = {}
    for deg, exprs in gradCCBSplines1DExprArr.items():
        gradDotGradCCBSplines1D[deg] = np.zeros((deg+1, deg+1, 2*deg+1), dtype=np.float64)

        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[0]):
                expr = exprs[i] * exprs[j]
                if isinstance(expr, sym.Number):
                    gradTerms = [((0,), float(expr))]
                else:
                    gradTerms = expr.as_poly().terms()
                for p, coeff in gradTerms:
                    gradDotGradCCBSplines1D[deg][i, j, p] = coeff
    
    # 2D
    gradDotGradCCBSplines2D = {}
    for deg, exprs in gradCCBSplines2DExprArr.items():
        gradDotGradCCBSplines2D[deg] = np.zeros((deg+1, deg+1, deg+1, deg+1, 2*deg+1, 2*deg+1), dtype=np.float64)

        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                for k in range(exprs.shape[0]):
                    for l in range(exprs.shape[1]):
                        expr = exprs[i, j, 0] * exprs[k, l, 0] + exprs[i, j, 1] * exprs[k, l, 1]
                        if isinstance(expr, sym.Number):
                            gradTerms = [((0,0), float(expr))]
                        else:
                            gradTerms = expr.as_poly().terms()
                        for (px, py), coeff in gradTerms:
                            gradDotGradCCBSplines2D[deg][i, j, k, l, px, py] = coeff

    # 3D
    gradDotGradCCBSplines3D = {}
    for deg, exprs in gradCCBSplines3DExprArr.items():
        gradDotGradCCBSplines3D[deg] = np.zeros((deg+1, deg+1, deg+1, deg+1, deg+1, deg+1, 2*deg+1, 2*deg+1, 2*deg+1), dtype=np.float64)

        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                for k in range(exprs.shape[2]):
                    for l in range(exprs.shape[0]):
                        for m in range(exprs.shape[1]):
                            for n in range(exprs.shape[2]):
                                expr = (exprs[i, j, k, 0] * exprs[l, m, n, 0] +
                                        exprs[i, j, k, 1] * exprs[l, m, n, 1] +
                                        exprs[i, j, k, 2] * exprs[l, m, n, 2])
                                if isinstance(expr, sym.Number):
                                    gradTerms = [((0,0,0), float(expr))]
                                else:
                                    gradTerms = expr.as_poly().terms()
                                for (px, py, pz), coeff in gradTerms:
                                    gradDotGradCCBSplines3D[deg][i, j, k, l, m, n, px, py, pz] = coeff

    # =============================================================================
    # Compute the dot products of the second derivatives in terms of polynomial coefficients.
    # These are stored as dictionaries where each key corresponds to a specific degree.
    # The first D axes correspond to the first basis function indices, the next D axes
    # to the second basis function indices, and the last D axes to the polynomial
    # coefficient indices of the resulting polynomial.
    # =============================================================================

    grad2DotGrad2CCBSplines1D = {}
    for deg, exprs in grad2CCBSplines1DExprArr.items():
        grad2DotGrad2CCBSplines1D[deg] = np.zeros((deg+1, deg+1, 2*deg+1), dtype=np.float64)

        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[0]):
                expr = exprs[i] * exprs[j]
                if isinstance(expr, sym.Number):
                    gradTerms = [((0,), float(expr))]
                else:
                    gradTerms = expr.as_poly().terms()
                for p, coeff in gradTerms:
                    grad2DotGrad2CCBSplines1D[deg][i, j, p] = coeff

    grad2DotGrad2CCBSplines2D = {}
    for deg, exprs in grad2CCBSplines2DExprArr.items():
        grad2DotGrad2CCBSplines2D[deg] = np.zeros((deg+1, deg+1, deg+1, deg+1, 2*deg+1, 2*deg+1), dtype=np.float64)

        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                for k in range(exprs.shape[0]):
                    for l in range(exprs.shape[1]):
                        expr = exprs[i, j, 0, 0] * exprs[k, l, 0, 0] + exprs[i, j, 0, 1] * exprs[k, l, 0, 1] +\
                               exprs[i, j, 1, 0] * exprs[k, l, 1, 0] + exprs[i, j, 1, 1] * exprs[k, l, 1, 1]
                        if isinstance(expr, sym.Number):
                            gradTerms = [((0,0), float(expr))]
                        else:
                            gradTerms = expr.as_poly().terms()
                        for (px, py), coeff in gradTerms:
                            grad2DotGrad2CCBSplines2D[deg][i, j, k, l, px, py] = coeff  

    grad2DotGrad2CCBSplines3D = {}
    for deg, exprs in grad2CCBSplines3DExprArr.items():
        gradDotGradCCBSplines3D[deg] = np.zeros((deg+1, deg+1, deg+1, deg+1, deg+1, deg+1, 2*deg+1, 2*deg+1, 2*deg+1), dtype=np.float64)

        for i in range(exprs.shape[0]):
            for j in range(exprs.shape[1]):
                for k in range(exprs.shape[2]):
                    for l in range(exprs.shape[0]):
                        for m in range(exprs.shape[1]):
                            for n in range(exprs.shape[2]):
                                expr = (exprs[i, j, k, 0, 0] * exprs[l, m, n, 0, 0] +\
                                        exprs[i, j, k, 0, 1] * exprs[l, m, n, 0, 1] +\
                                        exprs[i, j, k, 0, 2] * exprs[l, m, n, 0, 2] +\
                                        exprs[i, j, k, 1, 0] * exprs[l, m, n, 1, 0] +\
                                        exprs[i, j, k, 1, 1] * exprs[l, m, n, 1, 1] +\
                                        exprs[i, j, k, 1, 2] * exprs[l, m, n, 1, 2] +\
                                        exprs[i, j, k, 2, 0] * exprs[l, m, n, 2, 0] +\
                                        exprs[i, j, k, 2, 1] * exprs[l, m, n, 2, 1] +\
                                        exprs[i, j, k, 2, 2] * exprs[l, m, n, 2, 2])
                                if isinstance(expr, sym.Number):
                                    gradTerms = [((0,0,0), float(expr))]
                                else:
                                    gradTerms = expr.as_poly().terms()
                                for (px, py, pz), coeff in gradTerms:
                                    gradDotGradCCBSplines3D[deg][i, j, k, l, m, n, px, py, pz] = coeff        

    return (gradDotGradCCBSplines1D, gradDotGradCCBSplines2D, gradDotGradCCBSplines3D,
              grad2DotGrad2CCBSplines1D, grad2DotGrad2CCBSplines2D, grad2DotGrad2CCBSplines3D)

# The first time this module is imported/ran, the B-spline gradient dot products
# are computed and cached to a file. Subsequent imports/runs load the cached data.
from pathlib import Path
MODULE_DIR = Path(__file__).resolve().parent
CACHE_FILE = MODULE_DIR / 'gradDotGradCCBSplinesCache.npz'

if CACHE_FILE.exists() and __name__ != '__main__':
    data = np.load(CACHE_FILE, allow_pickle=True)
    gradDotGradCCBSplines1D = data['gradDotGradCCBSplines1D'].item()
    gradDotGradCCBSplines2D = data['gradDotGradCCBSplines2D'].item()
    gradDotGradCCBSplines3D = data['gradDotGradCCBSplines3D'].item()
    grad2DotGrad2CCBSplines1D = data['grad2DotGrad2CCBSplines1D'].item()
    grad2DotGrad2CCBSplines2D = data['grad2DotGrad2CCBSplines2D'].item()
    grad2DotGrad2CCBSplines3D = data['grad2DotGrad2CCBSplines3D'].item()
else:
    (gradDotGradCCBSplines1D, gradDotGradCCBSplines2D, gradDotGradCCBSplines3D,
     grad2DotGrad2CCBSplines1D, grad2DotGrad2CCBSplines2D, grad2DotGrad2CCBSplines3D) = computeGradDotGradBSplines()
    np.savez_compressed(CACHE_FILE, 
                        gradDotGradCCBSplines1D=gradDotGradCCBSplines1D, 
                        gradDotGradCCBSplines2D=gradDotGradCCBSplines2D,
                        gradDotGradCCBSplines3D=gradDotGradCCBSplines3D,
                        grad2DotGrad2CCBSplines1D=grad2DotGrad2CCBSplines1D,
                        grad2DotGrad2CCBSplines2D=grad2DotGrad2CCBSplines2D,
                        grad2DotGrad2CCBSplines3D=grad2DotGrad2CCBSplines3D)

