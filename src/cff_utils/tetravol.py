"""Module to compute arbitrary moments over tetrahedra in 3D with one vertex at the origin."""

import numpy as np
from numba import njit, types

@njit(cache=True)
def getCombs(n):
    """Generate all combinations of non-negative integers (a, b, c) such that a + b + c = n.
    
    Parameters
    ----------
    n : int
        The target sum for the combinations.
    
    Returns
    -------
    combs : np.ndarray
        A 2D array of shape (num_combs, 3) containing all combinations (a, b, c).
    """
    num_combs = (n+2) * (n+1) // 2
    combs = np.zeros((num_combs, 3), dtype=np.int64)
    idx = 0
    for a in range(n+1):
        for b in range(n - a + 1):
            c = n - a - b
            combs[idx, 0] = a
            combs[idx, 1] = b
            combs[idx, 2] = c
            idx += 1

    return combs

@njit(cache=True)
def build_combs_cache():
    """Build a cache of combinations of non-negative integers (a, b, c) for n = 0 to 19.

    Returns
    -------
    combs_cache : np.ndarray
        A 2D array containing all combinations for n = 0 to 19.
    inds1 : np.ndarray
        A 1D array where inds1[n] is the starting index in combs_cache for combinations summing to n.
    inds2 : np.ndarray
        A 1D array where inds2[n] is the ending index in combs_cache for combinations summing to n.
    """
    nCombs = 15
    inds1 = np.empty(nCombs, dtype=np.int64)
    inds2 = np.empty(nCombs, dtype=np.int64)
    ind_tot = 0
    for n in range(nCombs):
        num_solutions = (n+2) * (n+1) // 2
        inds1[n] = ind_tot
        inds2[n] = ind_tot + num_solutions
        ind_tot += num_solutions
    combs_cache = np.empty((inds2[-1], 3), dtype=np.int64)
    for n in range(nCombs):
        combs_cache[inds1[n]:inds2[n]] = getCombs(n)

    return combs_cache, inds1, inds2

# Precompute factorials up to 15!
factorial = np.array([1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880., 3628800., 39916800., 479001600., 6227020800., 87178291200., 1307674368000.])

# Precompute combinations cache for n = 0 to 14
combs_cache, inds1, inds2 = build_combs_cache()


@njit(cache=True)
def tetraMoment(x0, y0, z0, x1, y1, z1, x2, y2, z2, V, p, q, r):
    """Compute a single moment of a tetrahedron defined by a triangle and the origin.

    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the first vertex of the triangle.
    x1, y1, z1 : float
        Coordinates of the second vertex of the triangle.
    x2, y2, z2 : float
        Coordinates of the third vertex of the triangle.
    V : float
        Volume of the tetrahedron.
    p, q, r : int
        Degrees of the moment to compute.
    
    Returns
    -------
    moment : float
        The computed moment of the tetrahedron.
    """

    n = p + q + r
    preterm = 6*V*factorial[p]*factorial[q]*factorial[r]/factorial[n+3]

    total = 0.
    for mx in combs_cache[inds1[p]:inds2[p]]:
        for my in combs_cache[inds1[q]:inds2[q]]:
            for mz in combs_cache[inds1[r]:inds2[r]]:
                mut = mx + my + mz
                num = factorial[mut[0]] * factorial[mut[1]] * factorial[mut[2]]
                den = (factorial[mx[0]] * factorial[mx[1]] * factorial[mx[2]] *
                       factorial[my[0]] * factorial[my[1]] * factorial[my[2]] *
                       factorial[mz[0]] * factorial[mz[1]] * factorial[mz[2]])
                vProd = (x0**mx[0]) * (y0**my[0]) * (z0**mz[0]) * (x1**mx[1]) * (y1**my[1]) * (z1**mz[1]) * (x2**mx[2]) * (y2**my[2]) * (z2**mz[2])
                total += (num / den) * vProd
    return preterm * total


@njit(cache=True, inline='always')
def tetraVolume(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    """Compute the volume of a tetrahedron defined by a triangle and the origin.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Coordinates of the first vertex of the triangle.
    x1, y1, z1 : float
        Coordinates of the second vertex of the triangle.
    x2, y2, z2 : float
        Coordinates of the third vertex of the triangle.
        
    Returns
    -------
    V : float
        Volume of the tetrahedron.
    """
    V = (1/6.) * (x0*(y1*z2 - y2*z1) - y0*(x1*z2 - x2*z1) + z0*(x1*y2 - x2*y1))
    return V

@njit(cache=True)
def getTetraMoments(triangle, degs, deg):
    """Compute the moments of a tetrahedron defined by a triangle and the origin.
    
    Parameters
    ----------
    triangle : np.ndarray
        A 2D array of shape (3, 3) representing the vertices of the triangle.
    degs : np.ndarray
        A 2D array of shape (numDegs, 3) representing the degrees of the moments to compute.
    deg : int
        The maximum degree of the moments to compute.
        
    Returns
    -------
    moments : np.ndarray
        A 3D array of shape (deg+1, deg+1, deg+1) containing the computed fluid moments for the tetrahedron.
    """

    assert triangle.shape == (3, 3), "Triangle must be of shape (3, 3)"
    
    moments = np.empty((deg+1, deg+1, deg+1), dtype=np.float64)
    
    x0, y0, z0 = triangle[0]
    x1, y1, z1 = triangle[1]
    x2, y2, z2 = triangle[2]
    
    V = tetraVolume(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    for p, q, r in degs:
        if p == q == r == 0:
            moments[p, q, r] = V
        else:
            moments[p, q, r] = tetraMoment(x0, y0, z0, x1, y1, z1, x2, y2, z2, V, p, q, r)
    return moments
