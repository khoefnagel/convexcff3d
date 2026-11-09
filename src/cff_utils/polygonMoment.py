import math
import numpy as np
from numba import njit


# Precompute lookup table
N_MAX = 50
comb_table = np.zeros((N_MAX+1, N_MAX+1), dtype=np.int64)
for n in range(N_MAX+1):
    for k in range(n+1):
        comb_table[n, k] = math.comb(n, k)

@njit(cache=True, inline='always')
def polygonMoment(polygon, degx, degy):
    """
    Analytically find integral of a polynomial term (e.g. x^2*y^4) over a polygon.
    Based on the following article:
    https://api.semanticscholar.org/CorpusID:17506973

    Parameters
    ----------
    polygon : array_like
        Array of shape (Nv, 2) with Nv the number of vertices in the polygon.
        The first column corresponds to x-coordinates, the second to y-coordinates.
        The first and last vertex should be the same to close the polygon.
    degx : int
        Degree of x in the polynomial term (x^degx).
    degy : int
        Degree of y in the polynomial term (y^degy).
    """

    assert np.all(np.isclose(polygon[0], polygon[-1])), "First and last vertex of the polygon should be the same to close the polygon"

    p, q = degx, degy 
    
    leadingFrac = 1./((p + q + 2)*(p+q+1)*comb_table[p+q, p])
    isum = 0.
    for i in range(1, len(polygon)):
        xi, yi = polygon[i]
        xim1, yim1 = polygon[i-1]
        pqSum = 0.
        for k in range(p+1):
            for l in range(q+1):
                pqSum += comb_table[k+l, l]*comb_table[p+q-k-l, q-l]*\
                                                 xi**k*xim1**(p-k)*yi**l*yim1**(q-l)
        isum += (xim1*yi-xi*yim1)*pqSum            
    
    return leadingFrac*isum

@njit(cache=True, inline='always')
def edgeMoment(edge, degx, degy):
    """
    Analytically find integral of a polynomial term (e.g. x^2*y^4) over a polygon edge.
    Based on the following article:
    https://api.semanticscholar.org/CorpusID:17506973

    Parameters
    ----------
    edge : array_like
        Array of shape (2, 2) with the two vertices of the edge.
        The first column corresponds to x-coordinates, the second to y-coordinates.
    degx : int
        Degree of x in the polynomial term (x^degx).
    degy : int
        Degree of y in the polynomial term (y^degy).
    """

    assert edge.ndim == 2 and edge.shape == (2, 2), "edge must be of shape (2, 2)"

    p, q = degx, degy 
    
    leadingFrac = 1./((p + q + 2)*(p+q+1)*comb_table[p+q, p])

    xi, yi = edge[1]
    xim1, yim1 = edge[0]
    pqSum = 0.
    for k in range(p+1):
        for l in range(q+1):
            pqSum += comb_table[k+l, l]*comb_table[p+q-k-l, q-l]*\
                                                xi**k*xim1**(p-k)*yi**l*yim1**(q-l)
    isum = (xim1*yi-xi*yim1)*pqSum            
    
    return leadingFrac*isum

