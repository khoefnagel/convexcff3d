import numpy as np
from numba import njit



@njit(cache=True, inline='always')
def splitOnPlane2D(segment, plane_point, plane_normal):
    tol = 1e-12

    dists = segment - plane_point
    dists = dists[:, 0]*plane_normal[0] + dists[:, 1]*plane_normal[1]
    
    posSeg = np.empty((2, 2))
    negSeg = np.empty((2, 2))

    # Check if segment is coplanar
    if np.all(np.abs(dists) <= tol):
        negSeg = segment
        return negSeg, 2, posSeg, 0, True
    
    # Check if segment is on either side of the plane
    if np.all(dists < tol):
        negSeg = segment
        return negSeg, 2, posSeg, 0, False
    if np.all(dists > -tol):
        posSeg = segment
        return negSeg, 0, posSeg, 2, False

    # If segment is not coplanar and not on either side, it must be intersecting
    t = dists[0] / (dists[0] - dists[1])
    inter = segment[0] + t * (segment[1] - segment[0])

    if dists[0] > 0:
        posSeg[0] = segment[0]
        posSeg[1] = inter
        negSeg[0] = inter
        negSeg[1] = segment[1]
    else:
        negSeg[0] = segment[0]
        negSeg[1] = inter
        posSeg[0] = inter
        posSeg[1] = segment[1]
    
    return negSeg, 2, posSeg, 2, False
    


@njit(cache=True, inline='always')
def splitOnPlane3D(polygon, plane_point, plane_normal):
    tol = 1e-12
    n = polygon.shape[0]
    D = polygon.shape[-1]

    # dists = (polygon - plane_point) @ plane_normal
    dists = polygon - plane_point
    dists = dists[:, 0]*plane_normal[0] + dists[:, 1]*plane_normal[1] + dists[:, 2]*plane_normal[2]

    neg_poly = np.empty((n+2, D))
    pos_poly = np.empty((n+2, D))


    # Check if triangle is coplanar
    if np.all(np.abs(dists) <= tol):
        neg_poly = polygon
        return neg_poly, n, pos_poly, 0, True

    neg_count = 0
    pos_count = 0

    # Loop over edges
    for i in range(n):
        v0 = polygon[i]
        v1 = polygon[(i + 1) % n]
        d0 = dists[i]
        d1 = dists[(i + 1) % n]

        # Add vertices to negative polygon if on negative side or on plane
        if d0 <= tol:
            neg_poly[neg_count] = v0
            neg_count += 1
        # Add vertices to positive polygon if on positive side or on plane
        if d0 >= -tol:
            pos_poly[pos_count] = v0
            pos_count += 1

        # If edge crosses the plane, compute intersection
        if (d0 < -tol and d1 > tol) or (d0 > tol and d1 < -tol):
            t = d0 / (d0 - d1)
            inter = v0 + t * (v1 - v0)
            neg_poly[neg_count] = inter
            neg_count += 1
            pos_poly[pos_count] = inter
            pos_count += 1

    # Return slices containing only used vertices
    return neg_poly, neg_count, pos_poly, pos_count, False

@njit(cache=True, inline='always')
def splitOnPlane(polygon, plane_point, plane_normal):
    D = polygon.shape[1]
    if D == 2:
        return splitOnPlane2D(polygon, plane_point, plane_normal)
    elif D == 3:
        return splitOnPlane3D(polygon, plane_point, plane_normal)
    else:
        raise ValueError("D must be 2 or 3")

