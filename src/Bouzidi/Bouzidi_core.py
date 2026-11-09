import numpy as np
import shapely
from scipy import sparse
import trimesh
from numba import njit, types
from numba.typed import Dict

from Matrices.shapesRavels import ravelf
from Matrices.matrixUtil import dictToSparseMat 





@njit(cache=True)
def build_Bouzidi_matrices(xw_arr, Cj, i, iBar, N, shapef, CSM, Fi2Ftot):

    D = Cj.shape[0]



    Cj_nonzero_ind = np.where(np.abs(Cj) > 1e-14)[0][0] # Get the first non-zero index of Cj

    for xw in xw_arr:

        if np.min(np.abs(xw)) < 1e-14 or np.min(np.abs(xw - np.asarray(N))) < 1e-14:
            # Hit point is on the boundary of the domain, skip this hit
            continue

        q = (Cj*xw - 0.5 - np.floor(Cj*xw - 0.5))[Cj_nonzero_ind]

        # Coordinates of the two fluid points along the direction of Cj
        xf1 = xw - q*Cj
        xf2 = xf1 - Cj

        # Cell indices of the two fluid points
        xf1_idx = np.floor(xf1).astype(np.int64) % np.asarray(N)
        xf2_idx = np.floor(xf2).astype(np.int64) % np.asarray(N)

        # Remove stream from solid to fluid
        ri = ravelf(iBar, xf1_idx, shapef)
        ci = ravelf(iBar, xf1_idx, shapef)
        if (ri, ci) not in CSM:
            CSM[(ri, ci)] = 0.0
        else:
            raise Exception(f"Duplicate entry in CSM for (ri, ci) = ({ri}, {ci})")
        CSM[(ri, ci)] += -1.0

        a = 2*q if q < 0.5 else 0.5/q
        b = 1 - 2*q if q < 0.5 else (2*q - 1)/(2*q)

        ri = ravelf(iBar, xf1_idx, shapef)
        ci = ravelf(i, (xf1_idx + Cj) % np.asarray(N), shapef)
        if (ri, ci) not in CSM:
            CSM[(ri, ci)] = 0.0
        else:
            raise Exception(f"Duplicate entry in CSM for (ri, ci) = ({ri}, {ci})")
        CSM[(ri, ci)] += a

        ri = ravelf(iBar, xf1_idx, shapef)
        if q < 0.5: # Use f_i(xf2)
            ci = ravelf(i, (xf2_idx + Cj) % np.asarray(N), shapef)
        else: # q >= 0.5, use f_i_bar(xf1)
            ci = ravelf(iBar, (xf1_idx - Cj) % np.asarray(N), shapef)
        if (ri, ci) not in CSM:
            CSM[(ri, ci)] = 0.0
        else:
            raise Exception(f"Duplicate entry in CSM for (ri, ci) = ({ri}, {ci})")
        CSM[(ri, ci)] += b

        ## Forces
        
        for dim in range(D):
            ri = dim

            ci = ravelf(i, (xf1_idx + Cj) % np.asarray(N), shapef)
            # ci = ravelf(i, xf1_idx, shapef)
            val = Cj[dim]*(1 + a)
            if (ri, ci) not in Fi2Ftot:
                Fi2Ftot[(ri, ci)] = 0.0
            else:
                raise Exception(f"Duplicate entry in Fi2Ftot for (ri, ci) = ({ri}, {ci})")
            Fi2Ftot[(ri, ci)] += val

            if q < 0.5: # Use f_i(xf2)
                ci = ravelf(i, (xf2_idx + Cj) % np.asarray(N), shapef)
                # ci = ravelf(i, xf2_idx, shapef)
            else: # q >= 0.5, use f_i_bar(xf1)
                ci = ravelf(iBar, (xf1_idx - Cj) % np.asarray(N), shapef)
                # ci = ravelf(iBar, xf1_idx, shapef)
            val = Cj[dim]*b
            if (ri, ci) not in Fi2Ftot:
                Fi2Ftot[(ri, ci)] = 0.0
            else:
                raise Exception(f"Duplicate entry in Fi2Ftot for (ri, ci) = ({ri}, {ci})")
            Fi2Ftot[(ri, ci)] += val


def Bouzidi_core_1D():
    raise NotImplementedError("Bouzidi_core_1D is not implemented yet.")

def get_starting_points_2D(Nx, Ny, Cj):
    Cj = tuple(Cj)
    if Cj == (1, 0) or Cj == (-1, 0):  # Horizontal rays
        return np.zeros(Ny) + 0.5, np.arange(Ny) + 0.5
    elif Cj == (0, 1) or Cj == (0, -1):  # Vertical rays
        return np.arange(Nx) + 0.5, np.zeros(Nx) + 0.5
    elif Cj == (1, 1) or Cj == (-1, -1) or Cj == (1, -1) or Cj == (-1, 1):  # Diagonal rays (bottom-left to top-right)
        return np.arange(-Ny, Nx + Ny) + 0.5, np.zeros(Nx + 2*Ny) + 0.5
    else:
        raise ValueError(f"Unsupported Cj direction: {Cj}. Only cardinal and diagonal directions are supported.")
def Bouzidi_core_2D(solidsUnion, Cij, N, IBar):
    Nx, Ny = N
    Q = Cij.shape[0]
    shapef = (Q, Nx, Ny, 1)

    # starting_points_x = np.vstack([np.hstack([np.zeros(Ny) + 0.5, np.ones(Ny)*Nx - 0.5]), np.hstack([np.arange(Ny) + 0.5]*2)]).T
    # starting_points_y = np.vstack([np.hstack([np.arange(Nx) + 0.5]*2), np.hstack([np.zeros(Nx) + 0.5, np.ones(Nx)*Ny - 0.5])]).T


    key_type = types.UniTuple(types.int64, 2)
    value_type = types.float64
    CSM = Dict.empty(key_type, value_type)
    Fi2Ftot = Dict.empty(key_type, value_type)

    for i in range(Q):
        Cj = Cij[i].ravel()
        iBar = IBar[i]
        if np.all(Cj == 0):
            continue

        starting_points_x, starting_points_y = get_starting_points_2D(Nx, Ny, Cj)
        rays = shapely.linestrings(np.stack([np.vstack([starting_points_x, starting_points_y]).T - Cj*(Nx + Ny), 
                                             np.vstack([starting_points_x, starting_points_y]).T + Cj*(Nx + Ny)], axis=1))


        hits = shapely.intersection(rays, solidsUnion)

        lines = []
        for hit in hits:
            if hit.is_empty:
                continue
            if hit.geom_type == 'MultiLineString':
                lines.extend(hit.geoms)
            elif hit.geom_type == 'LineString':
                lines.append(hit)
            elif hit.geom_type == 'Point':
                continue  # Ignore point hits
            elif hit.geom_type == 'GeometryCollection':
                for hit_ in hit.geoms:
                    if hit_.geom_type == 'LineString':
                        lines.append(hit_)
            else:
                raise ValueError(f"Unexpected geometry type: {hit.geom_type}")

        xw_arr = []
        for line in lines:
            line_dot_Cj = np.diff(line.xy).ravel().dot(Cj)
            
            # Hit point on the wall is the first or last point of the line depending on the direction of Cj
            xw = np.array(line.xy)[:, 0] if line_dot_Cj > 0 else np.array(line.xy)[:, -1]        
            xw_arr.append(xw)
        


        build_Bouzidi_matrices(np.array(xw_arr), Cj, i, iBar, N, shapef, CSM, Fi2Ftot)

    return dictToSparseMat(CSM, (Q*Nx*Ny, Q*Nx*Ny)), dictToSparseMat(Fi2Ftot, (2, Q*Nx*Ny))


def get_starting_points_3D(Nx, Ny, Nz, Cj):
    Cj = tuple(Cj)
    if Cj == (1, 0, 0) or Cj == (-1, 0, 0):  # X-direction rays
        y, z = np.meshgrid(np.arange(Ny) + 0.5, np.arange(Nz) + 0.5, indexing='ij')
        return np.zeros(Ny*Nz) + 0.5, y.ravel(), z.ravel()
    elif Cj == (0, 1, 0) or Cj == (0, -1, 0):  # Y-direction rays
        x, z = np.meshgrid(np.arange(Nx) + 0.5, np.arange(Nz) + 0.5, indexing='ij')
        return x.ravel(), np.zeros(Nx*Nz) + 0.5, z.ravel()
    elif Cj == (0, 0, 1) or Cj == (0, 0, -1):  # Z-direction rays
        x, y = np.meshgrid(np.arange(Nx) + 0.5, np.arange(Ny) + 0.5, indexing='ij')
        return x.ravel(), y.ravel(), np.zeros(Nx*Ny) + 0.5
    elif Cj == (1, 1, 0) or Cj == (-1, -1, 0) or Cj == (1, -1, 0) or Cj == (-1, 1, 0):  # Diagonal rays in XY plane
        x, z = np.meshgrid(np.arange(-Ny, Nx + Ny) + 0.5, np.arange(Nz) + 0.5, indexing='ij')
        return x.ravel(), np.zeros((Nx + 2*Ny) * Nz) + 0.5, z.ravel()
    elif Cj == (1, 0, 1) or Cj == (-1, 0, -1) or Cj == (1, 0, -1) or Cj == (-1, 0, 1):  # Diagonal rays in XZ plane
        x, y = np.meshgrid(np.arange(-Nz, Nx + Nz) + 0.5, np.arange(Ny) + 0.5, indexing='ij')
        return x.ravel(), y.ravel(), np.zeros((Nx + 2*Nz) * Ny) + 0.5
    elif Cj == (0, 1, 1) or Cj == (0, -1, -1) or Cj == (0, 1, -1) or Cj == (0, -1, 1):  # Diagonal rays in YZ plane
        x, y = np.meshgrid(np.arange(Nx) + 0.5, np.arange(-Nz, Ny + Nz) + 0.5, indexing='ij')
        return x.ravel(), y.ravel(), np.zeros((Ny + 2*Nz) * Nx) + 0.5
    else:
        raise ValueError(f"Unsupported Cj direction: {Cj}. Only cardinal directions are supported in 3D.")
def Bouzidi_core_3D(solidsUnion, Cij, N, IBar):
    Nx, Ny, Nz = N
    Q = Cij.shape[0]
    shapef = (Q, Nx, Ny, Nz, 1)

    
    key_type = types.UniTuple(types.int64, 2)
    value_type = types.float64
    CSM = Dict.empty(key_type, value_type)
    Fi2Ftot = Dict.empty(key_type, value_type)

    for i in range(Q):
        Cj = Cij[i].ravel()
        iBar = IBar[i]
        if np.all(Cj == 0):
            continue

        starting_points_x, starting_points_y, starting_points_z = get_starting_points_3D(Nx, Ny, Nz, Cj)
        ray_starting_points = np.vstack([starting_points_x, starting_points_y, starting_points_z]).T

        # Perturb slightly to prevent issues where rays hit the solid exactly at a vertex or edge, which can cause numerical issues
        solidsUnionPerturbed = solidsUnion.apply_translation(np.array([1e-6, 1e-6, 1e-6]))

        locs, _, tri_ids = solidsUnionPerturbed.ray.intersects_location(
            ray_origins=ray_starting_points - 2*max(Nx, Ny, Nz)*Cj, # Start the rays far enough away from the solid to ensure they intersect
            ray_directions=np.tile(Cj, (ray_starting_points.shape[0], 1)), 
            multiple_hits=True
        )

        # Find hits into the solid (surface normal opposite Cj)
        valid_ids = np.where(solidsUnion.face_normals[tri_ids].dot(Cj) < 0)[0]

        # Hits on the wall
        xw_arr = locs[valid_ids]

        for j in range(len(xw_arr)):
            for k in range(j + 1, len(xw_arr)):
                if np.allclose(xw_arr[j], xw_arr[k], atol=1e-14):
                    raise ValueError(f"Duplicate hit points found: {xw_arr[j]} and {xw_arr[k]}")


        build_Bouzidi_matrices(np.array(xw_arr), Cj, i, iBar, N, shapef, CSM, Fi2Ftot)

    return dictToSparseMat(CSM, (Q*Nx*Ny*Nz, Q*Nx*Ny*Nz)), dictToSparseMat(Fi2Ftot, (3, Q*Nx*Ny*Nz))



def Bouzidi_core(solidsUnion, Cij, N, IBar):
    D = Cij.shape[-1]

    if D == 1:
        return Bouzidi_core_1D()
    elif D == 2:
        return Bouzidi_core_2D(solidsUnion, Cij, N, IBar)
    elif D == 3:
        return Bouzidi_core_3D(solidsUnion, Cij, N, IBar)
