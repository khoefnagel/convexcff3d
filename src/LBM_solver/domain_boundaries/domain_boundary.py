import numpy as np
import importlib

from ..solver_util.get_slice_inds import get_slice_inds

def apply_domain_boundaries_factory(ctx):
    """Create a function that applies domain boundary conditions to the distribution function F."""

    D = ctx.D
    Q = ctx.Q
    Cij_int = ctx.Cij_int
    float_t = ctx.float_t
    use_cuda = ctx.use_cuda

    if use_cuda:
        from numba import cuda

    domain_boundary_buffers = [] # Buffers to store the updates for the unknown populations at the domain boundaries.
    get_updates_functions = []  # Functions to get the updates for the unknown populations at the domain boundaries.
    apply_domain_boundaries_updates_functions = [] # Function to apply the updates for the unknown populations.

    for side_normal, domain_boundary in ctx.domain_boundaries.items():  # Iterate over each domain boundary.

        # Validate the side normal and domain boundary dictionary for the current domain boundary condition.
        validate_side_normal(side_normal, domain_boundary, D)

        # Indices of the unknown populations that need to be reconstructed by the boundary condition.
        unknown_populations = np.where(np.sum(np.array(side_normal) * Cij_int, axis=-1).reshape(Q) > 0)[0]

        # Index of the normal direction of the boundary condition (0 for x, 1 for y, 2 for z).
        normal_index = np.where(np.array(side_normal) != 0)[0][0]

        # Create a slice tuple to select the cells at the domain boundary
        slices_ijk = [slice(None)] * 3
        slices_ijk[normal_index] = slice(-1, None) if side_normal[normal_index] == -1 else slice(0, 1)
        slices_ijk = tuple(slices_ijk)

        # Get the shape of the domain boundary slice and the starting indices.
        # Both are tuples of length 3.
        n_, start_inds = get_slice_inds(ctx, slices_ijk)

        # Initialize a buffer to store the updates for the unknown populations at the domain boundary.
        buffer = cuda.device_array((Q, *n_, 1), dtype=float_t) if use_cuda else np.zeros((Q, *n_, 1), dtype=float_t)
        domain_boundary_buffers.append(buffer)

        # Create the function to obtain the updates for the unknown populations at the domain boundary.
        get_updates_factory = importlib.import_module(f"LBM_solver.domain_boundaries.{domain_boundary['type']}").get_domain_boundary_updates_factory
        get_updates = get_updates_factory(ctx, domain_boundary, unknown_populations, start_inds, n_, side_normal, normal_index)
        get_updates_functions.append(get_updates)

        # Create the function to apply the updates for the unknown populations at the domain boundary.
        apply_updates_factory = importlib.import_module(f"LBM_solver.domain_boundaries.{domain_boundary['type']}").apply_domain_boundary_updates_factory
        apply_updates = apply_updates_factory(ctx, domain_boundary, unknown_populations, start_inds, n_, side_normal, normal_index)
        apply_domain_boundaries_updates_functions.append(apply_updates)


    def apply_domain_boundaries(F, t):
        """Apply domain boundary conditions to the distribution function F.
        This is done in two steps:
        1) Get the updates for the unknown populations at the domain boundaries and store them in the buffers.
        2) Apply the updates from the buffers to the distribution function F.
        
        Args:
            F: Distribution function array of shape (Q, nx, ny, nz, 1), updated in place.
            t: Current timestep.
        """

        # 1) Get the updates for the unknown populations at the domain boundaries and store them in the buffers.
        for j, get_updates in enumerate(get_updates_functions):
            get_updates(F, domain_boundary_buffers[j], t)

        cuda.synchronize() if use_cuda else None

        # 2) Apply the updates from the buffers to the distribution function F.
        for j, apply_updates in enumerate(apply_domain_boundaries_updates_functions):
            apply_updates(F, domain_boundary_buffers[j], t)

        cuda.synchronize() if use_cuda else None

    return apply_domain_boundaries



def validate_side_normal(side_normal, domain_boundary, D):
    """Validate the side normal and domain boundary dictionary for a given domain boundary condition.
    
    Args:
        side_normal: Tuple of length D representing the normal direction of the domain boundary,
                      where inwards is positive.
        domain_boundary: Dictionary containing the domain boundary condition parameters.
        D: Number of spatial dimensions (1, 2, or 3).
    """
    if D == 1:
        assert side_normal in [(1), (-1)], f"Invalid side normal {side_normal} in domain_boundaries. Must be one of [(1), (-1)]"
    elif D == 2:
        assert side_normal in [(1, 0), (-1, 0), (0, 1), (0, -1)], f"Invalid side normal {side_normal} in domain_boundaries. Must be one of [(1, 0), (-1, 0), (0, 1), (0, -1)]"  
    elif D == 3:
        assert side_normal in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], f"Invalid side normal {side_normal} in domain_boundaries. Must be one of [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]"
    assert isinstance(domain_boundary, dict), f"Domain boundary for side normal {side_normal} must be a dictionary with keys 'type' and additional parameters depending on the type"
    assert 'type' in domain_boundary, f"Domain boundary for side normal {side_normal} must have a 'type' key"





