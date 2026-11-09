import numpy as np
from numba import cuda

def get_blocks_threads(nx, ny, nz):
    """
    Calculate the number of blocks and threads for a 3D CUDA kernel launch based on the
    dimensions of the computational domain (nx, ny, nz).
    Prefers to put threads in the z-direction, then y-direction, and finally x-direction.
    
    Args:
        nx (int): Number of cells in the x-direction.
        ny (int): Number of cells in the y-direction.
        nz (int): Number of cells in the z-direction.

    Returns:
        blocks: Tuple[int, int, int]: Number of blocks in the x, y, and z directions.
        threads: Tuple[int, int, int]: Number of threads per block in the x, y, and z directions.
    """
    threads_1D = cuda.get_current_device().WARP_SIZE
    threads_z = min(threads_1D, nz)
    threads_1D //= threads_z
    threads_y = min(threads_1D, ny)
    threads_1D //= threads_y
    threads_x = min(threads_1D, nx)
    threads = (int(threads_x), int(threads_y), int(threads_z))
    blocks = (int(np.ceil(nx / threads[0])), int(np.ceil(ny / threads[1])), int(np.ceil(nz / threads[2])))
    return blocks, threads