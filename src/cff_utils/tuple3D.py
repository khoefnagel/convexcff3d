from numba import njit


@njit(cache=True, inline='always')
def tuple3D(arr):
    """Convert a 1D array of length 1, 2 or 3 to a 3D tuple by padding with zeros as necessary.
    
    Parameters
    ----------
    arr : np.ndarray
        A 1D array of length 1, 2 or 3.
    
    Returns
    -------
    tup : tuple
        A tuple of length 3.
    """
    
    if len(arr) == 1: # 1D, pad to 3D
        return (arr[0], 0, 0)
    elif len(arr) == 2: # 2D, pad to 3D
        return (arr[0], arr[1], 0)
    elif len(arr) == 3: # 3D
        return (arr[0], arr[1], arr[2])
    else:
        raise ValueError("Array must be of length 1, 2 or 3.")