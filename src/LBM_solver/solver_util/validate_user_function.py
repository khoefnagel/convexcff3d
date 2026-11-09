import numpy as np


def validate_user_function(func_in, ctx, expected_signature={'x', 'y', 'z', 't'}):

    if callable(func_in):

        # Check signature is as expected
        import inspect
        sig = inspect.signature(func_in)
        if sig.parameters.keys() != expected_signature:
            raise ValueError(f"Function has an invalid signature. Expected {expected_signature}.")

        # Check whether the function is numba.njit / numba.cuda.jit compilable
        try:  
            return ctx.device_decorator(func_in)
        except Exception as e:
            raise ValueError(f"Function is not numba compilable. Error: {e}")

    else:
        val = ctx.float_t(func_in)
        def _make_const_func(val):
            def const_func(x, y, z, t):
                return val
            return const_func
        return ctx.device_decorator(_make_const_func(val))