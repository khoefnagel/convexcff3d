
import inspect
import textwrap

def apply_kernel_over_domain_factory(ctx, kernel, domain_shape, boundary_offset=(0, 0, 0)):
    """Create a kernel that applies the given device kernel acting on a single cell over the entire specified domain.
    Works for both CPU (numba.njit) and GPU (numba.cuda.jit) execution contexts.
    
    Args:
        ctx: Simulation context object.
        kernel: Device kernel function that operates on a single cell, taking (ijk, *args) as arguments,
                where ijk is a tuple of (i, j, k) cell indices.
        domain_shape: Tuple specifying the shape of the (sub)-domain (nx, ny, nz).
        boundary_offset: Tuple specifying the offset to apply to the indices (default is (0, 0, 0))."""

    # Namespace of variables needed for the dynamically generated kernel code.
    namespace = {"ctx": ctx, "kernel": kernel, "domain_shape": domain_shape, "boundary_offset": boundary_offset}

    # Get the kernel arguments and verify that the first argument is 'ijk'.
    kernel_func = getattr(kernel, "py_func", kernel)  # Get the underlying Python function if it's a Numba kernel
    kernel_args = list(inspect.signature(kernel_func).parameters)
    assert kernel_args[0] == 'ijk', f"First argument of the kernel must be 'ijk', but got {kernel_args[0]}"

    # Store the rest of the kernel arguments as a string
    signature = ', '.join(kernel_args[1:])  # Join the rest of the arguments as a string

    # Create wrapper kernel that adds the boundary offset to the indices, add to namespace
    apply_kernel_over_domain_kernel_template = textwrap.dedent(f"""
    @ctx.device_decorator
    def apply_kernel_over_domain_kernel(ijk, {signature}):
        i, j, k = ijk
        i2 = i + boundary_offset[0]
        j2 = j + boundary_offset[1]
        k2 = k + boundary_offset[2]
        kernel((i2, j2, k2), {signature})""")
    exec(apply_kernel_over_domain_kernel_template, namespace)

    # ==================================================================================================
    # GPU code 
    if ctx.use_cuda:
        from numba import cuda

        from .get_blocks_threads import get_blocks_threads
        blocks, threads = get_blocks_threads(*domain_shape)

        # Add CUDA-specific variables to the namespace
        namespace.update({"cuda": cuda, "blocks": blocks, "threads": threads})

        # Create the GPU kernel that applies the kernel over the entire domain
        apply_kernel_over_domain_GPU_template = textwrap.dedent(f"""
        @ctx.kernel_decorator
        def apply_kernel_over_domain_GPU({signature}):
            ijk = cuda.grid(3)
            if ijk[0] >= domain_shape[0] or ijk[1] >= domain_shape[1] or ijk[2] >= domain_shape[2]:
                return

            apply_kernel_over_domain_kernel(ijk, {signature})


        def wrapper({signature}):
            apply_kernel_over_domain_GPU[blocks, threads]({signature})""")
    
        exec(apply_kernel_over_domain_GPU_template, namespace)

        # Return the wrapper function that launches the GPU kernel over the entire domain
        return namespace["wrapper"]

    # ==================================================================================================
    # CPU code
    else:
        from numba import prange

        from .unravel_index import unravel_index_factory
        unravel_index = unravel_index_factory(ctx, domain_shape)

        # Add CPU-specific variables to the namespace
        namespace.update({"prange": prange, "unravel_index": unravel_index})

        # Create the CPU kernel that applies the kernel over the entire domain
        apply_kernel_over_domain_CPU_template = textwrap.dedent(f"""
        @ctx.kernel_decorator
        def apply_kernel_over_domain_CPU({signature}):
            for m in prange(domain_shape[0] * domain_shape[1] * domain_shape[2]):
                ijk = unravel_index(m)

                apply_kernel_over_domain_kernel(ijk, {signature})""")

        exec(apply_kernel_over_domain_CPU_template, namespace)

        # Return the CPU kernel that applies the kernel over the entire domain
        return namespace["apply_kernel_over_domain_CPU"]
        