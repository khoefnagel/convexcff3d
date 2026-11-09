from dataclasses import dataclass
import numpy as np
import importlib
from scipy import sparse

@dataclass
class Context:
    n: tuple
    tau: float
    body_force: tuple
    stencil: str
    D: int
    Q: int
    Cij_float: np.ndarray
    Cij_int: np.ndarray
    Wi_float: np.ndarray
    kernel_decorator: callable
    device_decorator: callable
    domain_boundaries: list
    initialize_model: str
    macroscopic_model: str
    equilibrium_model: str
    collide_model: str
    Fi2Ftot: sparse.coo_matrix
    Fi2fsurfInput: sparse.coo_matrix
    float_t: type
    int_t: type
    threads_1d: int
    use_cuda: bool
    debug: bool
    rows: np.ndarray = None
    cols: np.ndarray = None
    values: np.ndarray = None

    def __init__(self, 
                 n=(1, 1, 1),
                 Nt=1,
                 tau=1.0,
                 body_force=(0., 0., 0.),
                 stencil='d2q9', 
                 domain_boundaries=[],
                 initialize_model='initialize_equilibrium',
                 macroscopic_model='default_macroscopic',
                 equilibrium_model='default_equilibrium',
                 collide_model='BGK',
                 forcing_model=None,
                 Fi2Ftot=None,
                 Fi2fsurfInput=None,
                 cell_volumes=None,
                 cell_centroids=None,
                 wall_dists=None,
                 use_cuda=False,
                 debug=False, 
                 cache=False, 
                 threads=256, 
                 precision='single',
                 CSM=None,
                 ):
        
        self.stencil = stencil
        self.Nt = Nt
        self.tau = tau
        self.domain_boundaries = domain_boundaries
        self.initialize_model = initialize_model
        self.macroscopic_model = macroscopic_model
        self.equilibrium_model = equilibrium_model
        self.collide_model = collide_model
        self.Fi2Ftot = Fi2Ftot
        self.Fi2fsurfInput = Fi2fsurfInput
        self.debug = debug
        self.cache = cache
        self.threads_1d = threads
        self.precision = precision

        match precision:
            case 'half':
                self.float_t = np.float16
                self.int_t = np.int16
            case 'single':
                self.float_t = np.float32
                self.int_t = np.int32
            case 'double':
                self.float_t = np.float64
                self.int_t = np.int64
            case _:
                raise ValueError(f"Unsupported precision: {precision}")

        self.use_cuda = self._verify_cuda() if use_cuda else False

        self.n = np.asarray(n, dtype=self.int_t)

        stencil = importlib.import_module(f"LBM_solver.stencils.{stencil.lower()}")
        self.D = stencil.D
        self.Q = stencil.Q
        self.Cij_float = stencil.Cij.astype(self.float_t)
        self.Cij_int = stencil.Cij.astype(self.int_t)
        self.Wi_float = stencil.Wi.astype(self.float_t)

        if self.use_cuda:
            from numba import cuda
            self.kernel_decorator = cuda.jit(device=False, cache=cache)
            self.device_decorator = cuda.jit(device=True, cache=cache, inline='always')
        else:
            from numba import njit
            self.kernel_decorator = njit(parallel=True, cache=cache)
            self.device_decorator = njit(parallel=True, cache=cache, inline='always')

        ind_dtype = np.int32 if np.prod(n)*self.Q < np.iinfo(np.int32).max else np.int64
        self.rows = np.array(CSM.row, dtype=ind_dtype) if CSM is not None else None
        self.cols = np.array(CSM.col, dtype=ind_dtype) if CSM is not None else None
        self.values = np.array(CSM.data, dtype=self.float_t) if CSM is not None else None

        self.cell_volumes = cuda.to_device(cell_volumes) if self.use_cuda and cell_volumes is not None else cell_volumes
        self.cell_centroids = cuda.to_device(cell_centroids) if self.use_cuda and cell_centroids is not None else cell_centroids
        self.wall_dists = cuda.to_device(wall_dists) if self.use_cuda and wall_dists is not None else wall_dists

        self.initialize_factory = importlib.import_module(f"LBM_solver.initialization_models.{self.initialize_model}").initialize_factory
        self.macroscopic_factory = importlib.import_module(f"LBM_solver.macroscopic_models.{self.macroscopic_model}").macroscopic_factory
        self.equilibrium_factory = importlib.import_module(f"LBM_solver.equilibrium_models.{self.equilibrium_model}").equilibrium_factory
        self.collide_factory = importlib.import_module(f"LBM_solver.collision_models.{self.collide_model}").collide_factory


        self.zero_body_force = self._zero_body_force(body_force)
        self.body_force = self._prepare_body_force(body_force)

        if self.zero_body_force:
            self.forcing_model = 'none'
        else:
            self.forcing_model = 'Guo' if forcing_model is None else forcing_model
        self.force_shift_velocity_factory = importlib.import_module(f"LBM_solver.forcing.models.{self.forcing_model}").force_shift_velocity_factory
        self.add_source_term_factory = importlib.import_module(f"LBM_solver.forcing.models.{self.forcing_model}").add_source_term_factory



    @property
    def omega(self):
        """Return the BGK relaxation parameter."""
        return self.float_t(1.0) / self.tau


    def _verify_cuda(self):
        """Check whether Numba CUDA is available."""
        try:
            from numba import cuda
        except ImportError:
            print("Numba is not installed. Please install numba to use CUDA features. Falling back to CPU mode.")
            return False
        if not cuda.is_available():
            print("CUDA is not available. Please check your CUDA installation. Falling back to CPU mode.")
            return False
        return True

    def _zero_body_force(self, body_force):
        """Return whether all configured body-force components are zero."""
        for d in range(self.D):
            if callable(body_force[d]):
                return False
            elif body_force[d] != self.float_t(0.0):
                return False
        return True

    def _prepare_body_force(self, body_force_in):
        """Validate and convert body-force components to callable functions with signature ``(x, y, z, t)``.

        Args:
            body_force_in: Body-force components, specified as values or callables.

        Returns:
            A tuple of prepared body-force functions suitable for kernel execution.
        """
        body_force = []
        for d in range(self.D):
            if callable(body_force_in[d]):

                # Check signature is (x, y, z, t)
                import inspect
                sig = inspect.signature(body_force_in[d])
                if sig.parameters.keys() != {'x', 'y', 'z', 't'}:
                    raise ValueError(f"Body force function for dimension {d} has an invalid signature. Expected (x, y, z, t).")

                # Check whether the function returns a float_t
                test_value = body_force_in[d](0, 0, 0, 0)
                if not isinstance(test_value, (float, np.floating)):
                    raise ValueError(f"Body force function for dimension {d} must return a float or np.floating type, but got {type(test_value)}.")

                # Check whether the function is numba jit compilable
                try:
                    body_force.append(self.device_decorator(body_force_in[d]))
                except Exception as e:
                    raise ValueError(f"Body force function for dimension {d} is not numba compilable. Error: {e}")

            else:
                val = self.float_t(body_force_in[d])
                def _make_const_body_force_func(val):
                    def body_force_func(x, y, z, t):
                        return val
                    return body_force_func
                body_force.append(self.device_decorator(_make_const_body_force_func(val)))
        return tuple(body_force)

    
    def __repr__(self):
        return (f"Context(n={self.n}, tau={self.tau}, stencil='{self.stencil}', "
                f"D={self.D}, Q={self.Q}, macroscopic_model='{self.macroscopic_model}', "
                f"equilibrium_model='{self.equilibrium_model}', collide_model='{self.collide_model}', "
                f"float_t={self.float_t.__name__}, int_t={self.int_t.__name__}, "
                f"threads_1d={self.threads_1d}, use_cuda={self.use_cuda}, debug={self.debug})")