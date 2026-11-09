import numpy as np
from numba import cuda
import time
import sys
import warnings

from Context import Context

class LBM_Simulator:
    def __init__(self, 
                 nx=1, 
                 ny=1, 
                 nz=1, 
                 Nt=1, 
                 tau=1, 
                 body_force=(0.0, 0.0, 0.0),
                 domain_boundaries={},
                 writers=[],
                 CSM=None,
                 Fi2Ftot=None,
                 Fi2fsurfInput=None,
                 cell_volumes=None,
                 cell_centroids=None,
                 wall_dists=None,
                 vertices=None,
                 faces=None,
                 stencil='d3q19', 
                 collision_model='BGK', 
                 equilibrium_model='default_equilibrium', 
                 macroscopic_model='default_macroscopic',
                 initialize_model='initialize_equilibrium',
                 stream_only=False,
                 use_cuda=True,
                 debug=False,
                 precision='single'):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Nt = Nt
        self.tau = tau
        self.domain_boundaries = domain_boundaries
        self.writers = writers
        self.CSM = CSM
        self.Fi2Ftot = Fi2Ftot
        self.Fi2fsurfInput = Fi2fsurfInput
        self.vertices = vertices
        self.faces = faces
        self.stencil = stencil
        self.collision_model = collision_model
        self.body_force = body_force
        self.equilibrium_model = equilibrium_model
        self.macroscopic_model = macroscopic_model
        self.initialize_model = initialize_model
        self.stream_only = stream_only
        self.debug = debug
        self.precision = precision
        self.ignore_performance_warning = True

        """Initialize an LBM simulation and configure its selected models and execution context."""

        if self.ignore_performance_warning:
            warnings.filterwarnings(
                "ignore",
                message=".*Grid size .*under-utilization.*",
            )
        

        self.current_timestep = 0

        self.ctx = Context(
            stencil=stencil,
            n=(nx, ny, nz),
            Nt=Nt,
            tau=tau,
            body_force=body_force,
            domain_boundaries=domain_boundaries,
            initialize_model=initialize_model,
            macroscopic_model=macroscopic_model,
            equilibrium_model=equilibrium_model,
            collide_model=collision_model,
            Fi2Ftot=Fi2Ftot,
            Fi2fsurfInput=Fi2fsurfInput,
            cell_volumes=cell_volumes,
            cell_centroids=cell_centroids,
            wall_dists=wall_dists,
            use_cuda=use_cuda,
            debug=debug,
            precision=precision,
            CSM=CSM
        )

        self.Q = self.ctx.Q
        self.D = self.ctx.D
        self.use_cuda = self.ctx.use_cuda

        if self.stream_only:
            from LBM_solver.stream_only import lbm_stream_only_factory
            self.lbm_step = lbm_stream_only_factory(self.ctx)
        else:
            from LBM_solver.step import lbm_step_factory
            self.lbm_step = lbm_step_factory(self.ctx)
        
        from LBM_solver.streaming.apply_CSM import apply_CSM_factory
        self.apply_CSM = apply_CSM_factory(self.ctx)

        from LBM_solver.output.initialize_writers import initialize_writers
        self.field_writer_functions, self.force_writer_functions = initialize_writers(self.ctx, self.writers)

        from LBM_solver.domain_boundaries.domain_boundary import apply_domain_boundaries_factory
        self.apply_domain_boundaries = apply_domain_boundaries_factory(self.ctx)




    def initialize_f(self, rho0=1.0, u0=(0.0, 0.0, 0.0), F0=None):
        """Initialize the distribution-function buffers from the supplied density and velocity."""

        initialize = self.ctx.initialize_factory(self.ctx)

        self.F = initialize(rho0, u0, F0=F0)
        self.F_new = initialize(rho0, u0, F0=F0) 

    def collide_and_periodic_stream(self):
        self.lbm_step(self.F, self.F_new, self.current_timestep)
        self.current_timestep += 1
        self.F_new, self.F = self.F, self.F_new  # Swap references for next iteration

    def write_forces(self):
        for writer_func in self.force_writer_functions:
            if (self.current_timestep - writer_func.start_time) % writer_func.write_interval == 0 and \
                self.current_timestep >= writer_func.start_time and \
                    self.current_timestep <= writer_func.end_time:
                writer_func(self.F, self.current_timestep)

    def correct_boundaries(self):
        self.apply_CSM(self.F)  # Apply CSM after streaming
        self.apply_domain_boundaries(self.F, t=self.current_timestep)  # Apply domain boundaries after CSM

    def write_fields(self):
        for writer_func in self.field_writer_functions:
            if (self.current_timestep - writer_func.start_time) % writer_func.write_interval == 0 and \
                self.current_timestep >= writer_func.start_time and \
                    self.current_timestep <= writer_func.end_time:
                writer_func(self.F, self.current_timestep)

    def step(self):
        """Advance the simulation by one timestep."""

        # 1. COLLISION + 2. PERIODIC STREAMING
        self.collide_and_periodic_stream()

        # 2. WRITE OUTPUT OF FORCES
        self.write_forces()

        # 3. CORRECT FOR BOUNDARIES
        self.correct_boundaries()
        
        # 4. WRITE OUTPUT OF FIELDS
        self.write_fields()

    def simulate(self, rho0=1.0, u0=(0.0, 0.0, 0.0), F0=None):
        """Run the configured simulation for the specified number of timesteps."""

        self.initialize_f(rho0, u0, F0=F0)

        # 5. WRITE OUTPUT OF FIELDS
        for writer_func in self.field_writer_functions:
            if (self.current_timestep - writer_func.start_time) % writer_func.write_interval == 0 and \
                self.current_timestep >= writer_func.start_time and \
                    self.current_timestep <= writer_func.end_time:
                writer_func(self.F_new, self.current_timestep)

        t0 = time.perf_counter()
        for it in range(self.Nt):

            t1 = time.perf_counter()
            self.step()
            t2 = time.perf_counter()

            glups_step = (self.nx * self.ny * self.nz) / (t2 - t1) / 1e9

            # Progress bar
            progress = (it + 1) / self.Nt
            bar_width = 40
            filled = int(bar_width * progress)
            bar = "█" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\r[{bar}] {100*progress:6.2f}% | {glups_step:6.2f} GLUPS"
            )
            sys.stdout.flush()

        glups_avg = (self.nx * self.ny * self.nz * self.Nt) / (time.perf_counter() - t0) / 1e9
        sys.stdout.write(
            f"\r[{bar}] {100*progress:6.2f}% | {glups_avg:6.2f} GLUPS (average)"
        )
        sys.stdout.flush()        

        print()  # Newline after progress bar


        if self.use_cuda:
            cuda.synchronize()  # Ensure all GPU operations are complete before measuring time
            self.F = self.F.copy_to_host()  # Copy final state back to host

        return


def simulate(
        nx=1,
        ny=1, 
        nz=1, 
        Nt=1, 
        tau=1,
        rho0=1.0,
        u0=(0.0, 0.0, 0.0),
        F0=None,
        body_force=(0.0, 0.0, 0.0),
        domain_boundaries={},
        writers=[],
        CSM=None,
        Fi2Ftot=None,
        Fi2fsurfInput=None,
        cell_volumes=None,
        cell_centroids=None,
        wall_dists=None,
        vertices=None,
        faces=None,
        stencil='d3q19', 
        collision_model='BGK', 
        equilibrium_model='default_equilibrium', 
        macroscopic_model='default_macroscopic',
        initialize_model='initialize_equilibrium',
        stream_only=False,
        use_cuda=True,
        debug=False,
        precision='single'):
    lbm = LBM_Simulator(
        nx=nx,
        ny=ny,
        nz=nz,
        Nt=Nt,
        tau=tau,
        body_force=body_force,
        domain_boundaries=domain_boundaries,
        writers=writers,
        CSM=CSM,
        Fi2Ftot=Fi2Ftot,
        Fi2fsurfInput=Fi2fsurfInput,
        cell_volumes=cell_volumes,
        cell_centroids=cell_centroids,
        wall_dists=wall_dists,
        vertices=vertices,
        faces=faces,
        stencil=stencil,
        collision_model=collision_model,
        equilibrium_model=equilibrium_model,
        macroscopic_model=macroscopic_model,
        initialize_model=initialize_model,
        stream_only=stream_only,
        use_cuda=use_cuda,
        debug=debug,
        precision=precision
    )
    """Create and run an LBM simulation with the supplied configuration."""

    lbm.simulate(rho0, u0, F0=F0)
    return lbm
