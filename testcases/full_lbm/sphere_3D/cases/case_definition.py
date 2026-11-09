from dataclasses import dataclass
import numpy as np

@dataclass
class Case:
    # Common across all cases
    Nx_over_D = 20
    Ny_over_D = 10
    Nz_over_D = 10
    ux_inf = 0.1
    N_flow_through = 2.5
    x_c_over_N = np.array([0.2, 0.5, 0.5]) # Center of the sphere as a fraction of the domain size in each direction
    precision = 'single'

    # case-specific parameters
    Re: float | None = None # Reynolds number
    cells_per_radius: float | None = None


    # Derived properties
    @property
    def Nx(self):
        assert self.cells_per_radius is not None, "cells_per_radius must be set to compute Nx"
        return self.Nx_over_D*int(np.ceil(2*self.cells_per_radius))

    @property
    def Ny(self):
        assert self.cells_per_radius is not None, "cells_per_radius must be set to compute Ny"
        return self.Ny_over_D*int(np.ceil(2*self.cells_per_radius)) + 1  # Add 1 to ensure we have a point at the center of the sphere

    @property
    def Nz(self):
        assert self.cells_per_radius is not None, "cells_per_radius must be set to compute Nz"
        return self.Nz_over_D*int(np.ceil(2*self.cells_per_radius)) + 1  # Add 1 to ensure we have a point at the center of the sphere

    @property
    def D(self):
        assert self.cells_per_radius is not None, "cells_per_radius must be set to compute D"
        return 2 * self.cells_per_radius

    @property
    def t_flow_through(self):
        return int(self.Nx / self.ux_inf)
    
    @property
    def Nt(self):
        return int(self.N_flow_through * self.t_flow_through)
    
    @property
    def tau(self):
        assert self.Re is not None, "Re must be set to compute tau"
        return 0.5 + 3.0 * self.ux_inf * self.D / self.Re
    
    @property
    def x_c(self):
        return self.x_c_over_N * np.array([self.Nx, self.Ny, self.Nz])
    
    @property
    def Ftot_write_interval(self):
        return self.t_flow_through // 100
    
    @property
    def z_slice_write_interval(self):
        return self.Nt
    
    @property
    def center_profile_write_interval(self):
        return self.Nt
    
    @property
    def fsurf_write_interval(self):
        return self.Nt
