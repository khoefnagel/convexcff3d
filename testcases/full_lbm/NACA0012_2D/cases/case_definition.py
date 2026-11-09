from dataclasses import dataclass
import numpy as np

@dataclass
class Case:
    # Common across all cases
    Nx_over_c = 20
    Ny_over_c = 6
    ux_inf = 0.1
    N_flow_through = 3
    Re = 100 # Reynolds number
    x_c_over_N = np.array([0.25, 0.5]) # Center of the quarter chord as a fraction of the domain size in each direction
    precision = 'single'

    # case-specific parameters
    alpha_deg: float | None = None # Angle of attack in degrees
    cells_per_chord: float | None = None


    # Derived properties
    @property
    def Nx(self):
        assert self.cells_per_chord is not None, "cells_per_chord must be set to compute Nx"
        return int(np.ceil(self.Nx_over_c*self.cells_per_chord))

    @property
    def Ny(self):
        assert self.cells_per_chord is not None, "cells_per_chord must be set to compute Ny"
        return int(np.ceil(self.Ny_over_c*self.cells_per_chord))
    
    @property
    def chord(self):
        assert self.cells_per_chord is not None, "cells_per_chord must be set to compute chord"
        return self.cells_per_chord

    @property
    def t_flow_through(self):
        return int(self.Nx / self.ux_inf)
    
    @property
    def Nt(self):
        return self.N_flow_through * self.t_flow_through
    
    @property
    def tau(self):
        assert self.Re is not None, "Re must be set to compute tau"
        return 0.5 + 3.0 * self.ux_inf * self.chord / self.Re
    
    @property
    def x_c(self):
        return self.x_c_over_N * np.array([self.Nx, self.Ny])
    
    @property
    def Ftot_write_interval(self):
        return self.t_flow_through // 100
    
    @property
    def field_write_interval(self):
        return self.Nt
    
    @property
    def center_profile_write_interval(self):
        return self.Nt

