from dataclasses import dataclass
import numpy as np

@dataclass
class Case:
    # Common across all cases
    g_base = 8e-6 # g for Nx = 1
    tau_base = 0.50125 # tau for Nx = 1
    precision = 'double' # Precision for the simulation
    N_flow_throughs = 10 # Number of flow-throughs, based on the channel height and the maximum expected velocity.
    pad = 5 # Number of cells to add on each side of the domain for boundary conditions

    # case-specific parameters
    pfluid: float | None = None # Fluid fraction of boundary cells
    Nx: int | None = None


    # Derived properties
    @property
    def g(self):
        return self.g_base / self.Nx

    @property
    def tau(self):
        return 0.5 + (self.tau_base - 0.5) * self.Nx

    @property
    def Nx_domain(self):
        return self.Nx + 2 * self.pad + np.ceil(2*(1 - self.pfluid)).astype(int) # 5 cells on each side for boundary conditions and addition cells to always have channel height >= Nx

    @property
    def h(self):
        return self.Nx_domain - 2 * self.pad - 2*(1 - self.pfluid)

    @property
    def nu(self):
        return (self.tau - 0.5) / 3

    @property
    def u_max(self):
        return self.g * self.h**2 / (8 * self.nu)

    @property
    def Nt(self):
        return int(self.N_flow_throughs * self.h / self.u_max)

    @property
    def write_interval(self):
        return int(self.Nt / 100) # Write 100 times during the simulation

    def x_rel(self, x):
        return (x - self.Nx_domain / 2) / self.h

    def analytical_solution(self, x_rel):
        u_ref = self.g * self.h**2 / (2 * self.nu) * (1 / 4 - x_rel**2)
        return u_ref

