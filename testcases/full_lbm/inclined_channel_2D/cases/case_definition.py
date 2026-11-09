from dataclasses import dataclass
import numpy as np

@dataclass
class Case:
    # Common across all cases
    g_base = 8e-6 # g for Nx = 1
    tau_base = 0.50125 # tau for Nx = 1
    precision = 'double' # Precision for the simulation
    N_flow_throughs = 10 # Number of flow-throughs, based on the channel height and the maximum expected velocity.

    # case-specific parameters
    theta_deg: float | None = None # Channel angle in degrees
    h: int | None = None # Channel height in lattice units


    # Derived properties
    @property
    def height(self):
        return 2 * self.h

    @property
    def width(self):
        base_h = 8
        # Round base width to nearest even number
        return int(base_h / np.tan(np.deg2rad(self.theta_deg)) + 0.5) * 2 * int(self.h / base_h)

    @property
    def theta_deg_actual(self):
        # Actual theta is sligthly different from the specified one such that the channel width and height are integers.
        return np.rad2deg(np.arctan2(self.height, self.width))

    @property
    def g(self):
        return self.g_base / self.h

    @property
    def tau(self):
        return 0.5 + (self.tau_base - 0.5) * self.h

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

    def x_rel(self, wall_dist):
        return wall_dist / self.h - 0.5

    def analytical_solution(self, x_rel):
        u_ref = self.g * self.h**2 / (2 * self.nu) * (1 / 4 - x_rel**2)
        return u_ref

