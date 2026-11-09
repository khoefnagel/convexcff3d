
from ..derived.get_total_force import get_total_force_factory

def write_total_force_factory(ctx, filename):
    """Create a function to write total forces to a .csv file. This file is updated at each timestep."""

    D = ctx.D
    use_cuda = ctx.use_cuda

    filename += '.csv'

    # Function to compute the total force
    get_total_force = get_total_force_factory(ctx)

    # Create the output file and write the header
    Ftot_file = open(filename, 'w')
    header = ', '.join(['t', 'Fx', 'Fy', 'Fz'][:D+1])
    Ftot_file.write(f'{header}\n')  # Write header for CSV file

    def write_total_force(F, t):
        """Write the total forces to a .csv file at the given timestep.
        
        Args:
            F: Distribution function array of shape (Q, nx, ny, nz, 1).
            t: Current timestep.
        """

        # Compute the total force
        Ftot = get_total_force(F).copy_to_host() if use_cuda else get_total_force(F)

        # Write
        Ftot_file.write(f'{t},' + ', '.join(f'{Ftot[d]:.6e}' for d in range(D)) + '\n')
        Ftot_file.flush()  # Ensure data is written to file

    return write_total_force




