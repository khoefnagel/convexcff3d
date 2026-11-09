import numpy as np

from .write_fields import write_fields_factory
from .write_total_forces import write_total_force_factory
from .write_surface_forces import write_surface_forces_factory

def initialize_writers(ctx, writer_dicts):
    """Initialize the field and force writer functions based on the provided writer dictionaries."""

    # Lists of writer functions`to be returned`
    field_writer_functions = []
    force_writer_functions = []

    for writer in writer_dicts:

        # Get the writer type, default is 'field'
        writer_type = writer.get('type', 'field')

        # Get the start time of the writer, default is 0. If negative, it counts from the last time step.
        start_time = writer.get('start_time', 0)  # Default to starting at time 0 if not specified
        start_time = ctx.Nt + start_time if start_time < 0 else start_time  # Handle negative start times

        # Get the end time of the writer, default is ctx.Nt. If negative, it counts from the last time step.
        end_time = writer.get('end_time', ctx.Nt)  # Default to ending at the last time step if not specified
        end_time = ctx.Nt + end_time if end_time < 0 else end_time  # Handle negative end times

        # Get the write interval, default is the number of time steps such that it writes only the last time step.
        write_interval = writer.get('write_interval', ctx.Nt)  # Default to writing only at the end if not specified

        # Get whether to write cell data or point data. Default is cell data.
        write_cells = writer.get('write_cells', True)  # Default to writing cell data if not specified

        # Get the slices of the domain to write, default is the entire domain.
        slices_xyz_ = writer.get('slices_xyz', (slice(None), slice(None), slice(None)))

        # Convert any integer slice indices to slice objects to maintain consistent output shapes
        slices_xyz = correct_slices_xyz(slices_xyz_)


        match writer_type:
            
            case 'field':
                filename_template = writer.get('filename_template', 'field_{:05d}')
                field_writer_function = write_fields_factory(
                        ctx=ctx,
                        filename_template=filename_template,
                        fields=writer.get('fields', []),
                        slices_ijk=slices_xyz,
                        write_cells=write_cells
                    )
                field_writer_function.start_time = start_time  # Attach the start time to the function
                field_writer_function.end_time = end_time  # Attach the end time to the function
                field_writer_function.write_interval = write_interval  # Attach the write interval to the function
                field_writer_functions.append(field_writer_function)

            case 'total_force':
                # Number of leading zeros depending on Nt
                filename = writer.get('filename', 'output') 
                filename_template = filename + '_{:0' + str(len(str(ctx.Nt))) + 'd}'
                force_writer_function = write_total_force_factory(
                        ctx=ctx,
                        filename=filename
                    )
                force_writer_function.start_time = start_time  # Attach the start time to the function
                force_writer_function.end_time = end_time  # Attach the end time to the function
                force_writer_function.write_interval = write_interval  # Attach the write interval to the function
                force_writer_functions.append(force_writer_function)

            case 'surface_force':
                filename_template = writer.get('filename_template', 'field_{:05d}')
                force_writer_function = write_surface_forces_factory(
                        ctx=ctx,
                        filename_template=filename_template,
                        vertices=writer.get('vertices'),
                        faces=writer.get('faces')
                    )
                force_writer_function.start_time = start_time  # Attach the start time to the function
                force_writer_function.end_time = end_time  # Attach the end time to the function
                force_writer_function.write_interval = write_interval  # Attach the write interval to the function
                force_writer_functions.append(force_writer_function)

            case _:
                raise ValueError(f"Unknown writer type: {writer_type}")

    return field_writer_functions, force_writer_functions

def correct_slices_xyz(slices_ijk_):
        """Convert any integer slice indices to slice objects to maintain consistent output shapes
        
        Args:
            slices_ijk_: Tuple of slices or integers for each dimension (x, y, z). 
            If an integer is provided, it will be converted to a slice that selects only that index."""
        
        assert len(slices_ijk_) == 3, "Slice must be a tuple of 3 elements (x, y, z)"
        slices_ijk = list(slices_ijk_)  # Convert to list for mutability
        for d in range(3):
            if isinstance(slices_ijk_[d], (int, np.integer)):
                slices_ijk[d] = slice(slices_ijk_[d], slices_ijk_[d]+1)
        return tuple(slices_ijk)  # Convert back to tuple