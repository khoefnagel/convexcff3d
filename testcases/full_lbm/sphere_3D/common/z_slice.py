# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

import sys
sys.path.append('./definitions/')
from resolution_definition import case
Nt = case.Nt
Nz = case.Nz

print(Nt)


# create a new 'XML Image Data Reader'
slice_z_Nt_vti = XMLImageDataReader(FileName=[f'./output/slice_z_{Nt}.vti'])
slice_z_Nt_vti.CellArrayStatus = ['density', 'velocity']


# create a new 'Slice'
slice1 = Slice(Input=slice_z_Nt_vti)
slice1.SliceType = 'Plane'

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0, 0, Nz/2]

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 0.0, 1.0]



renderView1 = GetActiveViewOrCreate('RenderView')
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')
renderView1.ResetCamera(True)

# get display properties
slice1Display = GetDisplayProperties(slice1, view=renderView1)

# set scalar coloring
ColorBy(slice1Display, ('CELLS', 'velocity', 'Magnitude'))

SetActiveSource(slice_z_Nt_vti)

