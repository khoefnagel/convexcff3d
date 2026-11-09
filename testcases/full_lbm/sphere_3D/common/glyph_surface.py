# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML PolyData Reader'
fsurf_16000vtp = XMLPolyDataReader(FileName=['./output/fsurf_16000.vtp'])
fsurf_16000vtp.CellArrayStatus = ['force']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1481, 795]

# show data in view
fsurf_16000vtpDisplay = Show(fsurf_16000vtp, renderView1)

# get color transfer function/color map for 'force'
forceLUT = GetColorTransferFunction('force')

# trace defaults for the display properties.
fsurf_16000vtpDisplay.Representation = 'Surface'
fsurf_16000vtpDisplay.ColorArrayName = ['CELLS', 'force']
fsurf_16000vtpDisplay.LookupTable = forceLUT
fsurf_16000vtpDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
fsurf_16000vtpDisplay.SelectOrientationVectors = 'None'
fsurf_16000vtpDisplay.ScaleFactor = 1.6
fsurf_16000vtpDisplay.SelectScaleArray = 'force'
fsurf_16000vtpDisplay.GlyphType = 'Arrow'
fsurf_16000vtpDisplay.GlyphTableIndexArray = 'force'
fsurf_16000vtpDisplay.GaussianRadius = 0.08
fsurf_16000vtpDisplay.SetScaleArray = [None, '']
fsurf_16000vtpDisplay.ScaleTransferFunction = 'PiecewiseFunction'
fsurf_16000vtpDisplay.OpacityArray = [None, '']
fsurf_16000vtpDisplay.OpacityTransferFunction = 'PiecewiseFunction'
fsurf_16000vtpDisplay.DataAxesGrid = 'GridAxesRepresentation'
fsurf_16000vtpDisplay.SelectionCellLabelFontFile = ''
fsurf_16000vtpDisplay.SelectionPointLabelFontFile = ''
fsurf_16000vtpDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
fsurf_16000vtpDisplay.DataAxesGrid.XTitleFontFile = ''
fsurf_16000vtpDisplay.DataAxesGrid.YTitleFontFile = ''
fsurf_16000vtpDisplay.DataAxesGrid.ZTitleFontFile = ''
fsurf_16000vtpDisplay.DataAxesGrid.XLabelFontFile = ''
fsurf_16000vtpDisplay.DataAxesGrid.YLabelFontFile = ''
fsurf_16000vtpDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
fsurf_16000vtpDisplay.PolarAxes.PolarAxisTitleFontFile = ''
fsurf_16000vtpDisplay.PolarAxes.PolarAxisLabelFontFile = ''
fsurf_16000vtpDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
fsurf_16000vtpDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
fsurf_16000vtpDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'force'
forcePWF = GetOpacityTransferFunction('force')

# create a new 'Calculator'
calculator1 = Calculator(Input=fsurf_16000vtp)
calculator1.Function = ''

# set active source
SetActiveSource(fsurf_16000vtp)

# destroy calculator1
Delete(calculator1)
del calculator1

# create a new 'Generate Surface Normals'
generateSurfaceNormals1 = GenerateSurfaceNormals(Input=fsurf_16000vtp)

# show data in view
generateSurfaceNormals1Display = Show(generateSurfaceNormals1, renderView1)

# trace defaults for the display properties.
generateSurfaceNormals1Display.Representation = 'Surface'
generateSurfaceNormals1Display.ColorArrayName = ['CELLS', 'force']
generateSurfaceNormals1Display.LookupTable = forceLUT
generateSurfaceNormals1Display.OSPRayScaleArray = 'Normals'
generateSurfaceNormals1Display.OSPRayScaleFunction = 'PiecewiseFunction'
generateSurfaceNormals1Display.SelectOrientationVectors = 'None'
generateSurfaceNormals1Display.ScaleFactor = 1.6
generateSurfaceNormals1Display.SelectScaleArray = 'force'
generateSurfaceNormals1Display.GlyphType = 'Arrow'
generateSurfaceNormals1Display.GlyphTableIndexArray = 'force'
generateSurfaceNormals1Display.GaussianRadius = 0.08
generateSurfaceNormals1Display.SetScaleArray = ['POINTS', 'Normals']
generateSurfaceNormals1Display.ScaleTransferFunction = 'PiecewiseFunction'
generateSurfaceNormals1Display.OpacityArray = ['POINTS', 'Normals']
generateSurfaceNormals1Display.OpacityTransferFunction = 'PiecewiseFunction'
generateSurfaceNormals1Display.DataAxesGrid = 'GridAxesRepresentation'
generateSurfaceNormals1Display.SelectionCellLabelFontFile = ''
generateSurfaceNormals1Display.SelectionPointLabelFontFile = ''
generateSurfaceNormals1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
generateSurfaceNormals1Display.DataAxesGrid.XTitleFontFile = ''
generateSurfaceNormals1Display.DataAxesGrid.YTitleFontFile = ''
generateSurfaceNormals1Display.DataAxesGrid.ZTitleFontFile = ''
generateSurfaceNormals1Display.DataAxesGrid.XLabelFontFile = ''
generateSurfaceNormals1Display.DataAxesGrid.YLabelFontFile = ''
generateSurfaceNormals1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
generateSurfaceNormals1Display.PolarAxes.PolarAxisTitleFontFile = ''
generateSurfaceNormals1Display.PolarAxes.PolarAxisLabelFontFile = ''
generateSurfaceNormals1Display.PolarAxes.LastRadialAxisTextFontFile = ''
generateSurfaceNormals1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(fsurf_16000vtp, renderView1)

# show color bar/color legend
generateSurfaceNormals1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(Input=generateSurfaceNormals1)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.Function = ''

# show data in view
calculator1Display = Show(calculator1, renderView1)

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = ['CELLS', 'force']
calculator1Display.LookupTable = forceLUT
calculator1Display.OSPRayScaleArray = 'Normals'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'None'
calculator1Display.ScaleFactor = 1.6
calculator1Display.SelectScaleArray = 'force'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'force'
calculator1Display.GaussianRadius = 0.08
calculator1Display.SetScaleArray = ['POINTS', 'Normals']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = ['POINTS', 'Normals']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.SelectionCellLabelFontFile = ''
calculator1Display.SelectionPointLabelFontFile = ''
calculator1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
calculator1Display.DataAxesGrid.XTitleFontFile = ''
calculator1Display.DataAxesGrid.YTitleFontFile = ''
calculator1Display.DataAxesGrid.ZTitleFontFile = ''
calculator1Display.DataAxesGrid.XLabelFontFile = ''
calculator1Display.DataAxesGrid.YLabelFontFile = ''
calculator1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
calculator1Display.PolarAxes.PolarAxisTitleFontFile = ''
calculator1Display.PolarAxes.PolarAxisLabelFontFile = ''
calculator1Display.PolarAxes.LastRadialAxisTextFontFile = ''
calculator1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(generateSurfaceNormals1, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(generateSurfaceNormals1)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(generateSurfaceNormals1)

# Properties modified on generateSurfaceNormals1
generateSurfaceNormals1.ComputeCellNormals = 1

# Properties modified on calculator1
calculator1.AttributeType = 'Cell Data'
calculator1.ResultArrayName = 'F_normal'

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(calculator1)

# Properties modified on calculator1
calculator1.Function = 'force.Normals'

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator1Display, ('CELLS', 'F_normal'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(forceLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
calculator1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'F_normal'
f_normalLUT = GetColorTransferFunction('F_normal')

# get opacity transfer function/opacity map for 'F_normal'
f_normalPWF = GetOpacityTransferFunction('F_normal')

# create a new 'Calculator'
calculator2 = Calculator(Input=calculator1)
calculator2.Function = ''

# Properties modified on calculator2
calculator2.AttributeType = 'Cell Data'
calculator2.ResultArrayName = 'F_t'

# Properties modified on calculator2
calculator2.Function = 'force - F_normal*Normals'

# show data in view
calculator2Display = Show(calculator2, renderView1)

# trace defaults for the display properties.
calculator2Display.Representation = 'Surface'
calculator2Display.ColorArrayName = ['CELLS', 'F_normal']
calculator2Display.LookupTable = f_normalLUT
calculator2Display.OSPRayScaleArray = 'Normals'
calculator2Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator2Display.SelectOrientationVectors = 'None'
calculator2Display.ScaleFactor = 1.6
calculator2Display.SelectScaleArray = 'F_normal'
calculator2Display.GlyphType = 'Arrow'
calculator2Display.GlyphTableIndexArray = 'F_normal'
calculator2Display.GaussianRadius = 0.08
calculator2Display.SetScaleArray = ['POINTS', 'Normals']
calculator2Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator2Display.OpacityArray = ['POINTS', 'Normals']
calculator2Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator2Display.DataAxesGrid = 'GridAxesRepresentation'
calculator2Display.SelectionCellLabelFontFile = ''
calculator2Display.SelectionPointLabelFontFile = ''
calculator2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
calculator2Display.DataAxesGrid.XTitleFontFile = ''
calculator2Display.DataAxesGrid.YTitleFontFile = ''
calculator2Display.DataAxesGrid.ZTitleFontFile = ''
calculator2Display.DataAxesGrid.XLabelFontFile = ''
calculator2Display.DataAxesGrid.YLabelFontFile = ''
calculator2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
calculator2Display.PolarAxes.PolarAxisTitleFontFile = ''
calculator2Display.PolarAxes.PolarAxisLabelFontFile = ''
calculator2Display.PolarAxes.LastRadialAxisTextFontFile = ''
calculator2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(calculator1, renderView1)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()


# update the view to ensure updated data information
renderView1.Update()

# create a new 'Glyph'
glyph1 = Glyph(Input=calculator2,
    GlyphType='Arrow')
glyph1.OrientationArray = ['CELLS', 'F_t']
glyph1.ScaleArray = ['CELLS', 'F_normal']
glyph1.ScaleFactor = 1.6
glyph1.GlyphTransform = 'Transform2'

# Properties modified on glyph1
glyph1.ScaleArray = ['CELLS', 'F_t']
glyph1.ScaleFactor = 5.0

# show data in view
glyph1Display = Show(glyph1, renderView1)

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = ['POINTS', 'F_normal']
glyph1Display.LookupTable = f_normalLUT
glyph1Display.OSPRayScaleArray = 'F_normal'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'F_normal'
glyph1Display.ScaleFactor = 1.5973724365234376
glyph1Display.SelectScaleArray = 'F_normal'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'F_normal'
glyph1Display.GaussianRadius = 0.07986862182617188
glyph1Display.SetScaleArray = ['POINTS', 'F_normal']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = ['POINTS', 'F_normal']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.SelectionCellLabelFontFile = ''
glyph1Display.SelectionPointLabelFontFile = ''
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
glyph1Display.DataAxesGrid.XTitleFontFile = ''
glyph1Display.DataAxesGrid.YTitleFontFile = ''
glyph1Display.DataAxesGrid.ZTitleFontFile = ''
glyph1Display.DataAxesGrid.XLabelFontFile = ''
glyph1Display.DataAxesGrid.YLabelFontFile = ''
glyph1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
glyph1Display.PolarAxes.PolarAxisTitleFontFile = ''
glyph1Display.PolarAxes.PolarAxisLabelFontFile = ''
glyph1Display.PolarAxes.LastRadialAxisTextFontFile = ''
glyph1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# turn off scalar coloring
ColorBy(glyph1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(f_normalLUT, renderView1)

# set active source
SetActiveSource(calculator2)

# set active source
SetActiveSource(glyph1)

# Properties modified on glyph1
glyph1.ScaleFactor = 10.0

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 50.0

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.MaximumNumberOfSamplePoints = 20000

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on glyph1
glyph1.ScaleFactor = 100.0

# update the view to ensure updated data information
renderView1.Update()

# reset view to fit data
renderView1.ResetCamera()

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [64.0, 80.49191284179688, 135.2137252739355]
renderView1.CameraFocalPoint = [64.0, 80.49191284179688, 80.49254608154297]
renderView1.CameraParallelScale = 14.16288334545896

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
