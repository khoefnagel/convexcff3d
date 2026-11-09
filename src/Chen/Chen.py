from scipy import sparse
import numpy as np

from Concave.Concave import Concave2D, Concave3D
from CFF import CFF2D, CFF3D

from Bouzidi.Bouzidi_core import Bouzidi_core

class Chen2D(CFF2D):
    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='PC', deg=0,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        assert deg == 0, "Chen2D only supports deg=0 (piecewise constant)."
        assert projection == 'PC', "Chen2D only supports projection='PC' (piecewise constant)."
        self.Chen = True
        CFF2D.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)

class Chen3D(CFF3D):
    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='PC', deg=0,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        self.Chen = True
        assert deg == 0, "Chen3D only supports deg=0 (piecewise constant)."
        assert projection == 'PC', "Chen3D only supports projection='PC' (piecewise constant)."
        CFF3D.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)
