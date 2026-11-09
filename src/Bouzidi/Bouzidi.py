from scipy import sparse
import numpy as np

from Concave.Concave import Concave2D, Concave3D
from CFF import CFF

from Bouzidi.Bouzidi_core import Bouzidi_core

class Bouzidi2D(CFF):
    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='NIS', deg=1,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        CFF.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)
        Concave2D.__init__(self, N, solids, self.Cij, self.degs, self.degsCells, self.deg)
        self.CSM, self.Fi2Ftot = Bouzidi_core(self.solidsUnion, self.Cij, self.N, self.Ibar)
        self.row, self.col, self.data = self.CSM.row, self.CSM.col, self.CSM.data
        return

class Bouzidi3D(CFF):
    def __init__(self, velSetName, N, solids : list, forceDens=None, projection='NIS', deg=1,
                 onlyPgramOverlapped=False, minDeriv=1, concaveCorrectionMethod='approx'):
        CFF.__init__(self, velSetName, N, solids, forceDens=forceDens, projection=projection, deg=deg,
                     onlyPgramOverlapped=onlyPgramOverlapped, minDeriv=minDeriv, concaveCorrectionMethod=concaveCorrectionMethod)
        Concave3D.__init__(self, N, solids, self.Cij, self.degs, self.degsCells, self.deg)
        self.CSM, self.Fi2Ftot = Bouzidi_core(self.solidsUnion, self.Cij, self.N, self.Ibar)
        self.row, self.col, self.data = self.CSM.row, self.CSM.col, self.CSM.data
        return