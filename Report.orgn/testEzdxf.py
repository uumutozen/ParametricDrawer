import sys
sys.path.append('C:\Program Files\FreeCAD 0.21')  # Update this path

import os
import ezdxf
from FreeCAD import Vector, Draft

def Msg(message):
    print(message)

filePath, fileName = os.path.split(__file__)
drawing = ezdxf.readfile(os.path.join(filePath, "ezdxf_spline.dxf"))

modelspace = drawing.modelspace()
splines = modelspace.query('SPLINE')
Msg('n=%d\n' % len(splines))

points = splines[0].get_fit_points()
points2 = [Vector(i.x, i.y, i.z) for i in points]

line = Draft.makeWire(points2, closed=False, face=False, support=None)
spline = Draft.makeBSpline(points2, closed=False, face=False, support=None)
bez = Draft.makeBezCurve(points2, closed=False, support=None)

Msg('Done!\n\n')

