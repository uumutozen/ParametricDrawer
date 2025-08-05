import ezdxf
import numpy as np
import math
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

z_up = 1.0;
z_down = 0.0;

file_name = "dxf/two-splines"
f360 = ezdxf.readfile(file_name + ".dxf")

msp = f360.modelspace()

def base_angle(angle):
    if angle >= 180:
        new_angle = base_angle(angle - 360)
    elif angle < -180:
        new_angle = base_angle(angle + 360)
    else:
        new_angle = angle
    return new_angle


print("G92 E0;")
print("G1 Z" + str(z_up) + ";")
print("G1 X0 Y0;")
for entity in msp:
    print("G1 X" + str(0) + " Y" + str(0) + "; Spline Started")
    if entity.dxftype() == 'SPLINE':
        spline_entity = entity
        control_points = np.array(spline_entity.control_points)
        knots = np.array(spline_entity.knots)
        degree = spline_entity.dxf.degree
        bspline = BSpline(knots, control_points, degree)
        x_control_points = [point[0] for point in control_points]
        y_control_points = [point[1] for point in control_points]
        num_segments = 100  # Number of segments
        t = np.linspace(knots[degree], knots[-(degree + 1)], num_segments + 1)
        bspline_points = bspline(t)
        x_bspline_points = [point[0] for point in bspline_points]
        y_bspline_points = [point[1] for point in bspline_points]
        print("G1 X" + str(x_bspline_points[0]) + " Y" + str(y_bspline_points[0]) + ";")
        print("G1 Z" + str(z_down) + ";")
        plt.plot(x_bspline_points, y_bspline_points, 'b-')
        plt.plot(x_bspline_points, y_bspline_points, 'bo-', label='BSpline Points', alpha=0.75)
        dbspline = bspline.derivative(nu=1)
        dbspline_points = dbspline(t)
        dbspline_norms = [np.linalg.norm(point) for point in dbspline_points]
        x_dbspline_points = [point[0] / norm for point, norm in zip(dbspline_points, dbspline_norms)]
        y_dbspline_points = [point[1] / norm for point, norm in zip(dbspline_points, dbspline_norms)]
        plt.plot(x_dbspline_points, y_dbspline_points, 'go-', label='DBSpline Points', alpha=0.75)
        bspline_tangent_angle = []
        # Compute angles
        for i in range(len(x_bspline_points)):
            plt.arrow(x_bspline_points[i], y_bspline_points[i], x_dbspline_points[i], y_dbspline_points[i], width=0.10,
                      color='r', alpha=0.5)
            if x_dbspline_points[i] > 0 and y_dbspline_points[i] == 0:
                angle = 0
            elif x_dbspline_points[i] == 0 and y_dbspline_points[i] > 0:
                angle = 90
            elif x_dbspline_points[i] < 0 and y_dbspline_points[i] == 0:
                angle = 180
            elif x_dbspline_points[i] == 0 and y_dbspline_points[i] < 0:
                angle = 270
            elif x_dbspline_points[i] != 0 and y_dbspline_points[i] > 0:
                angle = math.degrees(np.arccos(np.dot([x_dbspline_points[i], y_dbspline_points[i]], [1, 0])))
            elif x_dbspline_points[i] != 0 and y_dbspline_points[i] < 0:
                angle = 360 - math.degrees(np.arccos(np.dot([x_dbspline_points[i], y_dbspline_points[i]], [1, 0])))
            bspline_tangent_angle.append(angle)
        # Arrange angles
        for i in range(len(bspline_tangent_angle)):
            if i == 0:
                if bspline_tangent_angle[i] < -180:
                    bspline_tangent_angle[i] = bspline_tangent_angle[i] + 360
                if bspline_tangent_angle[i] > 180:
                    bspline_tangent_angle[i] = bspline_tangent_angle[i] - 360
            else:
                if bspline_tangent_angle[i] - bspline_tangent_angle[i - 1] < -180:
                    bspline_tangent_angle[i] = bspline_tangent_angle[i] + 360
                if bspline_tangent_angle[i] - bspline_tangent_angle[i - 1] > 180:
                    bspline_tangent_angle[i] = bspline_tangent_angle[i] - 360
        for i in range(len(bspline_tangent_angle)):
            print("G1 X" + str(x_bspline_points[i]) + " Y" + str(y_bspline_points[i]) + " E" + str(
                bspline_tangent_angle[i]) + ";")
        print("G1 Z" + str(z_up) + ";")
        # reset_angle = 0
        print(base_angle(bspline_tangent_angle[-1]))
        print("G1 E" + str(0) + ";")

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-Spline Curve')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
