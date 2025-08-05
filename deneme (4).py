import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def interpolate_points_linear(start, end, steps):
    x_interp = np.linspace(start[0], end[0], steps)
    y_interp = np.linspace(start[1], end[1], steps)
    z_interp = np.linspace(start[2], end[2], steps)
    return x_interp, y_interp, z_interp

def interpolate_points_arc(center, radius, start_angle, end_angle, plane, steps):
    angles = np.linspace(start_angle, end_angle, steps)
    if plane == 'XY':
        x_interp = center[0] + radius * np.cos(angles)
        y_interp = center[1] + radius * np.sin(angles)
        z_interp = np.full_like(x_interp, center[2])
    elif plane == 'XZ':
        x_interp = center[0] + radius * np.cos(angles)
        z_interp = center[2] + radius * np.sin(angles)
        y_interp = np.full_like(x_interp, center[1])
    elif plane == 'YZ':
        y_interp = center[1] + radius * np.cos(angles)
        z_interp = center[2] + radius * np.sin(angles)
        x_interp = np.full_like(y_interp, center[0])
    return x_interp, y_interp, z_interp

def interpolate_points_bezier(control_points, steps):
    t_values = np.linspace(0, 1, steps)
    bezier_points = np.array([cubic_bezier(t, control_points) for t in t_values])
    return bezier_points[:, 0], bezier_points[:, 1], bezier_points[:, 2]

def cubic_bezier(t, control_points):
    bernstein_coeffs = np.array([comb(3, i, exact=True) * (1 - t)**(3 - i) * t**i for i in range(4)])
    return np.dot(bernstein_coeffs, control_points)

def update(frame):
    global current_index, x_path, y_path, z_path
    if frame >= len(x_path):
        return moving_point,
    moving_point.set_data(x_path[frame:frame+1], y_path[frame:frame+1])
    moving_point.set_3d_properties(z_path[frame:frame+1])
    return moving_point,

gcodefile = open("3d-with-g02-g03-g05.gcode", "r+")
gcodes = gcodefile.readlines()
gcodefile.close()
gcodes = [gcode.replace("\n", "") for gcode in gcodes]

coords_list = [[0.0, 0.0, 2.0]]
x_path, y_path, z_path = [], [], []

for gcode in gcodes:
    if gcode.startswith("G0") or gcode.startswith("G1"):
        x = take_coordinate(gcode, "X", coords_list[-1][0])
        y = take_coordinate(gcode, "Y", coords_list[-1][1])
        z = take_coordinate(gcode, "Z", coords_list[-1][2])
        x_interp, y_interp, z_interp = interpolate_points_linear(coords_list[-1], [x, y, z], steps=100)
        x_path.extend(x_interp)
        y_path.extend(y_interp)
        z_path.extend(z_interp)
        coords_list.append([x, y, z])
    elif gcode.startswith("G2") or gcode.startswith("G3"):
        x = take_coordinate(gcode, "X", coords_list[-1][0])
        y = take_coordinate(gcode, "Y", coords_list[-1][1])
        z = take_coordinate(gcode, "Z", coords_list[-1][2])
        i = take_coordinate(gcode, "I", 0)
        j = take_coordinate(gcode, "J", 0)
        center = [coords_list[-1][0] + i, coords_list[-1][1] + j, coords_list[-1][2]]
        radius = np.sqrt(i**2 + j**2)
        start_angle = np.arctan2(coords_list[-1][1] - center[1], coords_list[-1][0] - center[0])
        end_angle = np.arctan2(y - center[1], x - center[0])
        x_interp, y_interp, z_interp = interpolate_points_arc(center, radius, start_angle, end_angle, plane='XY', steps=100)
        x_path.extend(x_interp)
        y_path.extend(y_interp)
        z_path.extend(z_interp)
        coords_list.append([x, y, z])
    elif gcode.startswith("G5"):
        x = take_coordinate(gcode, "X", coords_list[-1][0])
        y = take_coordinate(gcode, "Y", coords_list[-1][1])
        z = take_coordinate(gcode, "Z", coords_list[-1][2])
        p = take_coordinate(gcode, "P", coords_list[-1][0])
        q = take_coordinate(gcode, "Q", coords_list[-1][1])
        control_points = np.array([coords_list[-1], [p, q, coords_list[-1][2]], [x, y, z]])
        x_interp, y_interp, z_interp = interpolate_points_bezier(control_points, steps=100)
        x_path.extend(x_interp)
        y_path.extend(y_interp)
        z_path.extend(z_interp)
        coords_list.append([x, y, z])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_path, y_path, z_path, label='Tool Path')
moving_point, = ax.plot([], [], [], 'ro')
ax.set_xlim([min(x_path), max(x_path)])
ax.set_ylim([min(y_path), max(y_path)])
ax.set_zlim([min(z_path), max(z_path)])

ani = FuncAnimation(fig, update, frames=len(x_path), interval=50, blit=True)
plt.legend()
plt.show()
