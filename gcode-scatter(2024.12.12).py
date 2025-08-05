import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# G0/G1 [x,y,z,e,f,i,j,p,q]
# [[x1,y1,z1,e1,f1,p1,q1],[x2,y2,z2,e2,f2,p2,q2],...,[xn,yn,zn,en,fn,pn,qn]]

#To Do
#1. Add Plotting Codes for G5 V
#2. Use Partition Norm to Determine the Number of Segments in a Curve X
#3. Use Arrows for G05 V
#4. Run the Cursor Along the Curve V
#5. Use Blade for Cursor Instead of Red Ball X
def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def rotation_matrix_to_align(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    k_matrix = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_mat = np.eye(3) + k_matrix + np.dot(k_matrix, k_matrix) * ((1 - c) / (s ** 2 if s else 1))
    return rotation_mat

def interpolate_points(x, y, z, steps=100):
    # İki nokta arasındaki tüm interpolasyon noktalarını bul
    x_interp = np.linspace(x[0], x[1], steps)
    y_interp = np.linspace(y[0], y[1], steps)
    z_interp = np.linspace(z[0], z[1], steps)
    return x_interp, y_interp, z_interp

def update(num):
    # Hareket eden noktanın çizgi üzerindeki konumunu günceller
    moving_point.set_data(x_interp_all[num:num+1], y_interp_all[num:num+1])
    moving_point.set_3d_properties(z_interp_all[num:num+1])
def cubic_bezier(t, control_points):
    '''
    Computes a point on a cubic Bézier curve.
    t: Parameter in [0, 1]
    Control Points = [P0, P1, P2, P3]: Control points (3D)
    '''
    #(1-t)**3 * control_points[0] + 3*(1-t)**2 * t * control_points[1] + 3*(1-t) * t**2 * control_points[2] + t**3 * control_points[3]
    bernstein_coeffs = np.array([comb(3, i, exact=True)*(1-t)**(3-i)*t**i for i in range(4)])
    return np.dot(bernstein_coeffs, control_points)

def take_coordinate(coordinates, coordinate_name, coordinate_previous):
    gcode_u_init = coordinates.find(coordinate_name)
    if gcode_u_init == -1:
        return coordinate_previous
    else:
        gcode_u_end = coordinates.find(" ", gcode_u_init + 1)
        if gcode_u_end != -1:
            gcode_u = float(coordinates[gcode_u_init + 1:gcode_u_end])
        else:
            gcode_u = float(coordinates[gcode_u_init + 1:])
        return gcode_u


gcodefile = open("tetrahedron.gcode", "r+")
gcodes = ""
# file1.seek(0)
gcodes = gcodefile.readlines()
gcodefile.close()
gcodes = [gcode.replace("\n", "") for gcode in gcodes]

# Initial Values
# [G*,X,Y,Z,E,F,I,J,P,Q]
gcode_list = [["G1", 0.0, 0.0, 2.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 0.0]]
coords_list = [[0.0, 0.0, 2.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 0.0]]

for gcode in gcodes:
    gcode_init = gcode.find(" ")
    gcode_code = gcode[0:gcode_init]
    gcode_end = gcode.find(";")
    if gcode_end == -1:
        gcode_coordinates = gcode[gcode_init + 1:]
    else:
        gcode_coordinates = gcode[gcode_init + 1:gcode_end]
    match gcode_code:
        case "G0" | "G1":
            x_current = take_coordinate(gcode_coordinates, "X", coords_list[-1][0])
            y_current = take_coordinate(gcode_coordinates, "Y", coords_list[-1][1])
            z_current = take_coordinate(gcode_coordinates, "Z", coords_list[-1][2])
            e_current = take_coordinate(gcode_coordinates, "E", coords_list[-1][3])
            f_current = take_coordinate(gcode_coordinates, "F", coords_list[-1][4])
            coord_current = [x_current, y_current, z_current, e_current, f_current, 0.0, 0.0, 0.0, 0.0]
            coords_list.append(coord_current)
            gcode_current = [gcode_code] + coord_current
            gcode_list.append(gcode_current)
        case "G2" | "G3":
            x_current = take_coordinate(gcode_coordinates, "X", coords_list[-1][0])
            y_current = take_coordinate(gcode_coordinates, "Y", coords_list[-1][1])
            z_current = take_coordinate(gcode_coordinates, "Z", coords_list[-1][2])
            e_current = take_coordinate(gcode_coordinates, "E", coords_list[-1][3])
            f_current = take_coordinate(gcode_coordinates, "F", coords_list[-1][4])
            i_current = take_coordinate(gcode_coordinates, "I", coords_list[-1][5])
            j_current = take_coordinate(gcode_coordinates, "J", coords_list[-1][6])
            coord_current = [x_current, y_current, z_current, e_current, f_current, i_current, j_current, 0.0, 0.0]
            coords_list.append(coord_current)
            gcode_current = [gcode_code] + coord_current
            gcode_list.append(gcode_current)
        case "G5":
            x_current = take_coordinate(gcode_coordinates, "X", coords_list[-1][0])
            y_current = take_coordinate(gcode_coordinates, "Y", coords_list[-1][1])
            z_current = take_coordinate(gcode_coordinates, "Z", coords_list[-1][2])
            e_current = take_coordinate(gcode_coordinates, "E", coords_list[-1][3])
            f_current = take_coordinate(gcode_coordinates, "F", coords_list[-1][4])
            i_current = take_coordinate(gcode_coordinates, "I", coords_list[-1][5])
            j_current = take_coordinate(gcode_coordinates, "J", coords_list[-1][6])
            p_current = take_coordinate(gcode_coordinates, "P", coords_list[-1][7])
            q_current = take_coordinate(gcode_coordinates, "Q", coords_list[-1][8])
            coord_current = [x_current, y_current, z_current, e_current, f_current, i_current, j_current, p_current, q_current]
            coords_list.append(coord_current)
            gcode_current = [gcode_code] + coord_current
            gcode_list.append(gcode_current)
        case "G92":
            x_offset = coord_current[0] - take_coordinate(gcode_coordinates, "X", coord_current[0])
            y_offset = coord_current[1] - take_coordinate(gcode_coordinates, "Y", coord_current[1])
            z_offset = coord_current[2] - take_coordinate(gcode_coordinates, "Z", coord_current[2])
            e_offset = coord_current[3] - take_coordinate(gcode_coordinates, "E", coord_current[3])
            f_offset = coord_current[4] - take_coordinate(gcode_coordinates, "F", coord_current[4])
            gcode_current = ["G92", x_offset, y_offset, z_offset, e_offset, f_offset, 0.0, 0.0, 0.0, 0.0]
            gcode_list.append(gcode_current)

# Remove G92, and transform all coordinates into absolute form
G92_count = sum(1 for gcode in gcode_list if gcode[0] == "G92")
for l in range(0, G92_count - 1):
    for k in range(len(gcode_list) - 1, 0, -1):
        if gcode_list[k][0] == "G92":
            for m in range(k + 1, len(gcode_list), 1):
                gcode_list[m] = [gcode_list[m][0], gcode_list[m][1] + gcode_list[k][1],
                                 gcode_list[m][2] + gcode_list[k][2], gcode_list[m][3] + gcode_list[k][3],
                                 gcode_list[m][4] + gcode_list[k][4], gcode_list[m][5] + gcode_list[k][5]]
            del gcode_list[k]

for k in range(0, len(gcode_list), 1):
    print(gcode_list[k])

x_coords = []
y_coords = []
z_coords = []
e_coords = []

x_coords = [point[1] for point in gcode_list]
y_coords = [point[2] for point in gcode_list]
z_coords = [point[3] for point in gcode_list]
e_coords = [point[4] for point in gcode_list]

x_interp_all = []
y_interp_all = []
z_interp_all = []
e_interp_all = []

# Creating figure
fig = plt.figure(figsize=(100, 100))
ax = fig.add_subplot(111, projection="3d")

route_length = 0
# Creating plot
ax.scatter3D(x_coords, y_coords, z_coords, color="red", alpha=0.25)
for k in range(1, len(gcode_list)):
    match gcode_list[k][0]:
        case "G1":
            ax.plot([gcode_list[k - 1][1], gcode_list[k][1]], [gcode_list[k - 1][2], gcode_list[k][2]],
                    [gcode_list[k - 1][3], gcode_list[k][3]], color="blue", alpha=0.5)
            line_length = 0
            p1 = np.array(gcode_list[k - 1][1:4])
            p2 = np.array(gcode_list[k][1:4])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="blue")
            line_length += np.linalg.norm(p2 - p1)
            route_length += line_length
        case "G2":
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            ax.scatter3D([cx], [cy], [(gcode_list[k - 1][3] + gcode_list[k][3]) / 2], color="gray", alpha=0.25)
            a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
            b = np.degrees(np.arctan2((gcode_list[k][2] - cy), (gcode_list[k][1] - cx)))
            if b > a:
                b = b - 360
            theta = np.linspace(np.radians(a), np.radians(b), 360)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            z = np.linspace(gcode_list[k - 1][3], gcode_list[k][3], 360)
            ax.plot(x, y, z, color="blue", alpha=0.5)
            arc_lengthcw = r * abs(np.radians(b) - np.radians(a))
            route_length += arc_lengthcw
        case "G3":
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            ax.scatter3D([cx], [cy], [(gcode_list[k - 1][3] + gcode_list[k][3]) / 2], color="gray", alpha=0.25)
            a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
            b = np.degrees(np.arctan2((gcode_list[k][2] - cy), (gcode_list[k][1] - cx)))
            if b < a:
                b = b + 360
            theta = np.linspace(np.radians(a), np.radians(b), 360)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            z = np.linspace(gcode_list[k - 1][3], gcode_list[k][3], 360)
            ax.plot(x, y, z, color="blue", alpha=0.5)
            arc_lengthccw = r * abs(np.radians(b) - np.radians(a))
            route_length += arc_lengthccw
        case "G5":
            P0 = np.array(gcode_list[k - 1][1:4])
            P1 = P0 + np.array([gcode_list[k][6], gcode_list[k][7], 0])
            P3 = np.array(gcode_list[k][1:4])
            P2 = P3 + np.array([gcode_list[k][8], gcode_list[k][9], 0])
            cpts = np.array([P0, P1, P2, P3])
            t_values = np.linspace(0, 1, 100)
            ax.scatter3D(cpts[:, 0], cpts[:, 1], cpts[:, 2], color="gray", alpha=0.25)
            curve_points = np.array([cubic_bezier(t, cpts) for t in t_values])
            ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], color="blue", alpha=0.5)
            for i in range(len(curve_points) - 1):
                route_length += np.linalg.norm(curve_points[i + 1] - curve_points[i])
        case _:
            print("Unknown G-Code")

print(route_length)

# Insert Codes for Formulating the Circular Arc
# Write It as a Function
# Insert into the Plo

'''
moving_point, = ax.plot([], [], [], marker="^", color='green')
plt.title("3D GCode Router")
'''


ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 300])
ax.set_yticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 300])
ax.set_zticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60])

ax.axes.set_xlim3d(left=0.0, right=80.0)
ax.axes.set_ylim3d(bottom=0.0, top=80.0)
ax.axes.set_zlim3d(bottom=0.0, top=5.0)

ax.view_init(90, 270)

# u = np.sin(np.pi * x_coords) * np.cos(np.pi * y_coords) * np.cos(np.pi * z_coords)
# v = -np.cos(np.pi * x_coords) * np.sin(np.pi * y_coords) * np.cos(np.pi * z_coords)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x_coords) * np.cos(np.pi * y_coords) * np.sin(np.pi * z_coords))

# Draw direction arrows for the route
for k in range(1, len(gcode_list)):
    match gcode_list[k][0]:
        case "G1":
            ax.quiver(gcode_list[k - 1][1], gcode_list[k - 1][2], gcode_list[k - 1][3],
                      gcode_list[k][1] - gcode_list[k - 1][1], gcode_list[k][2] - gcode_list[k - 1][2],
                      gcode_list[k][3] - gcode_list[k - 1][3], length=5.0, normalize=True, color='black')
        case "G2":
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
            b = np.degrees(np.arctan2((gcode_list[k][2] - cy), (gcode_list[k][1] - cx)))
            if b > a:
                b = b - 360
            ax.quiver(gcode_list[k - 1][1], gcode_list[k - 1][2], gcode_list[k - 1][3],
                      r * np.sin(np.radians(a)), -r * np.cos(np.radians(a)),
                      gcode_list[k][3] - gcode_list[k - 1][3], length=5.0, normalize=True, color='black')
        case "G3":
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
            b = np.degrees(np.arctan2((gcode_list[k][2] - cy), (gcode_list[k][1] - cx)))
            if b < a:
                b = b + 360
            ax.quiver(gcode_list[k - 1][1], gcode_list[k - 1][2], gcode_list[k - 1][3],
                      (-r) * np.sin(np.radians(a)), r * np.cos(np.radians(a)),
                      gcode_list[k][3] - gcode_list[k - 1][3], length=5.0, normalize=True, color='black')

        case "G5":
            cpts = np.array([P0, P1, P2, P3])
            t_values = np.linspace(0, 1, 100)
            bezier_points = np.array([cubic_bezier(t, cpts) for t in t_values])
            t_start = 0
            t_next = 0.01
            start = cubic_bezier(t_start, cpts)
            next_point = cubic_bezier(t_next, cpts)
            direction = next_point - start
            ax.quiver(start[0], start[1], start[2],
                    direction[0], direction[1], direction[2],
                    length=5.0, normalize=True, color='black')

        case _:
            print("Unknown G-Code")

for k in range(1, len(gcode_list)):
    match gcode_list[k][0]:
        case "G1":
            # Doğrusal interpolasyon
            x_interp, y_interp, z_interp,  = interpolate_points(
                [gcode_list[k - 1][1], gcode_list[k][1]],
                [gcode_list[k - 1][2], gcode_list[k][2]],
                [gcode_list[k - 1][3], gcode_list[k][3]],
                steps=10
            )
            x_interp_all.extend(x_interp)
            y_interp_all.extend(y_interp)
            z_interp_all.extend(z_interp)

        case "G2":
            # Saat yönünde dairesel interpolasyon
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            start_angle = np.arctan2(gcode_list[k - 1][2] - cy, gcode_list[k - 1][1] - cx)
            end_angle = np.arctan2(gcode_list[k][2] - cy, gcode_list[k][1] - cx)
            if end_angle > start_angle:
                end_angle -= 2 * np.pi
            angles = np.linspace(start_angle, end_angle, 10)
            x_interp = cx + r * np.cos(angles)
            y_interp = cy + r * np.sin(angles)
            z_interp = np.linspace(gcode_list[k - 1][3], gcode_list[k][3], 10)
            x_interp_all.extend(x_interp)
            y_interp_all.extend(y_interp)
            z_interp_all.extend(z_interp)

        case "G3":
            # Saat yönünün tersinde dairesel interpolasyon
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            start_angle = np.arctan2(gcode_list[k - 1][2] - cy, gcode_list[k - 1][1] - cx)
            end_angle = np.arctan2(gcode_list[k][2] - cy, gcode_list[k][1] - cx)
            if end_angle < start_angle:
                end_angle += 2 * np.pi
            angles = np.linspace(start_angle, end_angle, 10)
            x_interp = cx + r * np.cos(angles)
            y_interp = cy + r * np.sin(angles)
            z_interp = np.linspace(gcode_list[k - 1][3], gcode_list[k][3], 10)
            x_interp_all.extend(x_interp)
            y_interp_all.extend(y_interp)
            z_interp_all.extend(z_interp)

        case "G5":
            # Bézier eğrisi için interpolasyon yap
            t_values = np.linspace(0, 1, 20)
            bezier_points = np.array([cubic_bezier(t, cpts) for t in t_values])

            # G5 noktalarını hareket için güncelle
            x_interp_all.extend(bezier_points[:, 0])
            y_interp_all.extend(bezier_points[:, 1])
            z_interp_all.extend(bezier_points[:, 2])
# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=len(x_interp_all), interval=100, blit=False)
print(e_interp_all)
# Grafiği göster
plt.show()


