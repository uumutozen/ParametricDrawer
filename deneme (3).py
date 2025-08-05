import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# G0/G1 [x,y,z,e,f,i,j,p,q]
# [[x1,y1,z1,e1,f1,p1,q1],[x2,y2,z2,e2,f2,p2,q2],...,[xn,yn,zn,en,fn,pn,qn]]

def interpolate_points(x, y, z, steps=100):
    # İki nokta arasındaki tüm interpolasyon noktalarını bul
    x_interp = np.linspace(x[0], x[1], steps)
    y_interp = np.linspace(y[0], y[1], steps)
    z_interp = np.linspace(z[0], z[1], steps)
    return x_interp, y_interp, z_interp

def cubic_bezier(t, control_points):
    '''
    Computes a point on a cubic Bézier curve.
    t: Parameter in [0, 1]
    Control Points = [P0, P1, P2, P3]: Control points (3D)
    '''
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

# G-code dosyasını oku
gcodefile = open("3d-with-g02-g03-g05.gcode", "r+")
gcodes = gcodefile.readlines()
gcodefile.close()
gcodes = [gcode.replace("\n", "") for gcode in gcodes]

# Initial Values
gcode_list = [["G1", 0.0, 0.0, 2.0, 0.0, 5000.0, 0.0, 0.0, 0.0]]
coords_list = [[0.0, 0.0, 2.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 0.0]]

# G-code verilerini analiz et
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

# Başlangıç ve bitiş koordinatları
x_coords = [point[1] for point in gcode_list]
y_coords = [point[2] for point in gcode_list]
z_coords = [point[3] for point in gcode_list]

# 3D grafiği oluştur
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Grafik üzerine G-code noktalarını ekle
ax.scatter3D(x_coords, y_coords, z_coords, color="red", alpha=0.25)

# G-code komutlarına göre yolları çiz
for k in range(1, len(gcode_list)):
    match gcode_list[k][0]:
        case "G1":
            ax.plot([gcode_list[k - 1][1], gcode_list[k][1]], [gcode_list[k - 1][2], gcode_list[k][2]],
                    [gcode_list[k - 1][3], gcode_list[k][3]], color="blue", alpha=0.5)
        case "G2":
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
            b = np.degrees(np.arctan2((gcode_list[k][2] - cy), (gcode_list[k][1] - cx)))
            if b > a:
                b = b - 360
            theta = np.linspace(np.radians(a), np.radians(b), 360)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            z = np.linspace(gcode_list[k - 1][3], gcode_list[k][3], len(x))
            ax.plot(x, y, z, color="blue", alpha=0.5)
        case "G3":
            r = (gcode_list[k][6] ** 2 + gcode_list[k][7] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][6]
            cy = gcode_list[k - 1][2] + gcode_list[k][7]
            a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
            b = np.degrees(np.arctan2((gcode_list[k][2] - cy), (gcode_list[k][1] - cx)))
            if b < a:
                b = b + 360
            theta = np.linspace(np.radians(a), np.radians(b), 360)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            z = np.linspace(gcode_list[k - 1][3], gcode_list[k][3], len(x))
            ax.plot(x, y, z, color="blue", alpha=0.5)
        case "G5":
            p1 = np.array([gcode_list[k - 1][1], gcode_list[k - 1][2], gcode_list[k - 1][3]])
            p2 = np.array([gcode_list[k][1], gcode_list[k][2], gcode_list[k][3]])
            p3 = np.array([gcode_list[k][1] + gcode_list[k][6], gcode_list[k][2] + gcode_list[k][7], gcode_list[k][3]])
            p4 = np.array([gcode_list[k][1] + gcode_list[k][6], gcode_list[k][2] + gcode_list[k][7], gcode_list[k][3]])
            for t in np.linspace(0, 1, 100):
                point = cubic_bezier(t, [p1, p2, p3, p4])
                ax.scatter(point[0], point[1], point[2], color="green", alpha=0.1)

# Grafik başlıkları ve etiketleri
ax.set_title("G-code Path Simulation")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")

plt.show()
