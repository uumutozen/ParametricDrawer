import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# G0/G1 [x,y,z,e,f,i,j,p,q]
# [[x1,y1,z1,e1,f1,p1,q1],[x2,y2,z2,e2,f2,p2,q2],...,[xn,yn,zn,en,fn,pn,qn]]

#To Do
#1. Add Plotting Codes for G5
#2. Use Partition Norm to Determine the Number of Segments in a Curve

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


gcodefile = open("g5-code.gcode", "r+")
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
            print("G5 codes here")
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

'''
# Total route length
route_lenght = 0
for k in range(len(coords_list) - 1):
    p1 = np.array(coords_list[k])
    p2 = np.array(coords_list[k + 1])
    route_lenght = route_lenght + LA.norm(p2 - p1)
print(route_lenght)
'''

x_coords = []
y_coords = []
z_coords = []
e_coords = []

x_coords = [point[1] for point in gcode_list]
y_coords = [point[2] for point in gcode_list]
z_coords = [point[3] for point in gcode_list]

# Creating figure
fig = plt.figure(figsize=(100, 100))
ax = fig.add_subplot(111, projection="3d")

def cubic_bezier(t, P0, P1, P2, P3):
    """
    Computes a point on a cubic Bézier curve.
    t: Parameter in [0, 1]
    P0, P1, P2, P3: Control points (2D)
    """
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

# Creating plot
ax.scatter3D(x_coords, y_coords, z_coords, color="red", alpha=0.25)
for k in range(1, len(gcode_list)):
    match gcode_list[k][0]:
        case "G1":
            ax.plot([gcode_list[k - 1][1], gcode_list[k][1]], [gcode_list[k - 1][2], gcode_list[k][2]],
                    [gcode_list[k - 1][3], gcode_list[k][3]], color="blue", alpha=0.5)
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
        case "G5":
            Q_double_prime =
            Q_prime = x_prime = cs_x(t, 1)
            P_prime =
            P_double_prime =
            curvature = np.abs(P_prime * Q_double_prime - Q_prime * P_double_prime) / (P_prime**2 + Q_prime**2)**1.5

            center = np.array([gcode_list[k - 1][1] + gcode_list[k][8], [k - 1][2] + gcode_list[k][9]])
            
            # Define start and end points for the quarter-circle arc
            P0 = np.array([gcode_list[k][9], gcode_list[k][8]])  # Starting point on the circle (1, 0)
            P3 = np.array([gcode_list[k][8], gcode_list[k][9]])  # Ending point on the circle (0, 1)

            # Compute control points for the cubic Bézier approximation
            alpha = (4/3) * (np.sqrt(2) - 1) * r
            P1 = P0 + np.array([0, alpha])  # Control point near P0
            P2 = P3 + np.array([alpha, 0])  # Control point near P3

            # Generate points along the Bézier curve
            t_values = np.linspace(0, 1, 100)
            curve_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_values])
            
            circle = plt.Circle(center, r, color='gray', fill=False, linestyle='dashed', linewidth=5)
            fig, ax = plt.subplots()
            ax.add_artist(circle)
            ax.plot(curve_points[:, gcode_list[k][8]], curve_points[:, gcode_list[k][9]], label='Cubic Bézier Curve', color='blue')
            ax.plot([P0[gcode_list[k][8]], P1[gcode_list[k][8]], P2[gcode_list[k][8]], P3[gcode_list[k][8]]], [P0[gcode_list[k][9]], P1[gcode_list[k][9]], P2[gcode_list[k][9]], P3[gcode_list[k][9]]], 'ro-', label='Control Points')
        
            print("Unknown G-Code")

# Insert Codes for Formulating the Circular Arc
# Write It as a Function
# Insert into the Plot

moving_point, = ax.plot([], [], [], "go", markersize=10)
plt.title("3D GCode Router")

ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 300])
ax.set_yticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 300])
ax.set_zticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60])

ax.axes.set_xlim3d(left=0.0, right=150.0)
ax.axes.set_ylim3d(bottom=0.0, top=150.0)
ax.axes.set_zlim3d(bottom=0.0, top=60.0)

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
            r = (gcode_list[k][8] ** 2 + gcode_list[k][9] ** 2) ** 0.5
            cx = gcode_list[k - 1][1] + gcode_list[k][8]
            cy = gcode_list[k - 1][2] + gcode_list[k][9]
                
            # Define start and end points for the quarter-circle arc
            P0 = np.array([1, 0])  # Starting point on the circle (1, 0)
            P3 = np.array([0, 1])  # Ending point on the circle (0, 1)

            # Compute control points for the cubic Bézier approximation
            alpha = (4/3) * (np.sqrt(2) - 1) * r
            P1 = P0 + np.array([0, alpha])  # Control point near P0
            P2 = P3 + np.array([alpha, 0])  # Control point near P3

            # Generate points along the Bézier curve
            t_values = np.linspace(0, 1, 100)
            curve_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_values])
            


# Güncelleme fonksiyonu (her karede nokta bir sonraki konuma gider)
def update(num):
    # Hareket eden noktanın çizgi üzerindeki konumunu günceller
    moving_point.set_data(x_coords[num:num + 1], y_coords[num:num + 1])
    moving_point.set_3d_properties(z_coords[num:num + 1])
    moving_point.set_color("red")
    return moving_point,



# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=len(coords_list), interval=1000, blit=False)

# Grafiği göster
plt.show()
