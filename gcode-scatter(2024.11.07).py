import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.animation import FuncAnimation


# G0/G1 [x,y,z,e,f,i,j]
# [[x1,y1,z1,e1,f1],[x2,y2,z2,e2,f2],...,[xn,yn,zn,en,fn]]

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


gcodefile = open("3d-with-g02.gcode", "r+")
gcodes = ""
# file1.seek(0)
gcodes = gcodefile.readlines()
gcodefile.close()
gcodes = [gcode.replace("\n", "") for gcode in gcodes]

# Initial Values
# [G0,X,Y,Z,E,F,I,J]
gcode_list = [["G1", 0.0, 0.0, 2.0, 0.0, 5000.0,0.0,0.0]]
coords_list = [[0.0, 0.0, 2.0, 0.0, 5000.0,0.0,0.0]]

for gcode in gcodes:
    gcode_init = gcode.find(" ")
    gcode_code = gcode[0:gcode_init]
    gcode_end = gcode.find(";")
    if gcode_end == -1:
        gcode_coordinates = gcode[gcode_init + 1:]
    else:
        gcode_coordinates = gcode[gcode_init + 1:gcode_end]
    if (gcode_code == "G0" or gcode_code == "G1"):
        x_current = take_coordinate(gcode_coordinates, "X", coords_list[-1][0])
        y_current = take_coordinate(gcode_coordinates, "Y", coords_list[-1][1])
        z_current = take_coordinate(gcode_coordinates, "Z", coords_list[-1][2])
        e_current = take_coordinate(gcode_coordinates, "E", coords_list[-1][3])
        f_current = take_coordinate(gcode_coordinates, "F", coords_list[-1][4])
        coord_current = [x_current, y_current, z_current, e_current, f_current, 0.0, 0.0]
        coords_list.append(coord_current)
        gcode_current = [gcode_code] + coord_current
        gcode_list.append(gcode_current)
    elif (gcode_code == "G2" or gcode_code == "G3"):
        print("Insert Codes Here (CW/CCW)")
        x_current = take_coordinate(gcode_coordinates, "X", coords_list[-1][0])
        y_current = take_coordinate(gcode_coordinates, "Y", coords_list[-1][1])
        z_current = take_coordinate(gcode_coordinates, "Z", coords_list[-1][2])
        e_current = take_coordinate(gcode_coordinates, "E", coords_list[-1][3])
        f_current = take_coordinate(gcode_coordinates, "F", coords_list[-1][4])
        i_current = take_coordinate(gcode_coordinates, "I", coords_list[-1][5])
        j_current = take_coordinate(gcode_coordinates, "J", coords_list[-1][6])
        coord_current = [x_current, y_current, z_current, e_current, f_current, i_current, j_current]
        coords_list.append(coord_current)
        gcode_current = [gcode_code] + coord_current
        gcode_list.append(gcode_current)
    elif gcode_code == "G92":
        # print("Insert Codes Here")
        x_offset = coord_current[0] - take_coordinate(gcode_coordinates, "X", coord_current[0])
        y_offset = coord_current[1] - take_coordinate(gcode_coordinates, "Y", coord_current[1])
        z_offset = coord_current[2] - take_coordinate(gcode_coordinates, "Z", coord_current[2])
        e_offset = coord_current[3] - take_coordinate(gcode_coordinates, "E", coord_current[3])
        f_offset = coord_current[4] - take_coordinate(gcode_coordinates, "F", coord_current[4])
        gcode_current = ["G92", x_offset, y_offset, z_offset, e_offset, f_offset, 0.0, 0.0]
        gcode_list.append(gcode_current)

# Remove G92, and transform all coordinates into absolute form
G92_count = sum(1 for gcode in gcode_list if gcode[0] == "G92")
for l in range(0, G92_count - 1):
    for k in range(len(gcode_list)-1, 0, -1):
        if gcode_list[k][0] == "G92":
            for m in range(k + 1, len(gcode_list), 1):
                gcode_list[m] = [gcode_list[m][0], gcode_list[m][1] + gcode_list[k][1],
                                 gcode_list[m][2] + gcode_list[k][2], gcode_list[m][3] + gcode_list[k][3],
                                 gcode_list[m][4] + gcode_list[k][4], gcode_list[m][5] + gcode_list[k][5]]
            del gcode_list[k]

for k in range(0, len(gcode_list) , 1):
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

# Creating plot
ax.scatter3D(x_coords, y_coords, z_coords, color="red", alpha=0.25)
for k in range(1,len(gcode_list)):
    if gcode_list[k][0]=="G1":
        ax.plot([gcode_list[k-1][1],gcode_list[k][1]], [gcode_list[k-1][2],gcode_list[k][2]], [gcode_list[k-1][3],gcode_list[k][3]], color="blue", alpha=0.5)
    elif gcode_list[k][0]=="G2":
        print("G2")
        r = (gcode_list[k][6]**2 + gcode_list[k][7]**2)**0.5
        cx = gcode_list[k-1][1]+gcode_list[k][6]
        cy = gcode_list[k-1][2]+gcode_list[k][7]
        ax.scatter3D([cx], [cy], [(gcode_list[k-1][3]+gcode_list[k][3])/2], color="gray", alpha=0.25)
        a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
        b = np.degrees(np.arctan2((gcode_list[k][2]-cy), (gcode_list[k][1]-cx)))
        if b > a:
            b -= 2 * 180
        theta = np.linspace(np.radians(a), np.radians(b), 360)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        z = np.linspace(gcode_list[k-1][3], gcode_list[k][3], 360)
        ax.plot(x, y, z, color="blue", alpha=0.5)
    elif gcode_list[k][0]=="G3":
        print("G3")
        r = (gcode_list[k][6]**2 + gcode_list[k][7]**2)**0.5
        cx = gcode_list[k-1][1]+gcode_list[k][6]
        cy = gcode_list[k-1][2]+gcode_list[k][7]
        ax.scatter3D([cx], [cy], [(gcode_list[k-1][3]+gcode_list[k][3])/2], color="gray", alpha=0.25)
        a = np.degrees(np.arctan2(-gcode_list[k][7], -gcode_list[k][6]))
        b = np.degrees(np.arctan2((gcode_list[k][2]-cy), (gcode_list[k][1]-cx)))
        if b < a:
            b += 2 * 180
        theta = np.linspace(np.radians(a), np.radians(b), 360)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        z = np.linspace(gcode_list[k-1][3], gcode_list[k][3], 360)
        ax.plot(x, y, z, color="blue", alpha=0.5)

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
for k in range(len(x_coords) - 1):
    ax.quiver(x_coords[k], y_coords[k], z_coords[k], x_coords[k + 1] - x_coords[k], y_coords[k + 1] - y_coords[k],
              z_coords[k + 1] - z_coords[k], length=0.25, normalize=True, color='black')

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
