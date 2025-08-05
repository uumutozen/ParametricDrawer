import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as LA
from matplotlib.animation import FuncAnimation


# G0/G1 [x,y,z,e,f]
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


# Total route length
route_lenght = 0
for k in range(len(coords_list) - 1):
    p1 = np.array(coords_list[k])
    p2 = np.array(coords_list[k + 1])
    route_lenght = route_lenght + LA.norm(p2 - p1)
print(route_lenght)


x_coords = []
y_coords = []
z_coords = []
e_coords = []
i_coords = []
j_coords = []

x_coords = [point[1] for point in gcode_list]
y_coords = [point[2] for point in gcode_list]
z_coords = [point[3] for point in gcode_list]

# Creating figure
fig = plt.figure(figsize=(100, 100))
ax = fig.add_subplot(111, projection="3d")

# Creating plot
ax.scatter3D(x_coords, y_coords, z_coords, color="red", alpha=0.25)
ax.plot(x_coords[0:], y_coords[0:], z_coords[0:], color="blue", alpha=0.5)
ax.plot(x_coords[1:], y_coords[1:], z_coords[1:], color="blue", alpha=0.5)

# Insert Codes for Formulating the Circular Arc
g2_g3_listeleri = [liste for liste in gcode_list if liste[0] in ['G2', 'G3']]
'''
def calculate_g2_g3(coords_list):
    results = []
    current_position = [0.0, 0.0, 2.0]  # Başlangıç konumu (X, Y, Z)
    
    for g2_g3_listeleri in coords_list:
        cmd_type =  g2_g3_listeleri[0]
        X, Y, Z, E, F, I, J = g2_g3_listeleri[0][1:]

        if cmd_type == 'G2' or cmd_type == 'G3':
            # Merkez koordinatları
            center_x = current_position[0] + I
            center_y = current_position[1] + J
            
            # Çember yayı için başlangıç ve bitiş açısı
            start_angle = math.atan2(current_position[1] - center_y, current_position[0] - center_x)
            end_angle = math.atan2(Y - center_y, X - center_x)
            
            # Açı farkı hesaplanır (G2 için saat yönünde, G3 için saat yönünün tersinde)
            if cmd_type == 'G2':  # Saat yönünde
                if end_angle > start_angle:
                    end_angle -= 2 * math.pi
                angle_difference = start_angle - end_angle
            else:  # G3 - Saat yönünün tersinde
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                angle_difference = end_angle - start_angle
            
            # Çemberin yarıçapı
            radius = math.sqrt(I**2 + J**2)
            
            # Yay uzunluğunu hesapla
            arc_length = radius * angle_difference
            
            # Yay noktalarını hesapla
            num_points = 100
            angles = np.linspace(start_angle, end_angle, num_points)
            arc_x = center_x + radius * np.cos(angles)
            arc_y = center_y + radius * np.sin(angles)
            arc_z = np.linspace(current_position[2], Z, num_points)
            
            # Sonuçları sakla
            results.append({
                'g2_g3_listeleri': cmd_type,
                'start_position': current_position[:],
                'end_position': [X, Y, Z],
                'center': [center_x, center_y],
                'radius': radius,
                'angle_difference': math.degrees(angle_difference),
                'arc_length': arc_length,
                'feedrate': F,
                'arc_x': arc_x,
                'arc_y': arc_y,
                'arc_z': arc_z
            })
            
            # Geçerli konumu güncelle
            current_position = [X, Y, Z]
        else:
            # Diğer komutlar için pozisyonu güncelle
            current_position = [X, Y, Z]
    
    return results
results = calculate_g2_g3(coords_list)

for result in results:
    if result['gcode'] in ['G2', 'G3']:
        ax.plot(result['arc_x'], result['arc_y'], result['arc_z'], label=f"{result['g2_g3_listeleri']} - Radius: {result['radius']:.2f}")

results = calculate_g2_g3(coords_list)

'''
# Insert Codes for Formulating the Circular Arc
X, Y, Z, E, F, I, J = g2_g3_listeleri[0][1:]
center_x = x_current + I
center_y = y_current + J
start_angle = math.atan2(y_current - center_y,  x_current - center_x)
end_angle = math.atan2(Y - center_y, X - center_x)

if gcode_code == "G2":  # Saat yönünde
    if end_angle > start_angle:
        end_angle -= 2 * math.pi
    angle_difference = start_angle - end_angle
else:  # G3 - Saat yönünün tersinde
    if end_angle < start_angle:
        end_angle += 2 * math.pi
    angle_difference = end_angle - start_angle

# Çemberin yarıçapı
radius = math.sqrt(I**2 + J**2)

# Yay uzunluğunu hesapla
arc_length = radius * angle_difference

# Yay noktalarını hesapla
num_points = 100
angles = np.linspace(start_angle, end_angle, num_points)
arc_x = center_x + radius * np.cos(angles)
arc_y = center_y + radius * np.sin(angles)
arc_z = np.linspace(z_current, Z, num_points)



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

print(g2_g3_listeleri)


# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=len(coords_list), interval=100, blit=False)
# Grafiği göster
ax.legend()
plt.show()
