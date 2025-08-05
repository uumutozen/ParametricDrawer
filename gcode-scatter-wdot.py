import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def take_coordinate(coordinates, coordinate_name, coordinate_previous):
    gcode_u_init = coordinates.find(coordinate_name)
    if gcode_u_init == -1:
        return coordinate_previous
    else:
        gcode_u_end = coordinates.find(" ", gcode_u_init + 1)
        if gcode_u_end != -1:
            gcode_u = coordinates[gcode_u_init + 1:gcode_u_end]
        else:
            gcode_u = coordinates[gcode_u_init + 1:]
        return gcode_u


gcodefile = open("sample.gcode", "r+")
gcodes = ""
# file1.seek(0)
gcodes = gcodefile.readlines()
gcodefile.close()
gcodes = [gcode.replace("\n", "") for gcode in gcodes]

x_previous = 0.0
y_previous = 0.0
z_previous = 2.0
e_previous = 0.0
x_coords = []
y_coords = []
z_coords = []
e_coords = []

for gcode in gcodes:
    gcode_init = gcode.find(" ")
    gcode_code = gcode[0:gcode_init]
    if (gcode_code == "G0" or gcode_code == "G1"):
        gcode_end = gcode.find(";")
        if gcode_end == -1:
            gcode_coordinates = gcode[gcode_init + 1:]
        else:
            gcode_coordinates = gcode[gcode_init + 1:gcode_end]
        x_current = take_coordinate(gcode_coordinates, "X", x_previous)
        y_current = take_coordinate(gcode_coordinates, "Y", y_previous)
        z_current = take_coordinate(gcode_coordinates, "Z", z_previous)
        e_current = take_coordinate(gcode_coordinates, "E", e_previous)
        print(x_current, y_current, z_current, e_current)
        x_coords.append(float(x_current))
        y_coords.append(float(y_current))
        z_coords.append(float(z_current))
        e_coords.append(float(e_current))
        x_previous = x_current
        y_previous = y_current
        z_previous = z_current
        e_previous = e_current
    elif gcode_code == "G92":
        print("Insert Codes Here")

# Creating figure
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')


# Creating plot
ax.scatter3D(x_coords, y_coords, z_coords, color="red")
ax.plot(x_coords, y_coords, z_coords, color="blue")
moving_point, = ax.plot([], [], [], "o", markersize=8, color="chartreuse")
plt.title("3D GCode Router")



ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
ax.set_yticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
ax.set_zticks([0, 1, 2, 3, 4, 5])

ax.axes.set_xlim3d(left=0.0, right=10.0)
ax.axes.set_ylim3d(bottom=0.0, top=10.0)
ax.axes.set_zlim3d(bottom=0.0, top=3.0)

ax.view_init(90, 270)


# u = np.sin(np.pi * x_coords) * np.cos(np.pi * y_coords) * np.cos(np.pi * z_coords)
# v = -np.cos(np.pi * x_coords) * np.sin(np.pi * y_coords) * np.cos(np.pi * z_coords)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x_coords) * np.cos(np.pi * y_coords) * np.sin(np.pi * z_coords))

for k in range(len(x_coords) - 1):
    ax.quiver(x_coords[k], y_coords[k], z_coords[k], x_coords[k+1]-x_coords[k], y_coords[k+1]-y_coords[k], z_coords[k+1]-z_coords[k], length=0.25, normalize=True, color='black')

def interpolate_points(x, y, z, steps=100):
    # İki nokta arasındaki tüm interpolasyon noktalarını bul
    x_interp = np.linspace(x[0], x[1], steps)
    y_interp = np.linspace(y[0], y[1], steps)
    z_interp = np.linspace(z[0], z[1], steps)
    return x_interp, y_interp, z_interp

# Tüm scatter noktaları arasında interpolasyon yaparak adımları hazırlıyoruz 
x_interp_all = []
y_interp_all = []
z_interp_all = []

for i in range(len(x_coords) - 1):
    x_interp, y_interp, z_interp = interpolate_points([x_coords[i], x_coords[i+1]],
                                                      [y_coords[i], y_coords[i+1]],
                                                      [z_coords[i], z_coords[i+1]], steps=50)
    x_interp_all.extend(x_interp)
    y_interp_all.extend(y_interp)
    z_interp_all.extend(z_interp)
    
# Güncelleme fonksiyonu (her karede nokta bir sonraki konuma gider)
def update(num):
    # Hareket eden noktanın çizgi üzerindeki konumunu günceller
    moving_point.set_data(x_interp_all[num:num+1], y_interp_all[num:num+1])
    moving_point.set_3d_properties(z_interp_all[num:num+1])
    return moving_point,

# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=len(x_interp_all), interval=0.1, blit=False)

# Grafiği göster
plt.show()

print(x_coords)

