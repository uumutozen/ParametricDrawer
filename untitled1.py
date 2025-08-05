import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# G-code komutları
gcode_commands = [
    ['G1', 0.0, 0.0, 2.0, 0.0, 5000.0, 0.0, 0.0],
    ['G1', 0.0, 0.0, 2.0, 0.0, 3000.0, 0.0, 0.0],
    ['G1', 10.0, 20.0, 2.0, 0.0, 3000.0, 0.0, 0.0],
    ['G3', 45.0, 15.0, 10.0, 0.0, 3000.0, 20.0, 15.0],
    ['G2', 10.0, 20.0, 20.0, 0.0, 3000.0, -15.0, 20.0],
    ['G3', 45.0, 15.0, 20.0, 0.0, 3000.0, 20.0, 15.0]
]

# Başlangıç noktası
current_position = np.array([0.0, 0.0, 0.0])
positions = [current_position.copy()]

# Komutları işleyerek pozisyonları hesapla
for command in gcode_commands:
    g_code = command[0]
    x = command[1]
    y = command[2]
    z = command[3]

    if g_code == 'G1':
        # Doğrusal hareket
        current_position = np.array([x, y, z])
        positions.append(current_position.copy())
    elif g_code in ['G2', 'G3']:
        # Çembersel hareket
        center = np.array([command[5], command[6], command[3]])
        radius = np.linalg.norm(current_position[:2] - center[:2])
        angle_start = np.arctan2(current_position[1] - center[1], current_position[0] - center[0])
        angle_end = np.arctan2(y - center[1], x - center[0])

        if g_code == 'G2':  # Saat yönünde
            angle_end = angle_start + (angle_end - angle_start)
            angles = np.linspace(angle_start, angle_end, 100)
        else:  # Ters yönde
            angle_end = angle_start - (angle_end - angle_start)
            angles = np.linspace(angle_start, angle_end, 100)

        arc_x = center[0] + radius * np.cos(angles)
        arc_y = center[1] + radius * np.sin(angles)
        arc_z = np.full(arc_x.shape, z)

        for i in range(len(arc_x)):
            positions.append(np.array([arc_x[i], arc_y[i], arc_z[i]]))

# 3D görselleştirme
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Eksenleri ayarlama
ax.set_xlim([-100, 150])
ax.set_ylim([-100, 250])
ax.set_zlim([0, 250])
ax.set_xlabel('X ekseni', fontsize=12)
ax.set_ylabel('Y ekseni', fontsize=12)
ax.set_zlabel('Z ekseni', fontsize=12)
ax.set_title('3D G-code Animation', fontsize=14)

# Eksen çizgilerini kalınlaştırma
ax.xaxis.line.set_linewidth(2)
ax.yaxis.line.set_linewidth(2)
ax.zaxis.line.set_linewidth(2)

# Pozisyonları çizecek fonksiyon
def update(frame):
    ax.clear()
    ax.set_title('3D G-code Animation', fontsize=14)
    ax.set_xlabel('X ekseni', fontsize=12)
    ax.set_ylabel('Y ekseni', fontsize=12)
    ax.set_zlabel('Z ekseni', fontsize=12)

    # Tüm pozisyonları çiz
    x_data = [pos[0] for pos in positions[:frame+1]]
    y_data = [pos[1] for pos in positions[:frame+1]]
    z_data = [pos[2] for pos in positions[:frame+1]]

    ax.plot(x_data, y_data, z_data, marker='o')
    
    # Eksen sınırlarını tekrar ayarla
    ax.set_xlim([-100, 150])
    ax.set_ylim([-100, 250])
    ax.set_zlim([0, 250])

# Animasyonu başlat
ani = FuncAnimation(fig, update, frames=len(positions), interval=100)

plt.show()