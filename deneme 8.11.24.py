import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def calculate_g2_g3(commands):
    results = []
    current_position = [0.0, 0.0, 0.0]  # Başlangıç konumu (X, Y, Z)
    
    for command in commands:
        cmd_type = command[0]
        X, Y, Z, E, F, I, J = command[1:]

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
                'command': cmd_type,
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

# Komut dizisi örneği
commands = [
    ['G1', 0.0, 0.0, 2.0, 0.0, 5000.0, 0.0, 0.0],
    ['G3', 45.0, 15.0, 10.0, 0.0, 3000.0, 20.0, 15.0],
    ['G2', 10.0, 20.0, 20.0, 0.0, 3000.0, -15.0, 20.0]
]

# Sonuçları hesapla
results = calculate_g2_g3(commands)

# 3D Grafik oluşturma
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('G2 ve G3 Komutları ile Dairesel Hareket')

# G2 ve G3 yaylarını çizme
for result in results:
    if result['command'] in ['G2', 'G3']:
        ax.plot(result['arc_x'], result['arc_y'], result['arc_z'])

# Başlangıç ve bitiş noktalarını işaretle
for command in commands:
    cmd_type, X, Y, Z, *_ = command
    if cmd_type == 'G1':
        ax.scatter(X, Y, Z, color='blue', s=20)
    elif cmd_type == 'G2':
        ax.scatter(X, Y, Z, color='red', s=20)
    elif cmd_type == 'G3':
        ax.scatter(X, Y, Z, color='green', s=20)


ax.legend()
plt.show()
