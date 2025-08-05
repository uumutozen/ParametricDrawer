# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 01:13:35 2024

@author: Hope
"""

import re
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# G-code dosyasından G5 komutlarını ayıklayan fonksiyon
def extract_g5_points(gcode_file):
    points = []
    with open(gcode_file, 'r') as file:
        for line in file:
            # G5 X... Y... Z... formatındaki veriyi düzenli ifadelerle ayrıştır
            match = re.search(r'G5\s+X([-+]?[0-9]*\.?[0-9]+)\s+Y([-+]?[0-9]*\.?[0-9]+)', line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                points.append((x, y))
    return points

# Spline oluşturma ve çizim
def plot_spline_from_gcode(gcode_file):
    points = extract_g5_points(gcode_file)
    if not points:
        print("G5 komutu bulunamadı.")
        return
    
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Spline için parametre aralığı
    t = np.linspace(0, len(x) - 1, len(x))

    # CubicSpline kullanarak spline oluştur
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    # Daha yüksek çözünürlükte spline çizimi için
    t_fine = np.linspace(0, len(x) - 1, 500)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)

    # Çizim
    plt.figure(figsize=(8, 6))
    plt.plot(x_fine, y_fine, label="Spline", color="blue")
    plt.scatter(x, y, color="red", label="Control Points (G5)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Spline from G-code (G5 Commands)")
    plt.legend()
    plt.grid()
    plt.show()

# Örnek kullanım
gcode_path = "g5-code.gcode."  # G-code dosyanızın yolu
plot_spline_from_gcode(gcode_path)