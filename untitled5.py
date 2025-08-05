# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 02:20:22 2024

@author: Hope
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tek nokta için x, y, z değerleri
x = 5
y = 7
z = 3

# 3D ekseni oluşturma
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tek noktayı çizme
ax.scatter(x, y, z, color='red', s=100, alpha=1)  # Renk, boyut ve şeffaflık

# Eksen etiketleri
ax.set_xlabel('X Ekseni')
ax.set_ylabel('Y Ekseni')
ax.set_zlabel('Z Ekseni')

# Grafiği gösterme
plt.show()