from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 3D noktaları tanımlama
points = np.array([
    [0, 0, 0], 
    [1, 2, 1], 
    [2, 3, 4], 
    [4, 1, 2], 
    [5, 0, 5]
])
t = np.arange(len(points))  # Zaman parametresi
x, y, z = points[:, 0], points[:, 1], points[:, 2]

# Spline ile interpolasyon
t_interp = np.linspace(t[0], t[-1], 500)
spline_x = CubicSpline(t, x)(t_interp)
spline_y = CubicSpline(t, y)(t_interp)
spline_z = CubicSpline(t, z)(t_interp)

# Şekil ve eksen ayarı
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, "o-", label="Control Points")  # Kontrol noktaları
ax.plot(spline_x, spline_y, spline_z, label="Cubic Spline")  # Spline eğrisi
point, = ax.plot([], [], [], "ro", markersize=8)  # Hareketli nokta
ax.legend()
ax.set_xlim(min(x)-1, max(x)+1)
ax.set_ylim(min(y)-1, max(y)+1)
ax.set_zlim(min(z)-1, max(z)+1)

# Güncelleme fonksiyonu
def update(frame):
    point.set_data(spline_x[frame], spline_y[frame])  # 2D güncelleme
    point.set_3d_properties(spline_z[frame])  # 3D güncelleme
    return point,

# Animasyon
ani = FuncAnimation(fig, update, frames=len(t_interp), interval=20, blit=True)
plt.show()