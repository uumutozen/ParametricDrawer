import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametrik batmanlogo fonksiyonu
def batmanlogo(t):
    pi = np.pi
    if 0<=t<1:
        return [0.6 * t, 4]
    elif 1<=t<2:
        return [0.6 + 0.6 * (t - 1), 4 + 0.94 * (t - 1)]
    elif 2<=t<3:
        return [1.2, 4.9434 - 1.4434 * (t - 2)]
    elif 3<=t<4:
        s = (0.576 + pi) * (t - 3) - pi
        return [1.52 * np.cos(s) + 2.72, 1.52 * np.sin(s) + 3.5]
    elif 4<=t<5:
        return [8 * np.cos(1.83 * (5 - t) - 0.785), 5 * np.sin(1.83 * (5 - t) - 0.785)]
    elif 5<=t<6:
        s = 4.4 + 2.26 * (6 - t)
        return [-12.43 + 5.212 * s - 0.375 * s**2, -21.52 + 7.03 * s - 0.65 * s**2]
    elif 6<=t<7:
        s = 3.25 * (7 - t)
        return [s, -2 - 0.75 * (s-2)**2]
    elif 7<=t<8:
        s = 3.25 * (t - 7)
        return [-s, -2 - 0.75 * (s-2)**2]
    elif 8<=t<9:
        s = 4.4 + 2.26 * (t - 8)
        return [12.43 - 5.212 * s + 0.375 * s**2, -21.52 + 7.03 * s - 0.65 * s**2]
    elif 9<=t<10:
        s = 1.83 * (t - 9) - 0.785
        return [-8 * np.cos(s), 5 * np.sin(s)]
    elif 10<=t<11:
        s = (0.576 + pi) * (11 - t) - pi
        return [-1.52 * np.cos(s) - 2.72, 1.52 * np.sin(s) + 3.5]
    elif 11<=t<12:
        return [-1.2, 4.9434 - 1.4434 * (12 - t)]
    elif 12<=t<13:
        return [-0.6 - 0.6 * (13 - t), 4 + 0.94 * (13 - t)]
    elif 13<=t<14:
        return [-0.6 * (14 - t), 4]
    else:
        return [0, 0]

# t değerleri ve koordinatları hazırla
t_vals = np.linspace(0.001, 13.999, 2000)
coords = np.array([batmanlogo(t) for t in t_vals])
x_vals, y_vals = coords[:, 0], coords[:, 1]

# Grafik oluştur
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_vals, color=(0.368417, 0.506779, 0.709798))
dot, = ax.plot([], [], 'ro', markersize=8)  # kırmızı nokta

ax.set_xlim(-9, 9)
ax.set_ylim(-6, 6)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Batman")
ax.set_aspect('equal')

# Animasyon fonksiyonu
def update(frame):
    t = frame
    x, y = batmanlogo(t)
    dot.set_data([x], [y])
    return dot,

# Animasyon oluştur
ani = FuncAnimation(fig, update, frames=np.linspace(0.001, 13.999, 300),
                    interval=30, blit=True, repeat=True)

plt.show()
