import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Geometri hesap fonksiyonları
def tf(x, y, u, v):
    return np.pi - np.arctan2(v - y, u - x)

def l3f(x, y, z, u, v, w):
    return np.linalg.norm([u - x, v - y, w - z])

def af(l, k, h):
    return np.arccos((k**2 + h**2 - l**2) / (2 * k * h))

def bf(l, k, h):
    return af(k, l, h)

def df(z, w, h):
    return np.arccos(np.abs(w - z) / h)

def f1f(z, w, l, k, h):
    return bf(l, k, h) - df(z, w, h)

def f2f(z, w, l, k, h):
    return af(l, k, h) + df(z, w, h)

# Üçgen uç nokta yörüngesi (3D uzayda çizilecek)
def triangle_coords(t):
    t = np.asarray(t)
    total = 3
    edge = (t % (2 * np.pi)) / (2 * np.pi) * total
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)

    # Kenar 1 (A-B)
    idx = (edge >= 0) & (edge < 1)
    y[idx] = 2.7 * (1 - edge[idx])
    z[idx] = 9 + 2.5 * edge[idx]

    # Kenar 2 (B-C)
    idx = (edge >= 1) & (edge < 2)
    y[idx] = -2.7 + 5.4 * (edge[idx] - 1)
    z[idx] = 11.5 - 5.0 * (edge[idx] - 1)

    # Kenar 3 (C-A)
    idx = (edge >= 2) & (edge < 3)
    y[idx] = 2.7 - 2.7 * (edge[idx] - 2)
    z[idx] = 6.5 + 2.5 * (edge[idx] - 2)

    return x, y, z

# Yörünge fonksiyonları
def x_t(t): return 0
def y_t(t): return triangle_coords(np.atleast_1d(t))[1][0]
def z_t(t): return triangle_coords(np.atleast_1d(t))[2][0]

# Parametreler
w, h, d = 10, 8, 5
l1, l2 = 5, 10
x0, y0, z0 = 5, 0, 1

# Çizim ayarları
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
trail_x, trail_y, trail_z = [], [], []

def update(t):
    ax.cla()

    if np.isclose(t, 0.0):
        trail_x.clear()
        trail_y.clear()
        trail_z.clear()

    xt = float(x_t(t))
    yt = float(y_t(t))
    zt = float(z_t(t))

    l3 = l3f(x0, y0, z0, xt, yt, zt)
    theta = tf(x0, y0, xt, yt)
    a1 = f1f(z0, zt, l1, l2, l3)
    a2 = f2f(z0, zt, l1, l2, l3)

    joint_x = x0 + l1 * np.sin(a1) * np.cos(theta)
    joint_y = y0 - l1 * np.sin(a1) * np.sin(theta)
    joint_z = z0 + l1 * np.cos(a1)

    end_x = joint_x - l2 * np.sin(a2) * np.cos(theta)
    end_y = joint_y + l2 * np.sin(a2) * np.sin(theta)
    end_z = joint_z + l2 * np.cos(a2)

    # Tahta çizimi
    ax.plot([0, 0], [-w/2, w/2], [d, d], color='black')
    ax.plot([0, 0], [w/2, w/2], [d, d+h], color='black')
    ax.plot([0, 0], [w/2, -w/2], [d+h, d+h], color='black')
    ax.plot([0, 0], [-w/2, -w/2], [d+h, d], color='black')
    y_plane = np.linspace(-w / 2, w / 2, 10)
    z_plane = np.linspace(d, d + h, 10)
    Y, Z = np.meshgrid(y_plane, z_plane)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, color='ghostwhite', alpha=0.1, edgecolor='none')

    # Kol çizimi
    ax.plot([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='black', lw=2)
    ax.scatter([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='blue', s=25)

    # İz çizimi
    trail_x.append(end_x)
    trail_y.append(end_y)
    trail_z.append(end_z)
    ax.plot(trail_x, trail_y, trail_z, color='blue', linewidth=1)

    # Referans üçgen yörünge
    ts = np.linspace(0, 2 * np.pi, 300)
    xs, ys, zs = triangle_coords(ts)
    ax.plot(xs, ys, zs, color='gray', linestyle='--', alpha=0.3)

    # Sınırlar
    ax.set_xlim(-2, 10)
    ax.set_ylim(-w/2-2, w/2+2)
    ax.set_zlim(0, d+h+2)

    # Açı ve pozisyon bilgileri
    ax.text2D(1.10, 0.85, f"\u03C6₂: {np.degrees(a2):.2f}°", transform=ax.transAxes, fontsize=10, color='darkblue')
    ax.text2D(1.10, 0.80, f"\u03C6₁: {np.degrees(a1):.2f}°", transform=ax.transAxes, fontsize=10, color='darkgreen')
    ax.text2D(1.10, 0.75, f"\u03B8: {np.degrees(theta):.2f}°", transform=ax.transAxes, fontsize=10, color='darkred')
    ax.text2D(1.10, 0.25, f"X: {end_x:.2f}", transform=ax.transAxes, fontsize=10)
    ax.text2D(1.10, 0.20, f"Y: {end_y:.2f}", transform=ax.transAxes, fontsize=10)
    ax.text2D(1.10, 0.15, f"Z: {end_z:.2f}", transform=ax.transAxes, fontsize=10)

# Animasyonu başlat
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 120), interval=50)
plt.show()
