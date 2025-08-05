import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')

# --- Batman logo fonksiyonu (2B) ---
def batmanlogo(t):
    pi = np.pi
    def transform(x, y):
        return [x / 2, y / 2 + 8]  # küçült ve yukarı kaydır

    if 0<=t<1:
        return transform(0.6 * t, 4)
    elif 1<=t<2:
        return transform(0.6 + 0.6 * (t - 1), 4 + 0.94 * (t - 1))
    elif 2<=t<3:
        return transform(1.2, 4.9434 - 1.4434 * (t - 2))
    elif 3<=t<4:
        s = (0.576 + pi) * (t - 3) - pi
        return transform(1.52 * np.cos(s) + 2.72, 1.52 * np.sin(s) + 3.5)
    elif 4<=t<5:
        return transform(8 * np.cos(1.83 * (5 - t) - 0.785), 5 * np.sin(1.83 * (5 - t) - 0.785))
    elif 5<=t<6:
        s = 4.4 + 2.26 * (6 - t)
        return transform(-12.43 + 5.212 * s - 0.375 * s**2, -21.52 + 7.03 * s - 0.65 * s**2)
    elif 6<=t<7:
        s = 3.25 * (7 - t)
        return transform(s, -2 - 0.75 * (s-2)**2)
    elif 7<=t<8:
        s = 3.25 * (t - 7)
        return transform(-s, -2 - 0.75 * (s-2)**2)
    elif 8<=t<9:
        s = 4.4 + 2.26 * (t - 8)
        return transform(12.43 - 5.212 * s + 0.375 * s**2, -21.52 + 7.03 * s - 0.65 * s**2)
    elif 9<=t<10:
        s = 1.83 * (t - 9) - 0.785
        return transform(-8 * np.cos(s), 5 * np.sin(s))
    elif 10<=t<11:
        s = (0.576 + pi) * (11 - t) - pi
        return transform(-1.52 * np.cos(s) - 2.72, 1.52 * np.sin(s) + 3.5)
    elif 11<=t<12:
        return transform(-1.2, 4.9434 - 1.4434 * (12 - t))
    elif 12<=t<13:
        return transform(-0.6 - 0.6 * (13 - t), 4 + 0.94 * (13 - t))
    elif 13<=t<14:
        return transform(-0.6 * (14 - t), 4)
    else:
        return [0, 0]

# --- 2B Batman koordinatlarını 3B'ye çevir ---
def batmanlogo3d(t):
    x, y = batmanlogo(t)
    return [0, x, y]  # X sabit, Y ve Z koordinatları çizim alanına yansıtılır

# --- Robot kol hesaplamaları ---
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

# --- Parametreler ---
w, h, d = 10, 8, 5
l1, l2 = 5, 10
x0, y0, z0 = 5, 0, 1

# --- Hedef uç nokta koordinatları (Batman yörüngesi) ---
def x_t(t): return batmanlogo3d(t)[0]
def y_t(t): return batmanlogo3d(t)[1]
def z_t(t): return batmanlogo3d(t)[2]

# --- Grafik ayarları ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
trail_x, trail_y, trail_z = [], [], []

# --- Güncelleme fonksiyonu ---
def update(t):
    ax.cla()

    if np.isclose(t, 0.0):
        trail_x.clear()
        trail_y.clear()
        trail_z.clear()

    xt, yt, zt = x_t(t), y_t(t), z_t(t)
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

    # Tahta
    ax.plot([0, 0], [-w/2, w/2], [d, d], color='black')
    ax.plot([0, 0], [w/2, w/2], [d, d+h], color='black')
    ax.plot([0, 0], [w/2, -w/2], [d+h, d+h], color='black')
    ax.plot([0, 0], [-w/2, -w/2], [d+h, d], color='black')
    y_plane = np.linspace(-w / 2, w / 2, 10)
    z_plane = np.linspace(d, d + h, 10)
    Y, Z = np.meshgrid(y_plane, z_plane)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, color='ghostwhite', alpha=0.1, edgecolor='none')

    # Kol ve iz çizimi
    ax.plot([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='black', lw=2)
    ax.scatter([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='blue', s=25)

    trail_x.append(end_x)
    trail_y.append(end_y)
    trail_z.append(end_z)
    ax.plot(trail_x, trail_y, trail_z, color='blue', linewidth=1)

    # Batman şekli referansı (gri çizgi)
    ts = np.linspace(0.001, 14, 500)
    xs = [x_t(ti) for ti in ts]
    ys = [y_t(ti) for ti in ts]
    zs = [z_t(ti) for ti in ts]
    ax.plot(xs, ys, zs, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlim(-2, 10)
    ax.set_ylim(-w/2-2, w/2+2)
    ax.set_zlim(0, d+h+2)

    # Açı bilgileri
    ax.text2D(1.10, 0.85, f"\u03C6₂: {np.degrees(a2):.2f}°", transform=ax.transAxes, fontsize=10, color='darkblue')
    ax.text2D(1.10, 0.80, f"\u03C6₁: {np.degrees(a1):.2f}°", transform=ax.transAxes, fontsize=10, color='darkgreen')
    ax.text2D(1.10, 0.75, f"\u03B8: {np.degrees(theta):.2f}°", transform=ax.transAxes, fontsize=10, color='darkred')
    ax.text2D(1.10, 0.25, f"X: {end_x:.2f}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(1.10, 0.20, f"Y: {end_y:.2f}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(1.10, 0.15, f"Z: {end_z:.2f}", transform=ax.transAxes, fontsize=10, color='black')

# --- Animasyon başlat ---
ani = FuncAnimation(fig, update, frames=np.linspace(0.001, 13.999, 200), interval=50)
plt.show()
