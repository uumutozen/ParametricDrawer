import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def tf(x, y, u, v):
    return np.pi-np.arctan2(v - y, u - x)

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

# Parametreler
w, h, d = 10, 8, 5
l1, l2 = 5, 10
x0, y0, z0 = 5, 0, 1

# Uç nokta yörüngesi
def x_t(t): return np.zeros_like(t)
def y_t(t): return 1 + 3 * np.cos(t)
def z_t(t): return 9 + 3 * np.sin(t)

# Animasyon çizimi
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

trail_x, trail_y, trail_z = [], [], []

def update(t):
    ax.cla()

    # Uç nokta koordinatları
    xt, yt, zt = x_t(t), y_t(t), z_t(t)

    # Ara hesaplamalar
    l3 = l3f(x0, y0, z0, xt, yt, zt)
    theta = tf(x0, y0, xt, yt)
    a1 = f1f(z0, zt, l1, l2, l3)
    a2 = f2f(z0, zt, l1, l2, l3)

    # Eklem ve uç nokta pozisyonları
    joint_x = x0 + l1 * np.sin(a1) * np.cos(theta)
    joint_y = y0 - l1 * np.sin(a1) * np.sin(theta)
    joint_z = z0 + l1 * np.cos(a1)

    end_x = joint_x - l2 * np.sin(a2) * np.cos(theta)
    end_y = joint_y + l2 * np.sin(a2) * np.sin(theta)
    end_z = joint_z + l2 * np.cos(a2)

    # --- Platform çizimi ---
    ax.plot([0, 0], [-w/2, w/2], [d, d], color='black')
    ax.plot([0, 0], [w/2, w/2], [d, d+h], color='black')
    ax.plot([0, 0], [w/2, -w/2], [d+h, d+h], color='black')
    ax.plot([0, 0], [-w/2, -w/2], [d+h, d], color='black')
    y_plane = np.linspace(-w / 2, w / 2, 10)
    z_plane = np.linspace(d, d + h, 10)
    Y, Z = np.meshgrid(y_plane, z_plane)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, color='skyblue', alpha=0.5, edgecolor='none')

    # --- Kol çizimi ---
    ax.plot([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='black', lw=2)
    ax.scatter([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='blue', s=25)

    # --- Uç nokta izini kaydetme ---
    trail_x.append(end_x)
    trail_y.append(end_y)
    trail_z.append(end_z)

    # --- Çizim (iz) gösterimi ---
    ax.plot(trail_x, trail_y, trail_z, color='blue', linewidth=1)

    # --- Hedef yörünge (referans) ---
    ts = np.linspace(0, 2 * np.pi, 100)
    xs = x_t(ts)
    ys = y_t(ts)
    zs = z_t(ts)
    ax.plot(xs, ys, zs, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 20)


# --- Açıları derece cinsinden hesapla ---
    a1_deg = np.degrees(a1)
    a2_deg = np.degrees(a2)
    theta_deg = np.degrees(theta)

    # --- Açıları ekrana yazdır ---
    ax.text2D(0.02, 0.95, f"Açı 1 (a1): {a1_deg:.2f}°", transform=ax.transAxes, fontsize=10, color='darkgreen')
    ax.text2D(0.02, 0.90, f"Açı 2 (a2): {a2_deg:.2f}°", transform=ax.transAxes, fontsize=10, color='darkblue')
    ax.text2D(0.02, 0.85, f"Yön Açısı (theta): {theta_deg:.2f}°", transform=ax.transAxes, fontsize=10, color='darkred')
    ax.text2D(0.02, 0.80, f"X: {end_x:.2f}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(0.02, 0.75, f"Y: {end_y:.2f}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(0.02, 0.70, f"Z: {end_z:.2f}", transform=ax.transAxes, fontsize=10, color='black')

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 120), interval=50)
plt.show()