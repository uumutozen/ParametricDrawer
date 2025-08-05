import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, interact

trace = []

# Parametreler
w, h, d = 10, 8, 5      # Tahta boyutları
L1, L2 = 5, 10          # Kol uzunlukları
x0, y0, z0 = 5, 0, 2    # Robot kolunun tabanı (tahtadan uzaklaştırıldı)

def compute_positions(theta, phi1, phi2):
    # Dereceden radyana çevir
    theta = np.radians(theta)
    phi1 = np.radians(phi1)
    phi2 = np.radians(phi2)

    # İlk eklem
    x1 = x0 + L1 * np.sin(phi1) * np.cos(theta)
    y1 = y0 - L1 * np.sin(phi1) * np.sin(theta)
    z1 = z0 + L1 * np.cos(phi1)

    # Uç nokta
    x2 = x1 - L2 * np.sin(phi2) * np.cos(theta)
    y2 = y1 + L2 * np.sin(phi2) * np.sin(theta)
    z2 = z1 + L2 * np.cos(phi2)

    return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)

def draw_arm(theta, phi1, phi2):
    global trace
    base, joint1, joint2 = compute_positions(theta, phi1, phi2)
    trace.append(joint2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Robot kolu çiz
    xs, ys, zs = zip(base, joint1, joint2)
    ax.plot(xs, ys, zs, '-o', c='black', linewidth=4, markersize=8, label='Kol')

    # Tahta sabit x = 0 düzlemi içinde çiziliyor
    Y, Z = np.meshgrid(np.linspace(-w/2, w/2, 30), np.linspace(d, d + h, 30))
    X = np.zeros_like(Y)  # x = 0 düzlemi
    ax.plot_surface(X, Y, Z, alpha=0.3, color='blue', label='Tahta')

    # Uç nokta izini çiz
    if len(trace) > 1:
        xt, yt, zt = zip(*trace)
        ax.plot(xt, yt, zt, c='red', linewidth=2, label='İz')

    # Görsel ayarlar
    ax.set_xlim(-2, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, z0 + L1 + L2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Kol ve Sabit Tahta')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Etkileşimli arayüz
interact(
    draw_arm,
    theta=0,
    phi1=30,
    phi2=30
)
