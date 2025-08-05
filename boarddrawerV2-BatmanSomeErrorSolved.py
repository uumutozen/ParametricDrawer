import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


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
# parameters
w, h, d = 10, 8, 5
l1, l2 = 5, 10
x0, y0, z0 = 5, 0, 1


def x_t(t):
    return np.zeros_like(t)

def y_t(t):
    return (16 * np.sin(t)**3) * 0.2

def z_t(t):
    heart = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    heart_scaled = heart * 0.2 +9
    return heart_scaled


# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
trail_x, trail_y, trail_z = [], [], []

# Update function
def update(t):
    ax.cla()
    if np.isclose(t, 2*np.pi):
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

    # Board
    ax.plot([0, 0], [-w/2, w/2], [d, d], color='black')
    ax.plot([0, 0], [w/2, w/2], [d, d+h], color='black')
    ax.plot([0, 0], [w/2, -w/2], [d+h, d+h], color='black')
    ax.plot([0, 0], [-w/2, -w/2], [d+h, d], color='black')
    y_plane = np.linspace(-w / 2, w / 2, 10)
    z_plane = np.linspace(d, d + h, 10)
    Y, Z = np.meshgrid(y_plane, z_plane)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, color='ghostwhite', alpha=0, edgecolor='none')

    # arm and trace drawing
    ax.plot([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='black', lw=2)
    ax.scatter([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='blue', s=25)

    trail_x.append(end_x)
    trail_y.append(end_y)
    trail_z.append(end_z)

    # Drawing (trace) display
    ax.plot(trail_x, trail_y, trail_z, color='royalblue', linewidth=1.5)
    ax.view_init(elev=20, azim=-20)

    # Target orbit (reference)
    ts = np.linspace(0.00001, 2*np.pi, 140)
    xs = x_t(ts)
    ys = y_t(ts)
    zs = z_t(ts)
    ax.plot(xs, ys, zs, color='gray', linestyle = '--' ,alpha=0.3)

    ax.set_xlim(-2, 10)
    ax.set_ylim(-w/2-2, w/2+2)
    ax.set_zlim(0, d+h+2)

    # Calculate angles in degrees
    a1_deg = np.degrees(a1)
    a2_deg = np.degrees(a2)
    theta_deg = np.degrees(theta)

    # Print angles
    ax.text2D(1.10, 0.85, f"\u03C6₂: {a2_deg:.2f}°", transform=ax.transAxes, fontsize=10, color='darkblue')
    ax.text2D(1.10, 0.80, f"\u03C6₁: {a1_deg:.2f}°", transform=ax.transAxes, fontsize=10, color='darkgreen')
    ax.text2D(1.10, 0.75, f"\u03B8: {theta_deg:.2f}°", transform=ax.transAxes, fontsize=10, color='darkred')
    ax.text2D(1.10, 0.25, f"X: {0.00 if abs(end_x) < 1e-2 else end_x:.2f}", transform=ax.transAxes, fontsize=10)
    ax.text2D(1.10, 0.20, f"Y: {0.00 if abs(end_y) < 1e-2 else end_y:.2f}", transform=ax.transAxes, fontsize=10)
    ax.text2D(1.10, 0.15, f"Z: {0.00 if abs(end_z) < 1e-2 else end_z:.2f}", transform=ax.transAxes, fontsize=10)

ani = FuncAnimation(fig, update, frames=np.linspace(0.00001, 2*np.pi, 140), interval=50)
plt.show()



save_path = "C:/Users/Umut/Desktop/frames/"
os.makedirs(save_path, exist_ok=True)

n_frames = 200

for i, t in enumerate(np.linspace(0.00001, 2*np.pi, n_frames)):
    update(t)
    plt.pause(0.001)
    fig.savefig(f"{save_path}batman-{i+1}.png", format='png')  # <<< sadece burası değişti
    print(f"Kare {i+1} kaydedildi.")

plt.close(fig)
