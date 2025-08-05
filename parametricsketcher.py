import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Calculates the planar angle (theta) between two points in the XY plane
def tf(x, y, u, v):
    return np.pi - np.arctan2(v - y, u - x)
# Computes the 3D Euclidean distance between two points (used as l3)
def l3f(x, y, z, u, v, w):
    return np.linalg.norm([u - x, v - y, w - z])
# Calculates the angle (in radians) using the Law of Cosines
def af(l, k, h):
    return np.arccos((k**2 + h**2 - l**2) / (2 * k * h))
# Alternative version of af() with different argument order
def bf(l, k, h):
    return af(k, l, h)
# Calculates the vertical angle based on the Z-axis height difference
def df(z, w, h):
    return np.arccos(np.abs(w - z) / h)
# Calculates the first joint angle phi1
def f1f(z, w, l, k, h):
    return bf(l, k, h) - df(z, w, h)
# Calculates the second joint angle phi2
def f2f(z, w, l, k, h):
    return af(l, k, h) + df(z, w, h)

# parameters
w, h, d = 10, 8, 5
l1, l2 = 5, 10
x0, y0, z0 = 5, 0, 1

# The logo is parameterized over the interval t E [0, 14),
def batmanlogo_scalar(t):
    pi = np.pi
    # Transforms and scales the shape to fit the drawing area
    def transform(x, y):
        return [x / 2, y / 2 + 8]
    # Piecewise construction of the Batman logo using trigonometric and polynomial segments
    if 0 <= t < 1:
        return transform(0.6 * t, 4)
    elif 1 <= t < 2:
        return transform(0.6 + 0.6 * (t - 1), 4 + 0.94 * (t - 1))
    elif 2 <= t < 3:
        return transform(1.2, 4.9434 - 1.4434 * (t - 2))
    elif 3 <= t < 4:
        s = (0.576 + pi) * (t - 3) - pi
        return transform(1.52 * np.cos(s) + 2.72, 1.52 * np.sin(s) + 3.5)
    elif 4 <= t < 5:
        return transform(8 * np.cos(1.83 * (5 - t) - 0.785), 5 * np.sin(1.83 * (5 - t) - 0.785))
    elif 5 <= t < 6:
        s = 4.4 + 2.26 * (6 - t)
        return transform(-12.43 + 5.212 * s - 0.375 * s**2, -21.52 + 7.03 * s - 0.65 * s**2)
    elif 6 <= t < 7:
        s = 3.25 * (7 - t)
        return transform(s, -2 - 0.75 * (s - 2)**2)
    elif 7 <= t < 8:
        s = 3.25 * (t - 7)
        return transform(-s, -2 - 0.75 * (s - 2)**2)
    elif 8 <= t < 9:
        s = 4.4 + 2.26 * (t - 8)
        return transform(12.43 - 5.212 * s + 0.375 * s**2, -21.52 + 7.03 * s - 0.65 * s**2)
    elif 9 <= t < 10:
        s = 1.83 * (t - 9) - 0.785
        return transform(-8 * np.cos(s), 5 * np.sin(s))
    elif 10 <= t < 11:
        s = (0.576 + pi) * (11 - t) - pi
        return transform(-1.52 * np.cos(s) - 2.72, 1.52 * np.sin(s) + 3.5)
    elif 11 <= t < 12:
        return transform(-1.2, 4.9434 - 1.4434 * (12 - t))
    elif 12 <= t < 13:
        return transform(-0.6 - 0.6 * (13 - t), 4 + 0.94 * (13 - t))
    elif 13 <= t < 14:
        return transform(-0.6 * (14 - t), 4)
    else:
        return [0, 0]

# Returns x and y coordinates for the given time(s) t by evaluating the Batman logo path.
def batmanlogo(t):
    t = np.asarray(t)
    if t.ndim == 0:
        return batmanlogo_scalar(t)
    else:
        result = np.array([batmanlogo_scalar(ti) for ti in t])
        return result[:, 0], result[:, 1]
# Converts the 2D Batman logo path into 3D by assigning x = 0.
def batmanlogo3d(t):
    x, y = batmanlogo(t)
    return [0 * np.array(x), x, y]
# Helper functions to extract the X, Y, and Z coordinates from the 3D path.
def x_t(t): return batmanlogo3d(t)[0]
def y_t(t): return batmanlogo3d(t)[1]
def z_t(t): return batmanlogo3d(t)[2]

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
trail_x, trail_y, trail_z = [], [], []

# Update function
def update(t):
    ax.cla()
    if np.isclose(t, 13.9999):
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
    ax.plot_surface(X, Y, Z, color='ghostwhite', alpha=0.1, edgecolor='none')

    # arm and trace drawing
    ax.plot([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='black', lw=2)
    ax.scatter([x0, joint_x, end_x], [y0, joint_y, end_y], [z0, joint_z, end_z], color='blue', s=25)

    trail_x.append(end_x)
    trail_y.append(end_y)
    trail_z.append(end_z)

    # Drawing (trace) display
    ax.plot(trail_x, trail_y, trail_z, color='blue', linewidth=2)
    ax.view_init(elev=20, azim=-20)

    # Target orbit (reference)
    ts = np.linspace(0.0001, 13.9999, 140)
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

ani = FuncAnimation(fig, update, frames=np.linspace(0.0001, 13.9999, 200), interval=50)

#writer = FFMpegWriter(fps=30, metadata=dict(artist='Umut'), bitrate=1800)
#ani.save("BatmanLogo.mp4", writer=writer)

plt.show()
