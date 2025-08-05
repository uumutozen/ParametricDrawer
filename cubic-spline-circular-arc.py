import numpy as np
import matplotlib.pyplot as plt

def cubic_bezier(t, P0, P1, P2, P3):
    """
    Computes a point on a cubic Bézier curve.
    t: Parameter in [0, 1]
    P0, P1, P2, P3: Control points (2D)
    """
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

# Define parameters for the circle
radius = 1
center = np.array([0, 0])

# Define start and end points for the quarter-circle arc
P0 = np.array([1, 0])  # Starting point on the circle (1, 0)
P3 = np.array([0, 1])  # Ending point on the circle (0, 1)

# Compute control points for the cubic Bézier approximation
alpha = (4/3) * (np.sqrt(2) - 1) * radius
P1 = P0 + np.array([0, alpha])  # Control point near P0
P2 = P3 + np.array([alpha, 0])  # Control point near P3

# Generate points along the Bézier curve
t_values = np.linspace(0, 1, 100)
curve_points = np.array([cubic_bezier(t, P0, P1, P2, P3) for t in t_values])

# Plotting
circle = plt.Circle(center, radius, color='gray', fill=False, linestyle='dashed', linewidth=5)
fig, ax = plt.subplots()
ax.add_artist(circle)
ax.plot(curve_points[:, 0], curve_points[:, 1], label='Cubic Bézier Curve', color='blue')
ax.plot([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], 'ro-', label='Control Points')
ax.set_aspect('equal', 'box')
plt.title("Cubic Bézier Approximation of a Quarter-Circle Arc")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
