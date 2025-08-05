import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline


# Parametric curve definitions
def parametric_curve_arc(t):
    x = 5 * np.cos(t)
    y = 5 * np.sin(t)
    z = 0
    return np.array([x, y, z])


def parametric_curve_line(t):
    x = -5 + 10 * t / np.pi
    y = 5
    z = 0
    return np.array([x, y, z])


def parametric_curve_spline(t):
    points = np.array([
        [0, 0, 0],
        [3, 5, 0],
        [6, -2, 0],
        [10, 3, 0]
    ])
    times = np.linspace(0, 1, len(points))
    cs_x = CubicSpline(times, points[:, 0])
    cs_y = CubicSpline(times, points[:, 1])
    cs_z = CubicSpline(times, points[:, 2])
    x = cs_x(t)
    y = cs_y(t)
    z = cs_z(t)
    return np.array([x, y, z])


# Derivative functions for tangent vectors
def tangent_vector(func, t, delta=1e-5):
    p1 = func(t)
    p2 = func(t + delta)
    tangent = p2 - p1
    return tangent / np.linalg.norm(tangent)  # Normalize the tangent vector


# Function to create a rotation matrix to align one vector with another
def rotation_matrix_to_align(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    k_matrix = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_mat = np.eye(3) + k_matrix + np.dot(k_matrix, k_matrix) * ((1 - c) / (s ** 2 if s else 1))
    return rotation_mat


# Define the initial vertices of the tetrahedron
vertices = np.array([
    [0, 0, 0],  # Vertex 1 (apex, sharp point)
    [-1, 1, 10],  # Vertex 2
    [-1, -1, 10],  # Vertex 3
    [2, 0, 10],  # Vertex 4
])

# Select which curve to use
curve_type = "spline"  # Options: "arc", "line", "spline"

if curve_type == "arc":
    parametric_curve = parametric_curve_arc
elif curve_type == "line":
    parametric_curve = parametric_curve_line
elif curve_type == "spline":
    parametric_curve = parametric_curve_spline

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Update function for the animation
def update(frame):
    ax.clear()
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-5, 15)
    ax.set_title(f"Tetrahedron Following {curve_type.capitalize()} Path")

    # Get current position and tangent
    t = frame / 100  # Scale the frame to slow down movement
    t = np.clip(t, 0, 1)
    position = parametric_curve(t)
    tangent = tangent_vector(parametric_curve, t)

    # Rotate the tetrahedron to align its apex with the tangent
    rot_mat = rotation_matrix_to_align(np.array([0, 0, 1]), tangent)
    rotated_vertices = np.dot(vertices, rot_mat.T)

    # Move the tetrahedron to the new position
    moved_vertices = rotated_vertices + position

    # Plot the tetrahedron
    poly3d = [[moved_vertices[vert] for vert in [0, 1, 2]],
              [moved_vertices[vert] for vert in [0, 1, 3]],
              [moved_vertices[vert] for vert in [0, 2, 3]],
              [moved_vertices[vert] for vert in [1, 2, 3]]]
    ax.add_collection3d(Poly3DCollection(poly3d, alpha=0.7, color='red', edgecolor='k'))

    # Plot the curve
    t_values = np.linspace(0, np.pi if curve_type in ["arc", "line"] else 1, 500)
    curve = np.array([parametric_curve(ti) for ti in t_values])
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='blue', linewidth=2)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=500, interval=100)

# Show the animation
plt.show()


