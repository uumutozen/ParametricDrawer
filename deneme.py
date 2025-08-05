import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# G-code parsing function
def parse_gcode(gcode):
    lines = gcode.strip().split('\n')
    points = []
    for line in lines:
        if line.startswith('G1'):
            coords = line.split()
            x = y = z = 0.0
            for coord in coords:
                if coord.startswith('X'):
                    x = float(coord[1:])
                elif coord.startswith('Y'):
                    y = float(coord[1:])
                elif coord.startswith('Z'):
                    z = float(coord[1:])
            points.append((x, y, z))
    return points

# Generate a tetrahedron at a given point
def generate_tetrahedron(center, size=1.0):
    x, y, z = center
    vertices = [
        [x, y, z + size],
        [x - size, y - size, z],
        [x + size, y - size, z],
        [x, y + size, z],
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3]],
    ]
    return vertices, faces

# G-code string
gcode = """
G1 X0 Y0 Z0
G1 X5 Y5 Z0
G1 X10 Y0 Z5
G1 X15 Y5 Z10
G1 X20 Y0 Z15
"""

# Parse G-code
points = parse_gcode(gcode)

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_zlim(0, 20)

# Draw the path
x_vals, y_vals, z_vals = zip(*points)
ax.plot(x_vals, y_vals, z_vals, color='blue')

# Animation
tetrahedron = None  # Initialize tetrahedron

def update(frame):
    global tetrahedron
    if tetrahedron:
        tetrahedron.remove()  # Remove previous tetrahedron

    vertices, faces = generate_tetrahedron(points[frame], size=1.0)
    tetrahedron = Poly3DCollection(faces, alpha=0.5, linewidths=1, edgecolors='r')
    ax.add_collection3d(tetrahedron)

ani = FuncAnimation(fig, update, frames=len(points), interval=500, repeat=False)
plt.show()
