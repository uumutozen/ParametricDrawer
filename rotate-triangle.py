import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the initial vertices of the tetrahedron
initial_vertices = np.array([
    [0, 0, 0],  # Vertex 1 (reference point)
    [-1, 1, 10],  # Vertex 2
    [-1, -1, 10],  # Vertex 3
    [2, 0, 10],  # Vertex 4
])


def get_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3]],
    ]


# G-Code example data
gcode_list = [
    ("G1", 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("G1", 5, 5, 5, 0, 0, 0, 0, 0, 0),
    ("G1", 10, 5, 5, 0, 0, 0, 0, 0, 0),
    ("G1", 5, 10, 5, 0, 0, 0, 0, 0, 0),
]

# Plot initialization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot of tetrahedron
faces = get_faces(initial_vertices)
tetrahedron = Poly3DCollection(faces, edgecolor="k", alpha=0.6, linewidths=1, facecolors="gold")
tetra_ref = ax.add_collection3d(tetrahedron)
scatter = ax.scatter(initial_vertices[:, 0], initial_vertices[:, 1], initial_vertices[:, 2], color='red', s=50)
vertex_labels = [ax.text(v[0], v[1], v[2], f"V{i + 1}", color="blue") for i, v in enumerate(initial_vertices)]


# Update tetrahedron's position
def update_tetrahedron(new_v1_position):
    global initial_vertices
    offset = new_v1_position - initial_vertices[0]
    transformed_vertices = initial_vertices + offset
    faces = get_faces(transformed_vertices)

    # Clear previous collections (if needed)
    for collection in ax.collections:
        collection.remove()  # Remove old tetrahedron and scatter plot

    # Redraw the tetrahedron with updated vertices
    new_tetrahedron = Poly3DCollection(faces, edgecolor="k", alpha=0.6, linewidths=1, facecolors="gold")
    ax.add_collection3d(new_tetrahedron)

    # Update scatter points
    ax.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2], color='red', s=50)

    # Update vertex labels
    for i, v in enumerate(transformed_vertices):
        vertex_labels[i].set_position((v[0], v[1]))
        vertex_labels[i].set_3d_properties(v[2])


# G-Code path animation
def animate_gcode_path():
    for k in range(1, len(gcode_list)):
        prev_position = np.array(gcode_list[k - 1][1:4])
        new_position = np.array(gcode_list[k][1:4])

        # Animate movement
        steps = 100
        for i in range(steps):
            interpolated_position = prev_position + (new_position - prev_position) * (i / steps)
            update_tetrahedron(interpolated_position)
            plt.pause(0.01)  # Pause to visualize animation

        # Draw G-Code path
        ax.plot([gcode_list[k - 1][1], gcode_list[k][1]],
                [gcode_list[k - 1][2], gcode_list[k][2]],
                [gcode_list[k - 1][3], gcode_list[k][3]], color="blue", alpha=0.5)


animate_gcode_path()
plt.show()
