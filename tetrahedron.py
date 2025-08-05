import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the vertices of a tetrahedron
vertices = np.array([
    [0,0,0],    # Vertex 1
    [-1,1,10],  # Vertex 2
    [-1,-1,10],  # Vertex 3
    [2,0,10],  # Vertex 4
])

# Define the faces of the tetrahedron using the vertices
faces = [
    [vertices[0], vertices[1], vertices[2]],  # Face 1
    [vertices[0], vertices[1], vertices[3]],  # Face 2
    [vertices[0], vertices[2], vertices[3]],  # Face 3
    [vertices[1], vertices[2], vertices[3]],  # Face 4
]

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

poly3d = Poly3DCollection(faces, alpha=0.5, edgecolor='k')
ax.add_collection3d(poly3d)

# Set limits for better visualization
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3])
ax.set_zlim([-2, 12])

# Plot the tetrahedron
tetrahedron = Poly3DCollection(faces, edgecolor="k", alpha=0.6, linewidths=1, facecolors="gold")
ax.add_collection3d(tetrahedron)

# Add vertices as points
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=50)

coords_list = [[0.0, 0.0, 2.0], [1.0, 1.0, 3.0], [2.0, 1.5, 3.5]]
cleaned_coords_list = remove_commas_from_coords(coords_list)
print(cleaned_coords_list)

# Show the plot
plt.show()
