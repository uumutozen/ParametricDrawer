import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# Tetrahedron'un başlangıçtaki köşe koordinatları
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, np.sqrt(3)/2, 0],
    [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
])

# Tetrahedron'un yüzeylerini tanımlama
faces = [
    [vertices[0], vertices[1], vertices[2]],
    [vertices[0], vertices[1], vertices[3]],
    [vertices[1], vertices[2], vertices[3]],
    [vertices[2], vertices[0], vertices[3]]
]

# 3D çizim oluşturma
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tetrahedron'un başlangıçtaki yüzlerini çizen Poly3DCollection
poly3d = Poly3DCollection(faces, alpha=0.5, edgecolor='k')
ax.add_collection3d(poly3d)

# Eksen aralıklarını ayarla
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.set_zlim([-1, 2])

# Tetrahedron'un hareketi için bir yol tanımlama (örnek: bir daire)
t = np.linspace(0, 2 * np.pi, 100)
x_path = 0.5 * np.cos(t)
y_path = 0.5 * np.sin(t)
z_path = 0.2 * np.sin(2 * t)

# Güncelleme fonksiyonu

def update(frame):
    # Tetrahedron'u hareket ettirmek için yeni koordinatlar hesapla
    dx, dy, dz = x_path[frame], y_path[frame], z_path[frame]
    new_vertices = vertices + np.array([dx, dy, dz])

    new_faces = [
        [new_vertices[0], new_vertices[1], new_vertices[2]],
        [new_vertices[0], new_vertices[1], new_vertices[3]],
        [new_vertices[1], new_vertices[2], new_vertices[3]],
        [new_vertices[2], new_vertices[0], new_vertices[3]]
    ]
    poly3d.set_verts(new_faces)
    return poly3d,

# Animasyon oluşturma
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
print(type(x_path))
plt.show()
