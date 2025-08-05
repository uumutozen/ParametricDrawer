import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Örnek veri: scatter için rastgele X, Y, Z koordinatları
np.random.seed(0)
x_data = np.random.uniform(-10, 10, 25)
y_data = np.random.uniform(-10, 10, 25)
z_data = np.random.uniform(-10, 10, 25)

# Çizgi için örnek yol verisi
line_x = np.linspace(-10, 10, 100)
line_y = np.sin(line_x)
line_z = np.cos(line_x)

# 3D figür oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter noktalarını çizeceğimiz yer
scatter = ax.scatter(x_data, y_data, z_data, color='blue', s=50)

# Çizgi için başlangıç durumları
line, = ax.plot([], [], [], 'r-', lw=2)  # Kırmızı çizgi (plot)

# Eksen sınırlarını ayarla
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# Eksen etiketleri
ax.set_xlabel('X Ekseni')
ax.set_ylabel('Y Ekseni')
ax.set_zlabel('Z Ekseni')

# Güncelleme fonksiyonu (her karede çizgi hareket eder)
def update(num):
    # Çizginin o ana kadar olan kısımlarını çizer
    line.set_data(line_x[:num], line_y[:num])
    line.set_3d_properties(line_z[:num])
    return line,

# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=len(line_x), interval=100, blit=False)

# Grafiği göster
plt.show()
