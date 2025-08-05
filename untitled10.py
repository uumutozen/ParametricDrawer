import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Örnek veri: scatter ve hareket eden nokta için X, Y, Z koordinatları
np.random.seed(0)
x_data = np.random.uniform(-10, 10, 25)
y_data = np.random.uniform(-10, 10, 25)
z_data = np.random.uniform(-10, 10, 25)

# 3D figür oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter noktalarını çizeceğimiz yer (sabit noktalar)
scatter = ax.scatter(x_data, y_data, z_data, color='blue', s=50)

# Hareket eden tek bir nokta (başlangıçta boş)
moving_point, = ax.plot([], [], [], 'ro', markersize=8)  # Kırmızı renkli hareket eden nokta

# Eksen sınırlarını ayarla
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# Eksen etiketleri
ax.set_xlabel('X Ekseni')
ax.set_ylabel('Y Ekseni')
ax.set_zlabel('Z Ekseni')

# Güncelleme fonksiyonu (her karede nokta bir sonraki konuma gider)
def update(num):
    # Hareket eden noktanın konumunu günceller
    moving_point.set_data(x_data[num:num+1], y_data[num:num+1])
    moving_point.set_3d_properties(z_data[num:num+1])
    return moving_point,

# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=1000, interval=500, blit=False)

# Grafiği göster
plt.show()
