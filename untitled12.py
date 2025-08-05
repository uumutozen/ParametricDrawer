
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Örnek veri: scatter ve çizgi için X, Y, Z koordinatları
np.random.seed(0)
x_data = np.random.uniform(-10, 10, 10)
y_data = np.random.uniform(-10, 10, 10)
z_data = np.random.uniform(-10, 10, 10)

# 3D figür oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter noktalarını çizeceğimiz yer (sabit noktalar)
scatter = ax.scatter(x_data, y_data, z_data, color='blue', s=50)

# Scatter noktalarını birleştiren çizgiyi çiziyoruz
line, = ax.plot(x_data, y_data, z_data, color='green', lw=2)  # Yeşil çizgi

# Hareket eden tek bir nokta (başlangıçta boş)
moving_point, = ax.plot([], [], [], 'go', markersize=8)  # Kırmızı renkli hareket eden nokta

# Eksen sınırlarını ayarla
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# Eksen etiketleri
ax.set_xlabel('X Ekseni')
ax.set_ylabel('Y Ekseni')
ax.set_zlabel('Z Ekseni')

# Hareket eden noktanın çizgi üzerindeki ilerlemesi için interpolasyon fonksiyonu
def interpolate_points(x, y, z, steps=100):
    # İki nokta arasındaki tüm interpolasyon noktalarını bul
    x_interp = np.linspace(x[0], x[1], steps)
    y_interp = np.linspace(y[0], y[1], steps)
    z_interp = np.linspace(z[0], z[1], steps)
    return x_interp, y_interp, z_interp

# Tüm scatter noktaları arasında interpolasyon yaparak adımları hazırlıyoruz
x_interp_all = []
y_interp_all = []
z_interp_all = []

for i in range(len(x_data) - 1):
    x_interp, y_interp, z_interp = interpolate_points([x_data[i], x_data[i+1]],
                                                      [y_data[i], y_data[i+1]],
                                                      [z_data[i], z_data[i+1]], steps=50)
    x_interp_all.extend(x_interp)
    y_interp_all.extend(y_interp)
    z_interp_all.extend(z_interp)

# Güncelleme fonksiyonu (her karede nokta bir sonraki konuma gider)
def update(num):
    # Hareket eden noktanın çizgi üzerindeki konumunu günceller
    moving_point.set_data(x_interp_all[num:num+1], y_interp_all[num:num+1])
    moving_point.set_3d_properties(z_interp_all[num:num+1])
    return moving_point,

# Animasyon fonksiyonu
ani = FuncAnimation(fig, update, frames=len(x_interp_all), interval=50, blit=False)

# Grafiği göster
plt.show()
