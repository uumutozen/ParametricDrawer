import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation

def rota_ve_teget_vektor_animasyon(x_interp_all, y_interp_all, z_interp_all, num_frames):
    """
    3D uzayda bir rota boyunca hareket eden bir nokta ve teğet vektörünü animasyon olarak çizer.

    Parametreler:
        x_interp_all, y_interp_all, z_interp_all (list): Rota boyunca x, y, z noktaları.
        num_frames (int): Animasyonun toplam kare sayısı.
    """
    # Başlangıç ayarları
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel("X Eksen")
    ax.set_ylabel("Y Eksen")
    ax.set_zlabel("Z Eksen")

    # Quiver (Teğet vektör) ve rota
    quiver = Line3DCollection([], linewidths=2, colors='r')
    ax.add_collection3d(quiver)

    # Fonksiyon: `np.float64` listeyi numpy dizisine çevirir
    def convert_np_float64_list_to_ndarray(data):
        return np.array(data, dtype=np.float64)

    # Güncelleme fonksiyonu
    def update(frame):
        # Başlangıç noktasını al
        x_start = convert_np_float64_list_to_ndarray(x_interp_all)[frame]
        y_start = convert_np_float64_list_to_ndarray(y_interp_all)[frame]
        z_start = convert_np_float64_list_to_ndarray(z_interp_all)[frame]

        # Sonraki nokta (veya son karede sabit kalır)
        if frame < num_frames - 1:
            x_next = convert_np_float64_list_to_ndarray(x_interp_all)[frame + 1]
            y_next = convert_np_float64_list_to_ndarray(y_interp_all)[frame + 1]
            z_next = convert_np_float64_list_to_ndarray(z_interp_all)[frame + 1]
        else:
            x_next = x_start
            y_next = y_start
            z_next = z_start

        # Teğet vektörün hesaplanması
        dx = x_next - x_start
        dy = y_next - y_start
        dz = z_next - z_start

        # Quiver güncelleniyor
        quiver.set_segments([[[x_start, y_start, z_start],
                              [x_start + dx, y_start + dy, z_start + dz]]])
        return quiver,

    # Animasyonu başlat
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)
    plt.show()

# Örnek rota verileri
t_values = np.linspace(0, 2 * np.pi, 200)
x_interp_all = 5 * np.cos(t_values)
y_interp_all = 5 * np.sin(t_values)
z_interp_all = t_values

# Fonksiyonu çalıştır
rota_ve_teget_vektor_animasyon(x_interp_all, y_interp_all, z_interp_all, num_frames=200)

