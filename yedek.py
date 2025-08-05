'''
def rotation_matrix_to_align(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    if s == 0:  # v1 ve v2 paralel ise
        return np.eye(3)
    k_matrix = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + k_matrix + np.dot(k_matrix, k_matrix) * ((1 - c) / (s**2))
    return rotation_matrix
'''
'''
def update(num):
    # Hareket eden noktanın çizgi üzerindeki konumunu günceller
    moving_point.set_data(x_interp_all[num:num+1], y_interp_all[num:num+1])
    moving_point.set_3d_properties(z_interp_all[num:num+1])
'''
'''

def update(frame):
    # `frame` dizinin boyutunu aşmasın diye kontrol ekliyoruz
    max_index = len(x_interp_all) - 1
    frame = min(frame, max_index)

    # Başlangıç noktasını al
    x_start = convert_np_float64_list_to_ndarray(x_interp_all)[frame]
    y_start = convert_np_float64_list_to_ndarray(y_interp_all)[frame]
    z_start = convert_np_float64_list_to_ndarray(z_interp_all)[frame]

    # Sonraki nokta (eğer son karedeyse aynı noktada kalır)
    if frame < max_index:
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

'''

def update(frame):
    # Hareket vektörünü hesapla
    dx, dy, dz = (
        convert_np_float64_list_to_ndarray(x_interp_all)[frame],
        convert_np_float64_list_to_ndarray(y_interp_all)[frame],
        convert_np_float64_list_to_ndarray(z_interp_all)[frame]
    )
    movement_vector = np.array([dx, dy, dz])

    # Önceki ve sonraki hareket yönünü hesapla
    if frame > 0:
        prev_dx, prev_dy, prev_dz = (
            convert_np_float64_list_to_ndarray(x_interp_all)[frame - 1],
            convert_np_float64_list_to_ndarray(y_interp_all)[frame - 1],
            convert_np_float64_list_to_ndarray(z_interp_all)[frame - 1]
        )
        previous_vector = np.array([prev_dx, prev_dy, prev_dz])
    else:
        previous_vector = np.array([1, 0, 0])  # Varsayılan başlangıç yönü

    # Hareket yönünü normalize et
    movement_direction = movement_vector - previous_vector
    norm = np.linalg.norm(movement_direction)

    if norm > 0:  # Norm sıfır değilse normalize et
        movement_direction /= norm
    else:
        movement_direction = np.array([1, 0, 0])  # Varsayılan bir yön belirle

    # Başlangıç yönünden hareket yönüne dönmek için rotasyon hesapla
    start_direction = np.array([1, 0, 0])  # Varsayılan başlangıç yönü
    if not np.allclose(movement_direction, start_direction):
        rotation_vector = np.cross(start_direction, movement_direction)
        rotation_angle = np.arccos(np.dot(start_direction, movement_direction))
        rotation_matrix = R.from_rotvec(rotation_angle * rotation_vector).as_matrix()
    else:
        rotation_matrix = np.eye(3)  # Dönüşüm gerekmez

    # Tetrahedron köşe noktalarını döndür ve taşı
    rotated_vertices = (rotation_matrix @ vertices.T).T
    new_vertices = rotated_vertices + movement_vector

    # Yüzleri güncelle
    new_faces = [
        [new_vertices[0], new_vertices[1], new_vertices[2]],
        [new_vertices[0], new_vertices[1], new_vertices[3]],
        [new_vertices[1], new_vertices[2], new_vertices[3]],
        [new_vertices[2], new_vertices[0], new_vertices[3]]
    ]
    poly3d.set_verts(new_faces)
    return poly3d,