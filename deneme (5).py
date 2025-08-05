
import re
import matplotlib.pyplot as plt
import numpy as np

# Bezier eğrisi için temel fonksiyon
def cubic_bezier(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3

# G5 komutlarını dosyadan ayıkla
def extract_g5_commands(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    g5_commands = []
    for line in lines:
        if line.startswith('G5'):
            g5_commands.append(line.strip())
    return g5_commands

# G5 komutunu çözümle
def parse_g5_command(command):
    # G5 Xx Yy Ixi Jyi Pxj Qyj
    match = re.match(r'G5 X([-.\d]+) Y([-.\d]+) I([-.\d]+) J([-.\d]+) P([-.\d]+) Q([-.\d]+)', command)
    if match:
        x, y, ix, jy, px, qy = map(float, match.groups())
        return (ix, jy), (px, qy), (x, y)
    else:
        return None

# Bezier eğrisini çiz
def draw_bezier_curve(start_point, control_point1, control_point2, end_point):
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([
        cubic_bezier(t, start_point, control_point1, control_point2, end_point)
        for t in t_values
    ])
    plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier Curve')
    plt.scatter(*zip(start_point, control_point1, control_point2, end_point), color='red')
    plt.annotate('Start', start_point, textcoords="offset points", xytext=(-10,10), ha='center')
    plt.annotate('Control 1', control_point1, textcoords="offset points", xytext=(-10,10), ha='center')
    plt.annotate('Control 2', control_point2, textcoords="offset points", xytext=(-10,10), ha='center')
    plt.annotate('End', end_point, textcoords="offset points", xytext=(-10,10), ha='center')

# G-code dosyasını işleyip eğrileri çiz
def process_gcode(file_path):
    g5_commands = extract_g5_commands(file_path)
    plt.figure(figsize=(8, 8))
    for command in g5_commands:
        parsed = parse_g5_command(command)
        if parsed:
            control_point1, control_point2, end_point = parsed
            start_point = (0, 0) if not len(plt.gca().lines) else plt.gca().lines[-1].get_xydata()[-1]
            draw_bezier_curve(start_point, control_point1, control_point2, end_point)
    plt.axis('equal')
    plt.legend()
    plt.title('Cubic Bezier Curves from G5 Commands')
    plt.show()

# Kullanım
file_path = "C:/Users/Umut/Desktop/math 4079/g5-sample.np"  # G-code dosya yolunu buraya yazın
process_gcode(file_path)