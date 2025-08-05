import matplotlib.pyplot as plt
import numpy as np

def parse_gcode(gcode):
    commands = []
    for line in gcode.splitlines():
        if line.strip():
            commands.append(line.strip())
    return commands

def draw_gcode(commands):
    fig, ax = plt.subplots()
    x, y, z = 0, 0, 0
    last_angle = 0
    for command in commands:
        parts = command.split()
        cmd = parts[0]
        params = {p[0]: float(p[1:]) for p in parts[1:]}

        if cmd == "G0" or cmd == "G1":  # Linear movement
            new_x = params.get('X', x)
            new_y = params.get('Y', y)
            plt.plot([x, new_x], [y, new_y], 'b-')
            x, y = new_x, new_y

        elif cmd == "G2" or cmd == "G3":  # Arc movement
            clockwise = cmd == "G2"
            i = params.get('I', 0)
            j = params.get('J', 0)
            new_x = params.get('X', x)
            new_y = params.get('Y', y)
            center_x = x + i
            center_y = y + j
            radius = np.sqrt(i**2 + j**2)

            start_angle = np.arctan2(y - center_y, x - center_x)
            end_angle = np.arctan2(new_y - center_y, new_x - center_x)

            if clockwise and end_angle > start_angle:
                end_angle -= 2 * np.pi
            elif not clockwise and end_angle < start_angle:
                end_angle += 2 * np.pi

            angles = np.linspace(start_angle, end_angle, 100)
            arc_x = center_x + radius * np.cos(angles)
            arc_y = center_y + radius * np.sin(angles)
            plt.plot(arc_x, arc_y, 'r-')

            x, y = new_x, new_y

        # Compute and display blade direction
        angle = np.arctan2(y - center_y, x - center_x) if cmd in ["G2", "G3"] else last_angle
        last_angle = angle
        ax.arrow(x, y, 0.5 * np.cos(angle), 0.5 * np.sin(angle), head_width=0.5, color='g')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

gcode = """G0 X2 Y2 Z1 E0;
G1 X17 Y7 Z1 E18.43;
G3 X18.95 Y15.28 I-1.58 J4.74 E135;
G1 X17.54 Y16.69 E135;
G2 X11.68 Y30.84 I14.1423 J14.1453 E90;
G5 X75 Y75 I0 J100 P-15 Q-175 E85.10;"""

commands = parse_gcode(gcode)
