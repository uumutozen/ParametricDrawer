def take_coordinate(coordinates, coordinate_name, coordinate_previous):
    gcode_u_init = coordinates.find(coordinate_name)
    if gcode_u_init == -1:
        return coordinate_previous
    else:
        gcode_u_end = coordinates.find(" ", gcode_u_init + 1) 
        if gcode_u_end != -1:
            gcode_u = coordinates[gcode_u_init + 1:gcode_u_end]
        else:
            gcode_u = coordinates[gcode_u_init + 1:]
        return gcode_u


gcodefile = open("sample.gcode", "r+")
gcodes = ""
# file1.seek(0)
gcodes = gcodefile.readlines()
gcodefile.close()
gcodes = [gcode.replace("\n", "") for gcode in gcodes]


x_previous = 0.0
y_previous = 0.0
z_previous = 0.0
e_previous = 0.0

for gcode in gcodes:
    gcode_init = gcode.find(" ")
    gcode_code = gcode[0:gcode_init]
    if (gcode_code == "G0" or gcode_code == "G1"):
        gcode_end = gcode.find(";")
        if gcode_end == -1:
            gcode_coordinates = gcode[gcode_init + 1:]
        else:
            gcode_coordinates = gcode[gcode_init + 1:gcode_end]
        x_current = take_coordinate(gcode_coordinates, "X", x_previous)
        y_current = take_coordinate(gcode_coordinates, "Y", y_previous)
        z_current = take_coordinate(gcode_coordinates, "Z", z_previous)
        e_current = take_coordinate(gcode_coordinates, "E", e_previous)
        print(x_current, y_current, z_current, e_current)
        x_previous = x_current
        y_previous = y_current
        z_previous = z_current
        e_previous = e_current
    elif gcode_code == "G92":
        print("Insert Codes Here")
        