import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
ax.set_zticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

ax.axes.set_xlim3d(left=0.0, right=300.0)
ax.axes.set_ylim3d(bottom=0.0, top=300.0)
ax.axes.set_zlim3d(bottom=0.0, top=60.0)

