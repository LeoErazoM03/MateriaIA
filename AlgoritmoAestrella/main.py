# main.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import random_map
import a_star

plt.figure(figsize=(6, 6))

map = random_map.RandomMap()

ax = plt.gca()
ax.set_xlim([0, map.size])
ax.set_ylim([0, map.size])

# Dibujar mapa
for i in range(map.size):
    for j in range(map.size):
        if map.IsObstacle(i,j):
            rec = Rectangle((i, j), 1, 1, color='gray')
        else:
            rec = Rectangle((i, j), 1, 1, edgecolor='gray', facecolor='white')
        ax.add_patch(rec)

# Inicio (celda 1 → (0,0))
rec = Rectangle((0, 0), 1, 1, facecolor='blue')
ax.add_patch(rec)

# Meta (celda 76 → (8,3))
rec = Rectangle((3, 8), 1, 1, facecolor='red')
ax.add_patch(rec)

plt.axis('equal')
plt.axis('off')
plt.tight_layout()

# Ejecutar A*
astar = a_star.AStar(map)
astar.RunAndSaveImage(ax, plt)

plt.gca().invert_yaxis()
plt.show()
