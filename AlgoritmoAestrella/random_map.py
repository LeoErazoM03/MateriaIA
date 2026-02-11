# random_map.py

import numpy as np

class RandomMap:
    def __init__(self):
        self.size = 9

        # 0 = libre, 1 = pared
        self.grid = np.array([
            [0,0,1,0,0,0,0,0,1],
            [0,0,1,0,0,0,0,0,0],
            [0,0,1,0,1,0,1,0,0],
            [0,0,1,0,1,0,1,0,0],
            [0,0,1,0,1,0,1,1,0],
            [0,0,0,0,0,0,1,0,0],
            [0,0,1,1,1,1,1,1,0],
            [0,0,1,0,1,0,1,0,0],
            [0,0,1,0,0,0,0,0,0]
        ])

    def IsObstacle(self, x, y):
        return self.grid[y][x] == 1
