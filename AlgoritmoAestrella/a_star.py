# a_star.py

import sys
import time
import numpy as np
from matplotlib.patches import Rectangle

import point
import random_map

class AStar:
    def __init__(self, map):
        self.map = map
        self.open_set = []
        self.close_set = []
        self.start = point.Point(0,0)
        self.end = point.Point(3,8)

    def HeuristicCost(self, p):
        return 10 * (abs(self.end.x - p.x) + abs(self.end.y - p.y))

    def MoveCost(self, current, neighbor):
        if abs(current.x - neighbor.x) == 1 and abs(current.y - neighbor.y) == 1:
            return 14
        return 10

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0 or x >= self.map.size or y >= self.map.size:
            return False
        return not self.map.IsObstacle(x, y)

    def IsInPointList(self, p, point_list):
        return any(node.x == p.x and node.y == p.y for node in point_list)

    def IsInOpenList(self, p):
        return self.IsInPointList(p, self.open_set)

    def IsInCloseList(self, p):
        return self.IsInPointList(p, self.close_set)

    def IsEndPoint(self, p):
        return p.x == self.end.x and p.y == self.end.y

    def SaveImage(self, plt):
        millis = int(round(time.time() * 1000))
        plt.savefig('./' + str(millis) + '.png')

    def ProcessPoint(self, x, y, parent):
        if not self.IsValidPoint(x, y):
            return

        p = point.Point(x, y)
        if self.IsInCloseList(p):
            return

        move_cost = self.MoveCost(parent, p)
        tentative_g = parent.g + move_cost

        for node in self.open_set:
            if node.x == p.x and node.y == p.y:
                if tentative_g < node.g:
                    node.g = tentative_g
                    node.f = node.g + node.h
                    node.parent = parent
                return

        p.g = tentative_g
        p.h = self.HeuristicCost(p)
        p.f = p.g + p.h
        p.parent = parent
        self.open_set.append(p)

    def SelectPointInOpenList(self):
        min_f = sys.maxsize
        index = -1
        for i, p in enumerate(self.open_set):
            if p.f < min_f:
                min_f = p.f
                index = i
        return index

    def BuildPath(self, p, ax, plt, start_time):
        path = []
        while p:
            path.insert(0, p)
            p = p.parent

        print("\nRuta óptima:")
        for node in path:
            print(f"({node.x},{node.y}) G={node.g} H={node.h} F={node.f}")
            rec = Rectangle((node.x, node.y), 1, 1, color='g')
            ax.add_patch(rec)
            plt.draw()
            self.SaveImage(plt)

        print("\nCosto total:", path[-1].g)
        print("Tiempo:", round(time.time()-start_time,3), "segundos")

    def RunAndSaveImage(self, ax, plt):
        start_time = time.time()

        self.start.g = 0
        self.start.h = self.HeuristicCost(self.start)
        self.start.f = self.start.h

        self.open_set.append(self.start)

        while self.open_set:
            index = self.SelectPointInOpenList()
            p = self.open_set[index]

            if self.IsEndPoint(p):
                return self.BuildPath(p, ax, plt, start_time)

            del self.open_set[index]
            self.close_set.append(p)

            x, y = p.x, p.y
            self.ProcessPoint(x-1, y+1, p)
            self.ProcessPoint(x-1, y, p)
            self.ProcessPoint(x-1, y-1, p)
            self.ProcessPoint(x, y-1, p)
            self.ProcessPoint(x+1, y-1, p)
            self.ProcessPoint(x+1, y, p)
            self.ProcessPoint(x+1, y+1, p)
            self.ProcessPoint(x, y+1, p)

        print("No se encontró camino")
