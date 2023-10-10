import os
import sys
# move one directory up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from monte_carlo_chemokine_old import _check_neighbours
import numpy as np

box_size = [10, 20]

x = np.array([[0, 0],
              [0, 1],
              [0, 2],
              [9, 1]])
y = np.array([[9, 1],
              [1, 0],
              [0, 19],
              [5, 5],
              [5, 6]])

nx = len(x)
ny = len(y)

neighbours_xy, directions_xy = _check_neighbours(x, y, nx, ny, box_size)

neighbours_xy_no_reflection, directions_xy_no_reflection = _check_neighbours(x, y, nx, ny, box_size, reflection_x=False)

neighbours_xx, directions_xx = _check_neighbours(x, x, nx, nx, box_size)

neighbours_xx_no_reflection, directions_xx_no_reflection = _check_neighbours(x, x, nx, nx, box_size, reflection_x=False)


print("Neighbours of x and y with x reflection")
print(neighbours_xy)
print("\nDirections of x and y with x reflection")
print(directions_xy)
print("\nNeighbours of x and y without x reflection")
print(neighbours_xy_no_reflection)
print("\nDirections of x and y without x reflection")
print(directions_xy_no_reflection)
print("\nNeighbours of x and x")
print(neighbours_xx)
print("\nNeighbours of x and x")
print(directions_xx)
print("\nNeighbours of x and x without x reflection")
print(neighbours_xx_no_reflection)
print("\nNeighbours of x and x without x reflection")
print(directions_xx_no_reflection)