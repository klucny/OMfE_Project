import numpy as np


def calcBezierValue(control_points, t_val):
    n = len(control_points) - 1
    res = control_points[0]

def calcQuadBezier(control_points, t_val):
    x = (
            np.multiply(control_points[0][0], np.square(1-t_val))
            + np.multiply(control_points[1][0], np.multiply(t_val,(1-t_val)))
            + np.multiply(control_points[2][0],np.square(t_val)))
    y = np.multiply(control_points[0][1], np.square(1-t_val))+np.multiply(control_points[1][1], np.multiply(t_val,(1-t_val)))+np.multiply(control_points[2][1],np.square(t_val))
    return x,y

A = np.array([
    [np.square(1-0), 2*0*(0-1), np.square(0)],
    [np.square(1-0.5), 2*0.5*(0.5-1), np.square(0.5)],
    [np.square(1-1), 2*1*(1-1), np.square(1)]
])

b = np.array([
    [0,0],
    [0.5,1],
    [1,0]
])

sol = np.linalg.lstsq(A,b)[0]
print(sol)

def test_calcQuadBezier():
    control_points = np.array([[0, 0], [5, 5], [10, 0]])
    t_val = 0.5
    x, y = calcQuadBezier(control_points, t_val)
    print(x, y)


