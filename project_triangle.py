import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
import svgwrite


norm = np.linalg.norm

''' given 3 vertices returns a version porjected to x,y plane with longest
vertex (0,0)\
longest direction along x axis
and (0,0) vertex connected to mid size and longest edges
and goes to positive

'''
def projectTriangle(v1,v2,v3):
    #edges eab points from a to b. Try adding eab + va to see
    e12 = v2 - v1
    e23 = v3 - v2
    e31 = v1 - v3
    edgearray = [e12,e23,e31]
    ne12 = norm(e12)
    ne23 = norm(e23)
    ne31 = norm(e31)
    sortarray = [ne12, ne23, ne31]
    sortindices = np.argsort(sortarray)
    longnorm = sortarray[sortindices[-1]]
    midnorm = sortarray[sortindices[1]]
    longhat = edgearray[sortindices[-1]]/longnorm
    print(sortindices)
    w1 = [0,0]
    w2 = [longnorm,0]
    mid = edgearray[sortindices[1]]
    w3x = np.abs(np.dot(mid, longhat))
    w3y = np.sqrt(midnorm**2 - w3x**2)
    #w3y = norm( mid - w3x * longhat)
    w3 = [w3x,w3y]
    return [w1,w2,w3]

r1 = np.array([0,0,0])
r2 = np.array([10,0,0])
r3 = np.array([1,1,0])

print projectTriangle(r1,r2,r3)
