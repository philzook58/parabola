import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#size factor
L = 20
N=3
f=4.

#use coordinates for hexagonal grid (1,0), basis vectors
a = L*np.array([1.,0.])
b = L* np.array([1./2, np.sqrt(3)/2]);

offset = 10 + N*L

def hat(v):
    return v / np.linalg.norm(v)

def coords(i,j):
    return (offset+i*a[0]+j*b[0],offset+i*a[1]+j*b[1])

def arraycoords(i,j):
    return np.array([offset+i*a[0]+j*b[0],offset+i*a[1]+j*b[1]])


def z(xy):
 return 1./4/f * (xy[0]**2 + xy[1]**2)

def coords3(i,j):
    xy = coords(i,j)
    return np.array([xy[0],xy[1],z(xy)])

cross = np.cross

def normal(v1,v2,v3):
    e12 = v2 - v1
    e23 = v3 - v2
    e31 = v1 - v3
    normy = hat(cross(e12,e23))
    #make always point in the gernal up z direction
    if (normy[2]<0):
        normy = -1 * normy
    return normy

def normangle(n1,n2):
    return np.arccos(np.dot(n1,n2))

angles = []

def corners(i,j,LR):
    if(LR == 'L'):
        v1 = coords3(i,j)
        v2 = coords3(i+1,j)
        v3 = coords3(i,j+1)
        return [v1,v2,v3]
    if(LR == 'R'):
        v1 = coords3(i+1,j+1)
        v2 = coords3(i+1,j)
        v3 = coords3(i,j+1)
        return [v1,v2,v3]
# interior edge?

def plotedges():
    for i in range(-N,N):
        for j in range(-N+1,N):
            if(i+j+1 <= N and i+j >= -N):

                # suth edge borders i,j L, i-1, j-1 R
                triangle1 = corners(i,j,'L')
                triangle2 = corners(i-1,j-1,'R')
                #print triangle1[0]-triangle1[1]
                norm1 = normal(triangle1[0],triangle1[1],triangle1[2])

                norm2 = normal(triangle2[0],triangle2[1],triangle2[2])
                angles.append( (i,j,'S', normangle(norm1,norm2) ) )
                #dwg.add(dwg.line(start=coords(i,j), end=coords(i+1,j),stroke=stroke))

    for i in range(-N+1,N):
        for j in range(-N,N):
            if(i+j+1 <= N and i+j >= -N):

                # west edge borders i,j L, i-1, j R
                triangle1 = corners(i,j,'L')
                triangle2 = corners(i-1,j,'R')
                norm1 = normal(triangle1[0],triangle1[1],triangle1[2])

                norm2 = normal(triangle2[0],triangle2[1],triangle2[2])
                angles.append( (i,j,'W', normangle(norm1,norm2) ) )
                #dwg.add(dwg.line(start=coords(i,j), end=coords(i,j+1),stroke=stroke))

    for i in range(-N,N):
        for j in range(-N,N):
            if(i+j+1 <= N-1 and i+j+1 >= -N+1):
                #East border i,j L and i,j R
                triangle1 = corners(i,j,'L')
                triangle2 = corners(i,j,'R')
                norm1 = normal(triangle1[0],triangle1[1],triangle1[2])

                norm2 = normal(triangle2[0],triangle2[1],triangle2[2])
                angles.append( (i,j,'E', normangle(norm1,norm2) ) )
                #dwg.add(dwg.line(start=coords(i+1,j), end=coords(i,j+1),stroke=stroke))
plotedges()
print angles
