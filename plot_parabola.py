
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#size factor
L = 1.
N=3
totalsize = L * 2 * N

#focus
f = totalsize/3 #4.
#use coordinates for hexagonal grid (1,0), basis vectors
a = L*np.array([1.,0.])
b = L* np.array([1./2, np.sqrt(3)/2]);

offset = 0#10 + N*L


x =[]
y = []
z = []
vertices = []
vertnum = 0


def coords(i,j):
    return (offset+i*a[0]+j*b[0],offset+i*a[1]+j*b[1])


def generateVertices():
    vertnum = 0
    for i in range(-N,N+1):
        for j in range(-N,N+1):
            if(np.abs(i+j) <= N):
                xy = coords(i,j)
                x.append(xy[0])
                y.append(xy[1])
                myz = 1./4/f * (xy[0]**2 + xy[1]**2)
                z.append(myz)
                vertices.append([xy[0],xy[1],myz])
                vertnum = vertnum+1
    return vertnum

print(str(generateVertices()) + " Vertices" )
#print z
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.auto_scale_xyz([0, 4], [0, 4], [0, 7])
ax.set_zlim(0, totalsize)
ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

plt.show()
