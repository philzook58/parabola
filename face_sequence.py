import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import svgwrite

norm = np.linalg.norm

#size factor
L = 40.
N=2


f=4. #* L
#use coordinates for hexagonal grid (1,0), basis vectors
a = L* np.array([1.,0.])
b = L* np.array([1./2, np.sqrt(3)/2]);

offset = 10 + N*L


x =[]
y = []

vertices = []



dwg = svgwrite.Drawing('test.svg', profile='full')



def z(xy):
 return 1./4/f * (xy[0]**2 + xy[1]**2)
 #return 0


def coords(i,j):
    return (offset+i*a[0]+j*b[0],offset+i*a[1]+j*b[1])
yoffset = 100

def projectTriangle(v1,v2,v3, xoffset):
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
    #print(sortindices)
    w1 = (xoffset,yoffset)
    w2 = (longnorm + xoffset,yoffset)
    mid = edgearray[sortindices[1]]
    w3x = np.abs(np.dot(mid, longhat))
    w3y = np.sqrt(midnorm**2 - w3x**2)
    #w3y = norm( mid - w3x * longhat)
    w3 = (w3x + xoffset,w3y + yoffset)
    return [w1,w2,w3]


stroke=svgwrite.rgb(0, 10, 1, '%')
#stroke = stroke('black', width=5)

def notches(triangle):
    #midpoints

    #draw notches as rect
    dwg.add(dwg.rect(insert=(x,y), size=(width,height), rotate=angle))


def drawface(triangle, i, j, LR, ):
    print yo

def plotfaces():
    xoffset = 0.
    for i in range(-N,N):
        for j in range(-N,N):
            if(i+j+1 <= N and i+j >= -N):
                #Left face
                a1 = coords(i+1,j)
                a2 = coords(i,j+1)
                a3 = coords(i,j)

                v1 = np.array([a1[0],a1[1], z(a1)])
                v2 = np.array([a2[0],a2[1], z(a2)])
                v3 = np.array([a3[0],a3[1], z(a3)])

                triangle = projectTriangle(v1,v2,v3, xoffset)

                xoffset = triangle[1][0]
                midpoint = (triangle[1][0] + triangle[0][0])/2
                label =  str(i) + ',' + str(j) + 'L'
                dwg.add(dwg.text(label, insert=(midpoint, yoffset+10), fill='red'))
                dwg.add(dwg.polygon(points=triangle, fill='none',stroke='black', stroke_width=1))


    for i in range(-N,N):
        for j in range(-N,N):
            if(i+j+2 <= N and i+j+1 >= -N):
                a1 = coords(i+1,j)
                a2 = coords(i,j+1)
                a3 = coords(i+1,j+1)

                v1 = np.array([a1[0],a1[1], z(a1)])
                v2 = np.array([a2[0],a2[1], z(a2)])
                v3 = np.array([a3[0],a3[1], z(a3)])

                triangle = projectTriangle(v1,v2,v3, xoffset)

                xoffset = triangle[1][0]
                dwg.add(dwg.polygon(points=triangle,stroke=stroke))


plotfaces()

myrect = dwg.add(dwg.rect(insert=(10,10), size=(100,100)))
myrect.rotate(30)
dwg.save()
