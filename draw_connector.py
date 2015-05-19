import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import svgwrite


dwg = svgwrite.Drawing('test.svg', profile='full')


materialthickness = 25.
notchwidth = materialthickness
notchlength = 100.
connectorheight = 100.
halfheight = connectorheight/2.

angles = [
(1,2,'S', 0.1 ),
(1,-2,'E', 0.3 ),
(-1,2,'W', 0.5 ),
(1,-2,'S', 0.1 )
]

def degree(rad):
    return rad * 180. / np.pi

def midpoint(a,b):
    return ( a[0]/2+b[0]/2, a[1]/2+b[1]/2 )

def grabangle(obj):
    return obj[3]

def sortangles():
    return sorted(angles, key=grabangle)

angles = sortangles()
print angles

currenty = 0.
for angle in angles:
    #angle = angles[0]
    # if you are reading this welcome to hell.
    grouplabel =  str(angle[0]) + str(angle[1])  +  str(angle[2])
    group = dwg.add(dwg.g(id=grouplabel, fill='red'))
    myangle = angle[3]
    theta = myangle
    alpha = (np.pi - myangle)/2.
    midlength = 2 * notchlength + notchwidth / 2. * np.tan(theta/2)
    topx = midlength -  halfheight / np.tan(alpha)
    bottomx = midlength + halfheight / np.tan(alpha)


    a = (np.cos(myangle), np.sin(myangle))
    perp = (a[1],-a[0])

    topmid = ((a[0]+ 1) * midlength , a[1] * midlength)


    mypath = [
    (topx, halfheight),
    (0, halfheight),
    (0,-halfheight),
    (bottomx, -halfheight),
    (topmid[0] + perp[0] * halfheight, topmid[1] + perp[1] * halfheight),
    (topmid[0] - perp[0] * halfheight, topmid[1] - perp[1] * halfheight),


    ]

    piece = group.add(dwg.polygon(points=mypath, fill='none',stroke='black', stroke_width=1))
    piece.translate((0,currenty))

    myrect = group.add(dwg.rect(insert=(-notchwidth/2.,0),fill ='none', stroke='black', size=(notchwidth,notchlength)))
    #myrect.translate((mid[0][0],mid[0][1]))
    myrect.translate((0,currenty))
    myrect.rotate(270)

    myrect = group.add(dwg.rect(insert=(-notchwidth/2.,0),fill ='none', stroke='black', size=(notchwidth,notchlength)))
    #myrect.translate((mid[0][0],mid[0][1]))
    myrect.translate((0,currenty))
    myrect.translate(topmid)
    myrect.rotate(90 + degree(myangle))


    prettylabel =  '(' + str(angle[0]) + ','+ str(angle[1]) + ") " +  str(angle[2])

    label = group.add(dwg.text(prettylabel, insert=(0,0), fill='red'))
    label.translate( ( notchlength +5,currenty+5))
    label.rotate(myangle/2)

    currenty = currenty + connectorheight + 10



dwg.save()
