import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import svgwrite


materialthickness = 50.
notchwidth = materialthickness
notchlength = 100.



dwg = svgwrite.Drawing('test.svg', profile='tiny')

triangle = [
(0.,0.),
(1000.,0),
(900.,400.)
]

#order is long middle short
edges = [

(1,1,'E'),
(1,1,'S'),
(1,1,'W'),

]

i = -3
j = 4

trianglegroup = dwg.add(dwg.g(id='triangle' + str(i) + str(j), fill='red'))

midpoint = (triangle[1][0] + triangle[0][0])/2
label =  str(i) + ',' + str(j) + 'L'
ytextoffset =  triangle[2][1]/3
xtextoffset = -20
trianglegroup.add(dwg.text(label, insert=(midpoint + xtextoffset, ytextoffset+10), fill='red'))
trianglegroup.add(dwg.polygon(points=triangle, fill='none',stroke='black', stroke_width=1))


def degree(rad):
    return rad * 180. / np.pi

def midpoint(a,b):
    return ( a[0]/2+b[0]/2, a[1]/2+b[1]/2 )

#lower left angle
angle1 = np.arctan(triangle[2][1]/triangle[2][0])
angle1 = degree(angle1)
#lower right angle
angle2 = np.arctan(triangle[2][1]/(triangle[1][0] - triangle[2][0]))
angle2 = degree(angle2)
print angle2



mid = [0.,0.,0.]

for r in range(3):
    mid[r] = midpoint(triangle[r-1], triangle[r])


def edgelabel(angle):
    return  '(' + str(angle[0]) + ','+ str(angle[1]) + ") " +  str(angle[2])

#LONG
myrect = trianglegroup.add(dwg.text(edgelabel(edges[0]), insert=(-notchwidth/2.,0), fill='red'))
myrect.translate((mid[1][0] + notchwidth,mid[1][1] + notchlength * .2))
myrect.rotate(180.)

#SHORT LABEL
myrect = trianglegroup.add(dwg.text(edgelabel(edges[2]),  insert=(-notchwidth/2.,0), fill='red'))
myrect.translate((mid[2][0]  + notchwidth ,mid[2][1]  - notchlength * 1.4 ))
myrect.rotate(-angle2)

#MID Label
myrect = trianglegroup.add(dwg.text(edgelabel(edges[1]), insert=(-notchwidth/2.,0), fill='red'))
myrect.translate((mid[0][0] +1.2 * notchwidth , mid[0][1]))
myrect.rotate(angle1)

myrect = trianglegroup.add(dwg.rect(insert=(-notchwidth/2.,0), fill ='none', stroke='black', size=(notchwidth,notchlength)))
myrect.translate((mid[1][0],mid[1][1]))

myrect = trianglegroup.add(dwg.rect(insert=(-notchwidth/2.,0),fill ='none', stroke='black', size=(notchwidth,notchlength)))
myrect.translate((mid[2][0],mid[2][1]))
myrect.rotate(-angle2+180)

myrect = trianglegroup.add(dwg.rect(insert=(-notchwidth/2.,0),fill ='none', stroke='black', size=(notchwidth,notchlength)))
myrect.translate((mid[0][0],mid[0][1]))
myrect.rotate(angle1+180)


dwg.save()
