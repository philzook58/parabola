import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import svgwrite


#L = 225.  #panel size in mm.
L = 80.
N = 2      #Panel number radius of parabola
materialthickness = 4.
#connectorheight = 50.
connectorheight = 25.
cutwidth = 1.   #Should be set to 0.01mm for ponoko, but then I can't see. Do in inkscape afertwards
fontsize = 6


totalsize = L * 2 * N #approximate total diameter

print 'Diameter: ' + str(totalsize) + 'mm'
# ponoko cardcoard is 4mm


notchlength = L/6
notchwidth = materialthickness


halfheight = connectorheight/2.

f= totalsize / 3 #* L
#use coordinates for hexagonal grid (1,0), basis vectors
a = L* np.array([1.,0.])
b = L* np.array([1./2, np.sqrt(3)/2]);

offset = 0 #10 + N*L


norm = np.linalg.norm


cutstroke =svgwrite.rgb(0, 0, 255)
#should be 0.01 for ponoko but then I cna't see anything

textstroke = svgwrite.rgb(246,146,30)




dwg = svgwrite.Drawing('faces.svg', profile='full', size=('790mm', '384mm'),style="font-size:"+str(fontsize)+";", viewBox="0 0 790 384")
dwg.add(dwg.rect(insert=(0,0),fill ='none', stroke=cutstroke, stroke_width=cutwidth, size=(790,384)))



yoffset = 100

def z(xy):
 return 1./4/f * (xy[0]**2 + xy[1]**2)
 #return 0


def coords(i,j):
    return (offset+i*a[0]+j*b[0],offset+i*a[1]+j*b[1])


def projectTriangle(v1,v2,v3, xoffset=0):
    #edges eab points from a to b. Try adding eab + va to see
    e12 = v2 - v1
    e23 = v3 - v2
    e31 = v1 - v3
    edgearray = [e12,e23,e31]
    ne12 = norm(e12)
    ne23 = norm(e23)
    ne31 = norm(e31)
    #sortarray = [ne12, ne23, ne31]
    #sortindices = np.argsort(sortarray)
    #longnorm = sortarray[sortindices[-1]]
    #midnorm = sortarray[sortindices[1]]
    #longhat = edgearray[sortindices[-1]]/longnorm
    #print(sortindices)
    w1 = (xoffset,0)
    w2 = (ne12 + xoffset,0)
    #mid = edgearray[sortindices[1]]
    w3x = np.abs(np.dot(e12, e31))/ne12
    w3y = np.sqrt(ne31**2 - w3x**2)
    #w3y = norm( mid - w3x * longhat)
    w3 = (w3x + xoffset,w3y)
    return [w1,w2,w3]



def edgelabel(angle):
    return  '(' + str(angle[0]) + ','+ str(angle[1]) + ") " +  str(angle[2])

def degree(rad):
    return rad * 180. / np.pi

def midpoint(a,b):
    return ( a[0]/2+b[0]/2, a[1]/2+b[1]/2 )

def tuplehat(q):
    v = np.array([q[0],q[1]])
    return v / np.linalg.norm(v)

trianglecount = 0

def drawTriangle(triangle, edges, i , j, xoffset,LR):
    trianglegroup = dwg.add(dwg.g(id='triangle' + str(i) + str(j)))

    midpoint2 = (triangle[1][0] + triangle[0][0])/2
    label =  str(i) + ',' + str(j) + LR
    ytextoffset =  triangle[2][1]/3
    xtextoffset = -20
    trianglegroup.add(dwg.text(label, insert=(midpoint2 + xtextoffset, ytextoffset+10), stroke=textstroke, fill='none'))
    #trianglegroup.add(dwg.polygon(points=triangle, fill='none',stroke=cutstroke, stroke_width=1))

    s = (triangle[2][0] - triangle[1][0], triangle[2][1] - triangle[1][1])
    s = tuplehat(s)
    shat = (s[0], s[1])
    sperp = (shat[1],-shat[0])
    mhat = (tuplehat(triangle[2])[0], tuplehat(triangle[2])[1])
    mperp = (mhat[1],-mhat[0])
    lhat = (1,0)
    lperp = (0,1)


    #lower left angle
    angle1 = np.arctan(triangle[2][1]/triangle[2][0])
    angle1 = degree(angle1)
    #lower right angle
    angle2 = np.arctan(triangle[2][1]/(triangle[1][0] - triangle[2][0]))
    angle2 = degree(angle2)



    mid = [0.,0.,0.]


    for r in range(3):
        mid[r] = midpoint(triangle[r-1], triangle[r])


    fullx = triangle[1][0]
    mypath = [
    (0,0),

    (mid[1][0]-notchwidth/2, 0),
    (mid[1][0]-notchwidth/2, notchlength),
    (mid[1][0]+notchwidth/2, notchlength),
    (mid[1][0]+notchwidth/2, 0),

    (fullx , 0),

    (mid[2][0] - shat[0]* notchwidth/2, mid[2][1] - shat[1]* notchwidth/2),
    (mid[2][0] - shat[0]* notchwidth/2 - sperp[0] * notchlength, mid[2][1] - shat[1]* notchwidth/2 - sperp[1] * notchlength),
    (mid[2][0] + shat[0]* notchwidth/2 - sperp[0] * notchlength, mid[2][1] + shat[1]* notchwidth/2 - sperp[1] * notchlength),
    (mid[2][0] + shat[0]* notchwidth/2, mid[2][1] + shat[1]* notchwidth/2),

    (triangle[2][0], triangle[2][1]),

    (mid[0][0] + mhat[0]* notchwidth/2, mid[0][1] + mhat[1]* notchwidth/2),
    (mid[0][0] + mhat[0]* notchwidth/2 + mperp[0] * notchlength, mid[0][1] + mhat[1]* notchwidth/2 + mperp[1] * notchlength),
    (mid[0][0] - mhat[0]* notchwidth/2 + mperp[0] * notchlength, mid[0][1] - mhat[1]* notchwidth/2 + mperp[1] * notchlength),
    (mid[0][0] - mhat[0]* notchwidth/2, mid[0][1] - mhat[1]* notchwidth/2),



    ]

    piece = trianglegroup.add(dwg.polygon(points=mypath, fill='none',stroke=cutstroke, stroke_width=cutwidth))
    #piece.translate((0,currenty))



    #LONG is now e12
    myrect = trianglegroup.add(dwg.text(edgelabel(edges[0]), insert=(-notchwidth/2.,0), stroke=textstroke , fill='none'))
    myrect.translate((mid[1][0] + notchwidth,mid[1][1] + notchlength * .2))
    myrect.rotate(180.)

    #SHORT LABEL e23
    myrect = trianglegroup.add(dwg.text(edgelabel(edges[1]),  insert=(-notchwidth/2.,0), stroke=textstroke , fill='none'))
    myrect.translate((mid[2][0]  + notchwidth ,mid[2][1]  - notchlength * .7 ))
    myrect.rotate(-angle2)

    #MID Label e13
    myrect = trianglegroup.add(dwg.text(edgelabel(edges[2]), insert=(-notchwidth/2.,0), stroke=textstroke , fill='none'))
    myrect.translate((mid[0][0] +1.2 * notchwidth , mid[0][1] ))
    myrect.rotate(angle1)


    """
    myrect = trianglegroup.add(dwg.rect(insert=(-notchwidth/2.,0), fill ='none', stroke='black', size=(notchwidth,notchlength)))
    myrect.translate((mid[1][0],mid[1][1]))

    myrect = trianglegroup.add(dwg.rect(insert=(-notchwidth/2.,0),fill ='none', stroke='black', size=(notchwidth,notchlength)))
    myrect.translate((mid[2][0],mid[2][1]))
    myrect.rotate(-angle2+180)

    myrect = trianglegroup.add(dwg.rect(insert=(-notchwidth/2.,0),fill ='none', stroke='black', size=(notchwidth,notchlength)))
    myrect.translate((mid[0][0],mid[0][1]))
    myrect.rotate(angle1+180)
    """
    trianglegroup.translate((xoffset,20))

    #temp = trianglecount
    #trianglecount = trianglecount + 1



def plotfaces():
    xoffset = 0.
    trianglecount = 0
    for i in range(-N,N):
        for j in range(-N,N):
            if(i+j+1 <= N and i+j >= -N):
                #Left face
                a1 = coords(i,j)
                a2 = coords(i+1,j)
                a3 = coords(i,j+1)


                v1 = np.array([a1[0],a1[1], z(a1)])
                v2 = np.array([a2[0],a2[1], z(a2)])
                v3 = np.array([a3[0],a3[1], z(a3)])

                temp = projectTriangle(v1,v2,v3)
                triangle = temp
                #edgeorder = temp[1]
                tempedges = [
                (i,j,'S' ), #e12
                (i,j,'E' ),        #e23
                (i,j,'W' ),        #e13

                ]

                edges = tempedges
                #for q in range(3):
                #    edges[2-q] = tempedges[edgeorder[q]]

                drawTriangle(triangle,edges,i,j,xoffset,'L')

                xoffset = xoffset + triangle[1][0] + 10

                trianglecount = trianglecount + 1

#Right facw

    for i in range(-N,N):
        for j in range(-N,N):
            if(i+j+2 <= N and i+j+1 >= -N):
                a1 = coords(i+1,j)
                a2 = coords(i+1,j+1)
                a3 = coords(i,j+1)

                v1 = np.array([a1[0],a1[1], z(a1)])
                v2 = np.array([a2[0],a2[1], z(a2)])
                v3 = np.array([a3[0],a3[1], z(a3)])

                temp = projectTriangle(v1,v2,v3)
                triangle = temp
                #edgeorder = temp[1]
                tempedges = [
                (i+1,j,'W' ), #e12
                (i,j+1,'S' ),        #e23
                (i,j,'E' ),        #e13

                ]

                edges = tempedges
                #for q in range(3):
                #    edges[2-q] = tempedges[edgeorder[q]]

                drawTriangle(triangle,edges,i,j,xoffset,'R')

                xoffset = xoffset + triangle[1][0] + 10

                trianglecount = trianglecount + 1
    print str(trianglecount) + " faces made"

plotfaces()


dwg.save()


#connector section

dwg = svgwrite.Drawing('connectors.svg', profile='full', size=('791mm', '384mm'),style="font-size:"+str(fontsize)+";")
dwg.add(dwg.rect(insert=(0,0),fill ='none', stroke=cutstroke,stroke_width=cutwidth, size=(791,384)))

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

edgecount = 0

def plotedges():
    edgecount = 0
    for i in range(-N,N):
        for j in range(-N+1,N):
            if(i+j+1 <= N and i+j >= -N):

                # suth edge borders i,j L, i-1, j-1 R
                triangle1 = corners(i,j,'L')
                triangle2 = corners(i,j-1,'R')
                #print triangle1[0]-triangle1[1]
                norm1 = normal(triangle1[0],triangle1[1],triangle1[2])

                norm2 = normal(triangle2[0],triangle2[1],triangle2[2])
                angles.append( (i,j,'S', normangle(norm1,norm2) ) )
                #dwg.add(dwg.line(start=coords(i,j), end=coords(i+1,j),stroke=stroke))
                edgecount = edgecount + 1

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




def grabangle(obj):
    return obj[3]

def sortangles():
    return sorted(angles, key=grabangle)
print angles
angles = sortangles()
#print angles

def drawConnectors():
    currenty = 0.
    for angle in angles:

        #angle = angles[0]
        # if you are reading this welcome to hell.
        grouplabel =  str(angle[0]) + str(angle[1])  +  str(angle[2])
        group = dwg.add(dwg.g(id=grouplabel))
        myangle = angle[3]
        theta = myangle
        alpha = (np.pi - myangle)/2.
        midlength = 2 * notchlength + notchwidth / 2. * np.tan(theta/2)
        topx = midlength -  halfheight / np.tan(alpha)
        bottomx = midlength + halfheight / np.tan(alpha)


        a = (np.cos(myangle), np.sin(myangle))
        perp = (a[1],-a[0])

        topmid = ((a[0]+ 1) * midlength , a[1] * midlength)

        chopout = (halfheight - notchwidth/2)/3

        mypath = [
        (topx, halfheight),


        (chopout,halfheight),
        (0,halfheight-chopout),

        #(0, halfheight),
        (0, notchwidth/2),
        (notchlength,notchwidth/2),
        (notchlength,-notchwidth/2),
        (0, -notchwidth/2),

        (0,-halfheight+chopout),
        (chopout,-halfheight),

        #(0,-halfheight),
        (bottomx, -halfheight),

        (topmid[0] + perp[0] * halfheight - a[0]*chopout, topmid[1] + perp[1] * halfheight - a[1]*chopout),
        (topmid[0] + perp[0] * (halfheight-chopout), topmid[1] + perp[1] * (halfheight-chopout)),


        (topmid[0] + perp[0] * notchwidth/2, topmid[1] + perp[1] * notchwidth/2),

        (topmid[0] + perp[0] * notchwidth/2 - a[0]*notchlength, topmid[1] + perp[1] * notchwidth/2 - a[1]*notchlength),
        (topmid[0] - perp[0] * notchwidth/2 - a[0]*notchlength, topmid[1] - perp[1] * notchwidth/2 - a[1]*notchlength),

        (topmid[0] - perp[0] * notchwidth/2, topmid[1] - perp[1] * notchwidth/2 ),

        (topmid[0] - perp[0] * (halfheight-chopout), topmid[1] - perp[1] * (halfheight-chopout)),

        (topmid[0] - perp[0] * halfheight - a[0]*chopout, topmid[1] - perp[1] * halfheight - a[1]*chopout),


        ]

        piece = group.add(dwg.polygon(points=mypath, fill='none',stroke=cutstroke, stroke_width=cutwidth))
        piece.translate((0,currenty))

        prettylabel =  '(' + str(angle[0]) + ','+ str(angle[1]) + ") " +  str(angle[2])

        label = group.add(dwg.text(prettylabel, insert=(0,0), stroke=textstroke , fill='none'))
        label.translate( ( notchlength +5,currenty+5))
        label.rotate(myangle/2)

        currenty = currenty + connectorheight +3

        #edgecount = edgecount +1

drawConnectors()
print str(len(angles)) + ' connectors'

print 'N = ' + str(N)

facenum = 12 * N * N / 2
print 'expected faces:' + str(facenum)

#we don't get connectors for exterior edges.
print 'expected edges:' + str(facenum +6* N *(N+1)/2 - 6 *N)

dwg.save()
