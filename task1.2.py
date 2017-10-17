import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp





EPSILON = 1e-6

def surface(x0,y0,z0,x1,y1,z1,x2,y2,z2,x,y,z):
    a11=x-x0
    a12=x1-x0
    a13=x2-x0
    a21=y-y0
    a22=y1-y0
    a23=y2-y0
    a31=z-z0
    a32=z1-z0
    a33=z2-z0
    return a11*a22*a33+a12*a23*a31+a13*a21*a32-a13*a22*a31-a11*a23*a32-a12*a21*a33

def pointOfInterceptionSurfaceAndLine(a,b,c,d,e):
    U = np.cross(b - a, c - a)
    K = -U.dot(a)
    V = (U.dot(d) + K) / U.dot(d - e)
    point = d + V * (e - d)
    return point

def triangleArea(a, b, c):
    return np.linalg.norm(np.cross(b - a, c - a)) / 2

def isPointInsideTriangle(a, b, c, m):
    sq=triangleArea(a, b, m) + triangleArea(b, c, m) + triangleArea(c, a, m) - triangleArea(a, b, c)
    return  sq< EPSILON

def intersectTwoLines(a, b, c, d):
    V1=b-a
    V2=d-c
    P=c-a
    if np.linalg.norm(np.cross(V1, V2)) > EPSILON:
        t = np.linalg.norm(np.cross(P, V2)) / np.linalg.norm(np.cross(V1, V2))
        if abs(t)<1:
            return True
        else:
            return False
    return False

def isIntersection(coord1, coord2):#A, B, C, A2, B2, C2):
    print("tri=",coord1,coord2)
    vtx = coord1#sp.rand(3,3)
    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color(colors.rgb2hex(sp.rand(3)))
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)

    vtx2 = coord2#sp.rand(3,3)
    tri2 = a3.art3d.Poly3DCollection([vtx2,vtx])
    tri2.set_color(colors.rgb2hex(sp.rand(3)))
    tri2.set_edgecolor('k')
    ax.add_collection3d(tri2)
    A=coord1[0]
    B=coord1[1]
    C=coord1[2]
    A2=coord2[0]
    B2=coord2[1]
    C2=coord2[2]
    point_A2=surface(A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2], A2[0], A2[1], A2[2])
    point_B2 = surface(A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2], B2[0], B2[1], B2[2])
    point_C2 = surface(A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2], C2[0], C2[1], C2[2])
    print(point_A2, point_B2, point_C2)
    if point_A2*point_B2*point_C2==0:
        if 0 == point_A2 and 0 == point_B2 and 0 == point_C2:
            if (isPointInsideTriangle(A,B,C,A2) or isPointInsideTriangle(A,B,C,B2) or isPointInsideTriangle(A,B,C,C2)):
                return True
            else:
                return intersectTwoLines(A,B,A2,C2) or intersectTwoLines(A,C,A2,C2) or intersectTwoLines(B,C,A2,C2) or intersectTwoLines(A,B,A2,B2) or intersectTwoLines(A,C,A2,B2) or intersectTwoLines(B,C,A2,B2) or intersectTwoLines(A,B,B2,C2) or intersectTwoLines(A,C,B2,C2) or intersectTwoLines(B,C,B2,C2)
        elif 0 == point_A2 and 0 == point_C2:
            if isPointInsideTriangle(A, B, C, A2) or isPointInsideTriangle(A, B, C, C2):
                return True
            else:
                return intersectTwoLines(A,B,A2,C2) or intersectTwoLines(A,C,A2,C2) or intersectTwoLines(B,C,A2,C2)
        elif point_A2==0 and point_B2==0:
            if (isPointInsideTriangle(A, B, C, A2) or isPointInsideTriangle(A, B, C, B2)):
                return True
            else:
                return intersectTwoLines(A,B,A2,B2) or intersectTwoLines(A,C,A2,B2) or intersectTwoLines(B,C,A2,B2)
        elif (point_B2==0 and point_C2==0):
            if (isPointInsideTriangle(A, B, C, B2) or isPointInsideTriangle(A, B, C, C2)):
                return True
            else:
                return intersectTwoLines(A,B,B2,C2) or intersectTwoLines(A,C,B2,C2) or intersectTwoLines(B,C,B2,C2)
        elif point_A2==0:
            if (isPointInsideTriangle(A, B, C, A2) or (point_B2*point_C2>0)):
                return True
            else:
                intersectionPoint = pointOfInterceptionSurfaceAndLine(A, B, C, B2, C2)
                return isPointInsideTriangle(A, B, C, intersectionPoint) or intersectTwoLines(A,B,intersectionPoint,A2) or intersectTwoLines(A,C,intersectionPoint,A2) or intersectTwoLines(B,C,intersectionPoint,A2)
        elif point_B2==0:
            if (isPointInsideTriangle(A, B, C, B2) or (point_A2*point_C2>0)):
                return True
            else:
                intersectionPoint = pointOfInterceptionSurfaceAndLine(A, B, C, A2, C2)
                return isPointInsideTriangle(A, B, C, intersectionPoint) or intersectTwoLines(A,B,intersectionPoint,B2) or intersectTwoLines(A,C,intersectionPoint,B2) or intersectTwoLines(B,C,intersectionPoint,B2)
        else:
            if (isPointInsideTriangle(A, B, C, C2) or (point_B2*point_A2>0)):
                return True
            else:
                intersectionPoint = pointOfInterceptionSurfaceAndLine(A, B, C, B2, A2)
                return isPointInsideTriangle(A, B, C, intersectionPoint) or intersectTwoLines(A,B,intersectionPoint,C2) or intersectTwoLines(A,C,intersectionPoint,C2) or intersectTwoLines(B,C,intersectionPoint,C2)
    elif (point_A2*point_B2>0 and point_A2*point_C2>0):
        return False
    elif (point_A2*point_B2<0 and point_A2*point_C2<0):
        #A2-B2,C2
        intersectionPoint1=pointOfInterceptionSurfaceAndLine(A,B,C,A2,B2)
        intersectionPoint2 = pointOfInterceptionSurfaceAndLine(A, B, C, A2, C2)
        return isPointInsideTriangle(A, B, C, intersectionPoint1) or isPointInsideTriangle(A,B,C,intersectionPoint2)
    elif (point_B2*point_A2<0 and point_B2*point_C2<0):
        # B2-A2,C2
        intersectionPoint1=pointOfInterceptionSurfaceAndLine(A,B,C,B2,A2)
        intersectionPoint2 = pointOfInterceptionSurfaceAndLine(A, B, C, B2, C2)
        return isPointInsideTriangle(A, B, C, intersectionPoint1) or isPointInsideTriangle(A,B,C,intersectionPoint2)
    elif (point_C2*point_A2<0 and point_B2*point_C2<0):
        # C2-A2,B2
        intersectionPoint1=pointOfInterceptionSurfaceAndLine(A,B,C,C2,A2)
        intersectionPoint2 = pointOfInterceptionSurfaceAndLine(A, B, C, C2, B2)
        return isPointInsideTriangle(A, B, C, intersectionPoint1) or isPointInsideTriangle(A,B,C,intersectionPoint2)
    else:
        return False

#false
A=[0,0,0]
B=[0,0,5]
C=[5,0,5]
A2=[0,5,0]
B2=[0,5,5]
C2=[5,5,5]
#print(isIntersection(np.asarray(A),np.asarray(B),np.asarray(C),np.asarray(A2),np.asarray(B2),np.asarray(C2)) )


# print(isIntersection([[0,0,0],[0,5,0],[6,5,0]],[[1,4,0],[2,4,0],[2,3,0]]))
#
# print(isIntersection([[-1,0,0],[0,-1,0],[0,0,0]],[[0,0,0],[0,3,0],[5,0,0]]))

coord1=[[0,0,0],[0,5,0],[6,5,0]]
coord2=[[1,4,0],[2,4,0],[2,3,0]]

all_Coordinates= [[coord1,coord2]]
coord1=[[-1,0,0],[0,-1,0],[0,0,0]]
coord2=[[0,0,0],[0,3,0],[5,0,0]]
all_Coordinates.append([coord1,coord2])
coord1=[[-1,0,0],[0,2,0],[0,0,0]]
coord2=[[0,0,0],[5,0,0],[0,4,0]]
all_Coordinates.append([coord1,coord2])
p1 = [0, 0,-10]
p2 = [0, 2, -10]
p3 = [1, 0, -10]
tr1 = [p1, p2, p3]
q1 = [0, -1, 0]
q2 = [0, 3, 0]
q3 = [7, -1, 0]
tr2 = [q1, q2, q3]
all_Coordinates.append([tr1,tr2])

p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [1, 2, 0]
q2 = [1, 1, -3]
q3 = [0.5, 2, -2]
tr2 = [q1, q2, q3]
all_Coordinates.append([tr1,tr2])


p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [4, 0, 0]
q2 = [1, 1, -3]
q3 = [0.5, 2, -2]
tr2 = [q1, q2, q3]
all_Coordinates.append([tr1,tr2])

p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [1, 2, 0]
q2 = [2, 1, 0]
q3 = [0.5, 2, -2]
tr2 = [q1, q2, q3]
all_Coordinates.append([tr1,tr2])


p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [-1, 2, 2]
q2 = [0, 2, 2]
q3 = [0, 0, -2]
tr2 = [q1, q2, q3]
all_Coordinates.append([tr1,tr2])
# all_Coordinates=[]
p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [5, 0, 0]
tr1 = [p1, p2, p3]
q1 = [1, 1, 2]
q2 = [5, 6, -2]
q3 = [3, -4, -1]
tr2 = [q1, q2, q3]
all_Coordinates.append([tr1,tr2])



print(all_Coordinates)
# print(isIntersection(np.asarray(all_Coordinates[0][0]), np.asarray(all_Coordinates[0][1])))
# coord1=[[0,0,0],[0,0,5],[5,0,5]]
# coord2=[[0,5,0],[0,5,5],[5,5,5]]
# print(isIntersection(coord1, coord2))
#
# coord1=[[0,0,0],[0,0,5],[5,0,5]]
# coord2=[[0,5,0],[0,5,5],[5,5,5]]
# print(isIntersection(coord1, coord2))
for i in range(0,len(all_Coordinates)):
    ax = a3.Axes3D(pl.figure())
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    print(i)
    print(isIntersection(np.asarray(all_Coordinates[i][0]), np.asarray(all_Coordinates[i][1])) or isIntersection(np.asarray(all_Coordinates[i][1]), np.asarray(all_Coordinates[i][0])))
    pl.show()
    inputInt = int()
    # input()
    # continue
# pl.show()