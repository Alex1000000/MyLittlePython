import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import sys
import pywavefront

def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    height = glutGet(GLUT_SCREEN_HEIGHT)
    width = glutGet(GLUT_SCREEN_WIDTH)
    gluPerspective(np.degrees(np.arctan2(1, 1)), width * 1e0 / height, 1e-3, 1e3)
    global eye
    global lat
    global lon
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    center = eye + (np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon))
    up = -np.sin(lat) * np.sin(lon), np.cos(lat), -np.sin(lat) * np.cos(lon)
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])
    glScale(8, 8, 8)
    glTranslate(0, -0.25, -4)
    glEnableClientState(GL_VERTEX_ARRAY)
    # glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_DOUBLE, 0, pointdata)
    # glNormalPointer(GL_DOUBLE, 0, normals)
    glColorPointer(3, GL_DOUBLE, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0,  len(pointdata))
    glDisableClientState(GL_COLOR_ARRAY)
    # glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glRotate(np.degrees(np.arctan2(-1, 0)), 1, 0, 0)
    # glTranslate(0, 0, 0.5)
    # glColor(1, 0, 0)
    # #    glutSolidCylinder(0.5, 1e-3, 60, 1)
    # glTranslate(0, 0, 1e-3)
    # glColor(1, 1, 0)
    # glutSolidCone(0.5, 0.5, 60, 60)
    glutSwapBuffers()


# def keyboard(key, x, y):
#     print("hey")
#     global eye
#     global lat
#     global lon
#     if key == 'W':
#         print("w")
#         eye += np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
#     elif key.upper() == 'S':
#         print("s")
#         eye -= np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
#     elif key.upper() == 'A':
#         print("a")
#         eye += np.cos(lon), 0, -np.sin(lon)
#     elif key.upper() == 'D':
#         print("d")
#         eye += -np.cos(lon), 0, np.sin(lon)
#     elif key == '\x1b':
#         print("no")
#         glutLeaveMainLoop()


def specialkeys(key, x, y):
    print("hey")
    global eye
    global lat
    global lon
    if key == GLUT_KEY_UP:
        print("w")
        eye += np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    elif key == GLUT_KEY_DOWN:
        print("s")
        eye -= np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    elif key == GLUT_KEY_LEFT:
        print("a")
        eye += np.cos(lon), 0, -np.sin(lon)
    elif key == GLUT_KEY_RIGHT:
        print("d")
        eye += -np.cos(lon), 0, np.sin(lon)
    elif key == '\x1b':
        print("no")
        glutLeaveMainLoop()



def motion(x, y):
    height = glutGet(GLUT_SCREEN_HEIGHT)
    width = glutGet(GLUT_SCREEN_WIDTH)
    center = width / 2, height / 2
    if (x, y) == center:
        return
    glutWarpPointer(int(center[0]), int(center[1]))
    global lat
    lat = min(max(lat - np.arcsin(y * 2e0 / height - 1), -np.arctan2(1, 0)), np.arctan2(1, 0))
    global lon
    lon -= np.arcsin(x * 2e0 / width - 1)


if __name__ == '__main__':
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE | GLUT_RGBA)
    glutEnterGameMode()
    glutDisplayFunc(display)
    glutIdleFunc(display)
    # glutKeyboardFunc(keyboard)
    glutSpecialFunc(specialkeys)
    glutMotionFunc(motion)
    glutPassiveMotionFunc(motion)
    glutSetCursor(GLUT_CURSOR_NONE)
    height = glutGet(GLUT_SCREEN_HEIGHT)
    width = glutGet(GLUT_SCREEN_WIDTH)
    glutWarpPointer(int(width / 2), int(height / 2))
    glClearColor(0.2, 0.2, 0.2, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    vertex = create_shader(GL_VERTEX_SHADER, """
        varying vec4 vertex_color;

        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            vertex_color = gl_Color;
        }
        """)
    fragment = create_shader(GL_FRAGMENT_SHADER, """
        varying vec4 vertex_color;

        void main() {
            gl_FragColor = vertex_color;
        }
        """)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glUseProgram(program)
    pointdata =[[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5],
                 [0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, 0, 0.5],
                 [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
                 [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                 [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0, 0.5],
                 [0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, -0.5, 0.5],
                 [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0, -0.5],
                 [0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, -0.5, -0.5],
                 [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0, -0.5],
                 [0.5, 0, -0.5], [0.5, 0, 0.5], [0.5, -0.5, 0.5],
                 [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, 0, -0.5],
                 [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, -0.5, 0.5]]
    pointdata = np.asarray(pointdata)/4
    # pointdata = [mySubList[2]+1/4 for mySubList in pointdata]
    pointcolor = [[1, 1, 0], [1, 1, 0], [1, 1, 0], #r-y
                  [1, 1, 0], [1, 1, 0], [1, 1, 0], #r-y
                  [1, 1, 0], [1, 1, 0], [1, 1, 0], #g-y
                  [1, 1, 0], [1, 1, 0], [1, 1, 0], #g-y
                  [1, 0, 0], [1, 0, 0], [1, 0, 0],#r
                  [1, 0, 0], [1, 0, 0], [1, 0, 0],#r
                  [0, 1, 0], [0, 1, 0], [0, 1, 0],#g
                  [0, 1, 0], [0, 1, 0], [0, 1, 0],#g
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0]]
    # myList = [[x / myInt for x in mySubList] for mySubList in myList]
    # myList = np.asarray(myList) / myInt

    normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0],
               [0, 1, 0], [0, 1, 0], [0, 1, 0],
               [0, -1, 0], [0, -1, 0], [0, -1, 0],
               [0, -1, 0], [0, -1, 0], [0, -1, 0],
               [0, 0, 1], [0, 0, 1], [0, 0, 1],
               [0, 0, 1], [0, 0, 1], [0, 0, 1],
               [0, 0, -1], [0, 0, -1], [0, 0, -1],
               [0, 0, -1], [0, 0, -1], [0, 0, -1],
               [1, 0, 0], [1, 0, 0], [1, 0, 0],
               [1, 0, 0], [1, 0, 0], [1, 0, 0],
               [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
               [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    # # glutSolidCone(0.4, 0.4, 20, 20)
    # print("len(pointdata)=", len(pointdata))
    # print("len(pointcolor)=", len(pointcolor))
    # for angle in range(361):
    #     x = math.sin(math.radians(angle))/6
    #     y = math.cos(math.radians(angle))/6
    #     # glColor3f(1, 1, 0)
    #     # glVertex2f(x, y)
    #     tmp=[[x, y,0.0]]
    #     pointdata=np.concatenate((pointdata, tmp), axis=0)
    #     # pointdata.append([x,y,0])
    #     pointcolor.append([1,1,0])
    # print("len(pointdata)=", len(pointdata))
    # print("len(pointcolor)=", len(pointcolor))
    # for angle in range(361):
    #     x = math.sin(math.radians(angle))/6
    #     y = math.cos(math.radians(angle))/6
    #     # glColor3f(1, 0, 0)
    #     # glVertex2f(x, y)
    #     pointdata=np.concatenate((pointdata, [[x,y,0.0]]))
    #     # pointdata.append([x,y,0])
    #     pointcolor.append([1,0,0])

    step=1
    size_con=10
    for i in range(360):
        # glColor3f(1.0,1.0,0.0)
        # masconecolorlist.append([1.0,1.0,0.0])
        # masconevertslist.append([0, 0, 1])
        pointcolor.append([1.0, 1.0, 0.0])
        pointdata = np.concatenate((pointdata, [[0, 0, 1]]))
        # pointdata.append([0, 0, 1])
        # glVertex3f( 0, 0, 1.0)
        # glColor3f(1.0,1.0,0.0)
        # masconecolorlist.append([1.0,1.0,0.0])
        # masconevertslist.append([math.sin((i*math.pi)/180), math.cos((i*math.pi)/180), 0])
        pointcolor.append([1.0, 1.0, 0.0])
        pointdata = np.concatenate((pointdata, [[math.sin((i * math.pi) / 180), math.cos((i * math.pi) / 180), 0]]))
        # glVertex3f(math.sin((i*math.pi)/180), math.cos((i*math.pi)/180), 0)
        # glColor3f(1.0,1.0,0.0)
        # masconecolorlist.append([1.0,1.0,0.0])
        # masconevertslist.append([math.sin(((i+step)*math.pi)/180), math.cos(((i+step)*math.pi)/180), 0])
        pointcolor.append([1.0, 1.0, 0.0])
        pointdata = np.concatenate((pointdata, [[math.sin(((i + step) * math.pi) / 180), math.cos(((i + step) * math.pi) / 180), 0]]))
        # glVertex3f(math.sin(((i+step)*math.pi)/180), math.cos(((i+step)*math.pi)/180), 0)
    # glEnd()

    # glBegin(GL_TRIANGLES)
    for i in range(360):
        # glColor3f(1.0,0.0,0.0)
        # glVertex3f( 0, 0, 0)
        pointcolor.append([1.0, 0.0, 0.0])
        pointdata = np.concatenate((pointdata, [[0, 0, 0]]))
        # glColor3f(1.0, 0.0, 0.0)
        # glVertex3f(math.cos((i*math.pi)/180), math.sin((i*math.pi)/180), 0)
        pointcolor.append([1.0, 0.0, 0.0])
        pointdata = np.concatenate((pointdata, [[math.cos((i * math.pi) / 180), math.sin((i * math.pi) / 180), 0]]))
        # glColor3f(1.0,0.0,0.0)
        # glVertex3f(math.cos(((i+step)*math.pi)/180), math.sin(((i+step)*math.pi)/180), 0)
        pointcolor.append([1.0, 0.0, 0.0])
        pointdata = np.concatenate((pointdata, [[math.cos(((i + step) * math.pi) / 180), math.sin(((i + step) * math.pi) / 180), 0]]))
    # glEnd()
    endOfCon=len(pointdata)
    for i in range(36,len(pointdata)):
        # print(pointdata[i])
        pointdata[i] = np.asarray(pointdata[i]) / size_con
        # pointdata[i]=pointdata/size_con
    for i in range(0,36):
        # print(pointdata[i])
        pointdata[i] = [pointdata[i][0],pointdata[i][1],pointdata[i][2]-1/4]



    name = 'earth.obj'
    meshes = pywavefront.Wavefront(name)
    ps = pywavefront.ObjParser(meshes, name)
    ps.read_file(name)
    pointdata2 = ps.material.vertices
    N = len(pointdata2) // 24
    pointdata_earth = np.zeros((N, 3, 3))
    pointcolor_earth = np.zeros((N * 3, 3))
    pointcolor_earth+=1
    for i in range(0, N):
        for j in range(0, 3):
            pointdata_earth[i, j, 0:3] = pointdata2[24 * i + 8 * j + 5:24 * i + 8 * j + 8]
    pointdata_earth /= pointdata_earth.max()
    pointdata_earth_new = np.zeros((N * 3, 3))
    pointdata_earth_new = np.reshape(pointdata_earth, (N * 3, 3))

    # for i in range(0, N):
    pointcolor_earth[:, :] = [0.0, 0.0, 1.0]
        # pointcolor_earth[3 * i + 1, :] = [0.0, 0.0, 1.0]
        # pointcolor_earth[3 * i + 2, :] = [0.0, 0.0, 1.0]

    pointdata = np.concatenate((pointdata, pointdata_earth_new))
    pointcolor=np.concatenate((pointcolor, pointcolor_earth))

    for i in range(endOfCon,len(pointdata)):
        # print(pointdata[i])
        pointdata[i] = np.asarray(pointdata[i]) / 10 -1/4

    global eye
    eye = np.zeros(3)
    global lat
    lat = 0
    global lon
    lon = np.arctan2(0, -1)
    glutMainLoop()