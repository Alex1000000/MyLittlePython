import math
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from random import random
import numpy as np
import random
import pywavefront

from OpenGL.raw.GLU import gluLookAt

class ObjParser2(object):
    def __init__(self):
        self.vertices = []
        self.indices = []

    def read_file(self, file_name):
        for line in open(file_name, 'r'):
            self.parse(line)

    def parse(self, line):
        entries = line.split()
        if entries[0] == "v":
            self.vertices += list(map(float, entries[1:4]))
        elif entries[0] == "f":
            quad = list(map(int, map(lambda s: s.split('/')[0], entries[1:])))
            if len(quad) == 3:
                self.indices += [quad[0]-1, quad[1]-1, quad[2]-1]
            elif len(quad) == 4:
                self.indices += [quad[0]-1, quad[1]-1, quad[2]-1, quad[2]-1, quad[3]-1, quad[0]-1]

def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader


def draw():
    global eye
    global lat
    global lon
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glClear(GL_COLOR_BUFFER_BIT)
    # glLoadIdentity()
    center = eye + (np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon))
    up = -np.sin(lat) * np.sin(lon), np.cos(lat), -np.sin(lat) * np.cos(lon)
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])
    glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    # glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    # glNormalPointer(GL_DOUBLE, 0, normals)
    # glDisableClientState(GL_COLOR_ARRAY)
    # glDisableClientState(GL_NORMAL_ARRAY)
    # glDisableClientState(GL_VERTEX_ARRAY)
    glDrawArrays(GL_TRIANGLES, 0,50580)#10262)#2196)
    # glLoadIdentity()
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    # glTranslatef(0.0, 0.0, 0.003)  # Сдвинемся по оси Z на -0.3
    glColor(1, 1, 0)
    # glutSolidCone(0.5, 0.5, 60, 60)

    # # cone's pinnacle
    # # glBegin(GL_TRIANGLE_FAN)
    # glColor3f(1, 1, 0)
    # glVertex3f(0, 0, 1)
    # for angle in range(361):
    #     x = math.sin(math.radians(angle))/6
    #     y = math.cos(math.radians(angle))/6
    #     # glColor3f(1, 1, 0)
    #     # glVertex2f(x, y)
    #     # np.concatenate((pointdata, [x, y, 0]),axis=0)
    #     pointdata.append([x,y,0])
    #     pointcolor.append([1,1,0])
    # # glEnd()
    #
    # # bottom of cone
    # # glBegin(GL_TRIANGLE_FAN)
    # glColor3f(1, 0, 0)
    # glVertex3f(0, 0, 0)
    # for angle in range(361):
    #     x = math.sin(math.radians(angle))/6
    #     y = math.cos(math.radians(angle))/6
    #     # glColor3f(1, 0, 0)
    #     # glVertex2f(x, y)
    #     np.concatenate(pointdata, [x,y,0])
    #     # pointdata.append([x,y,0])
    #     pointcolor.append([1,0,0])
    # # glEnd()

    glutSwapBuffers()
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    pass


def specialkeys(key, x, y):
    # global eye
    # global lat
    # global lon
    # if key == GLUT_KEY_UP:
    #     eye += np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    # elif key == GLUT_KEY_DOWN:
    #     eye -= np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    # elif key == GLUT_KEY_LEFT:
    #     eye += np.cos(lon), 0, -np.sin(lon)
    # elif key == GLUT_KEY_RIGHT:
    #     eye += -np.cos(lon), 0, np.sin(lon)
    # elif key == '\x1b':
    #     glutLeaveMainLoop()
    global pointcolor
    if key == GLUT_KEY_UP:
        glRotatef(5, 1, 0, 0)
    if key == GLUT_KEY_DOWN:
        glRotatef(-5, 1, 0, 0)
    if key == GLUT_KEY_LEFT:
        glRotatef(5, 0, 1, 0)
    if key == GLUT_KEY_RIGHT:
        glRotatef(-5, 0, 1, 0)
    if key == GLUT_KEY_END:
        pointcolor = [[random(), random(), random()], [random(), random(), random()], [random(), random(), random()]]




# def setupView(self):
#     self.camera = Camera()
#     self.camera.set_position(v3(3, 0, 10))
#     self.camera.look_at(v3(0, 0, 0))

# def Cube():
#     vertices = (
#         (1, -1, -1),
#         (1, 1, -1),
#         (-1, 1, -1),
#         (-1, -1, -1),
#         (1, -1, 1),
#         (1, 1, 1),
#         (-1, -1, 1),
#         (-1, 1, 1)
#     )
#
#     edges = (
#         (0, 1),
#         (0, 3),
#         (0, 4),
#         (2, 1),
#         (2, 3),
#         (2, 7),
#         (6, 3),
#         (6, 4),
#         (6, 7),
#         (5, 1),
#         (5, 4),
#         (5, 7)
#     )
#     glBegin(GL_LINES)
#     for edge in edges:
#         for vertex in edge:
#             glVertex3fv(vertices[vertex])
#     glEnd()

if __name__ == '__main__':
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(50, 50)
    glutInit(sys.argv)
    glutCreateWindow(b"Shaders!")
    glutDisplayFunc(draw)
    glutIdleFunc(draw)
    glutSpecialFunc(specialkeys)
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



    parser = ObjParser2()
    parser.read_file("deer.obj")
    vertices = np.array(parser.vertices, dtype='f')
    indices = np.array(parser.indices, dtype='i')
    colors = np.array([random.random() for _ in indices], dtype='f')
    vertexRe= np.reshape(vertices, (len(vertices)/3, 3))
    colorsRe=np.reshape(colors, (len(colors)/3, 3))
    print("len(vertices)=", len(vertices))
    print ("vertices=", vertices)
    print("len(vertices)=", len(vertexRe))
    print ("vertices=", vertexRe)
    print("len(indices)=", len(indices))
    print("indices=",indices)
    newPointFromObj = []
    for i in range(0,len(indices)):
        # print(i)
        # x = int(indices[i])
        # print("x=",x)
        elem= vertices[indices[i]]
        # print("elem=", elem)
        newPointFromObj.append(elem)
    vertexRe2 = np.reshape(newPointFromObj, (len(newPointFromObj) / 3, 3))
    print("len(vertexRe2)=", len(vertexRe2))
    print("vertexRe2=",vertexRe2)

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


    # glTranslatef(0.0, 0.0, 0)                                # Сдвинемся по оси Z на -0.7
    # Рисуем цилиндр с радиусом 0.1, высотой 0.2
    # Последние два числа определяют количество полигонов
    # glutSolidCylinder(0.1, 0.2, 20, 20)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluLookAt(-10, -10, -10, 0, 0, 0, 0, 10, 0);
    # glTranslatef(0,0,5)
    # glViewport(0, 0, 20, 20);
    # x,y,width, height = glGetDoublev(GL_VIEWPORT)
    # gluPerspective(45, width/float (height or 1), 0.25, 200,)
    # gluLookAt(3, 2, 0, 0, 0, 0, 0, 10, 0);
    print(3*12+361*2*3)
    print("len(pointdata)_all=",len(pointdata)*len(pointdata[0]))
    print("len(pointdata)=", len(pointdata))
    print("len(pointcolor)=", len(pointcolor))
    global eye
    eye = np.zeros(3)
    global lat
    lat = 0
    global lon
    lon = np.arctan2(0, -1)
    glutMainLoop()