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
    glDrawArrays(GL_TRIANGLES, 0,len(pointdata)*3)#10262)#2196)
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
    if key == GLUT_KEY_PAGE_DOWN:
        for k in range(0,1):
            for i in range(NumberOfParts):
                Col =Temp_to_color_glob(temperature.T_cur[i], temperature.T_cur)
                for j in range(diapazon[i], diapazon[i + 1]):
                    # Col[2]=1.0
                    pointcolor[j] = [Col, Col, Col]
            temperature.next_step()
        print(temperature.T_cur)



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


def parseFile(file_name):
    count = 0
    start = 0
    list_of_count_of_vel = []
    list_of_vertex = []
    list_of_triang = []
    index = 0
    lst_vel = []
    lst_f = []
    total = 1
    count_of_v = 0

    for line in open(file_name, 'r'):
        values = line.split()
        if len(values) < 2:
            continue
        if(values[0]== '#' and values[1] == 'object' and count != 0):
            list_of_count_of_vel.append(count)
            list_of_vertex.append(lst_vel)
            list_of_triang.append(lst_f)
            index = index + 1
            total = total + count_of_v
            count_of_v = 0
            count = 0
            lst_vel = []
            lst_f = []
        if (values[0] == '#' and values[1] == 'object' and count == 0):
            start = 1

        if(values[0] == 'f' and count == 0):
            start = 1

        if(start == 1 and values[0] == 'f'):
            count = count + 1
            lst_f.append([float(values[1])-total,float(values[2])-total,float(values[3])-total])

        if (start == 1 and values[0] == 'v'):
            lst_vel.append([float(values[1]), float(values[2]), float(values[3])])
            count_of_v = count_of_v + 1

    list_of_vertex.append(lst_vel)
    list_of_triang.append(lst_f)
    list_of_count_of_vel.append(count)
    print("list_of_count_of_vel=",list_of_count_of_vel)
    print("list_of_triang=", list_of_triang)
    print("list_of_vertex=", list_of_vertex)
    print("list_of_count",sum(list_of_count_of_vel),3*sum(list_of_count_of_vel) )
    return  list_of_count_of_vel,list_of_triang,list_of_vertex






def Form_Triangles(list_of_vertex, list_of_tri):
    triangles = []
    diapazon = np.zeros(len(list_of_vertex) + 1, dtype=np.int)
    for i in range(len(list_of_vertex)):
        diapazon[i + 1] = diapazon[i] + len(list_of_tri[i])
        for el in list_of_tri[i]:
            triangle = np.array([list_of_vertex[i][int(el[0])], list_of_vertex[i][int(el[1])], list_of_vertex[i][int(el[2])]])
            triangles.append(triangle)
    return np.array(triangles), diapazon


def Temp_to_color_glob(temp, global_temp):
    s=[1,2,3]
    # global_temp=global_temp%100
    new_glob=np.zeros(len(global_temp))
    for i in range(len(global_temp)):
        new_glob[i]=global_temp[i]%1
        new_glob[i]=global_temp[i]
    new_temp=temp%1
    new_temp=temp
    print("global_temp=",global_temp)
    print("new_glob=", new_glob)
    max_temp=np.amax(new_glob)
    min_temp=np.amin(new_glob)
    avg_v=np.average(new_glob)
    print("avg_v=", avg_v)
    print("max_temp=",max_temp,"min_temp=",min_temp)
    val=(new_temp-min_temp)/(max_temp-min_temp)
    print("val=",val)
    return [val, val,val]



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




    name = 'model1bis.obj'
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
    pointdata_earth /= pointdata_earth.max()*1.2
    pointdata_earth_new = np.zeros((N * 3, 3))
    pointdata_earth_new = np.reshape(pointdata_earth, (N * 3, 3))

    # for i in range(0, N):
    pointcolor_earth[:, :] = [0.0, 0.0, 1.0]
    print("pointcolor_earth=",len(pointcolor_earth),"x ",len(pointcolor_earth[0]))
    for i in range(0, 384*3):
        pointcolor_earth[i, :]=[1.0,0.0,0.0]
    for i in range(384*3, (384+384)*3):
        pointcolor_earth[i, :]=[0.0,1.0,0.0]
    for i in range((384+384)*3, (384 + 384+384)*3):
        pointcolor_earth[i, :] = [1.0, 0.0, 1.0]
    for i in range((384 + 384+384)*3, (384 + 384 + 384+384)*3):
        pointcolor_earth[i, :] = [0.0, 1.0, 1.0]
    for i in range((384 + 384 + 384 +384)*3, (384 + 384 + 384 + 384+224)*3):
        pointcolor_earth[i, :] = [1.0, 1.0, 0.0]
        # pointcolor_earth[3 * i + 1, :] = [0.0, 0.0, 1.0]
        # pointcolor_earth[3 * i + 2, :] = [0.0, 0.0, 1.0]
    print("pointdata_earth_new=", len(pointdata_earth_new))
    pointdata = np.concatenate((pointdata, pointdata_earth_new))
    pointcolor=np.concatenate((pointcolor, pointcolor_earth))

    listOfDotsNumber, list_of_tri, list_of_vertex = parseFile('model1.obj')
    NumberOfParts = len(listOfDotsNumber)
    print("M=",NumberOfParts,"list=",listOfDotsNumber)
    triangles, diapazon = Form_Triangles(list_of_vertex, list_of_tri)

    def get_area(numberOfParts,trianglesArray, diapazon):
        areas = []
        for i in range(0, len(listOfDotsNumber)):
            area = 0
            for j in range(diapazon[i], diapazon[i + 1]):
                p1 = np.array(trianglesArray[j,0])
                p2 = np.array(trianglesArray[j,1])
                p3 = np.array(trianglesArray[j,2])
                area += np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / 2
            areas += [area]
        return areas
    print("get_area===",get_area(NumberOfParts,triangles, diapazon))


    triangles /= (2 * triangles.max())
    pointcolorTri = np.zeros((diapazon[len(list_of_vertex)], 3, 3))
    from random import random
    for i in range(0, len(listOfDotsNumber)):
        m = random()
        k = random()
        for j in range(diapazon[i], diapazon[i + 1]):
            pointcolorTri[j] = [k, 0.0, m]
    print("triangles=",triangles, "len=", len(triangles), " x ",len(triangles[0]),"=",len(triangles)*len(triangles[0]))
    print("diapazon=", diapazon)



    # -------------------------------------
    # draw only obj
    pointdata=pointdata_earth_new
    pointcolor=pointcolor_earth

    pointdata=triangles*2
    pointcolor=pointcolorTri
    # ----------------------------------------
    import scipy.optimize as so
    import scipy.integrate as si
    import copy


    def right_f(T, t, lambada, Q_R, c, eps, S):
        NumberOfParts = len(c)
        right_part = np.zeros(NumberOfParts)
        StephBolC = 5.67
        for i in range(NumberOfParts):
            for j in range(NumberOfParts):
                if i != j:
                    right_part[i] -= lambada[i, j] * S[i, j] * (T[i] - T[j])
            right_part[i] -= eps[i] * S[i, i] * StephBolC * (T[i] / 100) ** 4
            right_part[i] += Q_R[i](t)
            right_part[i] /= c[i]
        return right_part
    class TempComputer:
        def __init__(self, lambda_const, Q_R, c, eps, S, tau, NumberOfParts):
            self.lambada = lambda_const
            self.Q_R = Q_R
            self.c = c
            self.eps = eps
            self.S = S
            self.counter = 0
            # print("NumberOfParts=",NumberOfParts,"c=",c)
            # get стац решение
            T=np.array([ 0,   73.26218973, 0,  72.97086224,  72.93746353])
            T=np.zeros(NumberOfParts)
            self.T_cur = so.fsolve(right_f, np.zeros(NumberOfParts), args=(0, lambda_const, Q_R, c, eps, S))
            self.T = copy.copy(self.T_cur)
            self.tau = tau
            self.NumberOfParts = NumberOfParts

        def next_step(self):
            T_t = np.linspace((self.counter - 1) * self.tau, self.counter * self.tau, 2)
            print("T_t=",T_t)
            self.counter += 1
            self.T = si.odeint(right_f, self.T_cur, T_t, args=(self.lambada, self.Q_R, self.c, self.eps, self.S))
            self.T_cur = copy.copy(self.T[1])
            return self.T[1]
    # решаем систему
    # squareOfElem=np.array([[  36.1672017   , 12.46398959  ,  0.   ,         0.  ,          0.        ],
    #                         [  12.47141881 ,  99.40769069  , 12.46398959 ,   0.  ,          0.        ],
    #                         [   0.          , 12.46398959  , 36.1672017  ,   3.1196918  ,   0.        ],
    #                         [   0.           , 0.           , 3.1196918   , 12.36601433  ,  3.07374533],
    #                         [   0.           , 0.            ,0.           , 3.155      , 160.        ]])

    squareOfElem = np.array([[36.1672017 ,   12.4677042   ,  0.     ,       0.     ,       0.],
     [12.4677042 ,   99.40769069 ,  12.46398959  ,  0.      ,      0.],
    [0.,            12.46398959,   36.1672017,       3.1196918,    0.],
    [0.,            0.  ,          3.1196918  ,  12.36601433  ,  3.11437266],
    [0.,0.,0.,3.11437266,160.]])

    # print("square=", squareOfElem)
    #
    # for i in range(NumberOfParts):
    #     for j in range(i + 1, NumberOfParts):
    #         temp = (squareOfElem[i, j] + squareOfElem[j, i]) / 2
    #         squareOfElem[i, j] = temp
    #         squareOfElem[j, i] = temp
    print ("squareOfElem=",squareOfElem)
    eps = [0.1, 0.1, 0.1, 0.01, 0.05]
    c = [520, 520, 520, 840, 900]

    lambda_const = np.zeros((NumberOfParts, NumberOfParts))
    lambda_const[0, 1] = 20
    lambda_const[1, 0] = 20
    lambda_const[1, 2] = 20
    lambda_const[2, 1] = 20
    lambda_const[2, 3] = 10.5
    lambda_const[3, 2] = 10.5
    lambda_const[3, 4] = 119
    lambda_const[4, 3] = 119

    Q_R = []
    for i in range(NumberOfParts):
        f = lambda t: [0]
        Q_R.append(f)
    A = 2
    Q_R[1] = lambda t: [A * (20 + 3 * np.cos(t / 4))]
    tau = 10 ** 2
    print("pointcolor[6]=",pointcolor[6])
    temperature = TempComputer(lambda_const, Q_R, c, eps, squareOfElem, tau, NumberOfParts)
    for i in range(len(list_of_vertex)):
        Col = Temp_to_color_glob(temperature.T_cur[i], temperature.T_cur)
        print("temperature.T0[i]=", temperature.T_cur[i])
        print("Col=",Col)
        for j in range(diapazon[i], diapazon[i + 1]):
            pointcolor[j] = [Col, Col, Col]





    # len(triangles) * len(triangles[0])

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

    parseFile("model1.obj")

    global eye
    eye = np.zeros(3)
    global lat
    lat = 0
    global lon
    lon = np.arctan2(0, -1)
    glutMainLoop()