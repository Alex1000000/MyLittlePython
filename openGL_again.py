from OpenGL.GL import *
from OpenGL.GLUT import *
from random import random


def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader


def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointdata)
    glColorPointer(3, GL_FLOAT, 0, pointcolor)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glutSwapBuffers()
    pass


def specialkeys(key, x, y):
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
    pointdata = [[0, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]]
    pointcolor = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    glutMainLoop()