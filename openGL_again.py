import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *

from random import random

if __name__ == '__main__':
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(50, 50)
    glutInit(sys.argv)
    glutCreateWindow(b"Shaders!")
    glClearColor(0.2, 0.2, 0.2, 1)







