import tkinter as tk                    # imports
from tkinter import ttk
from tkinter import Text, END, INSERT,CURRENT,SEL_FIRST
from tkinter.colorchooser import *
from tkinter import filedialog

import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np
from numpy import arange
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


import scipy as sp
from scipy.integrate.odepack import odeint

def _quit():
    win.quit()     # stops mainloop
    win.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

class CircleParamsListToSave:
    listOfCircleParams = []
    def addCircleToList(self,x,y,radius,color):
        circleParams = {'x': x, 'y': y,'radius': radius ,'color': color}
        self.listOfCircleParams.append(circleParams)
    def clearList(self):
        self.listOfCircleParams=[]
    def addParamsList(self, newParamsList):
        self.listOfCircleParams=newParamsList

def Hello(event):
    print("Yet another hello world")

def IncreaseScale():
    global axis_size
    axis_size=axis_size/1.5
    a.set_ylim([(axis_size*-1), axis_size])
    a.set_xlim([(axis_size*-1), axis_size])
    canvas.draw()
    #print("+", axis_size)

def DecreaseScale():
    global axis_size
    axis_size=axis_size*1.5
    a.set_ylim([(axis_size*-1), axis_size])
    a.set_xlim([(axis_size*-1), axis_size])
    canvas.draw()
    #print("-", axis_size)

def sliderMove(event):
    entrySlider.delete(0, END)
    s=str(scaleMy.get())
    entrySlider.insert(0,s)
    #print ('slider')
def getColor():
    color=askcolor()
    print(color)

def getColorFromCombo(event):
    global currentColorFromCombo
    currentColorFromCombo=combo.current()
    print(currentColorFromCombo)


def onMouseClick(event):
    s=e.get()
    # print(x_mouse,";  ",y_mouse)
    # print(s)
    if s!="None;  None":
        # print("mose CLICK LEFT")
        if currentColorFromCombo==0:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='r')
            currentCircleList.addCircleToList(x_mouse,y_mouse,scaleMy.get(),'r')
        elif currentColorFromCombo==1:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='g')
            currentCircleList.addCircleToList(x_mouse, y_mouse, scaleMy.get(), 'g')
        elif currentColorFromCombo==2:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='b')
            currentCircleList.addCircleToList(x_mouse, y_mouse, scaleMy.get(), 'b')
        elif currentColorFromCombo==3:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='y')
            currentCircleList.addCircleToList(x_mouse, y_mouse, scaleMy.get(), 'y')
        elif currentColorFromCombo==4:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='m')
            currentCircleList.addCircleToList(x_mouse, y_mouse, scaleMy.get(), 'm')

        a.add_artist(circle1)
        canvas.draw()

def onMouseMove(event):
    #print(event.x, event.y, event.xdata, event.ydata)
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
    # ax=plt.axes() #ax.transAxes
    # print(ax.transData.transform_point([event.x, event.y]))
    global x_mouse,y_mouse,currentColorFromCombo
    x_mouse=event.xdata
    y_mouse=event.ydata
    e.delete(0, END)
    s=str(event.xdata)+";  "+str(event.ydata)
    e.insert(0, s)
    T.delete(CURRENT,END)
    T.insert(INSERT, s)
    #print ('move')

def file_save():
    #fi = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
    # if fi is None: # asksaveasfile return `None` if dialog closed with "cancel".
    #     return
    root = ET.Element('circlesInThePlot')
    ET.SubElement(root, 'settings',
                  {'xlim': str(a.get_xbound()[1]), 'ylim': str(a.get_ybound()[1])})

    for el in currentCircleList.listOfCircleParams:
        print(el)
        child = ET.SubElement(root, 'circle',
                              {'x': str(el['x']), 'y': str(el['y']), 'radius': str(el['radius']), 'color': el['color']})
    # ET.dump(root)
    fi = filedialog.asksaveasfilename(filetypes=[("XML files", "*.xml")])
    if fi is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    tree = ET.ElementTree(root)
    tree.write(fi)
    #text2save = "qwerty" # starts from `1.0`, not `0.0`
    #//fi.write(text2save)
    # fi.close() # `()` was missing.

def file_open():
    # filename = filedialog.askopenfilename(initialdir="/", title="Select file",
    #                                       filetypes=(("txt", "*.txt"), ("all files", "*.*")))
    # file = open(filename)
    # txt=file.read()
    # print(txt)
    # print("file open")
    a.cla()
    filename = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])
    tree = ET.parse(filename)
    currentCircleList.clearList()
    for node in tree.iter('settings'):
        xlim = float(node.attrib.get('xlim'))
        ylim = float(node.attrib.get('ylim'))
        a.set_ylim([-ylim, ylim])
        a.set_xlim([-xlim, xlim])
        canvas.draw()


    for node in tree.iter('circle'):
        x = float(node.attrib.get('x'))
        y = float(node.attrib.get('y'))
        radius = float(node.attrib.get('radius'))
        color = node.attrib.get('color')
        currentCircleList.listOfCircleParams.append({ 'x': x, 'y': y, 'radius': radius, 'color': color})
        circle1 = plt.Circle((x, y), radius, color=color)
        a.add_artist(circle1)
        canvas.draw()





def f_vect_scipy(z,t):
    # z is a vector with 12 entries
    # ordered in blocks of 4:
    # x[k](t),x’[k](t),y[k](t),y’[k](t)
    # for k = 1,2,3.
    L = [0 for k in range(12)]
    r = sp.array(L, float)

    # take three equal masses
    m = [597200000, 7.36, 597.2]
    # relabel input vector z
    x1 = z[0];
    u1 = z[1];
    y1 = z[2];
    v1 = z[3]
    x2 = z[4];
    u2 = z[5];
    y2 = z[6];
    v2 = z[7]
    x3 = z[8];
    u3 = z[9];
    y3 = z[10];
    v3 = z[11]
    # u and v are first derivatives of x and y
    r[0] = u1;
    r[2] = v1
    r[4] = u2;
    r[6] = v2
    r[8] = u3;
    r[10] = v3

    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    r3 = np.array([x3, y3])

    vel1 = np.array([u1, v1])
    vel2 = np.array([u2, v2])
    vel3 = np.array([u3, v3])

    dist12 = np.linalg.norm(r1 - r2)
    dist13 = np.linalg.norm(r1 - r3)
    dist32 = np.linalg.norm(r3 - r2)

    G = 10
    a1 = G * (m[1] * (r2 - r1) / (dist12 ** 3) + m[2] * (r3 - r1) / (dist13 ** 3))
    a2 = G * (m[0] * (r1 - r2) / (dist12 ** 3) + m[2] * (r3 - r2) / (dist32 ** 3))
    a3 = G * (m[0] * (r1 - r3) / (dist13 ** 3) + m[1] * (r2 - r3) / (dist32 ** 3))

    r[1] = a1[0]
    r[3] = a1[1]
    r[5] = a2[0]
    r[7] = a2[1]
    r[9] = a3[0]
    r[11] = a3[1]
    return r

def verlet(z,dt,N_of_iterations):
    # z is a vector with 12 entries
    # ordered in blocks of 4:
    # x[k](t),x’[k](t),y[k](t),y’[k](t)
    # for k = 1,2,3.
    L = [0 for k in range(12)]
    r = sp.array(L,float)

    # take three equal masses
    #sun 1.98892 × 10^30   217000
    #moon 7.36 × 10^22     1020
    #earth 5.972 x 10^24   30000 м/с
    # m = [597200000*(10**22), 7.36*(10**22), 597.2*(10**22)]
    m = [1.98892  * (10 ** 30), 7.34767309*(10**22), 597.2 * (10 ** 22)]
    # relabel input vector z
    x1 = z[0]; u1 = z[1]; y1 = z[2]; v1 = z[3]
    x2 = z[4]; u2 = z[5]; y2 = z[6]; v2 = z[7]
    x3 = z[8]; u3 = z[9]; y3 = z[10]; v3 = z[11]
    # u and v are first derivatives of x and y


    r1=np.array([x1,y1])
    r2=np.array([x2,y2])
    r3=np.array([x3,y3])

    vel1=np.array([u1,v1])
    vel2=np.array([u2,v2])
    vel3=np.array([u3,v3])

    dist12 = np.linalg.norm(r1 - r2)
    dist13 = np.linalg.norm(r1 - r3)
    dist32 = np.linalg.norm(r3 - r2)

    G = 6.67408*(10**-11)# 10^-11
    a1 = G * (m[1] * (r2 - r1) / (dist12 ** 3) + m[2] * (r3 - r1) / (dist13 ** 3))
    a2 = G * (m[0] * (r1 - r2) / (dist12 ** 3) + m[2] * (r3 - r2) / (dist32 ** 3))
    a3 = G * (m[0] * (r1 - r3) / (dist13 ** 3) + m[1] * (r2 - r3) / (dist32 ** 3))

    # N_of_iterations=10

    x1_n = np.zeros(N_of_iterations)
    y1_n = np.zeros(N_of_iterations)

    x2_n = np.zeros(N_of_iterations)
    y2_n = np.zeros(N_of_iterations)

    x3_n = np.zeros(N_of_iterations)
    y3_n = np.zeros(N_of_iterations)

    for i in range(0,N_of_iterations):
        delta_t=dt#t[i]

        r1_next = r1 + vel1 * delta_t + 0.5 * a1*delta_t*delta_t
        r2_next = r2 + vel2 * delta_t + 0.5 * a2*delta_t*delta_t
        r3_next = r3 + vel3 * delta_t + 0.5 * a3*delta_t*delta_t
        x1_n[i]=r1_next[0]
        y1_n[i]=r1_next[1]

        x2_n[i]=r2_next[0]
        y2_n[i]=r2_next[1]

        x3_n[i]=r3_next[0]
        y3_n[i]=r3_next[1]

        dist12 = np.linalg.norm(r1_next - r2_next)
        dist13 = np.linalg.norm(r1_next - r3_next)
        dist32 = np.linalg.norm(r3_next - r2_next)

        a1_next = G * (m[1] * (r2_next - r1_next) / (dist12 ** 3) + m[2] * (r3_next - r1_next) / (dist13 ** 3))
        a2_next = G * (m[0] * (r1_next - r2_next) / (dist12 ** 3) + m[2] * (r3_next - r2_next) / (dist32 ** 3))
        a3_next = G * (m[0] * (r1_next - r3_next) / (dist13 ** 3) + m[1] * (r2_next - r3_next) / (dist32 ** 3))

        vel1_next = vel1 + 0.5 * delta_t * (a1_next + a1)
        vel2_next = vel2 + 0.5 * delta_t * (a2_next + a2)
        vel3_next = vel3 + 0.5 * delta_t * (a3_next + a3)

        r1 = r1_next
        r2 = r2_next
        r3 = r3_next

        vel1=vel1_next
        vel2=vel2_next
        vel3=vel3_next

        a1=a1_next
        a2=a2_next
        a3=a3_next
    # plotting the trajectories
    canvas.draw()
    # fig = plt.figure("d")
    # a = fig.add_subplot(111)
    a.plot(x1_n, y1_n,"r", x2_n, y2_n,"g", x3_n, y3_n,"b")
    xMax = max(max(x1_n), max(x2_n), max(x3_n))
    xMim = min(min(x1_n), min(x2_n), min(x3_n))
    yMax = max(max(y1_n), max(y2_n), max(y3_n))
    yMim = min(min(y1_n), min(y2_n), min(y3_n))
    a.set_xlim([xMim - 100000, xMax + 100000])
    a.set_ylim([yMim - 100000, yMax + 100000])
    plt.show()

    return r



def scipy_n_body():
    # Plots the trajectories of 3 equal masses
    # forming a figure 8.

    # initial positions
    ip1 = [0, 0]  #the sun
    ip2 = [0, -1499.80000]  #the moon
    ip3 = [0, -1496.00000]  #the earth 380.000  150.000.000
    # initial velocities
    iv1=[0.0,0.0]#217000
    iv3 = [72.0, 72.0]#1020
    iv2 = [0, 300.00]#30000
    # input for initial righthandside vector
    initz = [ip1[0], iv1[0], ip1[1], iv1[1], \
    ip2[0], iv2[0], ip2[1], iv2[1], \
    ip3[0], iv3[0], ip3[1], iv3[1]]
    T = 5#2 * sp.pi / 3;
    n = 1000
    tspan = sp.linspace(0, T, n + 1)
    z_trajectories = odeint(f_vect_scipy, initz, tspan,mxstep=5000)
    # extracting the trajectories
    x1_traj = z_trajectories[:, 0];
    y1_traj = z_trajectories[:, 2];
    x2_traj = z_trajectories[:, 4];
    y2_traj = z_trajectories[:, 6];
    x3_traj = z_trajectories[:, 8];
    y3_traj = z_trajectories[:, 10];
    # # plotting the trajectories
    # fig = plt.figure("scipy: The Sun The Moon The Earth")
    # plt.plot(x1, y1,"r", x2, y2,"g", x3, y3,"b")
    # plt.show()
    # fig = plt.figure()
    # a = fig.add_subplot(111)
    xMax = max(max(x1_traj), max(x2_traj), max(x3_traj))
    xMim = min(min(x1_traj), min(x2_traj), min(x3_traj))
    yMax = max(max(y1_traj), max(y2_traj), max(y3_traj))
    yMim = min(min(y1_traj), min(y2_traj), min(y3_traj))
    a.set_xlim([xMim-100,xMax+100])
    a.set_ylim([yMim - 100, yMax + 100])
    canvas.draw()
    a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    print(x1_traj)
    plt.show()


def verlet_nBody():
    # Plots the trajectories of 3 equal masses
    # forming a figure 8.

    # initial positions
    ip1 = [0, 0]  # ip1 = [1.496*10e11, 0]  #the sun
    ip2 = [0, -1.496 * 1e11 - 384.467 * 1e6]  # [1022, -384467000]  #the moon
    ip3 = [0, -1.496 * 1e11]  # [0, -149600000]  #the earth 380.000  150.000.000
    # initial velocities
    iv1 = [0.0, 0.0]  # 217000
    iv2 = [1022 + 29.783 * 1e3, 0.0]  # [1022+29.783*1e3, 0.0]#-29.783*1e3]#1020
    iv3 = [29.783 * 1e3, 0.0]  # 30000
    # input for initial righthandside vector
    initz = [ip1[0], iv1[0], ip1[1], iv1[1], \
             ip2[0], iv2[0], ip2[1], iv2[1], \
             ip3[0], iv3[0], ip3[1], iv3[1]]
    T = 120  # 2 * sp.pi / 3;
    n = 200000
    # tspan = sp.linspace(0, T, n + 1)

    z = verlet(initz, T, n)
    return

def n_body_solving():
    print(v.get())
    modeOfComputation=v.get()
    if (modeOfComputation=="1"):
        print("_scipy")
        scipy_n_body()
    elif (modeOfComputation=="2"):
        print("_scipy")
        verlet_nBody()



x_mouse=0.0
y_mouse=0.0
win = tk.Tk()                           # Create instance
win.title("Python GUI")                 # Add a title
tabControl = ttk.Notebook(win)          # Create Tab Control
tab1 = ttk.Frame(tabControl, width= 70,height = 70)            # Create a tab
tabControl.add(tab1, text='Edit')      # Add the tab
tabControl.pack(expand=1, fill="both",side=tk.LEFT)  # Pack to make visible

mouse_label = tk.Label(tab1, text="Mouse coordinates:")
mouse_label.pack()

S = tk.Scrollbar(tab1)
S.pack(side=tk.RIGHT, fill=tk.Y)

e = tk.Entry(tab1)
e.pack()
# e.delete(0, END)
e.insert(0, "None;  None")

T = Text(tab1, height=2, width=30)
#T.pack()
T.insert(END, "Just a text Widget   ""HAMLET: To be, or not to be--that is the question:tis nobler in the mind to sufferThe slings and arrows of outrageous fortune")
S.config(command=T.yview)#######################
T.config(yscrollcommand=S.set)######################


slider_label = tk.Label(tab1, text="Slider value:")
slider_label.pack()
entrySlider = tk.Entry(tab1)
entrySlider.pack()
# e.delete(0, END)
entrySlider.insert(0, "1")
scaleMy=tk.Scale(tab1, from_=1.0, to=100.0, orient=tk.HORIZONTAL, command= sliderMove)
scaleMy.pack()

currentColorFromCombo=0
combo=ttk.Combobox(tab1)
combo['values']=('Red','Green','Blue', 'Yellow', 'Purple')
combo.set('Red')
combo.pack()
combo.bind("<<ComboboxSelected>>", getColorFromCombo)



tab2 = ttk.Frame(tabControl, width=70,height =70)            # Create a tab
tabControl.add(tab2, text='Model')      # Add the tab
tabControl.pack(expand=1, fill="both")  # Pack to make visible


currentCircleList= CircleParamsListToSave()
f = Figure(figsize=(4, 4), dpi=100)
a = f.add_subplot(111)
t = arange(-100.0, 100.0, 1.0)
s = np.exp(-t/2.) * np.sin(2*np.pi*t)#sin(2*pi*(t/100))

#a.plot(t, s)
axis_size=100
a.set_ylim([-100,100])
a.set_xlim([-100,100])

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=win)
canvas.show()
canvas.get_tk_widget().pack()

toolbar = NavigationToolbar2TkAgg(canvas, win)
toolbar.update()
canvas._tkcanvas.pack()

# btn = tk.Button(win,  #родительское окно
#              text="Click me",  #надпись на кнопке
#              width=10, height=2,  #ширина и высота
#              bg="white", fg="black") #цвет фона и надписи
# btn.bind("<Button-1>", Hello)       #при нажатии ЛКМ на кнопку вызывается функция Hello
# btn.pack()                          #расположить кнопку на главном окне

btn =tk.Button(master=win, text='+', bg="green",fg="white" ,command=IncreaseScale)
btn.pack()
btn =tk.Button(master=win, text='-', bg="red",fg="white" ,command=DecreaseScale)
btn.pack()
button = tk.Button(master=win, text='Quit', bg="blue",fg="white" ,command=_quit)
button.pack(side=tk.BOTTOM)
selectColor=tk.Button(tab2,text="selectColor", command=getColor)
selectColor.pack()
saveFile=tk.Button(tab1,text="saveFile", command=file_save)
saveFile.pack()
openFile=tk.Button(tab1,text="openFile", command=file_open)
openFile.pack()
# nBody=tk.Button(tab1,text="openFile", command=file_open)
# nBody.pack()


MODES = [
        ("scipy", "1"),
        ("verlet", "2"),
        ("verlet-threading", "3"),
        ("verlet-multiprocessing", "4"),
        ("verlet-cython", "5"),
        ("verlet-opencl", "6")
    ]

v = tk.StringVar()
v.set("1") # initialize

for text, mode in MODES:
    b = tk.Radiobutton(tab2, text=text,variable=v, value=mode, command=n_body_solving)
    b.pack(anchor=tk.W)

#ax = plt.axes()
#ax.transData.transform_point([x, y])
#inv = ax.transData.inverted()
cid = f.canvas.mpl_connect('motion_notify_event', onMouseMove)
# cid = f.canvas.mpl_connect('motion_notify_event', sliderMove)
win.bind("<Button-1>", onMouseClick)


win.mainloop()