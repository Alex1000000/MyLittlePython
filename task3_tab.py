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





def acc_get(m, coordOfplanets):
    G = 6.67408 * (10 ** -11);
    numbOfPlanets = coordOfplanets.size // m.size
    acc = np.zeros(coordOfplanets.size)
    for i in range(m.size):
        for j in range(m.size):
            if i != j:
                rDiff = coordOfplanets[j * numbOfPlanets:(j + 1) * numbOfPlanets] - coordOfplanets[
                                                                              i * numbOfPlanets:(i + 1) * numbOfPlanets]
                dist=np.linalg.norm(rDiff)
                acc[i * numbOfPlanets:(i + 1) * numbOfPlanets] += G * m[j] * rDiff / (dist ** 3)
    return acc

def scipy_n_body():
    def f_vect_scipy(z, t):
        r = np.zeros(z.shape)
        halfOfSize = z.size // 2
        r[halfOfSize:] = acc_get(m, z[:halfOfSize])
        r[:halfOfSize] = z[halfOfSize:]
        return r
    T = 60 * 60 * 24#2 * sp.pi / 3;
    n = 365 * 1

    m = np.array([ 1.98892e30, 7.34767309e22,5.972e24])# the sun  the moon the earth
    ip = np.array([0, 0,  (1.496e11 + 3.84467e8), 0, 1.496e11, 0,])
    iv = np.array([0, 0,0,(2.9783e4 + 1022), 0, 2.9783e4])

    tspan = np.arange(n) * T
    z_trajectories = odeint(f_vect_scipy, np.concatenate((ip, iv)), tspan)

    # extracting the trajectories
    x1_traj = z_trajectories[:, 0];
    y1_traj = z_trajectories[:, 1];
    x2_traj = z_trajectories[:, 2];
    y2_traj = z_trajectories[:, 3];
    x3_traj = z_trajectories[:, 4];
    y3_traj = z_trajectories[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    canvas.draw()
    print(x1_traj)
    plt.show()



def verlet_nBody():
    # Plots the trajectories of 3 equal masses
    delta_t =60 * 60 * 24
    n = 365 * 1

    G = 6.67408 * (10 ** -11);

    m = np.array([ 1.98892e30, 7.34767309e22,5.972e24])# the sun  the moon the earth
    ip = np.array([0, 0,  (1.496e11 + 3.84467e8), 0, 1.496e11, 0,])
    iv = np.array([0, 0,   0,(2.9783e4 + 1022),        0, 2.9783e4])

    times = np.arange(n) * delta_t
    numberOfPlanets = ip.size

    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))

    acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
    for j in np.arange(n - 1) + 1:
        #r
        coordinatesAndVel[j, :numberOfPlanets] = coordinatesAndVel[j - 1, :numberOfPlanets] + coordinatesAndVel[j - 1, numberOfPlanets:] * delta_t + 0.5 * acc * delta_t ** 2
        acc_next = acc_get(m, coordinatesAndVel[j, :numberOfPlanets])
        #vel
        coordinatesAndVel[j, numberOfPlanets:] = coordinatesAndVel[j - 1, numberOfPlanets:] + 0.5 * (acc + acc_next) * delta_t
        acc = acc_next
    # tspan = sp.linspace(0, T, n + 1)

    x1_traj = coordinatesAndVel[:, 0];
    y1_traj = coordinatesAndVel[:, 1];
    x2_traj = coordinatesAndVel[:, 2];
    y2_traj = coordinatesAndVel[:, 3];
    x3_traj = coordinatesAndVel[:, 4];
    y3_traj = coordinatesAndVel[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    canvas.draw()
    print(x1_traj)
    plt.show()
    # z = verlet(initz, T, n)
    return

def n_body_solving():
    print(v.get())
    modeOfComputation=v.get()
    if (modeOfComputation=="1"):
        print("it's_scipy")
        scipy_n_body()
    elif (modeOfComputation=="2"):
        print("it's_verlet")
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