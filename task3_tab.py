import tkinter as tk                    # imports
from tkinter import ttk
from tkinter import Text, END, INSERT,CURRENT,SEL_FIRST
from tkinter.colorchooser import *
from tkinter import filedialog
import multiprocessing as mp
import threading
import multiprocessing
from multiprocessing import Process, Lock, freeze_support
import time

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

def acceleration(body, masses, positions):
    G = 6.67408 * (10 ** -11);
    dimension = positions.size // masses.size

    res = np.zeros(dimension)
    for j in range(masses.size):
        if body != j:
            displacement = positions[j*dimension:(j+1) * dimension] - positions[body*dimension:(body+1)*dimension]
            res += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res

def acceleration_no_np(body, masses, positions):
    G = 6.67408 * (10 ** -11);
    dimension = len(positions) // len(masses)

    res = np.zeros(dimension)
    displacement = np.empty(dimension)
    for j in range(masses.size):
        if body != j:
            for k in range(dimension):
                displacement[k] = positions[j * dimension + k] - positions[body * dimension + k]
            res += G * masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res

def acc_get_oneT(body, masses, positions):
    G = 6.67408 * (10 ** -11);
    dimension = positions.size // masses.size

    res = np.zeros(dimension)
    for j in range(masses.size):
        if body != j:
            displacement = positions[j*dimension:(j+1) * dimension] - positions[body*dimension:(body+1)*dimension]
            res += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res


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

    # PrintOrbit


    a.cla()
    a.set_ylim([-(1.5e11),1.5e11])
    a.set_xlim([-(1.5e11),1.5e11])
    for i in range(int(n)):
        x1_traj = z_trajectories[i, 0];
        y1_traj = z_trajectories[i, 1];
        circle1 = plt.Circle((x1_traj, y1_traj), 0.1e11, color='r')
        a.add_artist(circle1)

        x1_traj = z_trajectories[i, 4];
        y1_traj = z_trajectories[i, 5];
        circle1 = plt.Circle((x1_traj, y1_traj), 0.01e11, color='b')
        a.add_artist(circle1)

        x1_traj = z_trajectories[i, 2];
        y1_traj = z_trajectories[i, 3];
        circle1 = plt.Circle((x1_traj, y1_traj), 0.01e11, color='y')
        a.add_artist(circle1)
        canvas.draw()
        # for j in range(0,len(m)):
        #     x1_traj = z_trajectories[i, j];
        #     y1_traj = z_trajectories[i, j+1];
        #     circle1 = plt.Circle((x1_traj, y1_traj), 0.01e11, color='b')
        #     a.add_artist(circle1)
        #     canvas.draw()
        import time
        # print(time)
        time.sleep(0.01)
        a.cla()

    # # extracting the trajectories
    # x1_traj = z_trajectories[:, 0];
    # y1_traj = z_trajectories[:, 1];
    # x2_traj = z_trajectories[:, 2];
    # y2_traj = z_trajectories[:, 3];
    # x3_traj = z_trajectories[:, 4];
    # y3_traj = z_trajectories[:, 5];
    # a.cla()
    # a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    # canvas.draw()
    # # print(x1_traj)
    # plt.show()



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

    a.cla()
    a.set_ylim([-(1.5e11),1.5e11])
    a.set_xlim([-(1.5e11),1.5e11])
    for i in range(int(n)):
        x1_traj = coordinatesAndVel[i, 0];
        y1_traj = coordinatesAndVel[i, 1];
        circle1 = plt.Circle((x1_traj, y1_traj), 0.1e11, color='r')
        a.add_artist(circle1)

        x1_traj = coordinatesAndVel[i, 4];
        y1_traj = coordinatesAndVel[i, 5];
        circle1 = plt.Circle((x1_traj, y1_traj), 0.01e11, color='b')
        a.add_artist(circle1)

        x1_traj = coordinatesAndVel[i, 2];
        y1_traj = coordinatesAndVel[i, 3];
        circle1 = plt.Circle((x1_traj, y1_traj), 0.01e11, color='y')
        a.add_artist(circle1)
        canvas.draw()
        # for j in range(0,len(m)):
        #     x1_traj = z_trajectories[i, j];
        #     y1_traj = z_trajectories[i, j+1];
        #     circle1 = plt.Circle((x1_traj, y1_traj), 0.01e11, color='b')
        #     a.add_artist(circle1)
        #     canvas.draw()
        import time
        # print(time)
        time.sleep(0.01)
        a.cla()

    #
    # x1_traj = coordinatesAndVel[:, 0];
    # y1_traj = coordinatesAndVel[:, 1];
    # x2_traj = coordinatesAndVel[:, 2];
    # y2_traj = coordinatesAndVel[:, 3];
    # x3_traj = coordinatesAndVel[:, 4];
    # y3_traj = coordinatesAndVel[:, 5];
    # a.cla()
    # a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    # canvas.draw()
    # # print(x1_traj)
    # plt.show()
    # # z = verlet(initz, T, n)
    return


def acc_getOne(m, coordOfplanets, jPlanet):
    G = 6.67408 * (10 ** -11);
    numbOfPlanets = coordOfplanets.size // m.size
    acc = np.asarray([0.0,0.0])


    acc = np.zeros(2)

    i=0
    for j in range(m.size):
        if jPlanet != j:
            rDiff = coordOfplanets[j * numbOfPlanets:(j + 1) * numbOfPlanets] - coordOfplanets[
                                                                                i * numbOfPlanets:(
                                                                                                  i + 1) * numbOfPlanets]
            dist = np.linalg.norm(rDiff)
            acc[i * numbOfPlanets:(i + 1) * numbOfPlanets] += G * m[j] * rDiff / (dist ** 3)
    return acc


def verlet_nBody_OneT(coordinatesAndVel, m, numberOfPlanets, jPlanet, n_iter, delta_t):
    acc = acc_getOne(m, coordinatesAndVel[0, : numberOfPlanets], jPlanet)
    for j in np.arange(n_iter - 1) + 1:
        coordinatesAndVel[j, jPlanet] = coordinatesAndVel[j - 1, jPlanet] + coordinatesAndVel[j - 1, (jPlanet+numberOfPlanets)] * delta_t + 0.5 * acc[0:2] * (delta_t ** 2)
        acc_next = acc_getOne(m, coordinatesAndVel[j, jPlanet], jPlanet)
        coordinatesAndVel[j, jPlanet+numberOfPlanets] = coordinatesAndVel[j - 1, jPlanet+numberOfPlanets] + 0.5 * (acc + acc_next) * delta_t
        acc = acc_next
    return




#sync between threads
def Sync(events, controlevent, loopevent, isfreeevent):
    print("Sync")
    N = len(events)
    while(1):
        loopevent.wait()
        if(isfreeevent.is_set()):
            break
        for i in range(N):
            events[i].wait()
            events[i].clear()
            # print("Event number: ", i)
        controlevent.set()
        loopevent.clear()


def verlet_nBody_treading3():

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
    dt=delta_t
    dimension = len(ip) // len(m)


    def solveForOneBody(body, event, controlevent, loopthread):
        cur_acceleration = acceleration(body, m, coordinatesAndVel[0, :numberOfPlanets])
        for j in np.arange(n - 1) + 1:
            event.set()
            loopthread.set()
            controlevent.wait()
            controlevent.clear()
            coordinatesAndVel[j, body*dimension:(body+1)*dimension] = \
                (coordinatesAndVel[j-1, body*dimension:(body+1)*dimension]
                 + coordinatesAndVel[j-1, numberOfPlanets+body*dimension:numberOfPlanets+(body+1)*dimension] * dt
                 + 0.5 * cur_acceleration * dt**2)
            event.set()
            loopthread.set()
            controlevent.wait()
            controlevent.clear()

            next_acceleration = acceleration(body, m, coordinatesAndVel[j, :numberOfPlanets])
            coordinatesAndVel[j, numberOfPlanets + body*dimension:numberOfPlanets + (body+1)*dimension] = \
                (coordinatesAndVel[j-1, numberOfPlanets + body*dimension:numberOfPlanets + (body+1)*dimension]
                 + 0.5 * (cur_acceleration + next_acceleration) * dt)
            cur_acceleration = next_acceleration
            event.set()
            loopthread.set()
            controlevent.wait()
            controlevent.clear()
        return

    events = []
    for body in m:
        events.append(threading.Event())
        events[-1].clear()
    events[0].set()

    isfreeevent = threading.Event()
    controlevent = threading.Event()
    loopevent = threading.Event()
    controlthread = threading.Thread(target=Sync, args=(events, controlevent, loopevent, isfreeevent))
    controlthread.start()

    t=[]
    for i in range(m.size):
        t.append(threading.Thread(target=solveForOneBody, args=(i,events[i], controlevent, loopevent)))
        t[i].start()
    for i in range(m.size):
        t[i].join()
    loopevent.set()
    isfreeevent.set()
    controlthread.join()

    x1_traj = coordinatesAndVel[:, 0];
    y1_traj = coordinatesAndVel[:, 1];
    x2_traj = coordinatesAndVel[:, 2];
    y2_traj = coordinatesAndVel[:, 3];
    x3_traj = coordinatesAndVel[:, 4];
    y3_traj = coordinatesAndVel[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj, "r", x2_traj, y2_traj, "g", x3_traj, y3_traj, "b")
    canvas.draw()
    print(x1_traj)
    plt.show()

    return coordinatesAndVel


def verlet_nBody_treading2():
    print("Treading function 4 treads 2")


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

    tspan = np.arange(n) * delta_t
    numberOfPlanets = ip.size

    dt=delta_t
    print("Tread starting")
    times = np.arange(n) * dt
    half = ip.size
    dimension = len(ip) // len(m)

    times = np.arange(n) * dt
    half = ip.size
    dimension = len(ip) // len(m)

    coordinatesAndVel = np.empty((times.size, half * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))

    def solveForOneBody(row_body, event_wait, event_set):
        for body in row_body:
            cur_acceleration = acceleration(body, m, coordinatesAndVel[0, :half])
            for j in np.arange(n - 1) + 1:
                event_wait.wait()
                event_wait.clear()
                coordinatesAndVel[j, body*dimension:(body+1)*dimension] = \
                (coordinatesAndVel[j-1, body*dimension:(body+1)*dimension]
                 + coordinatesAndVel[j-1, half+body*dimension:half+(body+1)*dimension] * dt
                 + 0.5 * cur_acceleration * dt**2)
                next_acceleration = acceleration(body, m, coordinatesAndVel[j, :half])
                coordinatesAndVel[j, half + body*dimension:half + (body+1)*dimension] = \
                (coordinatesAndVel[j-1, half + body*dimension:half + (body+1)*dimension]
                 + 0.5 * (cur_acceleration + next_acceleration) * dt)
                cur_acceleration = next_acceleration
                event_set.set()
        return

    events = []
    for body in range(m.size):
        events.append(threading.Event())
        events[-1].clear()
    events[0].set()

    threads = []
    number_of_cpus = mp.cpu_count()
    n_body_for1procc = int(np.ceil((len(m)) / number_of_cpus))

    for i in range(0, number_of_cpus):
        if (n_body_for1procc + i * n_body_for1procc) <= len(m):
            row = np.arange(n_body_for1procc) + i * n_body_for1procc;
            e_set=events[i]
            e_wait=events[i-1]
            threads.append(threading.Thread(target=solveForOneBody, args=(row, e_wait, e_set)))
            threads[-1].start()

        elif (n_body_for1procc + (i - 1) * n_body_for1procc) < len(m):
            e_set = events[i]
            e_wait = events[i - 1]
            row = np.arange(n_body_for1procc + (i - 1) * n_body_for1procc, len(m))
            threads.append(threading.Thread(target=solveForOneBody, args=(row,  e_wait, e_set)))
            threads[-1].start()


    for thread in threads:
        thread.join()

    print("Tread ending")

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
    return

def solveForOneBodyM(q, q_out, init_vel, shared_pos, body_row, events1, events2,m,delta_t,n, procId):
    dimension = len(shared_pos) // len(m)
    result_array=[]
    for body in body_row:
        my_cur_pos = np.array(shared_pos[body*dimension:(body+1)*dimension])
        cur_acceleration = acceleration_no_np(body, m, shared_pos)

    result = np.empty((n, dimension * 2))
    result[0, :] = np.concatenate((my_cur_pos, init_vel[body*dimension:(body+1)*dimension]))

    for j in np.arange(n - 1) + 1:
        for body in body_row:
            result[j, :dimension] = \
                (my_cur_pos
                 + result[j-1, dimension:] * delta_t
                 + 0.5 * cur_acceleration * delta_t**2)
        for body in body_row:
            q.put([body, result[j, :dimension]])
        events1[procId].set()
        # print("it=",j,"procId=",procId)
        if procId == 0:
            for i in range(len(events1)):
                # print("wait For=",i)
                events1[i].wait()
                # print("waitED For=", i)
                events1[i].clear()
            for i in range(len(m)):
                tmp = q.get()
                shared_pos[tmp[0]*dimension:(tmp[0]+1)*dimension] = tmp[1]
            for i in range(len(events2)):
                    events2[i].set()
        else:
            events2[procId].wait()
            events2[procId].clear()

        my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
        for body in body_row:
            next_acceleration = acceleration_no_np(body, m, shared_pos)

            result[j, dimension:] = \
                (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
            cur_acceleration = next_acceleration
            result_array.append([body, result])
    for body in body_row:
        q_out.put(result_array)
    return




def solveForOneBody( init_vel, shared_pos, body_row, events1, events2,m,delta_t,n):
    dimension = len(shared_pos) // len(m)
    result_array=[]
    for body in body_row:
        my_cur_pos = np.array(shared_pos[body*dimension:(body+1)*dimension])
        cur_acceleration = acceleration_no_np(body, m, shared_pos)

        result = np.empty((n, dimension * 2))
        result[0, :] = np.concatenate((my_cur_pos, init_vel[body*dimension:(body+1)*dimension]))

        for j in np.arange(n - 1) + 1:
            result[j, :dimension] = \
                (my_cur_pos
                 + result[j-1, dimension:] * delta_t
                 + 0.5 * cur_acceleration * delta_t**2)

            # q.put([body, result[j, :dimension]])
            events1[body].set()

            if body == 0:
                for i in range(len(m)):
                    events1[i].wait()
                    events1[i].clear()
                for i in range(len(m)):
                    # tmp = q.get()
                    shared_pos[body*dimension:(body+1)*dimension] = result[j, :dimension]
                for i in range(len(m)):
                    events2[i].set()
            else:
                events2[body].wait()
                events2[body].clear()

            my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
            next_acceleration = acceleration_no_np(body, m, shared_pos)

            result[j, dimension:] = \
                (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
            cur_acceleration = next_acceleration
            result_array.append([body, result])
    # q_out.put(result_array)
    return


def verlet_nBody_multiprocessing():
    number_of_cpus = mp.cpu_count()
    print("mul proc start")
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
    #

    times = np.arange(n) * delta_t

    coordinatesAndVel = np.empty((times.size, ip.size * 2))
    shared_pos = mp.Array('d', ip)

    n_body_for1procc = int(np.ceil((len(m)) / number_of_cpus))

    events1 = []
    events2 = []
    for i in range(0, number_of_cpus):
        if (n_body_for1procc + i * n_body_for1procc) <= len(m):
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[len(events1)-1].clear()
            events2[len(events1)-1].clear()
        elif (n_body_for1procc + (i - 1) * n_body_for1procc) < len(m):
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[len(events1)-1].clear()
            events2[len(events1)-1].clear()

    q = mp.Queue()
    q_out = mp.Queue()
    processes = []
    print("n_body_for1procc=",n_body_for1procc)
    for i in range(0, number_of_cpus):
        if (n_body_for1procc + i * n_body_for1procc) <= len(m):
            row = np.arange(n_body_for1procc) + i * n_body_for1procc;
            # print("row", row)
            p=mp.Process(target=solveForOneBodyM, args=(q, q_out, iv, shared_pos, row, events1, events2, m,delta_t,n,i))
            processes.append(p)
            p.start()
        elif (n_body_for1procc + (i - 1) * n_body_for1procc) < len(m):
            row = np.arange(n_body_for1procc + (i - 1) * n_body_for1procc, len(m))
            # print("row_=", row)
            p=mp.Process(target=solveForOneBodyM, args=(q, q_out, iv, shared_pos,row,  events1, events2, m,delta_t,n,i))
            processes.append(p)
            p.start()

    dim = ip.size // len(m)
    for i in processes:
        resFromq = q_out.get()
        for tmp in resFromq:
            coordinatesAndVel[:, tmp[0] * dim:(tmp[0] + 1) * dim] = tmp[1][:, :dim]
            coordinatesAndVel[:, ip.size + tmp[0] * dim: ip.size + (tmp[0] + 1) * dim] = tmp[1][:, dim:]

    for process in processes:
        process.join()
    #
    #
    x1_traj = coordinatesAndVel[:, 0];
    y1_traj = coordinatesAndVel[:, 1];
    x2_traj = coordinatesAndVel[:, 2];
    y2_traj = coordinatesAndVel[:, 3];
    x3_traj = coordinatesAndVel[:, 4];
    y3_traj = coordinatesAndVel[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    canvas.draw()
    # print(x1_traj)
    plt.show()
    # z = verlet(initz, T, n)
    print("mul proc end")
    return


def OneProcFor1BodyM2(q, q_out, init_vel, shared_pos, body, events1, events2,m,delta_t,n):
    dimension = len(shared_pos) // len(m)
    my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
    cur_acceleration = acceleration_no_np(body, m, shared_pos)

    result = np.empty((n, dimension * 2))
    result[0, :] = np.concatenate((my_cur_pos, init_vel[body * dimension:(body + 1) * dimension]))

    for j in np.arange(n - 1) + 1:
        result[j, :dimension] = \
            (my_cur_pos
             + result[j - 1, dimension:] * delta_t
             + 0.5 * cur_acceleration * delta_t ** 2)

        q.put([body, result[j, :dimension]])
        events1[body].set()

        # print(body,"wait", j)

        if body == 0:
            for i in range(len(m)):
                events1[i].wait()
                events1[i].clear()
            for i in range(len(m)):
                tmp = q.get()
                shared_pos[tmp[0] * dimension:(tmp[0] + 1) * dimension] = tmp[1]
            for i in range(len(m)):
                events2[i].set()
        else:
            events2[body].wait()
            events2[body].clear()

        my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
        next_acceleration = acceleration_no_np(body, m, shared_pos)

        result[j, dimension:] = \
            (result[j - 1, dimension:]
             + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
        cur_acceleration = next_acceleration
        # print(body, "not wait", j)
    q_out.put([body, result])


def verlet_nBody_multiprocessing2():
    number_of_cpus = mp.cpu_count()
    print("mul proc start")
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
    #


    # if __name__ == '__main__':
    times = np.arange(n) * delta_t

    coordinatesAndVel = np.empty((times.size, ip.size * 2))
    shared_pos = mp.Array('d', ip)
    n_body_for1procc = int(np.ceil((len(m)) / number_of_cpus))

    events1 = []
    events2 = []
    for body in m:
        events1.append(mp.Event())
        events2.append(mp.Event())
        events1[-1].clear()
        events2[-1].clear()

    q = mp.Queue()
    q_out = mp.Queue()
    processes = []
    for body in range(m.size):
        processes.append(
        mp.Process(target=OneProcFor1BodyM2, args=(q, q_out, iv, shared_pos, body, events1, events2,m,delta_t,n)))
        processes[-1].start()

    dim = ip.size // len(m)
    for i in range(len(m)):
        tmp = q_out.get()
        coordinatesAndVel[:, tmp[0] * dim:(tmp[0] + 1) * dim] = tmp[1][:, :dim]
        coordinatesAndVel[:, ip.size + tmp[0] * dim: ip.size + (tmp[0] + 1) * dim] = tmp[1][:, dim:]

    for process in processes:
        process.join()

    #
    x1_traj = coordinatesAndVel[:, 0];
    y1_traj = coordinatesAndVel[:, 1];
    x2_traj = coordinatesAndVel[:, 2];
    y2_traj = coordinatesAndVel[:, 3];
    x3_traj = coordinatesAndVel[:, 4];
    y3_traj = coordinatesAndVel[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj,"r", x2_traj, y2_traj,"g", x3_traj, y3_traj,"b")
    canvas.draw()
    # print(x1_traj)
    plt.show()
    # z = verlet(initz, T, n)
    print("mul proc end")
    return



# 2. Реализовать метод Верле на cython  без использования typed memoryview,
# с его использованием и
# без использования openmp и с его использованием
# ( всего 4 комбинации:
# без typed memoryview и без openmp,
# без typed memoryview и c openmp,
# с typed memoryview  и без openmp,
# с typed memoryview и c openmp).
# Добавить кнопки, соответствующие этим режимам на GUI.
# Добавить время выполнения расчета метода Верле на GUI.

def cton_nBody():
    print("cton begin")
    import MyLittlePython.verletCyton_noV_noO as vc

    coordinatesAndVel=vc.verlet_nBody()
    print(coordinatesAndVel)
    x1_traj = coordinatesAndVel[:, 0];
    y1_traj = coordinatesAndVel[:, 1];
    x2_traj = coordinatesAndVel[:, 2];
    y2_traj = coordinatesAndVel[:, 3];
    x3_traj = coordinatesAndVel[:, 4];
    y3_traj = coordinatesAndVel[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj, "r", x2_traj, y2_traj, "g", x3_traj, y3_traj, "b")
    canvas.draw()
    # print(x1_traj)
    plt.show()
    print("cton end")
    return

def openCL_nBody():
    print("openCL_nBody begin")
    import openCL_cpu as vcl
    m = np.array([1.98892e30, 7.34767309e22, 5.972e24])  # the sun  the moon the earth
    ip = np.array([0.0, 0, 149984467000.0, 0, 149600000000.0, 0 ])
    # ip=np.array([1.0,2.0,3.0,4.0,5.0,6.0])
    iv = np.array([0, 0, 0, (2.9783e4 + 1022), 0, 2.9783e4])
    # iv = np.array([7, 8, 9, 10, 11, 12])
    delta_t = 60 * 60 * 24
    n = 365 * 1

    coordinatesAndVel_old, loc_t=vcl.nBodiesVerlet_OpenCL(m, ip, iv, delta_t, n)
    coordinatesAndVel=np.reshape(coordinatesAndVel_old, (n, ip.size * 2))
    print(coordinatesAndVel)
    x1_traj = coordinatesAndVel[:, 0];
    y1_traj = coordinatesAndVel[:, 1];
    x2_traj = coordinatesAndVel[:, 2];
    y2_traj = coordinatesAndVel[:, 3];
    x3_traj = coordinatesAndVel[:, 4];
    y3_traj = coordinatesAndVel[:, 5];
    a.cla()
    a.plot(x1_traj, y1_traj, "r", x2_traj, y2_traj, "g", x3_traj, y3_traj, "b")
    canvas.draw()
    # print(x1_traj)
    plt.show()
    print("openCL_nBody end")
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
    elif (modeOfComputation=="3"):
        print("it's_verlet_treading")
        verlet_nBody_treading3()
    elif (modeOfComputation=="4"):
        print("it's_verlet_multiprocessing")
        verlet_nBody_multiprocessing()
    elif (modeOfComputation=="5"):
        print("cyton noV noO")
        cton_nBody()
    elif (modeOfComputation == "6"):
        print("openCL")
        openCL_nBody()




if __name__ == '__main__':
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