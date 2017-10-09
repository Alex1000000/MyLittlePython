import tkinter as tk                    # imports
from tkinter import ttk
from tkinter import Text, END, INSERT,CURRENT,SEL_FIRST
from tkinter.colorchooser import *
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np
from numpy import arange
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def _quit():
    win.quit()     # stops mainloop
    win.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


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
    print(s)
    print(x_mouse,";  ",y_mouse)
    if s!="None;  None":
        print("mose CLICK LEFT")
        if currentColorFromCombo==0:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='r')
        elif currentColorFromCombo==1:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='g')
        elif currentColorFromCombo==2:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='b')
        elif currentColorFromCombo==3:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='y')
        elif currentColorFromCombo==4:
            circle1 = plt.Circle((x_mouse, y_mouse), scaleMy.get(), color='m')

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
    fi = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
    if fi is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = "qwerty" # starts from `1.0`, not `0.0`
    fi.write(text2save)
    fi.close() # `()` was missing.

def file_open():
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("txt", "*.txt"), ("all files", "*.*")))
    file = open(filename)
    txt=file.read()
    print(txt)
    print("file open")

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


#ax = plt.axes()
#ax.transData.transform_point([x, y])
#inv = ax.transData.inverted()
cid = f.canvas.mpl_connect('motion_notify_event', onMouseMove)
# cid = f.canvas.mpl_connect('motion_notify_event', sliderMove)
win.bind("<Button-1>", onMouseClick)


win.mainloop()