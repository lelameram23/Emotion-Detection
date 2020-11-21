import tkinter
#from tkinter import Frame
from tkinter import messagebox
from tkinter import *

from emodataset import result
import status
window = tkinter.Tk()

window.title("Detection 5 Emotions From The Text")
window.geometry("900x500")
def clicked():

   result(entry1.get())

label1= Label(window)
label1.config(text="Link:")
label1.pack()

entry1=Entry(window)
entry1.pack()

button1= Button(window)
button1.config(text=" Say Something For Emotion Detection ")
button1.config(command=clicked, height=100,width=10)
button1.pack()



window.mainloop()