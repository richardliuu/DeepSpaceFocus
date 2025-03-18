import tkinter as tk
from tkinter import *

# r = root 
r = tk.Tk()
r.geometry("600x500")
r.title("Monitoring Concentration") # Replace with an application name 

w = Label(r, text = "Welcome to DeepSpaceFocus") 
w.pack()


# Break Timer

menu = Menu(r)
r.config(menu=menu)
break_timer_menu = Menu(menu)
menu.add_cascade(label= "Break Timer", menu=break_timer_menu)
break_timer_menu.add_command(label="Set Timer")
break_timer_menu.add_separator()
break_timer_menu.add_command(label="Exit", command=r.quit)
helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="Controls")
helpmenu.add_command(label="About")
Label(r, text = "Time")

button = tk.Button(r, text ="Stop", width = 20, command = r.destroy)
button.pack()

r.mainloop()
