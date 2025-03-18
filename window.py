import tkinter as tk
from tkinter import *
from tkinter import ttk

# r = root 
# Tkinter Window Setup
r = tk.Tk()
r.geometry("600x500")
r.title("Monitoring Concentration") # Replace with an application name 

w = Label(r, text = "Welcome to DeepSpaceFocus") 
w.pack()

# Break Timer

def show_timer(frame):
    frame.tkraise()

container = tk.Frame(r)

def open_break_timer():
    break_win =  Toplevel(r)
    break_win.title("Break Timer")
    break_win.geometry("600x500")

    Label(break_win, text="Set your timer").pack()
    Button(break_win, text="Start Timer").pack()
    Button(break_win, text="Close", command=break_win.destroy).pack()

menu = Menu(r)
r.config(menu=menu)
break_timer_menu = Menu(menu)
menu.add_cascade(label= "Break Timer", menu=break_timer_menu)
break_timer_menu.add_command(label="Set Timer")
break_timer_menu.add_separator()
break_timer_menu.add_command(label="Exit", command=r.quit)

# Selecting a time

def select_time(event):
    selected_time = combo_box.get()
    label.config(text="Select a time: " + selected_time)

label = tk.Label(text="Select a time: ")
label.pack(pady=40)

combo_box = ttk.Combobox(r, values=["Minutes", "Seconds", "Hours"])
combo_box.pack(pady=10)

combo_box.set("Minutes")
combo_box.bind("<<ComboboxSelected>>", select_time)

# Help Menu
helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="Controls")
helpmenu.add_command(label="About")
Label(r, text = "Time")

def open_help():
    help_win = Toplevel(r)
    help_win.title("Help")
    help_win.geometry("600x500")

    Label(help_win, text="Controls").pack()
    Button(help_win, text="Exit", command=help_win.destroy).pack()

button = tk.Button(r, text ="Stop", width = 20, command = r.destroy)
button.pack()

r.mainloop()
