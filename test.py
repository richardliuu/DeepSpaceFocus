import tkinter as tk
from tkinter import *
from tkinter import ttk

# r = root 
# Tkinter Window Setup
r = tk.Tk()
r.geometry("600x500")
r.title("Monitoring Concentration")  # Replace with an application name 

w = Label(r, text="Welcome to DeepSpaceFocus") 
w.pack()

container = tk.Frame(r)
container.pack(fill="both", expand=True)

frames = {}

# Menu Bar Config
menu = Menu(r)
r.config(menu=menu)

# Function to switch frames
def show_frame(frame):
    frame.tkraise()

"""Home Page"""
home_page = tk.Frame(container)
frames["home"] = home_page
home_label = tk.Label(home_page, text="Home Page", font=("Arial", 14))
home_label.pack(pady=20)
home_page.grid(row=0, column=0, sticky="nsew")

home_menu = Menu(menu)
menu.add_cascade(label="Home", menu=home_menu)
home_menu.add_command(label="Home", command=lambda: show_frame(home_page))

menu.add_separator()

"""Break Timer"""
# Timer Menu
timer_page = tk.Frame(container)
frames["timer"] = timer_page
timer_label = tk.Label(timer_page, text="Set Timer", font=("Arial", 14))
timer_label.pack(pady=20)

# Selecting a time (Now inside timer_page)
def select_time(event):
    selected_time = combo_box.get()
    label.config(text="Selected time: " + selected_time)

# ** Now these are inside timer_page **  
label = tk.Label(timer_page, text="Select a time: ")
label.pack(pady=10)

combo_box = ttk.Combobox(timer_page, values=["Minutes", "Seconds", "Hours"])
combo_box.pack(pady=10)

combo_box.set("Minutes")
combo_box.bind("<<ComboboxSelected>>", select_time)

timer_page.grid(row=0, column=0, sticky="nsew")

show_frame(home_page)

timer_menu = Menu(menu)
menu.add_cascade(label="Timer", menu=timer_menu)
timer_menu.add_command(label="Set Timer", command=lambda: show_frame(timer_page))

"""Help Menu"""
help_page = tk.Frame(container)
frames["help"] = help_page
help_label = tk.Label(help_page, text="Help Menu", font=("Arial", 14))
help_label.pack(pady=20)
help_page.grid(row=0, column=0, sticky="nsew")

help_menu = Menu(menu)
menu.add_cascade(label="Help", menu=help_menu)
menu.add_separator()
help_menu.add_command(label="Controls", command=lambda: show_frame(help_page))

r.mainloop()
