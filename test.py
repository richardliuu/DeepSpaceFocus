import tkinter as tk
from tkinter import *

# Root window
r = tk.Tk()
r.geometry("600x500")
r.title("Monitoring Concentration")

# Function to switch frames
def show_frame(frame):
    frame.tkraise()

# Create a container for pages
container = tk.Frame(r)
container.pack(fill="both", expand=True)

# Dictionary to store pages
frames = {}

# Menu Bar Config
menu = Menu(r)
r.config(menu=menu)

# Home Page
home_page = tk.Frame(container)
frames["home"] = home_page
home_label = tk.Label(home_page, text="Welcome to DeepSpaceFocus", font=("Arial", 14))
home_label.pack(pady=20)
stop_button = tk.Button(home_page, text="Stop", width=20, command=r.destroy)
stop_button.pack()
home_page.grid(row=0, column=0, sticky="nsew")

home_menu = Menu(menu)
menu.add_cascade(label="Home", menu=home_menu)

# Break Timer Page
break_timer_page = tk.Frame(container)
frames["break_timer"] = break_timer_page
break_label = tk.Label(break_timer_page, text="Break Timer Settings", font=("Arial", 14))
break_label.pack(pady=20)
back_button1 = tk.Button(break_timer_page, text="Back", command=lambda: show_frame(home_page))
back_button1.pack()
break_timer_page.grid(row=0, column=0, sticky="nsew")

timer_menu = Menu(menu)
menu.add_cascade(label="Home", menu=timer_menu)

# Help Page
help_page = tk.Frame(container)
frames["help"] = help_page
help_label = tk.Label(help_page, text="Help Information", font=("Arial", 14))
help_label.pack(pady=20)
back_button2 = tk.Button(help_page, text="Back", command=lambda: show_frame(home_page))
back_button2.pack()
help_page.grid(row=0, column=0, sticky="nsew")

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="Controls", command=lambda: show_frame(help_page))
helpmenu.add_command(label="About", command=lambda: show_frame(help_page))

# Set default page
show_frame(home_page)

break_timer_menu = Menu(menu)
menu.add_cascade(label="Break Timer", menu=break_timer_menu)
break_timer_menu.add_command(label="Set Timer", command=lambda: show_frame(break_timer_page))
break_timer_menu.add_separator()
break_timer_menu.add_command(label="Exit", command=r.quit)


r.mainloop()
