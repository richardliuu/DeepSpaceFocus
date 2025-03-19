import tkinter as tk
from tkinter import Menu, Label, Entry, StringVar
from tkinter import ttk

# Tkinter Window Setup
r = tk.Tk()
r.geometry("600x500")
r.title("DeepSpaceFocus")

# Create main container
container = tk.Frame(r)
container.pack(fill="both", expand=True)

# Dictionary to store frames
frames = {}

# Menu Bar Config
menu = Menu(r)
r.config(menu=menu)

# Function to switch frames
def show_frame(frame):
    frame.tkraise()

# ============= HOME PAGE =============
home_page = tk.Frame(container)
frames["home"] = home_page
home_page.grid(row=0, column=0, sticky="nsew")

# Home page widgets using grid
home_title = Label(home_page, text="Welcome to DeepSpaceFocus", font=("Arial", 16, "bold"))
home_title.grid(row=0, column=0, padx=20, pady=20, columnspan=3)

home_info = Label(home_page, text="Your productivity assistant")
home_info.grid(row=1, column=0, padx=20, pady=10, columnspan=3)

# Additional home content can be added here
home_description = Label(home_page, text="Use the menu above to navigate between features")
home_description.grid(row=2, column=0, padx=20, pady=30, columnspan=3)

# ============= TIMER PAGE =============
timer_page = tk.Frame(container)
frames["timer"] = timer_page
timer_page.grid(row=0, column=0, sticky="nsew")

# Timer page title
timer_title = Label(timer_page, text="Break Timer", font=("Arial", 14, "bold"))
timer_title.grid(row=0, column=0, padx=20, pady=20, columnspan=3)

# Time unit selection
time_unit_label = Label(timer_page, text="Select time unit:")
time_unit_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

time_unit_var = StringVar()
combo_box = ttk.Combobox(timer_page, textvariable=time_unit_var, 
                         values=["Minutes", "Seconds", "Hours"])
combo_box.grid(row=1, column=1, padx=10, pady=10, sticky="w")
combo_box.set("Minutes")

selected_unit_label = Label(timer_page, text="")
selected_unit_label.grid(row=1, column=2, padx=10, pady=10, sticky="w")

# Time value entry
time_value_label = Label(timer_page, text="Enter time value:")
time_value_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

time_entry = Entry(timer_page)
time_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

# Start button
start_button = tk.Button(timer_page, text="Start Timer", bg="#4CAF50", fg="white")
start_button.grid(row=3, column=0, columnspan=3, padx=20, pady=20)

# Function to update the selected time unit label
def select_time(event):
    selected_time = time_unit_var.get()
    selected_unit_label.config(text="Selected: " + selected_time)

combo_box.bind("<<ComboboxSelected>>", select_time)

# ============= HELP PAGE =============
help_page = tk.Frame(container)
frames["help"] = help_page
help_page.grid(row=0, column=0, sticky="nsew")

# Help page widgets using grid
help_title = Label(help_page, text="Help & Instructions", font=("Arial", 14, "bold"))
help_title.grid(row=0, column=0, padx=20, pady=20, columnspan=2)

help_subtitle = Label(help_page, text="How to use DeepSpaceFocus:", font=("Arial", 12))
help_subtitle.grid(row=1, column=0, padx=20, pady=10, sticky="w", columnspan=2)

help_text1 = Label(help_page, text="1. Use the Timer feature to set break reminders")
help_text1.grid(row=2, column=0, padx=30, pady=5, sticky="w", columnspan=2)

help_text2 = Label(help_page, text="2. Navigate using the menu bar at the top")
help_text2.grid(row=3, column=0, padx=30, pady=5, sticky="w", columnspan=2)

help_text3 = Label(help_page, text="3. Contact support at support@deepspacefocus.com")
help_text3.grid(row=4, column=0, padx=30, pady=5, sticky="w", columnspan=2)

# ============= MENU CONFIGURATION =============
# Home Menu
home_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Home", menu=home_menu)
home_menu.add_command(label="Home", command=lambda: show_frame(home_page))

# Timer Menu
timer_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Timer", menu=timer_menu)
timer_menu.add_command(label="Set Timer", command=lambda: show_frame(timer_page))

# Help Menu
help_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="Controls", command=lambda: show_frame(help_page))

# Configure grid weights to make frames expandable
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# Make all frames expand to fill the container
for frame in frames.values():
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

# Show home page initially

show_frame(frames["home"])

r.mainloop()