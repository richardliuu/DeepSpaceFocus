import tkinter as tk
from tkinter import *

# r = root 
r = tk.Tk()
r.geometry("600x500")
r.title("Monitoring Concentration") # Replace with an application name 

w = Label(r, text = "Welcome to DeepSpaceFocus") 
w.pack()

button = tk.Button(r, text ="Stop", width = 20, command = r.destroy)
button.pack()

r.mainloop()
