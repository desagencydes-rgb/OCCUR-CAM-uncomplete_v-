import tkinter as tk
from tkinter import ttk

print("Creating test window...")
root = tk.Tk()
print("Window created")

root.title("Test Window")
print("Title set")

label = ttk.Label(root, text="Hello World!")
label.pack()
print("Label added")

print("Starting main loop...")
root.mainloop()
print("Main loop ended")