from tkinter import *
from tkinter import ttk

width = 400
height = 500



# create a tkinter window
root = Tk()     
canvas = Canvas(root, bg = "white", width = width, height = height)
canvas.pack()        

def callback():
    Label(root, text="Hello World!", font=('Georgia 20 bold')).pack(pady=4)

# Create a Button
btn_right = ttk.Button(root, text="→", command= callback)
btn_left = ttk.Button(root, text="←", command= callback)
btn_back = ttk.Button(root, text="ↆ", command= callback)
btn_top = ttk.Button(root, text="𐍊", command= callback)
# btn.pack(ipadx=10)
# btn = Button(root, text = 'Click me !', bd = '5',command = root.destroy)

root.bind('<Return>',lambda event:callback())
# Set the position of button on the top of window.  
btn_right.pack(side = 'right')   
btn_top.pack(side = 'top') 
btn_left.pack(side = 'left')
btn_back.pack(side = 'bottom')  

root.mainloop()
