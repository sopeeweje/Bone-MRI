import tkinter as tk
from tkinter import filedialog
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from segmentation import calculate_largest_slice
from segmentation import select_slice
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import models
import keras
import tensorflow

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("CNN Bone-MRI GUI")
        master.geometry("650x600")
        master.configure(bg='#d9f0f0')

        self.leftFrame = tk.LabelFrame(master, text = "Image Loader...", width=300, height=300, bd=10, labelanchor='n')
        self.rightFrame = tk.LabelFrame(master, text="Choose Model...", width=300, height=300, bd=10, labelanchor='n')
        self.bottomFrame = tk.LabelFrame(master, text="Results...", width=600, height=240, bd=10, labelanchor='n')
        self.leftFrame.pack_propagate(False)
        self.rightFrame.pack_propagate(False)
        self.bottomFrame.pack_propagate(False)
        self.leftFrame.grid(row=0, column=0, padx = 10, pady= 10)
        self.rightFrame.grid(row=0, column=3, padx = 10, pady = 10)
        self.bottomFrame.grid(row=1, column=0, columnspan=4, padx=10, pady = 10)

        loadButton = tk.Button(self.leftFrame, text="Select an Image to Upload", anchor='center', command=self.upload_nrrd, borderwidth=4, padx = 2, pady = 6)
        loadButton.pack(pady = 10)

        models = ["v1", "v2", "v3", "v4"]
        modelVar = tk.StringVar(master)
        modelVar.set("--CHOOSE MODEL--")
        modelMenu = tk.OptionMenu(self.rightFrame, modelVar, *models)
        modelMenu.pack(pady=30)

        runButton = tk.Button(self.rightFrame, text="RUN MODEL", anchor='center', command= lambda: self.predict_image(None, modelVar.get()),
                               borderwidth=4, padx=2, pady=10, highlightbackground='#4EE814')
        runButton.pack(pady=30)

    def predict_image(self, image, model):
        #TODO: figure out how to make prediction, given model v1-v4
        # keras models have a .predict() function, but I don't think we have an actual finalized model yet
        if model == "--CHOOSE MODEL--":
            tk.messagebox.showerror("No Model Chosen", "Please choose a model from the drop-down menu before proceeding")
        print(model)
        modelString = str(model) #cast from stringVar to a string
        model = models[modelString] #turns string into real model, using dict
        # print(model.predict(image))

        # print the results of the model (which idk how to get yet) in the bottom frame
        self.display_result(.6, .5)

    def display_result(self, probability, threshold):
        if probability <= threshold:
            cancer = "BENIGN"
        else:
            cancer = "Malignant"
        text = tk.Text(self.bottomFrame)
        text.pack()
        text.insert(tk.END, "Probability of Malignancy = {}\nDiagnosis: {}".format(probability, cancer))

    def upload_nrrd(self):
        #TODO: allow user to change the file
        #Add confirm button?

        root.filename= filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("nrrd files", "*.nrrd"),))
        arr, _ = nrrd.read(root.filename)
        slice = calculate_largest_slice(arr)
        root.image, _ = select_slice(arr, arr, slice)
        f = Figure()
        a = f.add_subplot()
        a.imshow(root.image)

        canvas = FigureCanvasTkAgg(f, master = self.leftFrame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        canvas._tkcanvas.pack(side="top", fill="both", expand=1)
        # print("THE NUM IS: {}".format(image))
        # plt.imshow(image)
        # plt.show()

root = tk.Tk()
gui = GUI(root)
root.mainloop()