from tkinter import Tk, Label, Button, StringVar, filedialog
from tkinter import *
from tkinter import ttk
import datetime, Pmw, sys


class Aplicacion():
    def __init__(self):
        self.raiz = Tk()
        self.raiz.title("Suavizando fornteras")
        self.raiz.geometry('200x200')
        self.raiz.resizable(10, 10)

        self.boton0 = ttk.Button(self.raiz, text="Cargar Archivo",command=self.upFile)
        self.boton1 = ttk.Button(self.raiz, text="Salir", command=quit)

        self.boton0.pack(side=TOP,  expand=True, padx=10, pady=10)
        self.boton1.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
        self.raiz.mainloop()

    def upFile(self,event=None):
        filename = filedialog.askopenfilename()
        self.numAtb=IntVar()
        self.numAtb2=IntVar()
        self.raiz2 = Toplevel()
        self.raiz2.geometry('600x250')
        self.raiz2.resizable(10, 10)
        self.raiz2.title('Mostrar el archivo de datos')

        self.etiq0 = Label(self.raiz2, text="Atributos a graficar ").grid(row= 0, column=2)
        self.separ1 = Label(self.raiz2, text= " ").grid(row=1)
        self.etiq1 = Label(self.raiz2, text="Atributo: ").grid(row=2, column= 0)
        self.atributo1 = Entry(self.raiz2, textvariable=self.numAtb,width=10).grid(row=2,column=1)

        self.etiq2 = Label(self.raiz2, text="Atributo: ").grid(row=2, column=2)
        self.atributo2 = Entry(self.raiz2, textvariable=self.numAtb2,width=10).grid(row=2, column=3)

        self.separ2 = Label(self.raiz2, text=" ").grid(row=3)
        self.graficar = Button(self.raiz2,text ="Graficar", command = self.graficar).grid(row=4, column=1)
        self.suavizar = Button(self.raiz2,text ="Suavizar").grid(row=4, column=2)


        entry = Entry(self.raiz2, width=50, textvariable=filename)
        #with open(filename, "r") as f:
            #Label(self.raiz2, text=f.read()).grid(row= 5, column=0)

        self.raiz2.transient(master=self.raiz)
        self.raiz2.grab_set()
        self.raiz.wait_window(self.raiz2)


    def suavizar(self):
        self.raiz3 = Toplevel()
        self.raiz3.geometry('200x600')
        self.raiz3.resizable(10, 10)
        self.raiz3.title('Selecciona el origen de tu vuelo')

        self.raiz3.transient(master=self.raiz)
        self.raiz3.grab_set()
        self.raiz.wait_window(self.raiz3)

    def graficar(self):
        self.raiz4 = Toplevel()
        self.raiz4.geometry('200x600')
        self.raiz4.resizable(10, 10)
        self.raiz4.title('Selecciona tu Destino')

        print("Atributo seleccionado:  ",self.numAtb.get())
        print("Atributo seleccionado 2:  ",self.numAtb2.get())
        self.raiz4.transient(master=self.raiz)
        self.raiz4.grab_set()
        self.raiz.wait_window(self.raiz4)


def main():
    mi_app = Aplicacion()
    return 0


if __name__ == '__main__':
    main()


def read_file(file_name):  # Function to read file

    f = open(file_name, "r")
    lines = f.readlines()
    return lines
