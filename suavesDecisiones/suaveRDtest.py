from tkinter import Tk, Label, Button, StringVar, filedialog
import matplotlib.pyplot as plt
from math import sqrt
from tkinter import *
from tkinter import ttk
import numpy as np
import os.path
from sklearn.model_selection import train_test_split

def remove_last_element(arr):
    return arr[np.arange(arr.size - 1)]

class Aplicacion():

    #ventana inicial de la aplicación
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


    #Interfaz para cargar archivo y seleccionar los atributos y el numero de vecinos a graficar
    def upFile(self,event=None):
        self.filename = filedialog.askopenfilename()
        self.numAtb=IntVar() #variable que el almacena el primer atributo seleccionado
        self.numAtb2=IntVar() #variable que almacena el segundo atributo seleccionado
        self.k_vecinos=IntVar() #número de vecinos con el que el usuario desea suavizar
        self.filas = IntVar()
        self.numAtributos = IntVar()
        self.numClases = IntVar()
        self.raiz2 = Toplevel()
        self.raiz2.geometry('500x300')
        self.raiz2.resizable(10, 10)
        self.raiz2.title('Mostrar el archivo de datos')

        with open(self.filename, "r") as f:
           text = f.readlines()

           self.filas = text[0].strip()
           self.numAtributos= text[1].strip()
           self.numClases = text[2].strip()
        self.archivo = os.path.basename(self.filename)

        self.etiq0 = Label(self.raiz2, text="Atributos a graficar ").grid(row= 0, column=2)
        self.separ1 = Label(self.raiz2, text= " ").grid(row=1)
        self.etiq1 = Label(self.raiz2, text="Atributo: ").grid(row=2, column= 0)
        self.atributo1 = Entry(self.raiz2, textvariable=self.numAtb,width=8).grid(row=2,column=1)

        self.etiq2 = Label(self.raiz2, text="Atributo: ").grid(row=2, column=2)
        self.atributo2 = Entry(self.raiz2, textvariable=self.numAtb2,width=8).grid(row=2, column=3)

        self.separ2 = Label(self.raiz2, text=" ").grid(row=3)
        self.graficar = Button(self.raiz2,text ="Graficar", command = self.graficar).grid(row=4, column=1)

        self.etiq3 = Label(self.raiz2, text="#Vecinos").grid(row=4, column=2)
        self.numVecinos = Entry(self.raiz2, textvariable=self.k_vecinos,width=8).grid(row=4, column=3)
        self.suavizar = Button(self.raiz2,text ="Suavizar", command = self.suavizar).grid(row=4, column=4)

        self.separ3 = Label(self.raiz2, text=" ").grid(row=5)
        self.mostrar1= Label(self.raiz2,text="Nombre del dataset: ").grid(row=6, column=1)
        self.mostrar01= Label(self.raiz2,text=self.archivo).grid(row=6, column=2)

        self.mostar2 = Label(self.raiz2, text="Número de elementos: ").grid(row=7, column=1)
        self.mostrar02= Label(self.raiz2,text=self.filas).grid(row=7, column=2)

        self.mostar3= Label(self.raiz2, text="Número de Atributos: ").grid(row=8, column=1)
        self.mostrar03= Label(self.raiz2,text=self.numAtributos).grid(row=8, column=2)

        self.mostar4= Label(self.raiz2, text="Número de Clases: ").grid(row=9, column=1)
        self.mostrar04= Label(self.raiz2,text=self.numClases).grid(row=9, column=2)

        self.separ4 = Label(self.raiz2, text=" ").grid(row=10)
        self.partition = Button(self.raiz2,text ="Partición", command = self.separar_datos  ).grid(row=11, column=2)



        self.raiz2.transient(master=self.raiz)
        self.raiz2.grab_set()
        self.raiz.wait_window(self.raiz2)

    def separar_datos(self):
        alldata = np.loadtxt(self.filename, skiprows=3, delimiter=',')

        print(alldata)
        clases = [int(row[-1]) for row in alldata] #creamos un arreglo que guarda todas las clases de cada fila

        data = []
        for row in alldata:
            data.append(remove_last_element(row))

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        #randomstate = n permite que siempre tengas el mismo conjunto
        #se quita porque hago particion manual
        (trainX, testX, trainY, testY) = train_test_split(np.asarray(data), clases, test_size=0.25)


    #funcion que nos permite graficar el dataset
    def graficar(self):

        with open(self.filename, "r") as f:
           text = f.readlines()

           elementos = text[0].strip()
           atributos = text[1].strip()
           clase = text[2].strip()

           data = np.loadtxt(self.filename, skiprows=3, delimiter=',')
           nombre = os.path.basename(self.filename)

        atb1 = self.numAtb.get()
        atb2 = self.numAtb2.get()

        lista_graf1 = []
        lista_graf2 = []

        plt.figure()

        for i in range(int(clase)):
           for j in range(int(elementos)):
              if data[j][int(atributos)] == i:
                 lista_graf1.append(data[j][atb1])
                 lista_graf2.append(data[j][atb2])
           plt.scatter(lista_graf1, lista_graf2, marker='x')
           lista_graf1.clear()
           lista_graf2.clear()

        print("Atributo seleccionado:  ",self.numAtb.get())
        print("Atributo seleccionado 2:  ",self.numAtb2.get())

        plt.title("Graficando Dataset {}". format(nombre))
        plt.xlabel("Atributo {}".format(atb1))
        plt.ylabel("Atributo {}".format(atb2))
        plt.show()

    #función que se invoca al presionar el botón suavizar y llama a ENN para ejecutar el suavizado
    def suavizar(self):
        file = self.filename
        atb1 = self.numAtb.get()
        atb2 = self.numAtb2.get()
        numk = self.k_vecinos.get()

        print("atb1: ", atb1)
        print("atb2", atb2)
        print("numero de vecinos ",numk)
        suave_data = self.ENN(file,atb1,atb2,numk)


    def obtener_vecinos(self,data_train, fila_prueba, k_vecinos):
        distances = list()
        for train_row in data_train:
            dist = distacia_euclidiana(fila_prueba, train_row)
            distances.append((train_row, dist)) #almacena la fila y la distancia de esa fila a la fila de prueba
        distances.sort(key=lambda tup: tup[1]) #ordena los filas tomando en cuenta el valor de la distancia para obtener las más cercanas
        neighbors = list()

        for i in range(k_vecinos):
            neighbors.append(distances[i][0])
        return neighbors #retorna completas (con su clase) las k filas más cercanas


    def clasificar(self, data_train, fila_prueba, num_vecinos):
        neighbors = self.obtener_vecinos(data_train, fila_prueba, num_vecinos)
        vecinos_clase = [row[-1] for row in neighbors] #creamos una lista que extrae solo las clases de los vecinos mas cercanos
        #print("valores de salida ",vecinos_clase)
        return vecinos_clase

    def ENN(self,file,atb1,atb2,numk):

        suave_lista = []
        removidos = []

        with open(file, "r") as f:
            text = f.readlines()

            num_elementos = text[0].strip()
            num_atributos = text[1].strip()
            num_clases = text[2].strip()

        dataset = np.loadtxt(file, skiprows=3, delimiter=',')

        for fila in dataset:

            vecinos = self.clasificar(dataset,fila,numk)
            counter = 0
            mas_votado = vecinos[0]
            #print("Valor de vecinos [0] ",mas_votado)

            #realizamos la votación
            for i in vecinos:
                votacion = vecinos.count(i)
                if votacion > counter:
                   counter = votacion
                   mas_votado = i

            #numi= int(fila[int(num_atributos)])
            #print("clase predecida: ", num)
            #print("clase real ",numi)
            if int(mas_votado) == int(fila[int(num_atributos)]):
                suave_lista.append(fila)
            else:
                removidos.append(fila)

        list1 = []
        list2 = []

        # graficando el nuevo conjunto suavizado
        for i in range(int(num_clases)):
            for fila in suave_lista:
                if fila[int(num_atributos)] == i:
                    list1.append(fila[int(atb1)])
                    list2.append(fila[int(atb2)])

            plt.scatter(list1, list2, marker='x')
            list1.clear()
            list2.clear()


        plt.title("Suavizando con {} vecinos".format(numk))
        plt.xlabel("Attribute {}".format(atb1))
        plt.ylabel("Attribute {}".format(atb2))
        plt.show()

        print("Removidos: ",len(removidos))
        return suave_lista


#Calcula la distancia ecuclidiana entre dos vectores
def distacia_euclidiana(row1, row2):
    aux1 = row1
    aux2 = row2
    distance = 0.0
    for i in range(len(aux1) - 1):
        distance += (aux1[i] - aux2[i]) ** 2
    return sqrt(distance)

def main():
    mi_app = Aplicacion()
    return 0


if __name__ == '__main__':
    main()
