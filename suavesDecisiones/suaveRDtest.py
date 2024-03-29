from tkinter import Tk, Label, Button, StringVar, filedialog
import matplotlib.pyplot as plt #biblioteca para graficar
from math import sqrt #biblioteca para calcular la raiz
from tkinter import *
from tkinter import ttk #cargamos la biblioteca de la interfaz
import numpy as np # bibloiteca para hacer operaciones propia de python
import os.path # biblioteca para extraer el path y el nomre del archivo
from sklearn.model_selection import train_test_split #biblioteca para dividir en conjunto de entrenamiento y el de prueba
from sklearn.neural_network import MLPClassifier #biblioteca del clasificador multilayer perceptron
from sklearn.preprocessing import StandardScaler # bibbloiteca para normalizar los datos para el clasificador MLP
from sklearn.neighbors import KNeighborsClassifier # biblioteca del calsificador KNN
from sklearn.metrics import classification_report,confusion_matrix # biblioteca para mostrar la precision del calsificador
from sklearn import metrics

class Aplicacion():

    #Ventana inicial de la aplicación para cargar el archivo del dataset
    def __init__(self):
        #Creamos la ventana
        self.raiz = Tk()
        self.raiz.title("Suavizando fronteras")
        self.raiz.geometry('200x200')
        self.raiz.resizable(10, 10)

        self.boton0 = ttk.Button(self.raiz, text="Cargar Archivo",command=self.upFile)
        self.boton1 = ttk.Button(self.raiz, text="Salir", command=quit)

        self.boton0.pack(side=TOP,  expand=True, padx=10, pady=10)
        self.boton1.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
        self.raiz.mainloop()


    #Interfaz para seleccionar el archivo
    #Permite seleccionar los atributos y el número de vecinos a graficar
    def upFile(self,event=None):

        self.filename = filedialog.askopenfilename() #funcion para seleccionar el archivo y variable para almacenar su path
        self.numAtb=IntVar() #variable que el almacena el primer atributo seleccionado
        self.numAtb2=IntVar() #variable que almacena el segundo atributo seleccionado
        self.k_vecinos=IntVar() #número de vecinos con el que el usuario desea suavizar
        self.filas = IntVar()# número de elementos o de filas en el datset
        self.numAtributos = IntVar() #numero de atributos del dataset
        self.numClases = IntVar()#número de clases del dataset
        self.vecinos_knn=IntVar() #número de vecinos para el clasificador KNN

        #parametros de la vantana principal
        self.raiz2 = Toplevel()
        self.raiz2.geometry('700x400')
        self.raiz2.resizable(10, 10)
        self.raiz2.title('Mostrar el archivo de datos')

        with open(self.filename, "r") as f: # funcion para abrir el archivo
           text = f.readlines() #leemos las líneas del dataset

           self.filas = text[0].strip() #obtenemos el número de elementos del dataset
           self.numAtributos= text[1].strip()# obtenemos el número de atributos del datset
           self.numClases = text[2].strip() # obtenemos el número de clases del dataset
        self.archivo = os.path.basename(self.filename)#almacenamos el nombre del archivo del dataset no el path

        #Creamos la interfaz  con sus elementos como los botones, los entry, etc.
        #para realizar todas las funciones necesarias de el proyecto
        self.mostrar1 = Label(self.raiz2, text="Nombre del dataset: ").grid(row=0, column=0)
        self.mostrar01 = Label(self.raiz2, text=self.archivo).grid(row=0, column=1)

        self.mostar2 = Label(self.raiz2, text="Número de elementos: ").grid(row=1, column=0)
        self.mostrar02 = Label(self.raiz2, text=self.filas).grid(row=1, column=1)

        self.mostar3 = Label(self.raiz2, text="Número de Atributos: ").grid(row=2, column=0)
        self.mostrar03 = Label(self.raiz2, text=self.numAtributos).grid(row=2, column=1)

        self.mostar4 = Label(self.raiz2, text="Número de Clases: ").grid(row=3, column=0)
        self.mostrar04 = Label(self.raiz2, text=self.numClases).grid(row=3, column=1)

        self.separ5 = Label(self.raiz2, text=" ").grid(row=2,column=3)
        self.partition = Button(self.raiz2, text="Partición", command=self.separar_datos).grid(row=2, column=2)
        self.separ4 = Label(self.raiz2, text=" ").grid(row=4)


        self.etiq0 = Label(self.raiz2, text="Atributos a graficar ").grid(row= 5, column=2)
        self.separ1 = Label(self.raiz2, text= " ").grid(row=6)
        self.etiq1 = Label(self.raiz2, text="Atributo: ").grid(row=7, column= 0)
        self.atributo1 = Entry(self.raiz2, textvariable=self.numAtb,width=8).grid(row=7,column=1)

        self.etiq2 = Label(self.raiz2, text="Atributo: ").grid(row=7, column=2)
        self.atributo2 = Entry(self.raiz2, textvariable=self.numAtb2,width=8).grid(row=7, column=3)

        self.separ3 = Label(self.raiz2, text=" ").grid(row=8)
        self.graficar = Button(self.raiz2,text ="Graficar", command = self.graficar).grid(row=7, column=4)

        self.separ6 = Label(self.raiz2, text=" ").grid(row=8)
        self.etiq7 = Label(self.raiz2, text="#Vecinos").grid(row=9, column=2)
        self.numVecinos = Entry(self.raiz2, textvariable=self.k_vecinos,width=8).grid(row=9, column=3)
        self.suavizar = Button(self.raiz2,text ="Suavizar", command = self.suavizar).grid(row=9, column=4)

        self.separ2 = Label(self.raiz2, text=" ").grid(row=10)
        self.etiq3 = Label(self.raiz2, text="K vecinos ").grid(row=11, column=0)
        self.vecinos_KNN = Entry(self.raiz2, textvariable=self.vecinos_knn,width=8).grid(row=11, column=1)
        self.clasificadorKNN = Button(self.raiz2, text=" Clasificador KNN ", command=self.knn_clasificador).grid(row=12, column=1)
        self.multilayer_rna = Button(self.raiz2, text="Multilayer Perceptron", command=self.invocar_MP).grid(row=12,column=3)

        self.raiz2.transient(master=self.raiz)
        self.raiz2.grab_set()
        self.raiz.wait_window(self.raiz2)

    #funcion para separ el conjunto original en conjunto de entrenamiento y conjunto de prueba
    def separar_datos(self):

        alldata = np.loadtxt(self.filename, skiprows=3, delimiter=',') #cargamos el archivo de texto

        #creamos dos arreglos que nos van a permitir separ a los elementos de sus etiquetas de clase
        #para posteriormente obtener los conjunto de prueba y entrenamiento
        #creamos un arreglo que guarda unicamente todas las clases de cada elemento del conjunto completo
        clases = [int(row[-1]) for row in alldata]

        #creamos otro arreglo donde almacenamos cada elemento con sus atributos pero sin la clase
        data = []
        for row in alldata:
            data.append(remove_last_element(row))

        # diviendo el dataset en entrenamiento y prueba  usando  75% para el conjunto de entrenamiento
        # y el 25% restante para el de prueba
        #randomstate = n permite que siempre tengas el mismo conjunto
        #se quita porque hago particion manual
        (trainX, testX, trainY, testY) = train_test_split(np.asarray(data), clases, test_size=0.25)

        #unimos el conjunto de entrenamiento con sus respectivas etiquetas de clase
        #es necesario para su posterior  procesamiento de suavizado y entrenamiento
        self.train = []
        for i, row in enumerate(trainX, start=0):
            self.train.append(np.append(row, trainY[i]))

        #unimos el conjunto test con sus respectivas etiquetas
        self.test = []
        for i, row in enumerate(testX, start=0):
            self.test.append(np.append(row, testY[i]))

        #mostramos en la interfaz principal cuantos datos quedaron en el conjunto de entrenamiento
        self.mostarx = Label(self.raiz2, text="Train: ").grid(row=1, column=3)
        self.mostrar0x = Label(self.raiz2, text=len(self.train)).grid(row=1, column=4)

        # mostramos en la interfaz principal cuantos datos quedaron en el conjunto de prueba
        self.mostary = Label(self.raiz2, text="Test: ").grid(row=2, column=3)
        self.mostrar0y = Label(self.raiz2, text=len(self.test)).grid(row=2, column=4)

        self.X_train = trainX
        self.Y_train = trainY
        self.x_test = testX
        self.y_test = testY



    #funcion que nos permite graficar el dataset
    def graficar(self):

        train = np.asarray(self.train) # convertimos el conjunto de entrenamiento en un arreglo numpy

        # asignamos de manera local el atributo que selleciono el usuario a graficar
        atb1 = self.numAtb.get()
        atb2 = self.numAtb2.get()

        #creamos dos listas para graficar
        lista_graf1 = []
        lista_graf2 = []

        #graficamos el data train
        plt.figure()

        for i in range(int(self.numClases)):
           for j in range(len(train)):
              if train[j][int(self.numAtributos)] == i:
                 lista_graf1.append(train[j][atb1])
                 lista_graf2.append(train[j][atb2])
           plt.scatter(lista_graf1, lista_graf2, marker='x')
           lista_graf1.clear()
           lista_graf2.clear()

        #Etiquetas de la gráfica para saber que se está graficando
        plt.title("Graficando Dataset {}". format(self.archivo))
        plt.xlabel("Atributo {}".format(atb1))
        plt.ylabel("Atributo {}".format(atb2))
        plt.show()


    #función que se invoca al presionar el botón suavizar y llama a ENN para ejecutar el suavizado
    def suavizar(self):
        atb1 = self.numAtb.get()
        atb2 = self.numAtb2.get()
        numk = self.k_vecinos.get() #numero de vecinnos con los que se desea suavizar
        #convertimos el conjunto de entrenamiento en un arreglo numpy
        train_original = np.asarray(self.train)
        # se llama el algoritmo ENN y se pasan los parámetros necesarios,
        # retorna el conjunto suavizado
        self.suave_data = self.ENN(train_original,atb1,atb2,numk)

    #funcion que obtiene los vecinos más cercanos de un elemento en específico
    def obtener_vecinos(self,data_train, fila_prueba, k_vecinos):
        distances = list() # arreglo que almacena las distancias del elemento contra el resto de los elementos del conjunto
        for train_row in data_train:
            dist = distacia_euclidiana(fila_prueba, train_row)
            distances.append((train_row, dist)) #almacena la fila y la distancia de esa fila a la fila de prueba
        distances.sort(key=lambda tup: tup[1]) #ordena los filas tomando en cuenta el valor de la distancia para obtener las más cercanas

        neighbors = list() #lista que almacena a los k vecinos más cercanos
        for i in range(k_vecinos):
            neighbors.append(distances[i][0])
        return neighbors #retorna completas (con su clase) las k filas más cercanas


    def clasificar(self, data_train, fila_prueba, num_vecinos):
        neighbors = self.obtener_vecinos(data_train, fila_prueba, num_vecinos)
        vecinos_clase = [row[-1] for row in neighbors] #creamos una lista que extrae solo las clases de los vecinos mas cercanos
        return vecinos_clase

    def ENN(self,data,atb1,atb2,numk):

        suave_lista = []
        removidos = []

        #por cada elemento en el conjunto se buscan sus vecinos más cercanos y se compara las clases de sus vecinos con la de el mismo
        #para revisar si se encuentra en una zona fronteriza y si está traslapado con otras instancias
        for fila in data:

            vecinos = self.clasificar(data,fila,numk)
            counter = 0
            mas_votado = vecinos[0]

            #realizamos la votación para decidir si el elemento es romovido del conjunto
            for i in vecinos:
                votacion = vecinos.count(i)
                if votacion > counter:
                   counter = votacion
                   mas_votado = i


            #si la clase predominate es igual al verdadero valor del elemento se guarda
            #de lo contrario se asigna a la lista de elmentos removidos
            if int(mas_votado) == int(fila[int(self.numAtributos)]):
                suave_lista.append(fila)
            else:
                removidos.append(fila)

        list1 = []
        list2 = []

        # graficando el nuevo conjunto suavizado
        for i in range(int(self.numClases)):
            for fila in suave_lista:
                if fila[int(self.numAtributos)] == i:
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

    #en esta funcion invocamos al clasificador MLP y configuramos los parametros necesarios
    def invocar_MP(self):

        # creamos un arreglo que guarda todas las clases de cada fila del nuevo conjunto suavizado
        suaves_etiquetas = [int(row[-1]) for row in  self.suave_data]
        # creamos un arreglo que guarda todas las clases de cada fila del nuevo conjunto suavizado
        suaves_elementos = []
        for row in self.suave_data:
            suaves_elementos.append(remove_last_element(row))

        #convertimos los arreglos a arreglos numpy
        np.asarray(suaves_elementos)
        np.asarray(suaves_etiquetas)

        momentum = 0.9
        learning_rate =.1
        max_itera = 150
        nodos = 13 # específico para seg-data.txt

        #entrenamos la red con el cojunto sin suavizar y el test
        print("----- Multilayer Perceptron ----- Conjunto original")
        multi_layer_perceptron(self.X_train,self.x_test,self.Y_train,self.y_test, momentum, learning_rate, max_itera,nodos)

        #entrenamos la red con el conjunto suavizado y el mimsmo conjunto test
        print("----- Multilayer Perceptron ----- Conjunto suavizado")
        multi_layer_perceptron(suaves_elementos,self.x_test,suaves_etiquetas,self.y_test,momentum, learning_rate, max_itera,nodos)


    #clasificador KNN que utiliza por defecto la distancia ecuclidiana a menos que se indique lo contrario
    def knn_clasificador(self):

        etiquetas= []
        n_vecinos = self.vecinos_knn.get() #numero de vecinos con los que se quiere clasificar

        #se crea un arreglo para mostrar el numero de clases en el reporte
        i = 0
        while i < int(self.numClases):
            etiquetas.append(int(i))
            i = i+1

        # creamos un arreglo que guarda todas las clases de cada fila del nuevo conjunto suavizado
        suaves_etiquetas = [int(row[-1]) for row in self.suave_data]

        # creamos un arreglo que guarda todos los atributos de cada fila del nuevo conjunto suavizado
        suaves_elementos = []
        for row in self.suave_data:
            suaves_elementos.append(remove_last_element(row))

        #los convertimos a arreglos numpy para su procesamiento
        suaves_elementos = np.asarray(suaves_elementos)
        suaves_etiquetas = np.asarray(suaves_etiquetas)

        # entrenando knn con el conjunto sin suavizar
        print("      KNN --- Conjunto original con ", n_vecinos, " vecinos")
        model = KNeighborsClassifier(n_neighbors=n_vecinos)
        model.fit(self.X_train, self.Y_train)
        print(classification_report(self.y_test, model.predict(self.x_test), labels=etiquetas))

        #entrenando knn con el conjunto suavizado
        print("      KNN --- Conjunto suavizado con ", n_vecinos, "vecinos")
        mod2 = KNeighborsClassifier(n_neighbors=n_vecinos)
        mod2.fit(suaves_elementos, suaves_etiquetas)
        print(classification_report(self.y_test, mod2.predict(self.x_test), labels = etiquetas))



# funcion que nos ayuda a eliminar el último elmento (clase) de cada fila del dataset
# para dejar solo los filas con sus atributos
def remove_last_element(arr):
    return arr[np.arange(arr.size - 1)]


#Calcula la distancia ecuclidiana entre dos vectores
def distacia_euclidiana(row1, row2):
    aux1 = row1
    aux2 = row2
    distance = 0.0
    for i in range(len(aux1) - 1):
        distance += (aux1[i] - aux2[i]) ** 2
    return sqrt(distance)

#clasificador MLP
def multi_layer_perceptron(X_train, x_test, Y_train, y_test, momentum, learning_rate, max_itera,nodos):
    scaler = StandardScaler()

    # Ajustarse sólo a los datos de entrenamiento
    scaler.fit(X_train)

    # Ahora aplique las transformaciones a los datos:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(x_test)

    mlp = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes= (nodos),max_iter= max_itera ,momentum=momentum,
                        learning_rate='constant', learning_rate_init=learning_rate)

    mlp.fit(X_train, Y_train)

    predictions = mlp.predict(X_test)

    print(classification_report(y_test, predictions))
    #print("Matriz de confusión ", confusion_matrix(y_test, predictions))

    return metrics.accuracy_score(y_test, predictions)


def main():
    mi_app = Aplicacion()
    return 0


if __name__ == '__main__':
    main()
