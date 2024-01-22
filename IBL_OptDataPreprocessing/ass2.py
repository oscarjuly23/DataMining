#!/opt/local/bin/python3
# -*- coding: utf-8 -*-

# Pràctica 2 - Optimització, Preprocés i IBL aplicats amb scikit-learn
# Mineria de Dades, Curs 2021-22
# @author Oscar Julian Ponte(oscar.julian)
# 18 de Diciembre 2021

import numpy as np
import sklearn
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

def ass2():

    # [1] Digits Data Set
    digits = sklearn.datasets.load_digits()
    X = digits.data # Matriu X amb 1797 files i 64 columnes (Matrius 8x8)
    Y = digits.target # Variable Y amb 1797 files (posiciones imagenes)
    print(X.shape, Y.shape) # Mostra matriu
    print(digits.DESCR) # + Info
    plt.imshow(digits.images[0]) # Plot de la grafica de las imagenes
    digits.images[digits.target == 0] # Todas las imagenes que son 0
    plt.imshow(digits.images[digits.target == 9] [20])

    # Descripcion estadistica de como son los atributos:
    np.mean(X[Y == 7], axis = 0)  # Trazo del valor medio de pixeles de todas las imagenes con valor 7.
    plt.imshow(np.reshape(np.mean(X[Y == 3], axis = 0), [8,8])) # Este es el trazo medio con valor de las Y con valor de 3
    plt.imshow(np.reshape(np.mean(X[Y == 2], axis=0), [8, 8]))  # Este es el trazo medio con valor de las Y con valor de 2

    # Calculo de la desviacion típica  (+ Opcional con el plot)
    plt.imshow(np.reshape(np.std(X[Y == 3], axis = 0), [8,8])) # Nos dice donde hay más variaciones en el 3

    # Cuantos hay?
    np.sum(Y == 0) # Hay '178' 0, hay '182' 'x'.... (El que tiene mas el '3' i el q menos el '8') De esta manera podemos ver si hay algun problema de balance de clases.
    for i in range(10) :
        print("{} -> {}".format(i, np.sum(Y == i)))

    # Nos pide datos estadisticos, como medianas, desviaciones.... buscar q mas podemos sacar


    # [2] Division con train i test i normalización de datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, train_size = 0.7) # Nos devuelve las listas (dividimos entre train 70% i test 30%)
    print(X_train.shape) # 1257 imagenes que serviran para entrenar el modelo
    print(Y_train.shape) # Estan los dígitos
    print(Y_test.shape) # Estan los dígitos para hacer la comprobación en test
    print(X_test.shape) # 540 imagenes para testear el problema

    # Preprocessing: MIRAR LINK
    # http://scikit-learn.org/stable/modules/preprocessing.html

    # Normalitzamos los datos de train i test (escalamos los datos entre 0 i 1)
    X_train / np.max(X_train) # Todos los datos entre 0 i 1


    # [3] Proyección en diferentes componentes
    # Método 1 - PCA:
    model = decomposition.PCA(n_components = 2) # Model es un entrenador de PCA que va a tener 2 componentes principales (2 dimensiones)
    # Hemos hecho la descomposición en 2 componentes (2 dimensiones). Se puede hacer en 3 o más, pero entre 2 i 3 para hacer el plot i poder visualizarlo. (max 64 componentes)
    pca_train = model.fit(X_train) # Obtenemos vectores
    pca_train.components_.shape # (2, 64) --> 2 Vectores x64 píxeles (he rotado)
    X_train_pca = model.transform(X_train) # --> He reducido la dimensionalidad de 64 variables a 2 dimensiones
    # Ahora tenemos las 1257 imagenes en 2 posiciones del espacio, hemos hecho una rotacion de 64 dimensiones a 2. (Proyección)
    plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c = Y_train) # Nos quedan todos los puntos divididos -> Para escalarlo podemos aplicar: (Y_train / 10.0)

    # Método 2 - LDA:
    # Con este método conseguimos mejorar la separación de los datos respecto al PCA.
    # Hablar sobre la comparación de PCA y LDA (mirar teoria y mirar artículo)
    # NO ES OBLIGATORIO HACER LDA --> CLASSE 2 LO IMPLEMENTA?
    # Se puede hacer el PCD con SVD --> decomposition con TruncatedSVD, X_train_svd --> plt.clf() ax = Axes3D...
    # max num de componentes = num classes -1 (en este caso 10-1 = 9!)

    # Método 3 - TSNE:
    # TSNE hace una descomposicion en base a la 'T de studen??????' y nos muestra los valores muy separados
    # NO ES OBLIGATORIO
    model = manifold.TSNE
    X_train_tsne = model(n_components = 2, learning_rate = 'auto', init = 'pca').fit_transform(X_train, Y_train) # FALLO, LO VUELVE A EXPLICAR?

    # [4] Número óptimo de vecinos K.
    # Fent servir la funció sklearn.model_selection.KFold, farem una divisió en 10-fold cross validation per estimar primer el nombre òptim de veïns, el nombre òptim de dimensions.
    # DEFINIR FUNCION DE TEST
    # DEFINIR FUNCION DE TEST Q AVALUE IDONEITAT --> sklearn.metrics
    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # def compute_test(x_test, y_test, clf, cv): Kfolds = sklearn.model_selection.Kfold(…)
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    # scores = []
    # for i, j in Kfolds: …
        # scores.append(…)
    # return scores

    # Implementación búsqueda de vecinos K más cercanos:
    # KNeighborsClassifies nos ayuda a parametrizar respeto K elementos:
    parameters =  { 'n_neighbors': 3}
    knearest = KNeighborsClassifier(parameters) # N de paràmetros --> seran los 3 vecinos más cercanos
    model = knearest.fit(X_train,Y_train) # Ahora tenemos un modelo de knearest neighbor, entrenado con X_train i Y_train
    model.predict(X_test) # Podemos hacer una prediccion mediante Nearest Kneighbord
    model.predict(X_test) == Y_test # Podemos ver que algunos si y algunos no
    np.sum(model.predict(X_test) == Y_test)/Y_test.shape # Ha predecido un 99,88% de las posibles imagenes
    # Si subimos el num de vecinos y hacemos el fit podemos ver como el predict mejora un poco, pero muy da un resultado muy parecido.
    # Probar otros valores!, podemos mirar el mejor con un for en este sistema

    # Para tener una mejor classificación podemos usar el modelo GridSearch:
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    # CONCLUSIONS