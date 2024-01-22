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
import sklearn.metrics as metrics

def ass2():

    print('[1] Digits Data Set')
    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.target
    print('Matriu: ')
    print(X.shape, Y.shape)
    #print('Descripció: ')
    #print(digits.DESCR)
    plt.imshow(digits.images[0])
    plt.title("Images 0")
    plt.show()
    #digits.images[digits.target == 0]
    plt.imshow(digits.images[digits.target == 9] [20])
    plt.title("Images == 9")
    plt.show()

    np.mean(X[Y == 7], axis = 0)
    plt.imshow(np.reshape(np.mean(X[Y == 3], axis = 0), [8,8]))
    plt.title("Trazo medio Y=3")
    plt.show()

    plt.imshow(np.reshape(np.mean(X[Y == 2], axis=0), [8, 8]))
    plt.title("Trazo medio Y=2")
    plt.show()

    plt.imshow(np.reshape(np.std(X[Y == 3], axis = 0), [8,8]))
    plt.title("Variaciones Y=3")
    plt.show()

    # Cuantos hay?
    np.sum(Y == 0)
    print('Classes: ')
    for i in range(10) :
        print("{} -> {}".format(i, np.sum(Y == i)))

    print('')
    print('[2] Division con train i test i normalización de datos')

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, train_size = 0.7) # Nos devuelve las listas (dividimos entre train 70% i test 30%)
    print('Imagenes entrenamiento: ')
    print(X_train.shape)
    print('Dígitos: ')
    print(Y_train.shape)
    print('Dígitos test: ')
    print(Y_test.shape)
    print('Imagenes test: ')
    print(X_test.shape)

    print('Preprocessing:')
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_scaled.mean(axis=0)
    X_scaled.std(axis=0)
    print(X_scaled.mean(axis=0))

    print('Normalitzation:')
    X_normal = X_train / np.max(X_train)
    print(X_normal.std(axis=0))

    print('')
    print('[3] Proyección en diferentes componentes')
    print('Método 1 - PCA')
    model = decomposition.PCA(n_components = 2)

    print('Obtenemos vectores:')
    pca_train = model.fit(X_train)
    print(pca_train.components_.shape)
    X_train_pca = model.transform(X_train)
    plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c = Y_train)
    plt.title("PCA")
    plt.show()

    print('Método 2 - LDA')

    print('Método 3 - TSNE')
    model = manifold.TSNE
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3)
    tsne = model(n_components = 2, random_state = 0)
    X_2d = tsne.fit_transform(X_train) # (tarda en processar)
    print(X_2d)
    print('Shape Train: ')
    print(X_2d.shape)

    target_ids = range(len(digits.target))
    plt.figure(figsize = (6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, digits.target):
        plt.scatter(X_2d[Y_train == i, 0], X_2d[Y_train == i, 1], c = c, label = label)
    plt.legend()
    plt.title("TSNE - Train")
    plt.show()

    X_test[Y_test == 0, 0]
    X_2d = tsne.fit_transform(np.concatenate([X_train, X_test])) # (tarda en processar)
    X_2d = X_2d[1257:,]
    print('Shape Test: ')
    print(X_2d.shape)

    target_ids = range(len(digits.target))
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, digits.target):
        plt.scatter(X_2d[Y_test == i, 0], X_2d[Y_test == i, 1], c=c, label=label)
    plt.legend()
    plt.title("TSNE - Test")
    plt.show()

    print('')
    print('[4] Número óptimo de vecinos K.')
    from sklearn.metrics import classification_report
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    print('Cerca dels K més propers:')
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    TEST = classification_report(Y_test, y_pred)
    print('Idoneidad de test:')
    print(TEST)

    print('Predicción:')
    print(model.predict(X_test))
    model.predict(X_test) == Y_test
    np.sum(model.predict(X_test) == Y_test)/Y_test.shape
    print('% Prediccion:')
    print(np.sum(model.predict(X_test) == Y_test)/Y_test.shape)

    print('')
    print('GridSearch:')
    from sklearn.model_selection import GridSearchCV

    k_range = list(range(1, 11))
    tuned_parameters = dict(n_neighbors = k_range)
    scores = ["precision", "recall", "f1"]
    model = KNeighborsClassifier()

    for score in scores:
        print('')
        print("# Ajustes para: %s" % score)
        clf = GridSearchCV(model, tuned_parameters, cv = kf, scoring="%s_macro" % score)
        clf.fit(X_train, Y_train)
        print("Mejores parámetros:")
        print(clf.best_params_)
        print("Puntuaciones:")
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) para %r" % (mean, std * 2, params))

if __name__ == '__main__':
    ass2()

