#!/opt/local/bin/python3
# -*- coding: utf-8 -*-

# Pràctica 3 - Grid, Boosting / Xarxes Neuronals aplicats amb scikit-learn amb noves dades
# Mineria de Dades, Curs 2021-22
# @author Oscar Julian Ponte(oscar.julian)
# Gener 2022

import numpy as np
import sklearn
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
import sklearn.metrics as metrics


def ass3():
    # [0] RECAPITULACIÓ:

    # DIGITS DATA SET:
    # Carregarem el dataset de dígits en les variables X i Y.
    digits = sklearn.datasets.load_digits()
    X = digits.data # En la X carreguem Data, que serà una matriu amb 1797 files i 64 columnes (matriu 8x8).
    Y = digits.target # En la Y carreguem Target, que serà una variable amb les posicions de les imatges (1797 posicions).

    # DIVISIÓ AMB TRAIN & TEST:
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, train_size = 0.7) # Nos devuelve las listas (dividimos entre train 70% i test 30%)
    # Dividim les dades entre 70% Train i 30% Test i ho tenim en llistes.
    # X_train: 1257 imatges que serviran per entrenar el nostre model.
    # Y_train: Dígits.
    # X_test: 540 imatges per a testejar el problema.
    # Y_test: Dígits per a fer la comprovació de test.


    print("# [1] Nombre Òptim de veïns K amb validació creuada:")
    # En l'anterior pràctica ja ho vem aplicar, pero ho tornem a fer per a major claretat

    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import make_scorer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    # Generate possible values of the exploration Grid
    k = np.arange(20) + 1
    # Infer all the exploratory surface into parameters struct
    parameters = {'n_neighbors': k}
    # Create Learner Factory
    knearest = sklearn.neighbors.KNeighborsClassifier()
    # Instantiate a GridSearch with the a) Learner Factory b) Exploratory parameters c) CrossValidation param d) Compute Test Accuracy
    #clf = GridSearchCV(knearest, parameters, cv=10)
    clf = GridSearchCV(knearest, parameters, cv=kf, scoring="accuracy")
    # Perform exploratory grid search over TrainingData
    clf.fit(X_train, Y_train)
    # Obtrain the point of the grid that yielded the best train-accuracy
    clf.best_params_['n_neighbors']

    print("- Ajustes para Accuracy -")

    print("Mejores parámetros:")
    print(clf.best_params_)
    print("Puntuaciones:")

    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]

    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) para %r" % (mean, std * 2, params))

    print("")

    print("- MLPClassifier: -")
    from sklearn.neural_network import MLPClassifier

    # Make a 2-hidden-layer Neural Network with 100 Rectifier Units with a learning_rate of 0.02 e an 10 iterations
    nn = MLPClassifier(hidden_layer_sizes=(100, 100,), activation='relu', solver='sgd', learning_rate='constant', learning_rate_init=0.02, max_iter=10)
    # Probem amb la primera capa 100 neurones, la segona 100, des de 0,02 de learning-rate i amb 10 iteracions
    # Fit the data
    nn.fit(X_train, Y_train)
    print("Mínima perdida:")
    print(nn.best_loss_)
    print("")

    print("- AdaBoostClassifier -")
    from sklearn.ensemble import AdaBoostClassifier
    # Fit an AdaBoost classifier with 100 weak learners:
    abclf = AdaBoostClassifier(n_estimators=100)
    #Fit the data
    abclf.fit(X_train, Y_train)
    print("Error de cada estimador:")
    for i in abclf.estimator_errors_:
        print(i)
    print("")


    print("# [2] Nou conjunt de dades")

    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html (bag of words)
    #from sklearn.feature_extraction.text import CountVectorizer

    from sklearn.datasets import fetch_20newsgroups
    # https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

    # Primer vam fer la prova amb 4 categories i finalment amb totes.
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    #newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    #newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    # Tenim un conjunt de documents (de notícies) en el qual tenim:
    print("Total de notícies (train):")
    print(len(newsgroups_train.data)) # Un total de 11314 noticies
    print("Total de notícies (test):")
    print(len(newsgroups_test.data)) # Un total de 7532 noticies
    print("")
    print("Primera notícia (exemple): ")
    print(newsgroups_train.data[0]) # Primera notícia
    print("Categoria primera notícia (exemple): ")
    print(newsgroups_train.target[0]) # Categoria 7
    print(newsgroups_train.target_names[0])
    print("Totes les categories: ")
    print(newsgroups_train.target_names) # Un total de 20
    print("")

    print("- TfidfVectorizer -")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer() # Convertim textos en matriu
    vectorized_data = vectorizer.fit_transform(newsgroups_train.data) # Entrenamos i convertimos datos de train
    print("Matriu vectoritzada de les notcícies:")
    print(vectorized_data.shape) # matriu de (11314 documents x 130107 paraules que surten)
    # Aquest vector ens diu cuants cops surt una paraula en el document
    print("Paraules del primer file (primera notícia): ")
    #print(np.where(vectorized_data[0,:].todense()))
    print(vectorized_data[0,:].todense()) # La majoria son 0
    print("Numero de valors: ")
    print(vectorized_data[0,:].nnz) # 89 Valors

    print("")
    print("- KNeighborsClassifier: -")
    # Ara usant KNeighborsClassifier podem agafar els 3 veïns més propers dels vectors
    clf = KNeighborsClassifier(n_neighbors=3)

    # En el meu PC em dona un error: numpy.core._exceptions.MemoryError: Unable to allocate 650. MiB for an array with shape (7532, 11314) and data type float64
    # He estat buscant en Google i es degut a la Memoria virtual configurada en el meu sistema, no he aconseguit solucionar-ho i he hagut de fer les proves només
    # aplicant KNN en 4 categories ja que la matriu queda més petita i no em dona problemes de memòria.
    # Crec que al executar-ho des de un altre PC ha de funcionar correctament, ho deixo comentat per si es vol probar (s'hauria de comentar la següent part llavors)
    # clf.fit(vectorized_data, newsgroups_train.target)
    test_vectors = vectorizer.transform(newsgroups_test.data)
    # print(clf.predict(test_vectors)) # Fem un predict per els casos de test
    # print(clf.predict(test_vectors).shape) # 7532 noticies de test classificades
    # # Podem veure que amb KNearestNeighbor, l'Accuracy que hem obtingut es molt baixa:
    # # Fem el Classification_report entre la 'classe' real i la 'classe' precict
    # print(classification_report(newsgroups_test.target, clf.predict(test_vectors)))
    # # Hem encertat cada categoria amb un 68% de precisió, 65% de covertura i 66% score

    # Amb aquest classificador, si ho apliquem amb totes les dades no funciona en el meu PC així que hem fet la prova amb 4 categories
    newsgroups_train_KNN = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test_KNN = fetch_20newsgroups(subset='test', categories=categories)
    vectorized_data_KNN = vectorizer.fit_transform(newsgroups_train_KNN.data)
    clf.fit(vectorized_data_KNN, newsgroups_train_KNN.target)
    test_vectors_KNN = vectorizer.transform(newsgroups_test_KNN.data)
    #print(test_vectors_KNN)
    print("Categoria dels test:")
    print(clf.predict(test_vectors_KNN))
    print(clf.predict(test_vectors_KNN).shape)
    print(classification_report(newsgroups_test_KNN.target, clf.predict(test_vectors_KNN)))
    # Fent la prova amb 4 categories veiem que el predict de cada categoria s'ha fet amb més precisió
    print("")

    print("- MLPClassifier: -")
    # Make a 2-hidden-layer Neural Network with 100 Rectifier Units with a learning_rate of 0.02 e an 10 iterations
    nn = MLPClassifier(hidden_layer_sizes=(100, 100,), activation='relu', solver ='sgd', learning_rate ='constant', learning_rate_init = 0.02, max_iter = 10)
    # Probem amb la primera capa 300 neurones, la segona 100 i la tercera 40, des de 0,4 de learning-rate i amb 20 iteracions
    # La activació 'relu' --> Unidad lineal rectificada
    # Solver 'sgd' --> Estocastic gradient descendent
    # Al probar amb aquesta configuració de Neural Network em dona el seguent error: numpy.core._exceptions.MemoryError: Unable to allocate 298. MiB for an array with shape (130107, 300) and data type float64
    # Així que ho deixem amb la prova inicial.
    #nn = MLPClassifier(hidden_layer_sizes=(300, 100, 40), activation='relu', solver ='sgd', learning_rate ='constant', learning_rate_init = 0.04, max_iter = 20)
    # Fit the data
    nn.fit(vectorized_data, newsgroups_train.target)
    print(classification_report(newsgroups_test.target, nn.predict(test_vectors)))
    print("")

    print("- AdaBoostClassifier -")
    # Fit an AdaBoost classifier with 100 weak learners:
    abclf = AdaBoostClassifier(n_estimators=100)
    #Fit the data
    abclf.fit(vectorized_data, newsgroups_train.target)

    print(classification_report(newsgroups_test.target, abclf.predict(test_vectors)))
    print("")

if __name__ == '__main__':
    ass3()

