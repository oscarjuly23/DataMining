# Grid, Boosting / Neural Nwtworks
Este proyecto presenta la aplicación de algoritmos de optimización, boosting, y redes neuronales utilizando scikit-learn en un nuevo conjunto de datos. Se exploran técnicas como la búsqueda de hiperparámetros mediante GridSearch, así como la aplicación de KNeighborsClassifier, MLPClassifier, y AdaBoostClassifier en un conjunto de datos de noticias vectorizadas.

## Funcionalidades
### Número Óptimo de Vecinos (K) con Validación Cruzada
- Se realiza la búsqueda del número óptimo de vecinos utilizando GridSearchCV con validación cruzada.
- Importación de los paquetes necesarios de sklearn (grid_search y cross_validation).
- Utilización de GridSearchCV para optimizar el clasificador KNeighborsClassifier.
- Cálculo del accuracy y obtención de los parámetros que proporcionan la mejor precisión en el conjunto de entrenamiento.

### Nuevos Datos con Redes Neuronales, Bagging & Boosting
- Se aplica la vectorización de un conjunto de datos de noticias utilizando TfidfVectorizer.
- Exploración de distintos clasificadores, incluyendo KNeighborsClassifier, MLPClassifier, y AdaBoostClassifier.
- Evaluación del rendimiento de cada clasificador mediante el cálculo del accuracy.

### TfidfVectorizer
- Profundización en la utilidad de TfidfVectorizer para el análisis de texto.
- Descripción del proceso de tokenización, conteo y normalización de tokens.
- Explicación de cómo se aplica la ponderación tf-idf para asignar pesos a las palabras en función de su importancia en los documentos.

##
El script proporciona resultados y análisis relacionados con la optimización de hiperparámetros, la aplicación de diversos clasificadores y la utilización de TfidfVectorizer en un conjunto de datos de noticias. Explora las secciones del código para comprender mejor cada funcionalidad.
- @author: Oscar Julian
- @date: Enero 2022
