# PageRank Algorithm

Este repositorio contiene la implementación del algoritmo de PageRank para todas las páginas web de California. El algoritmo se basa en un grafo dirigido creado con coo_matrix (9664, 9664), que incluye algunos "spider-traps" y "dead-ends".

### Estructura:
- 'main.py': Contiene la implementación principal del algoritmo de PageRank.
- 'utils.py': Archivo de utilidades con funciones auxiliares.
- 'data/': Carpeta que contiene los datos necesarios para la ejecución del algoritmo.
- 'results/': Carpeta donde se almacenan los resultados de la ejecución.

### Ejecución:
- Ejecute main.py para calcular el PageRank de cada página web en el grafo dirigido.
- Los resultados se guardarán en la carpeta results/, indicando la importancia de cada página web.

### Detalles:
El algoritmo se basa en la teoría de PageRank, calculando la importancia de cada página web en función de la cantidad y calidad de los enlaces que la apuntan. Se han realizado experimentos con diferentes valores de β (probabilidad de teletransportación aleatoria), observando cómo afecta al número de iteraciones y al tiempo de convergencia.

### Resultados:
Después de varios experimentos, se ha encontrado que un valor de β entre 0.8 y 0.85 proporciona resultados óptimos en términos de convergencia y precisión. El tiempo de convergencia oscila entre 15-27 segundos, y se han realizado optimizaciones para mejorar la eficiencia del algoritmo.

##
Este proyecto proporciona una implementación eficiente del algoritmo de PageRank para analizar la importancia de las páginas web en un grafo dirigido. Los resultados obtenidos destacan la página web de la Universidad Davis como la más importante en el contexto de las páginas web de California.
- @date: 17 Octubre 2021
- @author: Oscar Julian
