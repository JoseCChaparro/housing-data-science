# End-to-end project
## UACH - INGENIERÍA EN CIENCIAS DE LA COMPUTACIÓN 
### José Carlos Chaparro Morales - 329613

Repositoprio del programa de predicción en base al dataset de Housing California


Este repositorio contiene los archivos necesarios para entrenar y utilizar un modelo de inteligencia artificial que predice los precios de casas en California utilizando el conjunto de datos California housing prices.

El proyecto consta de los siguientes archivos y carpetas:

- **End_to_End.ipynb**: un cuaderno de Jupyter que contiene el código para la exploración de datos, entrenamiento del modelo y evaluación de su rendimiento. Se puede acceder al cuaderno en una versión de colab con el siguiente [enlace a colab](https://colab.research.google.com/drive/1q6PG_T4YMy1sKQe70Lghm7mfFfiYRpLa?usp=sharing).

- **housing.csv**: el conjunto de datos de viviendas de California utilizado para el entrenamiento del modelo.

- **forest_reg.pkl**: un archivo pickle que contiene el modelo de aprendizaje automático entrenado.

- [Enlace al deploy en Streamlit](https://josecchaparro-housing-data-science-main-hmf0p4.streamlit.app/).

- **README.md**: un archivo de texto que contiene información sobre el proyecto.

Para utilizar el modelo entrenado, se debe entrar al siguiente [enlace](https://josecchaparro-housing-data-science-main-hmf0p4.streamlit.app/) en el cual se
realizó el despliegue del modelo, en esta página se podrán ingresar los datos de entrada para hacer una predicción con el modelo. A continuación, una muestra de como luce la página del enlace:

![Imagen interfaz streamlit](https://github.com/JoseCChaparro/housing-data-science/blob/main/images/Captura%20de%20pantalla%202023-03-01%20103909.png)

Para hacer la predicción solo es necesario ir hasta el final de la página después de haber ingresado los datos y dar click en el botón de Predecir como sigue:

El cuaderno contiene instrucciones detalladas sobre cómo cargar los datos, entrenar el modelo y evaluar su rendimiento.

![Imagen botón de predecir](https://github.com/JoseCChaparro/housing-data-science/blob/main/images/Captura%20de%20pantalla%202023-03-01%20104620.png)

El modelo entrenado está almacenado en el archivo forest_reg.pkl, que puede ser cargado y utilizado para hacer predicciones de precios de casas en California.

Este proyecto puede ser utilizado como punto de partida para aplicaciones de predicción de precios de viviendas en California, así como para la exploración de técnicas de aprendizaje automático para problemas de regresión.
