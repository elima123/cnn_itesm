# Red Neuronal Convolucional
CNN, por sus siglas en inglés, es un algoritmo de aprendizaje profundo (Deep Learning) diseñado específicamente para procesar y clasificar datos visuales, como imágenes y videos. Se inspira en la corteza visual del cerebro humano y es el estándar principal en el campo de la visión artificial.

# Descripción del dataset
El dataset utilizado en este proyecto fue obtenido de la plataforma [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) y corresponde a un conjunto de imágenes de resonancia magnética cerebral (MRI) orientado a la detección de tumores cerebrales. El conjunto de datos está dividido en dos categorías principales: imágenes etiquetadas como **“yes”**, que contienen anomalías o esferoides asociados con enfermedades cerebrales y posibles tumores, e imágenes etiquetadas como **“no”**, que no presentan dichas anomalías.

El dataset completo contiene un total de 253 imágenes, distribuidas en 155 imágenes para la clase **yes** y 98 imágenes para la clase **no**. Para el entrenamiento y evaluación del modelo, los datos fueron separados en tres subconjuntos: entrenamiento, prueba y validación.

Para la clase **yes**, se utilizaron 93 imágenes para entrenamiento, 31 imágenes para prueba y 31 imágenes para validación, manteniendo un total de 155 imágenes. Por otro lado, la clase **no** se dividió en 60 imágenes para entrenamiento, 19 imágenes para prueba y 19 imágenes para validación, con un total de 98 imágenes.

Esta distribución permite entrenar el modelo de aprendizaje automático utilizando la mayor parte de los datos disponibles, mientras que los conjuntos de validación y prueba permiten evaluar el desempeño y la capacidad de generalización del modelo frente a imágenes no vistas previamente.

Se aplicaron las siguientes transformaciones a las imagenes para aumentar el rendimiento del modelo y mejorar su entrenamiento.
- Eliminar bordes negros alrededor de las imagenes MRI.
- Normalizar el tamaño de las imagenes MRI a 256 x 256.