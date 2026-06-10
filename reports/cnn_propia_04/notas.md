# Notas
- Mejora notable en cuanto a la Specificity, el problema fue que aprendió a contestar todo como positivo. Pero ahora ya empezó a indicar en las pruebas correctamente negativos como negativos en un 25%.
- El cambio de flatten a GlobalAveragePooling2D, y el hecho de quitar dense ayudó mucho al modelo.
- Puede que la mejora provenga únicamente de remover dense, debido a que dense generaba muchas neuronas.
- Pienso que también el learning rate (0.0001) ayudo a evitar el problema relacionado con el recall alto debido a que permitia iteraciones más lentas de aprendizaje.
- El modelo tiene un f1 score menor (-10%) con respecto al anterior, sin embargo el problema d specificty mejoró.

# Cambios para el siguiente modelo
- Se aumentó el número de bloques y capas de cada bloque tomando en cuenta el tamaño de la cnn.
- Se duplicó el número de epochs para permitir que mejorara más tiempo (de cualquier manera, early stopping detiene el entrenamiento no necesario)
- Probé intermediamente con un tercer bloque identico (Total params: 74,913), pero regresó el problema de val_recall 1.0 
- Noté que si el modelo alcanza una epoch mayor a 55 el recall empieza a bajar y empieza a predicir imagenes como negativas (Lo cual es bueno, ya que no predecia imagenes negativas) 
- Ese último comportamiento puede indicar dos cosas, que el modelo está haciendo overfitting y por eso también bajó
- Paciencia de early stopping: 10 <- 15 (revertida).
- Total params: 47,169 (184.25 KB)

```text
# Modelo inspirado de VGG19 https://doi.org/10.48550/arXiv.1512.03385
model = Sequential()
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(data_augmentation)

# Block 1
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Head
model.add(GlobalAveragePooling2D())
model.add(Dropout(DROPOUT))

# Output
model.add(Dense(1, activation=ACTIVATION)) # Binary = 1
model.summary()
```