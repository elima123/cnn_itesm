# Notas
- Mejora notable en la grafica de recall, ya no calcula 1.0.
- El cambio de flatten a GlobalAveragePooling2D, y el hecho de quitar dense ayudó mucho al modelo.
- Puede que la mejora provenga únicamente de remover dense, debido a que dense generaba muchas neuronas.
- Pienso que también el learning rate (0.0001) ayudo a evitar el recall debido a que permitia iteraciones más lentas de aprendizaje.
- El modelo tiene un f1 score menor (-5%) con respecto al anterior, sin embargo el problema del recall mejoró bastante.

# Cambios para el siguiente modelo
- Se aumentó el número de bloques y capas de cada bloque tomando en cuenta el tamaño de la cnn.
- Se duplicó el número de epochs.
- Probé intermediamente con un tercer bloque identico (Total params: 74,913), pero regresó el problema de val_recall 1.0 (Esta mini-prueba fue hasta el epoch 42, y la grafica del modelo de este reporte mejoró hasta el epoch 55) (65 también)
- Paciencia de early stopping: 10 <- 15 (revertida).
- Total params: 47,169 (184.25 KB)

```text
# Modelo inspirado de VGG16 https://doi.org/10.48550/arXiv.1512.03385
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