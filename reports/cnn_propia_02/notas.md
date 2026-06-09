# Notas
El modelo sigue sin detectar negatives después de aumentar el learning rate.
El error puede provenir de la arquitectura que estoy usando, tiene muy pocas neuronas.
Tocará probar con modelos entrenados.
La caida de loss de 2.5 a 0.6 en una sola epoch es muy raro.
La grafica se detiene a las 10 epochs, y esa cantidad es mencionada en early stopping, puede que esté afectando.

# Cambios para el siguiente modelo
- Learning rate: 0.01 <- 0.001
- Paciencia de EarlyStopping: 15 <- 10
- Guardar el mejor según el f1score con respecto a la validación
- Cambio de las neuronas: 8,417,377 params <- ~500k params

```text
# Modelo inspirado de VGG16 https://doi.org/10.48550/arXiv.1512.03385
model = Sequential()
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(data_augmentation)

# Block 1
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Head
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(DROPOUT))

# Output
model.add(Dense(1, activation=ACTIVATION)) # Binary = 1
model.summary()
```