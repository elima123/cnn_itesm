# Notas
- Sigue prediciendo todo como positivo.
- accuracy: 0.6159 - auc: 0.5825 - f1score: 0.7623 - loss: 0.6649 - precision: 0.6159 - recall: 1.0000 - val_accuracy: 0.6078 - val_auc: 0.6306 - val_f1score: 0.7561 - val_loss: 0.6591 - val_precision: 0.6078 - val_recall: 1.0000
- Recall sigue calculandose como 1.0, la métrica clave se llama Specificity.

Voy a pasar a transfer learning, mi dataset sigue siendo muy pequeño.

Encontré que puede que la capa dense esté empeorando el rendimiento del modelo debido a que: 

- model.add(Flatten()) 
Convierte todo el feature map en un vector gigante.
1 x 1 x 100,000 example
“guarda cada pixel de activación”

- model.add(GlobalAveragePooling2D())
Como su nombre lo indica, los promedia en su cantidad de neuronas final
if 7 x 7 x 2048, then 2048 valores output
“resume qué tan fuerte se activó cada detector”

# Cambios para el siguiente modelo
- Cambie flatten por GlobalAveragePooling2D, y removí la capa densa de 32.
- Pienso que esto lo mejora debido a que flatten hace que se pierdan información importante al momento de la activación del output.
- También, quité dense porque pienso que aumenta mucho los parámetros que tengo, aunque había pensando en tener una red de 500k parámetros inicialmente.
- Regresé el valor de learning rate a 0.0001
- Total params: 28,673

```text
# Modelo inspirado de VGG19 https://doi.org/10.48550/arXiv.1512.03385
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
model.add(GlobalAveragePooling2D())
model.add(Dropout(DROPOUT))

# Output
model.add(Dense(1, activation=ACTIVATION)) # Binary = 1
model.summary()
```