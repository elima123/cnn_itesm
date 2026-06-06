## Diagnóstico del modelo

* **Accuracy:** Train ≈ 0.64 y Validation ≈ 0.60–0.61, con una brecha pequeña entre ambas.
* **Loss:** Train y Validation convergen cerca de 0.65 y se estabilizan temprano.

**Interpretación:** No hay evidencia de overfitting, ya que las curvas de entrenamiento y validación permanecen cercanas. Sin embargo, el modelo presenta **underfitting moderado**, pues la precisión se estanca en valores relativamente bajos y la pérdida deja de mejorar rápidamente. Esto indica que el modelo no está capturando suficientes patrones discriminativos de las imágenes MRI.

**Posibles causas:**

1. Dataset reducido (256 imágenes).
2. Arquitectura CNN con capacidad insuficiente.
3. Data augmentation limitado.
4. Resolución o complejidad de las MRI superior a la capacidad del modelo.
5. Learning rate no óptimo.

**Mejoras recomendadas:**

* Aplicar **transfer learning** (VGG16, ResNet50, DenseNet121 o EfficientNet-B0).
* Sustituir `Flatten` por `GlobalAveragePooling2D`.
* Incrementar progresivamente los filtros (16→32→64→128).
* Ajustar la tasa de aprendizaje para evitar estancamiento.

### Se incrementará el learning rate 0.0001 -> 0.001