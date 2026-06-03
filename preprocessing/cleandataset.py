from pathlib import Path
from PIL import Image
import numpy as np

# CONFIG
INPUT_DIR = "dataset_raw"
OUTPUT_DIR = "dataset_clean"
TARGET_SIZE = 256
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 1. CROP
def crop_black_background(img, threshold=30):
    """
    Elimina bordes negros alrededor de la imagen.
    """

    gray = img.convert("L") # Convierte la imagen en una escala de grises
    arr = np.array(gray) # Después, convierte la imagen gris en una Matriz de dos dimensiones (pixel array)

    mask = arr > threshold # Convierte la matriz de números en una matriz de booleanos.
                           # Marcan como false pixeles con WhiteScale (0 -> 255) menores a 11.

    rows = np.any(mask, axis=1) # Obtiene los sub arrays de filas donde exista al menos un pixel aceptado (TRUE)
    cols = np.any(mask, axis=0) # Obtiene los sub arrays de columnas donde exista al menos un pixel aceptado (TRUE)

    # Obtener las esquinas de la nueva imagen para después ejecutar un crop
    y_min, y_max = np.where(rows)[0][[0, -1]] # [0] extrae el arreglo de índices.
    x_min, x_max = np.where(cols)[0][[0, -1]] # [[0, -1]] toma el primer y último índice encontrados.
                   # np.where(x) devuelve las posiciones donde existe al menos un píxel útil en cada x.

    return img.crop(( x_min, y_min, x_max + 1, y_max + 1 )) # img.crop((left, upper, right, lower)), +1 para incluir last array

# 2. RESIZE
def resize_with_padding(img, target_size=256):
    """
    Ajusta el tamaño de la imagen según target_size.
    VGG16 necesita un tamaño 256x256.
    """

    # Consigue los valores de la nueva imagen según Width y Height de la imagen actual
    w, h = img.size
    scale = min( target_size/w, target_size/h ) # Obtiene el valor de proporción para reajuste del tamaño 
    new_w = int(w * scale) # Aplica la escala sin decimales
    new_h = int(h * scale) # Aplica la escala sin decimales

    # Usando el algoritmo de reducción LANCZOS, se aplica img.resize de python
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # LANCZOS, NEAREST, BILINEAR...

    # Crea un lienzo en negro donde se añadiran los nuevos valores RGB
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # Obtener los valores para centrar la imagen en el nuevo lienzo
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # Paste n Exit
    canvas.paste(img, (x_offset, y_offset))
    return canvas


for class_dir in Path(INPUT_DIR).iterdir():

    if not class_dir.is_dir():
        continue

    output_class = Path(OUTPUT_DIR) / class_dir.name
    output_class.mkdir(parents=True, exist_ok=True)

    for image_path in class_dir.glob("*"):
        try:

            img = Image.open(image_path).convert("RGB")

            # 1. Quitar fondo negro
            img = crop_black_background(img)

            # 2. Normalizar tamaño
            img = resize_with_padding(img, TARGET_SIZE)

            # 3. Guardar las imagenes en PNG para conservar la calidad
            output_file = output_class / image_path.name
            output_file = output_file.with_suffix(".png")
            img.save(output_file)
            print("OK:", image_path.name)

        except Exception as e:
            print("ERROR:", image_path.name, e)

print("Dataset cleaned.")