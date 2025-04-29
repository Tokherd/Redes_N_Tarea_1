# canal de blanco y negro = 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
def extraer_imagenes(directorio_train: str, directorio_test: str):
    def cargar_imagenes(directorio):
        clases = []
        imagenes = []
        lista_carpetas = os.listdir(directorio)
        contador = 0
        for carpeta_clases in lista_carpetas:
            for imagen in os.listdir(os.path.join(directorio, carpeta_clases)):
                foto = image.load_img(os.path.join(directorio, carpeta_clases, imagen), target_size=(256, 256, 1),color_mode='grayscale')
                imagenes.append(np.array(foto))
                clases.append(contador)
            contador += 1
        clases = np.array(clases)
        clases = to_categorical(clases)
        imagenes = np.array(imagenes, dtype=np.float32) / 255  # Normalizar imágenes
        return imagenes, clases

    # Cargar imágenes y clases para entrenamiento y prueba
    X_train, y_train = cargar_imagenes(directorio_train)
    X_test, y_test = cargar_imagenes(directorio_test)

    return X_train, X_test, y_train, y_test

# Directorios de entrenamiento y prueba
directorio_train = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
directorio_test = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'

# Llamar a la función
X_train, X_test, y_train, y_test = extraer_imagenes(directorio_train, directorio_test)

# Mezclar los datos de entrenamiento y prueba
# para que no asigne de manera secuencial
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Imprimir formas de los datos
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

