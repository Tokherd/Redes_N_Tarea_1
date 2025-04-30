import os
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import math
import time
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, concatenate
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========== CONFIGURACIÓN ==========

directorio_train = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
directorio_test = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'
batch_size = 8
epochs = 200
data_augmentation = True
growth_rate = 12
depth = 100
num_dense_blocks = 3
compression_factor = 0.5
use_max_pool = False

# ========== FUNCIÓN DE CARGA ==========
def extraer_imagenes(directorio_train: str, directorio_test: str):
    def cargar_imagenes(directorio):
        clases = []
        imagenes = []
        lista_carpetas = sorted(os.listdir(directorio))
        for idx, carpeta in enumerate(lista_carpetas):
            carpeta_path = os.path.join(directorio, carpeta)
            for imagen_archivo in os.listdir(carpeta_path):
                img_path = os.path.join(carpeta_path, imagen_archivo)
                foto = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
                imagenes.append(np.expand_dims(np.array(foto), axis=-1))  # Expandir canal
                clases.append(idx)
        clases = to_categorical(clases)
        imagenes = np.array(imagenes, dtype=np.float32) / 255.0
        return imagenes, clases
    X_train, y_train = cargar_imagenes(directorio_train)
    X_test, y_test = cargar_imagenes(directorio_test)
    return shuffle(X_train, y_train, random_state=42), shuffle(X_test, y_test, random_state=42)

