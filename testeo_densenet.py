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
batch_size = 4
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
                foto = image.load_img(img_path, target_size=(256, 256,1), color_mode='grayscale')
                imagenes.append(np.expand_dims(np.array(foto), axis=-1))  # Expandir canal
                clases.append(idx)
        clases = to_categorical(clases)
        imagenes = np.array(imagenes, dtype=np.float32) / 255.0
        return imagenes, clases

    X_train, y_train = cargar_imagenes(directorio_train)
    X_test, y_test = cargar_imagenes(directorio_test)
    return shuffle(X_train, y_train, random_state=42), shuffle(X_test, y_test, random_state=42)

# ========== CARGA DE DATOS ==========
(X_train, y_train), (X_test, y_test) = extraer_imagenes(directorio_train, directorio_test)
input_shape = (256, 256, 1)
num_classes = y_train.shape[1]

# ========== MODELO ==========
num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
num_filters_bef_dense_block = 2 * growth_rate

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate:', lr)
    return lr

inputs = Input(shape=input_shape)
x = BatchNormalization()(inputs)
x = Activation('relu')(x)
x = Conv2D(num_filters_bef_dense_block, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
x = concatenate([inputs, x])

for i in range(num_dense_blocks):
    for j in range(num_bottleneck_layers):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(4 * growth_rate, kernel_size=1, padding='same', kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(growth_rate, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
        x = concatenate([x, y])
    
    if i == num_dense_blocks - 1:
        continue

    num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
    num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
    y = BatchNormalization()(x)
    y = Conv2D(num_filters_bef_dense_block, kernel_size=1, padding='same', kernel_initializer='he_normal')(y)
    x = AveragePooling2D(pool_size=2)(y)

x = AveragePooling2D(pool_size=4)(x)
x = Flatten()(x)
outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(1e-3), metrics=['accuracy'])
model.summary()

# ========== CALLBACKS ==========
save_dir = os.path.join(os.getcwd(), f'saved_models_{epochs}_epocas')
model_name = 'fer_densenet_model.{epoch:02d}.h5'
os.makedirs(save_dir, exist_ok=True)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=2, save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]

# ========== ENTRENAMIENTO ==========
start = time.time()
if not data_augmentation:
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, y_test), shuffle=True, callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              epochs=epochs, steps_per_epoch=steps_per_epoch,
              validation_data=(X_test, y_test), callbacks=callbacks, verbose=2)

fin = time.time()
print('RUNNING TIME:', fin - start)

# ========== EVALUACIÓN ==========
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])