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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
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
compression_factors = [0.3, 0.5, 0.7]
use_max_pool = False

# ========== PLAN DE APRENDIZAJE ==========
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 150:
        lr *= 0.1**2
    elif epoch > 100:
        lr *= 0.1
    print(f"Learning rate for epoch {epoch}: {lr}")
    return lr

# ========== FUNCIÓN DE CARGA ==========
def extraer_imagenes(directorio_train, directorio_test):
    def cargar_imagenes(directorio):
        clases = []
        imagenes = []
        carpetas = sorted(os.listdir(directorio))
        for idx, carpeta in enumerate(carpetas):
            carpeta_path = os.path.join(directorio, carpeta)
            for img_archivo in os.listdir(carpeta_path):
                img_path = os.path.join(carpeta_path, img_archivo)
                foto = image.load_img(img_path, target_size=(64, 64), color_mode='grayscale')
                imagen = np.array(foto).astype(np.float32) / 255.0
                imagen = np.expand_dims(imagen, axis=-1)
                imagenes.append(imagen)
                clases.append(idx)
        return np.array(imagenes), to_categorical(clases)

    X_train, y_train = cargar_imagenes(directorio_train)
    X_test, y_test = cargar_imagenes(directorio_test)
    return shuffle(X_train, y_train, random_state=42), (X_test, y_test)

# ========== CARGA ==========
(X_train, y_train), (X_test, y_test) = extraer_imagenes(directorio_train, directorio_test)
input_shape = (64, 64, 1)
num_classes = y_train.shape[1]

# ========== LOOP PARA CADA COMPRESIÓN ==========
for compression_factor in compression_factors:
    print(f"\n=== Entrenando DenseNet con compression_factor = {compression_factor} ===\n")

    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
    num_filters_bef_dense_block = 2 * growth_rate

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

        if i != num_dense_blocks - 1:
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            y = BatchNormalization()(x)
            y = Conv2D(num_filters_bef_dense_block, kernel_size=1, padding='same', kernel_initializer='he_normal')(y)
            x = AveragePooling2D(pool_size=2)(y)

    x = AveragePooling2D(pool_size=4)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=1e-3), metrics=['accuracy'])
    model.summary()

    # ========== CALLBACKS ==========
    tag = f"densenet_c{int(compression_factor*100)}"
    save_dir = os.path.join(os.getcwd(), f'saved_models_{tag}')
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, f'{tag}_best.keras')
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, min_lr=5e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]

    # ========== ENTRENAMIENTO ==========
    start_time = time.time()
    if not data_augmentation:
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(X_test, y_test), shuffle=True, callbacks=callbacks)
    else:
        print('Using real-time data augmentation...')
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(X_train)
        steps_per_epoch = math.ceil(len(X_train) / batch_size)
        model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                  steps_per_epoch=steps_per_epoch,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  callbacks=callbacks,
                  verbose=2)

    end_time = time.time()
    print(f"[TIME] Compression {compression_factor} -> {end_time - start_time:.2f} seconds")

    # ========== EVALUACIÓN ==========
    model.load_weights(best_model_path)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Compression {compression_factor} -> Test loss: {scores[0]:.4f}, Test accuracy: {scores[1]:.4f}")

