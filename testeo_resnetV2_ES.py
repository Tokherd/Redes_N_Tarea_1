import time
import subprocess
import numpy as np
import pandas as pd
import os
# Desactivar cuDNN BatchNorm para mayor precision
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'

# Esperar y seleccionar GPU libre automaticamente
gpu_id = None
while True:
    # Consultar uso de memoria de cada GPU
    result = subprocess.run([
        'nvidia-smi',
        '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = result.stdout.decode().splitlines()
    for line in lines:
        idx, mem = line.strip().split(', ')
        if int(mem) == 0:
            gpu_id = idx
            break
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"GPU libre encontrada: {gpu_id}")
        break
    print("Esperando GPU libre...")
    time.sleep(5)

import csv
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import (Input, Dense, Flatten, Conv2D, BatchNormalization,
                          Activation, AveragePooling2D, Dropout, add)
from keras.models import Model
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             ReduceLROnPlateau, EarlyStopping)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Configurar memory growth en GPU seleccionada
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print('Error al configurar memory growth:', e)

# ---------- CONFIGURACION GENERAL ----------
directorio_train = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
directorio_test  = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'
batch_size       = 32  # reducir si OOM
num_classes      = 7
epochs           = 200
data_augmentation= True
model_type       = 'ResNet20v2'
depth            = 101
input_shape      = (64, 64, 1)
target_size      = (64, 64)

# ---------- CALCULO DE PESOS DE CLASE ----------
def calcular_pesos_de_clase(train_dir, batch_size=batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )
    freqs = np.bincount(gen.classes)
    total = len(gen.classes)
    weights_array = total / (len(freqs) * freqs)
    return {i: w for i, w in enumerate(weights_array)}

# ---------- GENERADORES CON AUMENTACION ----------
def crear_generadores(train_dir, test_dir, target_size, batch_size, augmentation=True):
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.1,
            shear_range=0.05,
            brightness_range=[0.9, 1.1]
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, test_gen

train_gen, test_gen = crear_generadores(
    directorio_train, directorio_test,
    target_size, batch_size, data_augmentation
)
class_weights = calcular_pesos_de_clase(directorio_train)

# ---------- CAPA RESNET ----------
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True,
                 conv_first=True, dropout_rate=None):
    x = inputs
    conv = Conv2D(
        num_filters, kernel_size=kernel_size, strides=strides,
        padding='same', kernel_initializer='he_normal'
    )
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        x = conv(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

# ---------- RED RESNET V2 ----------
def resnet_v2(input_shape, depth, num_classes):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth debe ser 9n+2')
    num_filters = 16
    num_blocks  = (depth - 2) // 9
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters)
    for stage in range(3):
        for block in range(num_blocks):
            strides = 1
            if stage > 0 and block == 0:
                strides = 2
            y = resnet_layer(x, num_filters, strides=strides,
                             activation=None, batch_normalization=False,
                             conv_first=False)
            y = resnet_layer(y, num_filters, conv_first=False)
            out_filters = num_filters * (4 if stage == 0 else 2)
            y = resnet_layer(y, out_filters, kernel_size=1, conv_first=False)
            if block == 0:
                x = resnet_layer(x, out_filters, kernel_size=1,
                                 strides=strides, activation=None,
                                 batch_normalization=False)
            x = add([x, y])
        num_filters = out_filters
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# ---------- SCHEDULE LR ----------
def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

model = resnet_v2(input_shape, depth, num_classes)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=lr_schedule(0)),
    metrics=['accuracy']
)
model.summary()

# ---------- CALLBACKS ----------
save_dir = os.path.join(os.getcwd(), 'saved_models')
os.makedirs(save_dir, exist_ok=True)
filepath = os.path.join(save_dir, f'ResnetV2_{model_type}.h5')
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1,
    save_best_only=False, save_weights_only=False, mode='max', save_freq='epoch'
)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer   = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, min_lr=0.5e-6)
early_stop   = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
callbacks    = [checkpoint, lr_reducer, lr_scheduler, early_stop]

# ---------- ENTRENAMIENTO ----------
start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=2,
    steps_per_epoch=len(train_gen),
    validation_steps=len(test_gen),
    callbacks=callbacks,
    class_weight=class_weights
)
end_time = time.time()
print(f'Tiempo de entrenamiento: {end_time - start_time:.2f} s')

# ---------- EVALUACION FINAL ----------
scores = model.evaluate(test_gen, verbose=0)
print(f'Test loss: {scores[0]:.4f}, Test accuracy: {scores[1]:.4f}')

# ---------- GUARDAR METRICAS ----------
result_dir = 'resultados_resnetv2'
os.makedirs(result_dir, exist_ok=True)
np.save(os.path.join(result_dir, 'test_loss.npy'), scores[0])
np.save(os.path.join(result_dir, 'test_accuracy.npy'), scores[1])
np.save(os.path.join(result_dir, 'execution_time.npy'), end_time - start_time)

# ---------- GRAFICOS ----------
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Loss train')
plt.plot(history.history['val_loss'], label='Loss val')
plt.title('Curva de perdida')
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'curva_perdida.png'))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Acc train')
plt.plot(history.history['val_accuracy'], label='Acc val')
plt.title('Curva de precision')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'curva_accuracy.png'))
plt.close()
