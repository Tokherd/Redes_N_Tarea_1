import os
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Input, Conv2D, Dense, GlobalAveragePooling2D, 
                                     Dropout, Activation, BatchNormalization, add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import Counter

# ========== CONFIGURACIÓN ==========
train_dir = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
test_dir = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'
batch_size = 64
epochs = 200
depth = 104
input_shape = (64, 64, 1)
target_size = (64, 64)

# ========== AUGMENTACIÓN Y GENERADORES ==========
def crear_generadores(train_dir, test_dir, target_size=(64, 64), batch_size=64):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.3
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

train_gen, val_gen, test_gen = crear_generadores(train_dir, test_dir, target_size, batch_size)
num_classes = train_gen.num_classes

# ========== PESOS DE CLASE ==========
def calcular_pesos_de_clase(train_gen):   # para manejar el desbalanceo de clases
    class_totals = Counter(train_gen.classes)
    total_samples = len(train_gen.classes)
    return {i: total_samples / (num_classes * class_totals[i]) for i in class_totals}

class_weights = calcular_pesos_de_clase(train_gen)

# ========== LR SCHEDULE ==========
def lr_schedule(epoch):
    lr = 0.0005  # menor valor inicial
    if epoch > 180:
        lr *= 0.1e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:   
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate:', lr)
    return lr

# ========== BLOQUE RESNET ==========
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                  padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))
    x = inputs
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
    return x

# ========== RESNETv1 ==========
def resnet_v1(input_shape, depth, num_classes=7):
    if (depth - 2) % 6 != 0:
        raise ValueError("Depth must be 6n+2")
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(x, num_filters=num_filters, strides=strides)
            y = resnet_layer(y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(x, num_filters=num_filters, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax',
                    kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    return Model(inputs, outputs)

# ========== COMPILACIÓN ==========
model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary() 
# ========== CALLBACKS ==========
save_dir = os.path.join(os.getcwd(), 'saved_models')
os.makedirs(save_dir, exist_ok=True)
filepath = os.path.join(save_dir, 'best_resnetv1_model.h5')

callbacks = [
    ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    LearningRateScheduler(lr_schedule),
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
]

# ========== ENTRENAMIENTO ==========
start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)
end_time = time.time()

# ========== EVALUACIÓN Y GUARDADO ==========
scores = model.evaluate(test_gen, verbose=2)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

output_dir = "resultados_resnetv1"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'test_loss.npy'), scores[0])
np.save(os.path.join(output_dir, 'test_accuracy.npy'), scores[1])
np.save(os.path.join(output_dir, 'execution_time.npy'), end_time - start_time)

# Guardar historia
pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'history.csv'))

# Gráficos
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss ')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
plt.close()


