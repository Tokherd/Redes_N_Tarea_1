import os
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Flatten,
                                     GlobalAveragePooling2D, Dropout, Activation,
                                     BatchNormalization, AveragePooling2D, add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import time
from collections import Counter
import pandas as pd
from tensorflow.keras.utils import to_categorical

# ========== CONFIGURACIÓN ==========

# Definición de directorios
directorio_train = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
directorio_test = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'
batch_size = 64
epochs = 200
depth = 104  # Profundidad de la red ResNet
version = 1  # Usamos ResNet v1

input_shape = (64, 64, 1)
target_size = (64, 64)

# ========== CARGA DE DATOS ==========

# Función para crear generadores de datos con más augmentaciones
def crear_generadores(train_dir, test_dir, target_size=(64, 64), batch_size=64, augmentation=True):
    if augmentation:
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
        validation_split=0.3)
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

train_gen, test_gen = crear_generadores(directorio_train, directorio_test, target_size=target_size, batch_size=batch_size, augmentation=True)
num_classes = train_gen.num_classes

# ========== CÁLCULO DE PESOS DE CLASE ==========

def calcular_pesos_de_clase(train_gen):
    labels = train_gen.classes
    class_totals = Counter(labels)
    total_samples = len(labels)
    class_weights = {
        i: total_samples / (train_gen.num_classes * class_totals[i])
        for i in class_totals
    }
    return class_weights

# ========== FUNCIONES DE RESNET ==========

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

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=7):
    if (depth - 2) % 6 != 0:
        raise ValueError('La profundidad debe ser 6n+2 (por ejemplo: 20, 32, 44, 56, 110)')

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters,
                                 kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ========== COMPILACIÓN Y CALLBACKS ==========

model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

model.summary()

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'resnet_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=2,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]

# ========== MEDIR TIEMPO DE ENTRENAMIENTO ==========

start_time = time.time()

# ========== ENTRENAMIENTO ========== 
print('Usando generadores con aumento de datos...')
history = model.fit(train_gen,
                    epochs=epochs,
                    validation_data=test_gen,
                    callbacks=callbacks,
                    class_weight=calcular_pesos_de_clase(train_gen),  # Agregar los pesos de clase
                    verbose=2)

end_time = time.time()
execution_time = end_time - start_time
print("Tiempo total de entrenamiento (segundos):", execution_time)

# ========== EVALUACIÓN FINAL ==========

scores = model.evaluate(test_gen, verbose=2)
print('Pérdida en test:', scores[0])
print('Precisión en test:', scores[1])

# ========== GUARDAR RESULTADOS ==========

output_dir = 'resultados_resnetv1'
os.makedirs(output_dir, exist_ok=True)

# Guardar métricas finales
np.save(os.path.join(output_dir, 'test_loss.npy'), scores[0])
np.save(os.path.join(output_dir, 'test_accuracy.npy'), scores[1])
np.save(os.path.join(output_dir, 'execution_time.npy'), execution_time)

# Guardar historial de entrenamiento como CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(output_dir, 'history.csv'))

# ========== GRAFICAR CURVAS ==========

# Curva de pérdida
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Curva de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'curva_perdida.png'))
plt.close()

# Curva de precisión
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Curva de precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'curva_precision.png'))
plt.close()

