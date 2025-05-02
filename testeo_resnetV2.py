import time
import numpy as np
import os
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import csv
import math
from keras import backend as K
from keras.layers import (Input, Dense, Flatten, Conv2D, BatchNormalization, Activation,
                          AveragePooling2D, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------- CONFIGURACIÓN GENERAL ----------
directorio_train = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
directorio_test = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'
batch_size = 128
num_classes = 7
epochs = 200
data_augmentation = True
model_type = 'ResNet20v2'
depth = 100
input_shape = (128, 128, 1)
target_size = (128, 128)

# ---------- CARGA DE DATOS Y CALCULO DE PESOS DE CLASE ----------
def calcular_pesos_de_clase(train_dir, batch_size=128):
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Calcular la frecuencia de las clases
    class_frequencies = np.bincount(train_gen.classes)
    total_samples = len(train_gen.classes)
    
    # Calcular los pesos de las clases
    class_weights = total_samples / (len(class_frequencies) * class_frequencies)
    
    return class_weights

# Función de generadores de datos con augmentación avanzada
def crear_generadores(train_dir, test_dir, target_size=(128, 128), batch_size=128, augmentation=True):
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            rotation_range=30,  # Rotación aleatoria de las imágenes
            zoom_range=0.2,  # Zoom aleatorio
            shear_range=0.2,  # Corte aleatorio
            brightness_range=[0.8, 1.2],  # Ajuste de brillo
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

# ---------- GENERAR DATOS Y CALCULAR PESOS ----------
train_gen, test_gen = crear_generadores(directorio_train, directorio_test, target_size=target_size, batch_size=batch_size, augmentation=True)
class_weights = calcular_pesos_de_clase(directorio_train, batch_size=batch_size)
num_classes = train_gen.num_classes

# ---------- DEFINICIÓN DE FUNCIÓN RESNET_LAYER ----------
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                  padding='same', kernel_initializer='he_normal', kernel_regularizer=None)
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

# ---------- RESNET V2 ----------
def resnet_v2(input_shape, depth, num_classes=7):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (e.g., 110 in [b])')
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1,
                             strides=strides, activation=activation,
                             batch_normalization=batch_normalization, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)
            if res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1,
                                 strides=strides, activation=None, batch_normalization=False)
            x = add([x, y])
        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ---------- DEFINICIÓN DE PROGRAMACIÓN DE LR ----------
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
    return lr

# ---------- COMPILACIÓN ----------
model = resnet_v2(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
# ---------- CALLBACKS ----------
save_dir = os.path.join(os.getcwd(), 'saved_models')
os.makedirs(save_dir, exist_ok=True)
model_name = f'ResnetV2_{model_type}.h5'
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# ENTRENAMIENTO
start_time = time.time()

if data_augmentation:
    history = model.fit(train_gen,
                        validation_data=test_gen,
                        epochs=epochs,
                        verbose=2,
                        steps_per_epoch=len(train_gen),
                        validation_steps=len(test_gen),
                        callbacks=callbacks,
                        class_weight=class_weights)  # Usar los pesos de las clases
else:
    history = model.fit(train_gen,
                        validation_data=test_gen,
                        epochs=epochs,
                        verbose=2,
                        steps_per_epoch=len(train_gen),
                        validation_steps=len(test_gen),
                        callbacks=callbacks,
                        class_weight=class_weights)  # Usar los pesos de las clases

end_time = time.time()
training_time = end_time - start_time

# EVALUACIÓN FINAL
scores = model.evaluate(test_gen, verbose=0)
final_accuracy = scores[1]
final_loss = scores[0]

# ---------- GUARDAR CURVAS ----------
results_dir = 'resultados_resnetv2'
os.makedirs(results_dir, exist_ok=True)

np.save(os.path.join(results_dir, 'train_loss.npy'), history.history['loss'])
np.save(os.path.join(results_dir, 'val_loss.npy'), history.history['val_loss'])
np.save(os.path.join(results_dir, 'train_accuracy.npy'), history.history['accuracy'])
np.save(os.path.join(results_dir, 'val_accuracy.npy'), history.history['val_accuracy'])

# ---------- GUARDAR METRICAS FINALES ----------
result_dir = 'resultados_resnetv2'
os.makedirs(result_dir, exist_ok=True)

np.save(os.path.join(result_dir, 'test_loss.npy'), scores[0])
np.save(os.path.join(result_dir, 'test_accuracy.npy'), scores[1])
np.save(os.path.join(result_dir, 'execution_time.npy'), training_time)

# Guardar curvas de pérdida y accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

np.save(os.path.join(result_dir, 'train_loss.npy'), train_loss)
np.save(os.path.join(result_dir, 'val_loss.npy'), val_loss)
np.save(os.path.join(result_dir, 'train_accuracy.npy'), train_acc)
np.save(os.path.join(result_dir, 'val_accuracy.npy'), val_acc)

# ----------------------------- GRÁFICOS -----------------------------
# Pérdida
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Pérdida entrenamiento')
plt.plot(val_loss, label='Pérdida validación')
plt.title('Curva de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'curva_perdida.png'))
plt.close()

# Accuracy
plt.figure(figsize=(8, 6))
plt.plot(train_acc, label='Precisión entrenamiento')
plt.plot(val_acc, label='Precisión validación')
plt.title('Curva de precisión')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'curva_accuracy.png'))
plt.close()
