# ======================== CONFIGURACIÓN INICIAL ========================
import os
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, Dropout,
                                     concatenate, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import img_to_array
# ======================== GPU: Limitar uso de memoria ========================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ======================== CONFIGURACIÓN ========================
directorio_train = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train'
directorio_test = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test'
batch_size = 64
epochs = 200
growth_rate = 16
depth = 100
num_dense_blocks = 5    
compression_factors = [0.3, 0.5, 0.7]
input_shape = (128, 128, 1)
target_size = (128, 128)

# ======================== FUNCIONES ========================
def calcular_pesos_clase(generator):
    etiquetas = generator.classes
    pesos = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(etiquetas), y=etiquetas)
    return dict(enumerate(pesos))

def crear_generadores_con_validacion(train_dir, target_size=(128, 128), batch_size=128, augmentation=True, val_split=0.3):
    if augmentation:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.4,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=val_split
        )
    else:
        datagen = ImageDataGenerator(rescale=1/255, validation_split=val_split)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    return train_gen, val_gen



# ======================== GENERADORES Y PESOS ========================
train_gen , val_gen = crear_generadores_con_validacion(directorio_train, target_size=target_size, batch_size=batch_size, augmentation=True)
pesos_clase = calcular_pesos_clase(train_gen)
steps_per_epoch = len(train_gen)

num_classes = train_gen.num_classes

# ======================== LEARNING RATE SCHEDULER ========================
def lr_schedule(epoch):
    lr = 2e-3  # Tasa de aprendizaje inicial
    if epoch > 10:
        lr = 1e-4  # Reducir la tasa de aprendizaje después de 10 épocas
    return lr

# ======================== LOOP PRINCIPAL ========================
for compression_factor in compression_factors:
    print(f"\n=== Entrenando DenseNet con compression_factor = {compression_factor} ===\n")
    start_time = time.time()

    # ==== CALLBACKS ====
    tag = f"densenet_c{int(compression_factor*100)}"
    save_dir = os.path.join(os.getcwd(), f'saved_models_{tag}')
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, f'{tag}_best.keras')
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, min_lr=5e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [checkpoint, lr_reducer, early_stopping, lr_scheduler]

    # ==== MODELO ====
    inputs = Input(shape=input_shape)
    x = Conv2D(2 * growth_rate, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    num_bottleneck_layers = (depth - 4) // num_dense_blocks // 2
    num_filters_bef_dense_block = 2 * growth_rate

    for i in range(num_dense_blocks):
        for j in range(num_bottleneck_layers):
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv2D(4 * growth_rate, kernel_size=1, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            y = Dropout(0.3)(y)  # Regularización Dropout
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(growth_rate, kernel_size=3, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            x = concatenate([x, y])

        if i != num_dense_blocks - 1:
            num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
            num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
            y = BatchNormalization()(x)
            y = Conv2D(num_filters_bef_dense_block, kernel_size=1, padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            y = Dropout(0.3)(y)  # Regularización Dropout
            x = AveragePooling2D(pool_size=2)(y)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)  # Regularización Dropout
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # ==== ENTRENAMIENTO ====
    history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=epochs,
    callbacks=callbacks,
    class_weight=pesos_clase,
    verbose=2
)



    # ==== EVALUACIÓN FINAL ====
    end_time = time.time()
    model.load_weights(best_model_path)

    # Genera el generador de test
    def crear_generador_test(test_dir, target_size=(128, 128), batch_size=128):
        test_datagen = ImageDataGenerator(rescale=1./255)  # Mismo reescalado que en entrenamiento
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical'
        )
        return test_gen

    # ======================== GENERADOR DE TEST ========================
    test_gen = crear_generador_test(directorio_test, target_size=target_size, batch_size=batch_size)

    scores = model.evaluate(test_gen, steps=len(test_gen), verbose=2)
    print(f"[RESULT] Compression {compression_factor} -> Test loss: {scores[0]:.4f}, Test accuracy: {scores[1]:.4f}")

    # ==== GUARDADO ====
    h5_model_path = os.path.join(save_dir, f'{tag}_best_model.h5')
    model.save(h5_model_path)
    print(f"[SAVE] Modelo guardado en formato HDF5: {h5_model_path}")

    output_dir = os.path.join(save_dir, "resultados")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'test_loss.npy'), scores[0])
    np.save(os.path.join(output_dir, 'test_accuracy.npy'), scores[1])
    np.save(os.path.join(output_dir, 'execution_time.npy'), end_time - start_time)
    np.save(os.path.join(output_dir, 'train_loss.npy'), history.history['loss'])
    np.save(os.path.join(output_dir, 'val_loss.npy'), history.history['val_loss'])
    np.save(os.path.join(output_dir, 'train_acc.npy'), history.history['accuracy'])
    np.save(os.path.join(output_dir, 'val_acc.npy'), history.history['val_accuracy'])

    # ==== GRÁFICAS ====
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.legend()
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title(f'Pérdida durante el entrenamiento: {tag}')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.legend()
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title(f'Precisión durante el entrenamiento: {tag}')
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    
    gc.collect()  # Limpiar memoria



