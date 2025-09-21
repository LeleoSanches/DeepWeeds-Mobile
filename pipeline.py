
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import image_dataset_from_directory
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV3Large

from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import callbacks

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter

from tensorflow.keras import mixed_precision

from sklearn.metrics import confusion_matrix, classification_report


import plot_training_results

# Mixed Precision - Entra FP16 
## Tem que garantir que a saída do modelo é float32
mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy()) 

# Global - Paths
IMG_DIR = "/home/leo/Documentos/DeepWeeds-master/images/"
LABEL_DIR = "/home/leo/Documentos/DeepWeeds-master/labels/"


# Global - Parâmetros
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
classes = [0,1,2,3,4,5,6,7,8]

mixed_precision.set_global_policy("mixed_float16")


data = pd.read_csv(LABEL_DIR + "labels.csv")
data['Label'] = data["Label"].astype(str)
data['Filename'] = data['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))


# 1) Split estratificado - Balanceamento das Classes
df_train, df_val = train_test_split(
    data, test_size=0.3, stratify=data['Label'], random_state=42
)

# 2) Classes consistentes (ordenadas)
classes = sorted(data['Label'].unique().tolist())

# 3) Preprocess do backbone (NÃO use rescale junto)
#preprocess_fn = mobilenet_v2.preprocess_input

preprocess_fn = mobilenet_v3.preprocess_input

# 4) Dois datagens: augment só no treino
"""
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)


train_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode="constant",
    shear_range=0.2,
    zoom_range=(0.5, 1),
    horizontal_flip=True,
    rotation_range=360,
    channel_shift_range=25,
    brightness_range=(0.75, 1.25))

"""

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    fill_mode = 'constant',
    shear_range = 0.2,
    rotation_range=360,
    channel_shift_range=25,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=(0.1,1),
    horizontal_flip=True,
    brightness_range=(0.75, 1.25)
)



val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn
)

# 5) Generators SEM validation_split/subset
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="Filename",
    y_col="Label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col="Filename",
    y_col="Label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    shuffle=False
)

# 6) Valida os índices e classes
print(train_generator.class_indices)
print(val_generator.class_indices)
assert train_generator.class_indices == val_generator.class_indices
assert set(train_generator.filenames).isdisjoint(set(val_generator.filenames))

# 7) class_weight p/ desbalanceamento
cw = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(classes)),
    y=train_generator.classes
)
class_weight = {i: float(w) for i, w in enumerate(cw)}
print("Distribuição treino:", Counter(train_generator.classes))
print("Distribuição val   :", Counter(val_generator.classes))

"""
##MODEL
### MOBILENETV2
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Congela a base inicialmente
]
"""

base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)


base_model.trainable = False
##Transfer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(len(classes), activation='sigmoid', dtype="float32" )(x)

model = Model(inputs=base_model.input, outputs=outputs)



# 8) Compile + callbacks
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
"""
cbs = [
    callbacks.ModelCheckpoint("best_head.keras", monitor="val_accuracy",
                              save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                            restore_best_weights=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, min_lr=1e-6),
]
"""
cbs = [
    callbacks.ModelCheckpoint("best_head.keras", monitor="val_accuracy",
                              save_best_only=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, min_lr=1e-6),
    callbacks.CSVLogger("treino_log.csv", append=False),
    callbacks.TensorBoard(log_dir="tb_logs", histogram_freq=1)
]


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=200,
    callbacks=cbs,
    class_weight=class_weight,   # <— comente esta linha se não quiser usar
    verbose=1
)



#Fine-Tunning
model.load_weights("best_head.keras")  # garante melhor ponto de partida

unfreeze_from = int(len(base_model.layers) * 0.6)  # ajuste 0.6–0.75 conforme VRAM/estabilidade
base_model.trainable = True
for i, L in enumerate(base_model.layers):
    if i < unfreeze_from:
        L.trainable = False
    elif isinstance(L, layers.BatchNormalization):
        L.trainable = False
    else:
        L.trainable = True

# LR bem baixo + (opcional) weight decay
opt = optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5)


model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

cbs_ft = [
    callbacks.ModelCheckpoint("best_v3_finetune.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]

history_ft = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=cbs_ft,
    verbose=1
)



plot_training_curves(history, out_png="curvas_mobilenetv3.png", title="MobileNetV3 - DeepWeeds")
save_history_csv(history, out_csv="historico_mobilenetv3.csv")

plot_confusion_and_report(
    model, val_generator, train_generator.class_indices,
    cm_png="cm_mobilenetv3.png",
    report_txt="report_mobilenetv3.txt",
    normalize=True
)


plot_training_curves(history_ft, out_png="curvas_mobilenetv3_finetunning.png", title="MobileNetV3 Tuned- DeepWeeds")
save_history_csv(history_ft, out_csv="historico_mobilenetv3_finetunning.csv")
