
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


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter



# Global - Paths
IMG_DIR = "/home/leo/Documentos/DeepWeeds-master/images/"
LABEL_DIR = "/home/leo/Documentos/DeepWeeds-master/labels/"


# Global - Parâmetros
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
classes = [0,1,2,3,4,5,6,7,8]




data = pd.read_csv(LABEL_DIR + "labels.csv")
data['Label'] = data["Label"].astype(str)
data['Filename'] = data['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))


# 1) Split estratificado (você já tem)
df_train, df_val = train_test_split(
    data, test_size=0.2, stratify=data['Label'], random_state=42
)

# 2) Classes consistentes (ordenadas)
classes = sorted(data['Label'].unique().tolist())

# 3) Preprocess do backbone (NÃO use rescale junto)
preprocess_fn = mobilenet_v2.preprocess_input

# 4) Dois datagens: augment só no treino
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
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

# 6) Sanity checks
print(train_generator.class_indices)
print(val_generator.class_indices)
assert train_generator.class_indices == val_generator.class_indices
assert set(train_generator.filenames).isdisjoint(set(val_generator.filenames))

# 7) (Opcional) class_weight p/ desbalanceamento
cw = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(classes)),
    y=train_generator.classes
)
class_weight = {i: float(w) for i, w in enumerate(cw)}
print("Distribuição treino:", Counter(train_generator.classes))
print("Distribuição val   :", Counter(val_generator.classes))


##MODEL
### MOBILENETV2
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Congela a base inicialmente

##Transfer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)



# 8) Compile + callbacks
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cbs = [
    callbacks.ModelCheckpoint("best_head.keras", monitor="val_accuracy",
                              save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                            restore_best_weights=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, min_lr=1e-6),
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=60,
    callbacks=cbs,
    class_weight=class_weight,   # <— comente esta linha se não quiser usar
    verbose=1
)