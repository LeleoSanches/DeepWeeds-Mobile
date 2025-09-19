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

"""
train = pd.read_csv(LABEL_DIR + "train_subset0.csv")
train['Label'] = train["Label"].astype(str)
train['Filename'] = train['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))

test = pd.read_csv(LABEL_DIR + "test_subset0.csv")
test['Label'] = test["Label"].astype(str)
test['Filename'] = test['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))

val = pd.read_csv(LABEL_DIR + "test_subset0.csv")
val['Label'] = val["Label"].astype(str)
val['Filename'] = val['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))
"""


df_train, df_val = train_test_split(data, test_size=0.2, stratify=data['Label'], random_state=42)


classes = sorted(data['Label'].unique().tolist())
#Ou utiliza o backbone ou faz resize
preprocess_fn = mobilenet_v2.preprocess_input

#Train datagen com augmentation e Split
"""
Para utilizar o datagen com augmentation, precisamos splitar o dataframe antes, pois se não o modelo valida com o augmentation.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range=0.1
)
"""
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    validation_split=0.2,
)


# Gerador para treino
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    x_col="Filename",
    y_col="Label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    classes=classes,
    seed=42
)



# Gerador para validação
val_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    x_col="Filename",
    y_col="Label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    classes=classes,
    seed=42
)


# debbug class_indices:
print(train_generator.class_indices)
print(val_generator.class_indices)
assert train_generator.class_indices == val_generator.class_indices



##MODEL
### MOBILENETV2
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Congela a base inicialmente

##MOBILENETV3
"""
base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
"""


##Transfer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)


###CALLBACKS
cbs = [
    callbacks.ModelCheckpoint("best_head.keras", monitor="val_accuracy",
                              save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                            restore_best_weights=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, min_lr=1e-6),
]
#####




model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=150,
    callbacks=cbs,
    verbose=1
)

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia por época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss por época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")

    plt.show()


final_metrics = {
    "final_train_acc": history.history['accuracy'][-1],
    "final_val_acc": history.history['val_accuracy'][-1],
    "final_train_loss": history.history['loss'][-1],
    "final_val_loss": history.history['val_loss'][-1]
}

print(final_metrics)



df_metrics = pd.DataFrame(history.history)
df_metrics.to_csv("historico_treinamento.csv", index=False)

plot_history(history, save_path='treinamento_mobilenetV3Small.png')