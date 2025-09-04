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

# Global - Paths
IMG_DIR = "/home/leo/Documentos/DeepWeeds-master/images/"
LABEL_DIR = "/home/leo/Documentos/DeepWeeds-master/labels/"


# Global - Parâmetros
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


data = pd.read_csv(LABEL_DIR + "labels.csv")
data['Label'] = data["Label"].astype(str)
data['Filename'] = data['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))
df_train, df_val = train_test_split(data, test_size=0.2, stratify=data['Label'], random_state=42)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


# Gerador para treino
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="Filename",
    y_col="Label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Gerador para validação
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col="Filename",
    y_col="Label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
"""
##MODEL
### MOBILENETV2
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Congela a base inicialmente
"""

##MOBILENETV3SMALL
base_model = MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

##Transfer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(9, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=150)



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