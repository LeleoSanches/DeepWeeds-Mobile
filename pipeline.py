import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from collections import Counter

import tensorflow as tf, keras

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Models and Backbone's
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small, MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
from tensorflow.keras.applications import efficientnet_v2

#Utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from plot_training_results import utils

# Mixed Precision - Entra FP16 
## Tem que garantir que a saída do modelo é float32
mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy()) 
print("Keras version:", keras.__version__)
K.set_image_data_format('channels_last')


BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
LABELS_DIR = BASE_DIR / "labels"

if not IMG_DIR.exists():
    raise FileNotFoundError(
        f"A pasta de imagens não foi encontrada em: {IMG_DIR} \n"
        f"Crie a pasta com as imagens do Deepweeds"
    )

for directory in [RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Global - Parâmetros
MODEL_NAME = "mobilenetv2"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DEFAULT_IMG_SIZE = {
    "mobilenetv2":      224,
    "mobilenetv3large": 224,
    "mobilenetv3small": 224,
    "efficientnetv2b0": 224,
    "efficientnetv2b1": 240,
    "efficientnetv2b2": 260,
    "efficientnetv2b3": 300,
}
AUTOTUNE = tf.data.AUTOTUNE
classes = [0,1,2,3,4,5,6,7,8]
TRAINING_EPOCHS = 200
FINETUNNING_EPOCHS = 100
TEST_SIZE = 0.2

#Global
SUPPORTED_MODELS = ["mobilenetv2", "mobilenetv3large", "mobilenetv3small", "efficientnetb0", "efficientnetb1", "efficientnetb2", "efficientnetb3",
    "efficientnetv2b0", "efficientnetv2b1", "efficientnetv2b2", "efficientnetv2b3"]


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Treino DeepWeeds com modelos mobile (transfer learning)",
        epilog=f"Modelos suportados: {', '.join(SUPPORTED_MODELS)}"
    )
    parser.add_argument(
        "--model", "-m",
        default="mobilenetv3large",
        choices=SUPPORTED_MODELS,
        help="Backbone a utilizar. Opções: %(choices)s (padrão: %(default)s)"
    )
    return parser


def load_split_data(label_dir: str, test_size: float):
    #Load Labels df
    try:
        data = pd.read_csv(LABEL_DIR + "labels.csv")
        data['Label'] = data["Label"].astype(str)
        data['Filename'] = data['Filename'].apply(lambda x: os.path.join(IMG_DIR, x))
    
    except FileNotFoundError as e:
        print(f"[ERRO] labels.csv não encontrado: {csv_path} "
              f"(verifique se 'label_dir' está correto).")
        raise
    
    #Train test split
    df_train, df_val = train_test_split(
        data, test_size=test_size, stratify=data['Label'], random_state=75
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # 2) Classes consistentes (ordenadas)
    classes = sorted(data['Label'].unique().tolist())
    
    #Debbug
    print("[DEBUG] Filename resolvido:", data["Filename"].iloc[0])
    print(f"[DEBUG] Tamanho total: {len(data)} | Treino: {len(df_train)} | Validação: {len(df_val)} "
          f"({len(df_val)/len(data)*100:.1f}% val)")

    
    return df_train, df_val, classes


def get_backbone(name: str, img_size):
    h, w = img_size
    input_shape = (h, w, 3)
    print(f"[DEBBUG] Input Shape Backbone: {input_shape}")
    name = name.lower()
    if name == "mobilenetv2":
        preprocess_fn = mobilenet_v2.preprocess_input
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    elif name == "mobilenetv3large":
        preprocess_fn = mobilenet_v3.preprocess_input
        base_model = MobileNetV3Large(input_shape=input_shape, include_top=False, weights="imagenet")
    elif name == "mobilenetv3small":
        preprocess_fn = mobilenet_v3.preprocess_input
        base_model = MobileNetV3Small(input_shape=input_shape, include_top=False, weights="imagenet")

    # -------- EfficientNet V2 --------
    elif name == "efficientnetv2b0":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B0(input_shape=input_shape, include_top=False, weights="imagenet")

    elif name == "efficientnetv2b1":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B1(input_shape=input_shape, include_top=False, weights="imagenet")

    elif name == "efficientnetv2b2":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B2(input_shape=input_shape, include_top=False, weights="imagenet")

    elif name == "efficientnetv2b3":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B3(input_shape=input_shape, include_top=False, weights="imagenet")

    else:
        raise ValueError("--model deve ser: mobilenetv2, mobilenetv3large, mobilenetv3small, "
                         "efficientnetv2b0/b1/b2/b3")

    return preprocess_fn, base_model


# 4) Dois datagens - augment só no treino
def set_generators(preprocess_fn, df_train, df_val,img_size ,debbug: bool):
    
    #Augmentation
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
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        shuffle=True,
        seed=122
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col="Filename",
        y_col="Label",
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        shuffle=False
    )

    if debbug:
        print(train_generator.class_indices)
        print(val_generator.class_indices)
        
        assert train_generator.class_indices == val_generator.class_indices
        assert set(train_generator.filenames).isdisjoint(set(val_generator.filenames))

        print("[DEBBUG] Distribuição treino:", Counter(train_generator.classes))
        print("[DEBBUG] Distribuição val   :", Counter(val_generator.classes))

    return train_generator, val_generator


def set_transferlearning(base_model):

    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(len(classes), activation='sigmoid', dtype="float32" )(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    #Compile Model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    return model


def fit_model(model, train_generator, val_generator, name:str, class_weight:bool, epochs:int):

    cbs = [
        callbacks.ModelCheckpoint("best_head.keras", monitor="val_accuracy",
                                save_best_only=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=16, min_lr=1e-6),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=32,
                                restore_best_weights=True, mode="max"),
        callbacks.CSVLogger(f"treino_log{name}.csv", append=False),
        callbacks.TensorBoard(log_dir="tb_logs", histogram_freq=1)
    ]

    if class_weight:
        cw = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(classes)),
            y=train_generator.classes
        )
        class_weight = {i: float(w) for i, w in enumerate(cw)}



        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=cbs,
            class_weight=class_weight,
            verbose=1
        )
    
    else:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=cbs,
            verbose=1
        )

    return history


#Fine-Tunning
def set_finetunning(model):
    model.load_weights("best_head.keras")

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
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

    return model

def fit_finetunning(model,train_generator, val_generator, epochs: int ,name:str):
    cbs_ft = [
        callbacks.ModelCheckpoint(f"best_{name}_finetune.keras", monitor="val_accuracy", mode="max", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history_ft = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=cbs_ft,
        verbose=1
    )
    return history_ft

#Ajusta img_size pelo modelo
def auto_img_size(model_name):
    FALLBACK_IMG_SIZE = 224
    s = DEFAULT_IMG_SIZE.get(model_name.lower(), FALLBACK_IMG_SIZE)
    print(f"[INFO] Ajustando IMG_SIZE para {s}x{s} para {model_name}")
    return (s, s)



if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    MODEL_NAME = str(args.model) 

    print(f"[INFO] Modelo selecionado: {MODEL_NAME}")
    IMG_SIZE = auto_img_size(MODEL_NAME)
    print(f"[INFO] IMG_SIZE De {MODEL_NAME} Ajustado para: {IMG_SIZE}")

    #Training
    df_train, df_val, classes = load_split_data(label_dir = LABEL_DIR, test_size=TEST_SIZE)
    preprocess_fn, base_model = get_backbone(name=MODEL_NAME, img_size=IMG_SIZE)
    train_generator, val_generator = set_generators(preprocess_fn, df_train, df_val,IMG_SIZE ,debbug=True)
    model = set_transferlearning(base_model)
    history = fit_model(model, train_generator, val_generator, name = MODEL_NAME,class_weight=False,epochs=TRAINING_EPOCHS )
    model = set_finetunning(model)
    history_ft = fit_finetunning(model, train_generator, val_generator,epochs = FINETUNNING_EPOCHS ,name = MODEL_NAME)


    #Plotting
    print('iniciando Plotting Results')
    plots = utils()

    plots.plot_training_curves(history, out_png=f"curvas_{MODEL_NAME}.png", title="MobileNetV3 - DeepWeeds")
    plots.save_history_csv(history, out_csv=f"historico_{MODEL_NAME}.csv")
    plots.plot_confusion_and_report(
        model, val_generator, train_generator.class_indices,
        cm_png=f"cm_{MODEL_NAME}.png",
        report_txt=f"report_{MODEL_NAME}.txt",
        normalize=True
    )
    plots.plot_training_curves(history_ft, out_png=f"curvas_{MODEL_NAME}_finetunning.png", title=f"{MODEL_NAME} Tuned- DeepWeeds")
    plots.save_history_csv(history_ft, out_csv=f"historico_{MODEL_NAME}_finetunning.csv")
    joined_dict, df_hist = plots.concat_histories(history, history_ft)
    df_hist.to_csv(f"historico_{MODEL_NAME}_joined.csv", index=False)
    split_epoch = len(history.history.get("accuracy", []))
    plots.plot_history_joined(joined_dict, out_png=f"{MODEL_NAME}_head_plus_ft.png",
                        title=f"{MODEL_NAME} - DeepWeeds (Head + FT)",
                        split_epoch=split_epoch)


