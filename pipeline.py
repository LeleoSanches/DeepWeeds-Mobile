import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from paths import get_project_dirs
from plot_training_results import utils

from collections import Counter

import tensorflow as tf, keras

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Models and Backbone's
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.applications import resnet, resnet_v2
from tensorflow.keras.applications import InceptionV3, NASNetMobile
from tensorflow.keras.applications import inception_v3, nasnet
from tensorflow.keras.applications import (
    MobileNetV3Large,
    MobileNetV3Small,
    MobileNetV2,
    ResNet50,
    ResNet101V2,
)

from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
)
from tensorflow.keras.applications import efficientnet_v2

# Utils
# from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Mixed Precision - Entra FP16
## Tem que garantir que a saída do modelo é float32
# mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy())
print("Keras version:", keras.__version__)
K.set_image_data_format("channels_last")
print("TF:", tf.__version__)
print("GPUs visíveis:", tf.config.list_physical_devices("GPU"))


# Global - Path, cooldown=2s
DIRS = get_project_dirs()
IMG_DIR = str(DIRS["images"]) + "/"
RESULTS_DIR = str(DIRS["results"]) + "/"
MODELS_DIR = str(DIRS["models"]) + "/"
LABEL_DIR = str(DIRS["labels"]) + "/"
OUTPUT_DIR = RESULTS_DIR

# Global - Parâmetros
MODEL_NAME = "mobilenetv2"

IMG_SIZE = (224, 224)
DEFAULT_IMG_SIZE = {
    "mobilenetv2": 224,
    "mobilenetv3large": 224,
    "mobilenetv3small": 224,
    "efficientnetv2b0": 224,
    "efficientnetv2b1": 240,
    "efficientnetv2b2": 260,
    "efficientnetv2b3": 300,
    "resnet50": 224,
    "resnet101v2": 224,
    "nasnetmobile": 224,
    "inceptionv3": 299,
}
AUTOTUNE = tf.data.AUTOTUNE
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]


# Global
SUPPORTED_MODELS = [
    "mobilenetv2",
    "mobilenetv3large",
    "mobilenetv3small",
    "efficientnetv2b0",
    "efficientnetv2b1",
    "efficientnetv2b2",
    "efficientnetv2b3",
    "resnet50",
    "resnet101v2",
    "nasnetmobile",
    "inceptionv3",
]


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Treino DeepWeeds com modelos mobile (transfer learning)",
        epilog=f"Modelos suportados: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mobilenetv3large",
        choices=SUPPORTED_MODELS,
        help="Backbone a utilizar. Opções: %(choices)s (padrão: %(default)s)",
    )
    parser.add_argument(
        "--fold",
        "-f",
        default="fold_1",
        help="Fold a utilizar. Opções: fold_1, fold_2, fold_3 (padrão: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=32,
        type=int,
        help="Tamanho do batch. (padrão: %(default)s)",
    )
    parser.add_argument(
        "--training_epochs",
        "-te",
        default=200,
        type=int,
        help="Número de épocas para treinamento. (padrão: %(default)s)",
    )
    parser.add_argument(
        "--finetunning_epochs",
        "-fe",
        default=100,
        type=int,
        help="Número de épocas para fine-tuning. (padrão: %(default)s)",
    )
    return parser


def load_split_data(label_dir: str, fold: str = "fold_1"):
    fold_path = Path(label_dir) / fold

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fold '{fold}' não encontrado em {label_dir}. "
            f"Verifique se os folds foram gerados corretamente."
        )
    for split in ["train.csv", "val.csv", "test.csv"]:
        if not (fold_path / split).exists():
            raise FileNotFoundError(
                f"Arquivo '{split}' não encontrado em {fold_path}. "
                f"Verifique se os arquivos de split foram gerados corretamente."
            )
    df_train = pd.read_csv(fold_path / "train.csv")
    df_train["Label"] = df_train["Label"].astype(str)
    df_train["Filename"] = df_train["Filename"].apply(
        lambda x: os.path.join(IMG_DIR, x)
    )

    df_val = pd.read_csv(fold_path / "val.csv")
    df_val["Label"] = df_val["Label"].astype(str)
    df_val["Filename"] = df_val["Filename"].apply(lambda x: os.path.join(IMG_DIR, x))

    df_test = pd.read_csv(fold_path / "test.csv")
    df_test["Label"] = df_test["Label"].astype(str)
    df_test["Filename"] = df_test["Filename"].apply(lambda x: os.path.join(IMG_DIR, x))

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    classes = sorted(df_train["Label"].unique())

    print(f"[INFO] Fold '{fold}' carregado com sucesso:")
    print(
        f"[DEBUG] Tamanho total: {len(df_train) + len(df_val) + len(df_test)} | "
        f"Treino: {len(df_train)} | Validação: {len(df_val)} | Teste: {len(df_test)} "
        f"({len(df_val)/(len(df_train) + len(df_val) + len(df_test))*100:.1f}% val)"
    )

    return df_train, df_val, df_test, classes


def get_backbone(name: str, img_size):
    h, w = img_size
    input_shape = (h, w, 3)
    print(f"[DEBBUG] Input Shape Backbone: {input_shape}")
    name = name.lower()
    if name == "mobilenetv2":
        preprocess_fn = mobilenet_v2.preprocess_input
        base_model = MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif name == "mobilenetv3large":
        preprocess_fn = mobilenet_v3.preprocess_input
        base_model = MobileNetV3Large(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif name == "mobilenetv3small":
        preprocess_fn = mobilenet_v3.preprocess_input
        base_model = MobileNetV3Small(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

    # -------- EfficientNet V2 --------
    elif name == "efficientnetv2b0":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

    elif name == "efficientnetv2b1":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B1(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

    elif name == "efficientnetv2b2":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

    elif name == "efficientnetv2b3":
        preprocess_fn = efficientnet_v2.preprocess_input
        base_model = EfficientNetV2B3(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    # -------- ResNet --------
    elif name == "resnet50":
        preprocess_fn = resnet.preprocess_input
        base_model = ResNet50(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif name == "resnet101v2":
        preprocess_fn = resnet_v2.preprocess_input
        base_model = ResNet101V2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif name == "nasnetmobile":
        preprocess_fn = nasnet.preprocess_input
        base_model = NASNetMobile(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif name == "inceptionv3":
        preprocess_fn = inception_v3.preprocess_input
        base_model = InceptionV3(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(
            "--model deve ser: mobilenetv2, mobilenetv3large, mobilenetv3small, "
            "efficientnetv2b0/b1/b2/b3, resnet50, resnet101v2, nasnetmobile, inceptionv3"
        )

    return preprocess_fn, base_model


# 4) Dois datagens - augment só no treino
def set_generators(
    preprocess_fn, df_train, df_val, df_test, img_size, BATCH_SIZE, debbug: bool
):

    # Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        fill_mode="constant",
        shear_range=0.2,
        rotation_range=360,
        channel_shift_range=25,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=(0.1, 1),
        horizontal_flip=True,
        brightness_range=(0.75, 1.25),
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

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
        seed=122,
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col="Filename",
        y_col="Label",
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        shuffle=False,
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col="Filename",
        y_col="Label",
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        shuffle=False,
    )

    if debbug:
        print(train_generator.class_indices)
        print(val_generator.class_indices)
        print(test_generator.class_indices)

        assert train_generator.class_indices == val_generator.class_indices
        assert set(train_generator.filenames).isdisjoint(set(val_generator.filenames))
        assert set(train_generator.filenames).isdisjoint(set(test_generator.filenames))

        print("[DEBBUG] Distribuição treino:", Counter(train_generator.classes))
        print("[DEBBUG] Distribuição val   :", Counter(val_generator.classes))
        print("[DEBBUG] Distribuição test  :", Counter(test_generator.classes))

    return train_generator, val_generator, test_generator


def set_transferlearning(base_model):

    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    outputs = Dense(len(classes), activation="softmax", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def fit_model(
    model,
    train_generator,
    val_generator,
    name: str,
    class_weight: bool,
    epochs: int,
    fold: str,
):

    cbs = [
        callbacks.ModelCheckpoint(
            "best_head.keras", monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=16, min_lr=1e-6
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=32,
            min_delta=0,
            restore_best_weights=True,
            mode="max",
        ),
        callbacks.CSVLogger(f"treino_log{name}_{fold}.csv", append=False),
        callbacks.TensorBoard(log_dir="tb_logs", histogram_freq=1),
    ]

    if class_weight:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(classes)),
            y=train_generator.classes,
        )
        class_weight = {i: float(w) for i, w in enumerate(cw)}

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=cbs,
            class_weight=class_weight,
            verbose=1,
        )

    else:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=cbs,
            verbose=1,
        )

    return history


# Fine-Tunning
def set_finetunning(model):
    model.load_weights("best_head.keras")

    unfreeze_from = int(len(base_model.layers) * 0.6)  # ajuste 0.6–0.75
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
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def fit_finetunning(
    model, train_generator, val_generator, epochs: int, name: str, fold: str
):
    cbs_ft = [
        callbacks.ModelCheckpoint(
            f"best_{name}_{fold}_finetune.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        callbacks.CSVLogger(f"finetune_log{name}_{fold}.csv", append=False),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="max",
            patience=16,
            min_delta=0,
            restore_best_weights=True,
        ),
        callbacks.TensorBoard(log_dir="tb_logs_finetune", update_freq="epoch"),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    history_ft = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=cbs_ft,
        verbose=1,
    )
    return history_ft


def set_predict(model_name, fold: str):
    model_path = f"best_{model_name}_{fold}_finetune.keras"
    print(f"[INFO] Carregando modelo para predição: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


"""
def fit_predict(model, val_generator, model_name):
    print(f"[INFO] Avaliando modelo {model_name} na validação...")
    y_true = val_generator.classes
    y_prob = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=val_generator.class_indices.keys(),
        output_dict=True,
    )
    print(f"Val Accuracy: {acc:.4f}")
    print("Classification Report:", report_dict)

    with open(f"report_{model_name}_finetune.txt", "w") as f:
        f.write(f"Val Accuracy: {acc:.4f}\n")
        f.write("Classification Report:\n")
        for label, metrics in report_dict.items():
            if label in val_generator.class_indices.keys():
                f.write(f"{label}:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")
    return acc, report_dict
"""


def fit_predict(model, val_generator, model_name, output_dir=".", fold_name=None):
    print(f"[INFO] Avaliando modelo {model_name} na validação...")

    os.makedirs(output_dir, exist_ok=True)

    y_true = val_generator.classes
    y_prob = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)

    idx_to_class = {v: k for k, v in val_generator.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )

    print(f"Val Accuracy: {acc:.4f}")
    print("Classification Report:", report_dict)

    # ===== relatório agregado =====
    report_suffix = f"_{fold_name}" if fold_name else ""
    report_path = os.path.join(
        output_dir, f"report_{model_name}{report_suffix}_finetune.txt"
    )

    with open(report_path, "w") as f:
        f.write(f"Val Accuracy: {acc:.4f}\n")
        f.write("Classification Report:\n")
        for label, metrics in report_dict.items():
            if label in class_names:
                f.write(f"{label}:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")

    # ===== saída por instância =====
    df_pred = pd.DataFrame(
        {
            "filename": val_generator.filenames,
            "y_true_idx": y_true,
            "y_pred_idx": y_pred,
            "y_true": [idx_to_class[i] for i in y_true],
            "y_pred": [idx_to_class[i] for i in y_pred],
            "correct": (y_true == y_pred).astype(int),
            "pred_confidence": np.max(y_prob, axis=1),
        }
    )

    # adiciona probabilidade de cada classe
    for i, class_name in enumerate(class_names):
        df_pred[f"prob_{class_name}"] = y_prob[:, i]

    pred_suffix = f"_{fold_name}" if fold_name else ""
    pred_path = os.path.join(
        output_dir, f"predictions_{model_name}{pred_suffix}_finetune.csv"
    )
    df_pred.to_csv(pred_path, index=False)

    print(f"[INFO] Predições por instância salvas em: {pred_path}")

    return acc, report_dict, df_pred


# Ajusta img_size pelo modelo
def auto_img_size(model_name):
    FALLBACK_IMG_SIZE = 224
    s = DEFAULT_IMG_SIZE.get(model_name.lower(), FALLBACK_IMG_SIZE)
    print(f"[INFO] Ajustando IMG_SIZE para {s}x{s} para {model_name}")
    return (s, s)


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    MODEL_NAME = str(args.model)
    FOLD_NAME = str(args.fold)
    BATCH_SIZE = int(args.batch_size)
    TRAINING_EPOCHS = int(args.training_epochs)
    FINETUNNING_EPOCHS = int(args.finetunning_epochs)

    print(f"[INFO] Modelo selecionado: {MODEL_NAME}")
    IMG_SIZE = auto_img_size(MODEL_NAME)
    print(f"[INFO] IMG_SIZE De {MODEL_NAME} Ajustado para: {IMG_SIZE}")

    # Training
    df_train, df_val, df_test, classes = load_split_data(
        label_dir=LABEL_DIR, fold=FOLD_NAME
    )
    preprocess_fn, base_model = get_backbone(name=MODEL_NAME, img_size=IMG_SIZE)
    train_generator, val_generator, test_generator = set_generators(
        preprocess_fn,
        df_train,
        df_val,
        df_test,
        IMG_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        debbug=True,
    )
    model = set_transferlearning(base_model)
    history = fit_model(
        model,
        train_generator,
        val_generator,
        name=MODEL_NAME,
        class_weight=False,
        epochs=TRAINING_EPOCHS,
        fold=FOLD_NAME,
    )
    model = set_finetunning(model)
    history_ft = fit_finetunning(
        model,
        train_generator,
        val_generator,
        epochs=FINETUNNING_EPOCHS,
        name=MODEL_NAME,
        fold=FOLD_NAME,
    )

    model = set_predict(MODEL_NAME, FOLD_NAME)
    # acc, report_dict = fit_predict(model, test_generator, MODEL_NAME)
    acc, report_dict, df_pred = fit_predict(
        model,
        val_generator,
        MODEL_NAME,
        output_dir=OUTPUT_DIR,
        fold_name=FOLD_NAME,
    )

    # Plotting
    print("iniciando Plotting Results")
    plots = utils()

    plots.plot_training_curves(
        history,
        out_png=f"curvas_{MODEL_NAME}_{FOLD_NAME}.png",
        title=f"{MODEL_NAME} - DeepWeeds (Head) - {FOLD_NAME}",
    )
    plots.save_history_csv(history, out_csv=f"historico_{MODEL_NAME}_{FOLD_NAME}.csv")
    plots.plot_confusion_and_report(
        model,
        val_generator,
        train_generator.class_indices,
        cm_png=f"cm_{MODEL_NAME}_{FOLD_NAME}.png",
        report_txt=f"report_{MODEL_NAME}_{FOLD_NAME}.txt",
        normalize=True,
    )
    plots.plot_training_curves(
        history_ft,
        out_png=f"curvas_{MODEL_NAME}_{FOLD_NAME}_finetunning.png",
        title=f"{MODEL_NAME} Tuned- DeepWeeds",
    )
    plots.save_history_csv(
        history_ft, out_csv=f"historico_{MODEL_NAME}_{FOLD_NAME}_finetunning.csv"
    )
    joined_dict, df_hist = plots.concat_histories(history, history_ft)
    df_hist.to_csv(f"historico_{MODEL_NAME}_{FOLD_NAME}_joined.csv", index=False)
    split_epoch = len(history.history.get("accuracy", []))
    plots.plot_history_joined(
        joined_dict,
        out_png=f"{MODEL_NAME}_{FOLD_NAME}_head_plus_ft.png",
        title=f"{MODEL_NAME} - DeepWeeds (Head + FT)",
        split_epoch=split_epoch,
    )
    try:
        plt.close("all")
    except:
        pass
