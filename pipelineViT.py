import os
import tensorflow as tf
import pandas as pd
import numpy as np
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
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Models and Backbone's
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.applications import (
    MobileNetV3Large,
    MobileNetV3Small,
    MobileNetV2,
)

from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
)
from tensorflow.keras.applications import efficientnet_v2

from transformers import AutoProcessor, TFMobileViTForImageClassification
from tf_keras import callbacks as kcallbacks
from tf_keras import optimizers as koptimizers
from tf_keras import losses as klosses
from tf_keras import metrics as kmetrics
from tf_keras import layers as klayers

# Utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

try:
    from tf_keras import backend as Ktfk

    Ktfk.set_image_data_format("channels_last")
except Exception:
    pass

try:
    from tensorflow.keras import backend as Ktf

    Ktf.set_image_data_format("channels_last")
except Exception:
    pass
# Mixed Precision - Entra FP16
## Tem que garantir que a saída do modelo é float32
# mixed_precision.set_global_policy("mixed_float16")
print(mixed_precision.global_policy())
print("Keras version:", keras.__version__)
K.set_image_data_format("channels_last")
print("TF:", tf.__version__)
print("GPUs visíveis:", tf.config.list_physical_devices("GPU"))


# Global - Paths
DIRS = get_project_dirs()
IMG_DIR = str(DIRS["images"]) + "/"
RESULTS_DIR = str(DIRS["results"]) + "/"
MODELS_DIR = str(DIRS["models"]) + "/"
LABEL_DIR = str(DIRS["labels"]) + "/"

# Global - Parâmetros
MODEL_NAME = "mobilenetv2"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DEFAULT_IMG_SIZE = {
    "mobilenetv2": 224,
    "mobilenetv3large": 224,
    "mobilenetv3small": 224,
    "efficientnetv2b0": 224,
    "efficientnetv2b1": 240,
    "efficientnetv2b2": 260,
    "efficientnetv2b3": 300,
}
AUTOTUNE = tf.data.AUTOTUNE
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TRAINING_EPOCHS = 200
FINETUNNING_EPOCHS = 100
TEST_SIZE = 0.2

# Global
SUPPORTED_MODELS = [
    "mobilenetv2",
    "mobilenetv3large",
    "mobilenetv3small",
    "efficientnetv2b0",
    "efficientnetv2b1",
    "efficientnetv2b2",
    "efficientnetv2b3",
    "mobilevit-small",
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
    return parser


def load_split_data(label_dir: str, test_size: float):
    # Load Labels df
    try:
        data = pd.read_csv(LABEL_DIR + "labels.csv")
        data["Label"] = data["Label"].astype(str)
        data["Filename"] = data["Filename"].apply(lambda x: os.path.join(IMG_DIR, x))

    except FileNotFoundError as e:
        print(
            f"[ERRO] labels.csv não encontrado: {csv_path} "
            f"(verifique se 'label_dir' está correto)."
        )
        raise

    # Train test split
    df_train, df_val = train_test_split(
        data, test_size=test_size, stratify=data["Label"], random_state=75
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # 2) Classes consistentes (ordenadas)
    classes = sorted(data["Label"].unique().tolist())

    # Debbug
    print("[DEBUG] Filename resolvido:", data["Filename"].iloc[0])
    print(
        f"[DEBUG] Tamanho total: {len(data)} | Treino: {len(df_train)} | Validação: {len(df_val)} "
        f"({len(df_val)/len(data)*100:.1f}% val)"
    )

    return df_train, df_val, classes


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

    else:
        raise ValueError(
            "--model deve ser: mobilenetv2, mobilenetv3large, mobilenetv3small, "
            "efficientnetv2b0/b1/b2/b3"
        )

    return preprocess_fn, base_model


# 4) Dois datagens - augment só no treino
def set_generators(preprocess_fn, df_train, df_val, img_size, debbug: bool):

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
    x = Dense(256, activation="relu")(x)
    outputs = Dense(len(classes), activation="sigmoid", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def fit_model(
    model, train_generator, val_generator, name: str, class_weight: bool, epochs: int
):

    cbs = [
        callbacks.ModelCheckpoint(
            "best_head.keras", monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=16, min_lr=1e-6
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=32, restore_best_weights=True, mode="max"
        ),
        callbacks.CSVLogger(f"treino_log{name}.csv", append=False),
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

    unfreeze_from = int(
        len(base_model.layers) * 0.6
    )  # ajuste 0.6–0.75 conforme VRAM/estabilidade
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


def fit_finetunning(model, train_generator, val_generator, epochs: int, name: str):
    cbs_ft = [
        callbacks.ModelCheckpoint(
            f"best_{name}_finetune.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True
        ),
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


# Ajusta img_size pelo modelo
def auto_img_size(model_name):
    FALLBACK_IMG_SIZE = 224
    s = DEFAULT_IMG_SIZE.get(model_name.lower(), FALLBACK_IMG_SIZE)
    print(f"[INFO] Ajustando IMG_SIZE para {s}x{s} para {model_name}")
    return (s, s)


#####################VIT PIPELINE #########################
def build_mobilevit_datasets_with_aug(
    df_train,
    df_val,
    classes,
    model_id="apple/mobilevit-small",
    batch_size=32,
    use_aug=True,
):

    processor = AutoProcessor.from_pretrained(model_id)
    image_size = processor.size.get("shortest_edge", processor.size.get("height", 256))

    def _get_image_size(proc, default=256):
        # tenta 'shortest_edge' (v2), depois 'height' (v1), depois inteiro
        size = getattr(proc, "size", None)
        if isinstance(size, dict):
            return size.get("shortest_edge", size.get("height", default))
        if isinstance(size, int):
            return size
        return default

    def _get_mean_std(proc):
        mean = getattr(proc, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(proc, "image_std", [0.5, 0.5, 0.5])
        return mean, std

    image_size = _get_image_size(processor, default=256)
    mean_vals, std_vals = _get_mean_std(processor)

    mean = tf.constant(mean_vals, dtype=tf.float32)  # shape (3,)
    std = tf.constant(std_vals, dtype=tf.float32)  # shape (3,)
    # mean = tf.constant(processor.image_mean, dtype=tf.float32)  # (3,)
    # std = tf.constant(processor.image_std, dtype=tf.float32)  # (3,)

    aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=10 / 180),  # ~±10°
            layers.RandomZoom(
                height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)
            ),  # ±10%
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # ±5%
        ],
        name="aug_mobilevit",
    )

    label2id = {c: i for i, c in enumerate(classes)}

    def _load_decode_rgb(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)  # uint8
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        return img

    def _preprocess(img, training=False):
        # resize primeiro (como no generator)
        img = tf.image.resize(img, (image_size, image_size), method="bilinear")
        # aug só no treino
        if training:
            img = aug(img, training=True)
        # normaliza com mean/std do processor (sem rescale adicional)
        img = (img - mean) / std
        return img

    def _make(paths, labels, training):
        y = tf.convert_to_tensor([label2id[str(l)] for l in labels], dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        if training:
            ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

        def _map(p, lab):
            img = _load_decode_rgb(p)
            img = _preprocess(img, training=training and use_aug)
            return img, lab

        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make(
        df_train["Filename"].values, df_train["Label"].values, training=True
    )
    val_ds = _make(df_val["Filename"].values, df_val["Label"].values, training=False)
    return train_ds, val_ds, image_size


def build_mobilevit_model(classes, model_id="apple/mobilevit-small"):
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}
    model = TFMobileViTForImageClassification.from_pretrained(
        model_id,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # recria a head para 9 classes
    )
    model.compile(
        optimizer=koptimizers.Adam(1e-4),
        loss=klosses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[kmetrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    cbs = [
        kcallbacks.ModelCheckpoint(
            "mvits_best_head.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        kcallbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max"
        ),
        kcallbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        kcallbacks.CSVLogger("mvits_history_head.csv", append=False),
    ]

    return model


###########################################################


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    MODEL_NAME = str(args.model)

    print(f"[INFO] Modelo selecionado: {MODEL_NAME}")
    IMG_SIZE = auto_img_size(MODEL_NAME)
    print(f"[INFO] IMG_SIZE De {MODEL_NAME} Ajustado para: {IMG_SIZE}")

    # Training
    df_train, df_val, classes = load_split_data(
        label_dir=LABEL_DIR, test_size=TEST_SIZE
    )

    ############MOBILEVIT TRAINING AND FINETUNNING################
    if MODEL_NAME.startswith("mobilevit"):
        train_ds, val_ds, mvit_size = build_mobilevit_datasets_with_aug(
            df_train,
            df_val,
            classes,
            model_id=f"apple/{MODEL_NAME}",  # ex.: apple/mobilevit-small
            batch_size=BATCH_SIZE,
            use_aug=True,
        )
        model = build_mobilevit_model(classes, model_id=f"apple/{MODEL_NAME}")
        cbs = [
            kcallbacks.ModelCheckpoint(
                "mvits_best_head.keras",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
            ),
            kcallbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                mode="max",
            ),
            kcallbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
            kcallbacks.CSVLogger("mvits_history_head.csv", append=False),
        ]

        # treina head (backbone congelado)
        for xb, yb in train_ds.take(1):
            print("[DBG] batch", xb.shape)  # deve ser (B, H, W, 3)

        if hasattr(model, "mobilevit"):
            model.mobilevit.trainable = False
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=TRAINING_EPOCHS,
            callbacks=cbs,
            verbose=1,
        )

        # fine-tuning leve (descongelar ~35% final, exceto BN), LR menor
        if hasattr(model, "mobilevit"):
            bb = model.mobilevit
            N = len(bb.layers)
            cut = int(N * 0.65)
            for i, L in enumerate(bb.layers):
                if i < cut or isinstance(L, klayers.BatchNormalization):
                    L.trainable = False
                else:
                    L.trainable = True

        model.compile(
            optimizer=koptimizers.Adam(1e-5),
            loss=klosses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[kmetrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINETUNNING_EPOCHS,
            callbacks=cbs,
            verbose=1,
        )

    # … (seus plots/saves iguais aos outros modelos)

    else:
        preprocess_fn, base_model = get_backbone(name=MODEL_NAME, img_size=IMG_SIZE)
        train_generator, val_generator = set_generators(
            preprocess_fn, df_train, df_val, IMG_SIZE, debbug=True
        )
        model = set_transferlearning(base_model)
        history = fit_model(
            model,
            train_generator,
            val_generator,
            name=MODEL_NAME,
            class_weight=False,
            epochs=TRAINING_EPOCHS,
        )
        model = set_finetunning(model)
        history_ft = fit_finetunning(
            model,
            train_generator,
            val_generator,
            epochs=FINETUNNING_EPOCHS,
            name=MODEL_NAME,
        )

    # Plotting
    print("iniciando Plotting Results")
    plots = utils()

    plots.plot_training_curves(
        history, out_png=f"curvas_{MODEL_NAME}.png", title=f"{MODEL_NAME} - DeepWeeds"
    )
    plots.save_history_csv(history, out_csv=f"historico_{MODEL_NAME}.csv")
    plots.plot_confusion_and_report(
        model,
        val_generator,
        train_generator.class_indices,
        cm_png=f"cm_{MODEL_NAME}.png",
        report_txt=f"report_{MODEL_NAME}.txt",
        normalize=True,
    )
    plots.plot_training_curves(
        history_ft,
        out_png=f"curvas_{MODEL_NAME}_finetunning.png",
        title=f"{MODEL_NAME} Tuned- DeepWeeds",
    )
    plots.save_history_csv(
        history_ft, out_csv=f"historico_{MODEL_NAME}_finetunning.csv"
    )
    joined_dict, df_hist = plots.concat_histories(history, history_ft)
    df_hist.to_csv(f"historico_{MODEL_NAME}_joined.csv", index=False)
    split_epoch = len(history.history.get("accuracy", []))
    plots.plot_history_joined(
        joined_dict,
        out_png=f"{MODEL_NAME}_head_plus_ft.png",
        title=f"{MODEL_NAME} - DeepWeeds (Head + FT)",
        split_epoch=split_epoch,
    )
