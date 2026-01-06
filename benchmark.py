from __future__ import annotations
import os, time, argparse, json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling, Normalization

AUTOTUNE = tf.data.AUTOTUNE


# Prepara GPU (opcional)
def _enable_mem_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# Registry de preprocess por backbone (lazy init)
_REGISTRY: Dict[str, Any] = {}


def _init_registry():
    from tensorflow.keras.applications import (
        mobilenet_v2,
        mobilenet_v3,
        efficientnet_v2,
        resnet,
        resnet_v2,
        inception_v3,
        nasnet,
    )

    _REGISTRY.update(
        {
            "mobilenetv2": mobilenet_v2.preprocess_input,
            "mobilenetv3small": mobilenet_v3.preprocess_input,
            "mobilenetv3large": mobilenet_v3.preprocess_input,
            "efficientnetv2b0": efficientnet_v2.preprocess_input,
            "efficientnetv2b1": efficientnet_v2.preprocess_input,
            "efficientnetv2b2": efficientnet_v2.preprocess_input,
            "efficientnetv2b3": efficientnet_v2.preprocess_input,
            "resnet50": resnet.preprocess_input,  # V1
            "resnet101v2": resnet_v2.preprocess_input,  # V2
            "inceptionv3": inception_v3.preprocess_input,
            "nasnetmobile": nasnet.preprocess_input,
        }
    )


def _has_baked_preprocess(model: tf.keras.Model) -> bool:
    for L in model.layers[:5]:
        if (
            isinstance(L, (Rescaling, Normalization))
            or L.__class__.__name__ == "Lambda"
        ):
            return True
    return False


def resolve_preprocess(
    model: tf.keras.Model, model_key: Optional[str], model_path: str
) -> Optional[Any]:
    """
    Retorna a função de preprocess adequada OU None se já estiver embutida no grafo.
    Ordem:
      1) Se o grafo tiver preprocess (Rescaling/Normalization/Lambda) -> None
      2) Se model_key foi passado -> registry
      3) Se existir <.keras>.meta.json com {"model_key": "..."} -> registry
      4) Erro
    """
    if _has_baked_preprocess(model):
        return None

    if model_key:
        key = model_key.lower()
        if not _REGISTRY:
            _init_registry()
        if key not in _REGISTRY:
            raise ValueError(f"model_key não suportado: {model_key}")
        return _REGISTRY[key]

    sidecar = Path(model_path).with_suffix(".keras.meta.json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            key = str(meta.get("model_key", "")).lower()
            if key:
                if not _REGISTRY:
                    _init_registry()
                if key not in _REGISTRY:
                    raise ValueError(f"model_key do sidecar não suportado: {key}")
                return _REGISTRY[key]
        except Exception:
            pass

    raise ValueError(
        "Não foi possível resolver o preprocess. "
        "Passe --model-key (ex.: mobilenetv2, inceptionv3, efficientnetv2b0) "
        "ou salve o modelo com o preprocess embutido (Rescaling/Normalization/Lambda) "
        "ou crie um sidecar <modelo>.keras.meta.json com {'model_key': '...'}."
    )


def load_paths_and_labels(
    labels_csv: Optional[str], filelist: Optional[str], images_root: Optional[str]
) -> Tuple[List[str], Optional[List[int]]]:
    """
    Retorna (filepaths, labels_int_ou_None).
    - Se labels_csv: espera colunas 'Filename' e 'Label'. Junta com images_root se fornecido.
    - Se filelist: arquivo texto com um caminho por linha (labels = zeros).
    """
    if labels_csv:
        df = pd.read_csv(labels_csv)
        if "Filename" not in df.columns or "Label" not in df.columns:
            raise ValueError("labels_csv deve ter colunas 'Filename' e 'Label'.")
        fn = df["Filename"].astype(str).tolist()
        if images_root:
            fn = [os.path.join(images_root, x) for x in fn]
        lbl = df["Label"].astype(int).tolist()
        return fn, lbl

    if filelist:
        paths = [
            ln.strip()
            for ln in Path(filelist).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if images_root:
            paths = [os.path.join(images_root, p) for p in paths]
        return paths, None

    raise ValueError("Informe --labels-csv ou --filelist.")


# Dataset de avaliação (sem augmentation)
def _decode_resize_preprocess(path, label, img_size, preprocess_fn):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, img_size, antialias=True)
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img) if preprocess_fn is not None else (img / 255.0)
    return img, label


def build_eval_dataset(
    filepaths: List[str],
    labels: Optional[List[int]],
    img_size: Tuple[int, int],
    preprocess_fn: Optional[Any],
    batch_size: int = 1,
    shuffle: bool = False,
    cache: bool = True,
    deterministic: bool = True,
) -> tf.data.Dataset:
    fp = tf.convert_to_tensor(filepaths, dtype=tf.string)
    if labels is None:
        labels = tf.zeros((tf.shape(fp)[0],), dtype=tf.int32)
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((fp, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=tf.shape(fp)[0], reshuffle_each_iteration=False)

    map_fn = lambda p, y: _decode_resize_preprocess(p, y, img_size, preprocess_fn)
    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic)

    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


# Benchmarks de latência/FPS
def benchmark_dataset(
    model: tf.keras.Model, ds: tf.data.Dataset, steps: Optional[int]
) -> Dict[str, Any]:
    it = iter(ds)
    times = []
    n_steps = 0
    last_batch = None

    while True:
        if steps is not None and n_steps >= steps:
            break
        try:
            batch = next(it)
        except StopIteration:
            break

        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
        last_batch = xb

        t0 = time.perf_counter()
        y = model(xb, training=False)
        _ = (y.logits if hasattr(y, "logits") else y).numpy()  # sincroniza
        times.append((time.perf_counter() - t0) * 1e3)
        n_steps += 1

    if not times:
        raise RuntimeError("Dataset não produziu batches suficientes para benchmark.")

    arr = np.asarray(times, dtype=np.float64)
    p50, p95, mean = np.median(arr), np.percentile(arr, 95), arr.mean()

    try:
        bs = int(last_batch.shape[0])
    except Exception:
        bs = 1

    return {
        "steps": int(n_steps),
        "batch": bs,
        "lat_ms_p50": round(float(p50), 3),
        "lat_ms_p95": round(float(p95), 3),
        "lat_ms_mean": round(float(mean), 3),
        "fps_p50": round(1000.0 / float(p50) * bs, 2),
        "fps_p95": round(1000.0 / float(p95) * bs, 2),
        "fps_mean": round(1000.0 / float(mean) * bs, 2),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de inferência no dataset real (sem augmentation)."
    )
    parser.add_argument(
        "--model-path", required=True, help="Caminho do .keras (ou SavedModel)."
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Chave do backbone para resolver preprocess (ex.: mobilenetv2, efficientnetv2b0, inceptionv3). "
        "Opcional se o modelo tiver preprocess embutido ou sidecar .meta.json.",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="CSV com colunas Filename e Label. Alternativa a --filelist.",
    )
    parser.add_argument(
        "--filelist",
        default=None,
        help="Arquivo texto com um caminho por linha (labels assumidos = zeros). Alternativa a --labels-csv.",
    )
    parser.add_argument(
        "--images-root",
        default=None,
        help="Prefixo para juntar com paths do CSV/filelist, se necessário.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch para o dataset (use 1 p/ latência).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Número de lotes a medir (None = todo dataset).",
    )
    parser.add_argument(
        "--out-csv", default="inference_bench.csv", help="Arquivo CSV de saída."
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Embaralhar ordem antes de medir."
    )
    args = parser.parse_args()

    _enable_mem_growth()

    # Carrega modelo
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Caminho não encontrado: {args.model_path}")
    model = load_model(args.model_path, compile=False)

    # Resolve input size do modelo
    ish = model.input_shape
    if isinstance(ish, (list, tuple)) and isinstance(ish[0], (list, tuple)):
        ish = ish[0]
    H, W = int(ish[1]), int(ish[2])

    # Resolve preprocess
    preprocess_fn = resolve_preprocess(model, args.model_key, args.model_path)

    # Carrega caminhos e rótulos
    filepaths, labels = load_paths_and_labels(
        args.labels_csv, args.filelist, args.images_root
    )
    if len(filepaths) == 0:
        raise ValueError("Nenhum caminho de imagem encontrado.")

    # Monta dataset real (sem augmentation)
    ds = build_eval_dataset(
        filepaths=filepaths,
        labels=labels,
        img_size=(H, W),
        preprocess_fn=preprocess_fn,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        cache=True,
        deterministic=True,
    )

    # Benchmark
    metrics = benchmark_dataset(model, ds, steps=None if args.steps < 0 else args.steps)

    # Resumo + CSV
    row = {
        "file": Path(args.model_path).name,
        "path": str(Path(args.model_path).resolve()),
        "input_h": H,
        "input_w": W,
        "batch": metrics["batch"],
        "steps": metrics["steps"],
        "lat_ms_p50": metrics["lat_ms_p50"],
        "lat_ms_p95": metrics["lat_ms_p95"],
        "lat_ms_mean": metrics["lat_ms_mean"],
        "fps_p50": metrics["fps_p50"],
        "fps_p95": metrics["fps_p95"],
        "fps_mean": metrics["fps_mean"],
        "model_key": (args.model_key or ""),
        "preprocess_embutido": int(preprocess_fn is None),
    }

    print("[RESULT]")
    for k, v in row.items():
        print(f"{k}: {v}")

    pd.DataFrame([row]).to_csv(args.out_csv, index=False)
    print(f"[OK] CSV salvo em: {args.out_csv}")


if __name__ == "__main__":
    main()
