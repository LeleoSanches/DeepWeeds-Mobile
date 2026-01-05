import pandas as pd
import numpy as np
import time
import re
import os
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import json

# Global Paths
MODELS_PATH = "models/"
RESULTS_PATH = "results/"
REPORTS_PATH = "results/reports/"


class report:
    @staticmethod
    def list_models(models_dir: str):

        models_path = Path(models_dir)
        model_paths = sorted(models_path.glob("*.keras"))

        if not model_paths:
            raise FileNotFoundError(
                f"Os modelos .keras não foram encontrados, verifique se os modelos estão em {models_path.resolve()}"
            )

        model_files = [p.name for p in model_paths]
        stems = [p.stem for p in model_paths]

        # normaliza nomes: remove prefixo 'best_' e sufixo '_finetune'
        model_names = []
        for s in stems:
            s = re.sub(r"^best_", "", s)  # remove 'best_' no início
            s = re.sub(r"_finetune$", "", s)  # remove '_finetune' no fim
            model_names.append(s)

        return model_names, model_files, model_paths

    @staticmethod
    def list_results(results_dir=REPORTS_PATH):
        results_path = Path(results_dir).resolve()
        results_files = sorted(results_path.glob("*.txt"))

        if not results_files:
            raise FileNotFoundError(
                f"Os arquivos de resultados .txt não foram encontrados, verifique se os resultados estão em {REPORTS_PATH}"
            )
        return list(results_path.glob("*.txt"))


class process_results:
    @staticmethod
    def process_txt(result_files):
        todos_dados = []

        for path in result_files:
            model_name = path.stem.replace("report_", "")
            linhas = path.read_text().splitlines()

            for linha in linhas:
                partes = linha.split()
                if len(partes) == 5 and partes[0].isdigit():
                    classe, precision, recall, f1, support = partes
                    todos_dados.append(
                        {
                            "model": model_name,
                            "class": classe,
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1),
                            "support": int(support),
                        }
                    )

        return todos_dados

    @staticmethod
    def process_global_metrics(result_files):
        globais = []
        for path in result_files:
            model_name = path.stem.replace("results_", "")
            linhas = path.read_text().splitlines()

            accuracy = None
            macro = None  # (precision, recall, f1, support)
            weighted = None  # (precision, recall, f1, support)

            for linha in linhas:
                partes = linha.split()

                if len(partes) == 3 and partes[0] == "accuracy":
                    accuracy = float(partes[1])

                if len(partes) == 6 and partes[0] == "macro" and partes[1] == "avg":
                    macro = (
                        float(partes[2]),
                        float(partes[3]),
                        float(partes[4]),
                        int(partes[5]),
                    )

                if len(partes) == 6 and partes[0] == "weighted" and partes[1] == "avg":
                    weighted = (
                        float(partes[2]),
                        float(partes[3]),
                        float(partes[4]),
                        int(partes[5]),
                    )

            globais.append(
                {
                    "model": model_name,
                    "accuracy": accuracy,
                    "macro_precision": macro[0],
                    "macro_recall": macro[1],
                    "macro_f1": macro[2],
                    "weighted_precision": weighted[0],
                    "weighted_recall": weighted[1],
                    "weighted_f1": weighted[2],
                    "support_total": weighted[3],  # total de amostras
                }
            )

        return globais


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


class KerasModelInspector:
    """
    Inspeciona um modelo salvo em .keras (ou SavedModel via load_model):
      - Carrega o modelo (compile=False)
      - Conta parâmetros (treináveis / não-treináveis / total)
      - Estima memória dos pesos (MB) via NumPy (.numpy().nbytes)
      - Breakdown por dtype real
      - Calcula FLOPs (forward, batch=1) e GMAC (conv. 1 MAC = 2 FLOPs)
      - Exporta resumo em CSV
    Observação: métodos de memória convertem pesos para NumPy na CPU.
    """

    def __init__(
        self,
        model_path: str,
        mac_is_2_flops: bool = True,
        input_shape_override: tuple | None = None,
    ):
        """
        mac_is_2_flops: se True, GMAC = GFLOPs/2 (convenção comum em CV).
        input_shape_override: tuple (H, W, 3). Se None, infere do modelo salvo.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Caminho não encontrado: {model_path}")

        self.model_path = model_path
        self.model = None
        self.mac_is_2_flops = mac_is_2_flops
        self.input_shape_override = input_shape_override  # (H, W, 3)

    # -----------------------------
    # Ciclo de vida
    # -----------------------------
    def load(self):
        # compile=False para carregar sem dependências de optimizer/loss
        self.model = load_model(self.model_path, compile=False)
        return self

    def _ensure_loaded(self):
        if self.model is None:
            raise RuntimeError("Modelo não carregado. Chame .load() primeiro.")

    def clear(self):
        """Libera o modelo e limpa a sessão Keras/TF."""
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        tf.keras.backend.clear_session()

    # -----------------------------
    # Helpers internos
    # -----------------------------
    def _infer_hw(self):
        """Infere (H, W) da entrada do modelo ou usa override."""
        if self.input_shape_override is not None:
            H, W, _ = self.input_shape_override
            return int(H), int(W)

        ish = self.model.input_shape
        # Modelos com múltiplas entradas -> usa a primeira
        if isinstance(ish, (list, tuple)) and isinstance(ish[0], (list, tuple)):
            ish = ish[0]
        H, W = int(ish[1]), int(ish[2])
        return H, W

    # -----------------------------
    # Contagens de parâmetros
    # -----------------------------
    def count_params(self):
        """Retorna (trainable, non_trainable, total) como inteiros."""
        self._ensure_loaded()
        trainable = int(
            np.sum([np.prod(v.shape) for v in self.model.trainable_variables])
        )
        non_trainable = int(
            np.sum([np.prod(v.shape) for v in self.model.non_trainable_variables])
        )
        total = trainable + non_trainable
        return trainable, non_trainable, total

    # -----------------------------
    # Memória via NumPy (robusto)
    # -----------------------------
    def weights_memory_mb(self):
        self._ensure_loaded()
        total_bytes = 0
        for w in self.model.weights:
            total_bytes += w.numpy().nbytes
        return round(total_bytes / (1024**2), 2)

    def dtype_breakdown(self):
        self._ensure_loaded()
        agg = {}
        for w in self.model.weights:
            arr = w.numpy()
            key = str(arr.dtype.name)  # ex.: 'float32', 'float16'
            if key not in agg:
                agg[key] = {"params": 0, "bytes": 0}
            agg[key]["params"] += arr.size
            agg[key]["bytes"] += arr.nbytes
        return {
            k: {"params": v["params"], "MB": round(v["bytes"] / (1024**2), 2)}
            for k, v in agg.items()
        }

    # -----------------------------
    # FLOPs / GMAC (forward, batch=1)
    # -----------------------------
    def _count_flops_forward(self, H: int, W: int) -> int:
        """
        Conta FLOPs do forward (batch=1, float32) via profiler v1.
        Retorna 0 se não conseguir perfilar.
        """

        @tf.function
        def _f(x):
            return self.model(x, training=False)

        spec = tf.TensorSpec([1, H, W, 3], tf.float32)
        conc = _f.get_concrete_function(spec)
        frozen = convert_variables_to_constants_v2(conc)
        graph_def = frozen.graph.as_graph_def()

        with tf.compat.v1.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            prof = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, options=opts
            )

        return 0 if prof is None else int(prof.total_float_ops)

    def compute_flops_gmac(self):
        """
        Retorna dict com FLOPs, GFLOPs e GMAC (forward, batch=1) no tamanho de entrada do modelo.
        """
        self._ensure_loaded()
        H, W = self._infer_hw()
        flops = self._count_flops_forward(H, W)
        gflops = float(flops) / 1e9
        gmac = gflops / 2.0 if self.mac_is_2_flops else gflops
        return {
            "input_h": H,
            "input_w": W,
            "FLOPs": flops,
            "GFLOPs": gflops,
            "GMAC": gmac,
        }

    @tf.function(jit_compile=False)
    def _infer_fn(self, x):
        return self.model(x, training=False)

    def _dummy_input(self, H, W, batch=1, dtype=tf.float32):
        return tf.random.uniform([batch, H, W, 3], 0.0, 1.0, dtype=dtype)

    def _benchmark_dummy(self, runs: int, warmup: int, batch: int):
        """
        Mede latência em ms usando dummy input (batch fixo), sincronizando GPU com .numpy().
        Retorna dict com p50/p95/média e FPS.
        """
        H, W = self._infer_hw()
        x = self._dummy_input(H, W, batch=batch)

        # warm-up
        for _ in range(warmup):
            y = self._infer_fn(x)
            _ = y.numpy()  # sincroniza GPU

        # medições
        times_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            y = self._infer_fn(x)
            _ = y.numpy()  # sincroniza GPU
            times_ms.append((time.perf_counter() - t0) * 1e3)

        arr = np.asarray(times_ms, dtype=np.float64)
        p50 = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        mean = float(arr.mean())

        return {
            "bench_batch": int(batch),
            "bench_runs": int(runs),
            "bench_warmup": int(warmup),
            "lat_ms_p50": round(p50, 3),
            "lat_ms_p95": round(p95, 3),
            "lat_ms_mean": round(mean, 3),
            "fps_p50": round((1000.0 / p50) * batch, 2),
            "fps_p95": round((1000.0 / p95) * batch, 2),
            "fps_mean": round((1000.0 / mean) * batch, 2),
        }

    # -----------------------------
    # Saída consolidada (inclui benchmark dummy)
    # -----------------------------
    def summary_dict(
        self, bench_runs: int = 100, bench_warmup: int = 20, bench_batch: int = 1
    ):
        """
        Retorna dicionário consolidado com:
          - params/memória/dtypes
          - FLOPs/GFLOPs/GMAC
          - latência/FPS (dummy input) com configurações bench_* fornecidas
        """
        self._ensure_loaded()

        tr, ntr, tot = self.count_params()
        mem_mb = self.weights_memory_mb()
        dtypes = self.dtype_breakdown()
        comp = self.compute_flops_gmac()
        bench = self._benchmark_dummy(
            runs=bench_runs, warmup=bench_warmup, batch=bench_batch
        )

        return {
            "file": os.path.basename(self.model_path),
            "path": os.path.abspath(self.model_path),
            "trainable": tr,
            "non_trainable": ntr,
            "total": tot,
            "memory_MB": mem_mb,
            "dtype_breakdown": dtypes,
            "input_h": comp["input_h"],
            "input_w": comp["input_w"],
            "FLOPs": comp["FLOPs"],
            "GFLOPs": round(comp["GFLOPs"], 6),
            "GMAC": round(comp["GMAC"], 6),
            "bench_batch": bench["bench_batch"],
            "bench_runs": bench["bench_runs"],
            "bench_warmup": bench["bench_warmup"],
            "lat_ms_p50": bench["lat_ms_p50"],
            "lat_ms_p95": bench["lat_ms_p95"],
            "lat_ms_mean": bench["lat_ms_mean"],
            "fps_p50": bench["fps_p50"],
            "fps_p95": bench["fps_p95"],
            "fps_mean": bench["fps_mean"],
            "inputs": [tuple(t.shape) for t in getattr(self.model, "inputs", [])]
            or None,
            "outputs": [tuple(t.shape) for t in getattr(self.model, "outputs", [])]
            or None,
        }

    def to_csv(
        self,
        out_csv: str = "model_params_summary.csv",
        bench_runs: int = 100,
        bench_warmup: int = 20,
        bench_batch: int = 1,
    ):
        s = self.summary_dict(
            bench_runs=bench_runs, bench_warmup=bench_warmup, bench_batch=bench_batch
        )
        row = {
            "file": s["file"],
            "path": s["path"],
            "trainable": s["trainable"],
            "non_trainable": s["non_trainable"],
            "total": s["total"],
            "memory_MB": s["memory_MB"],
            "input_h": s["input_h"],
            "input_w": s["input_w"],
            "FLOPs": s["FLOPs"],
            "GFLOPs": s["GFLOPs"],
            "GMAC": s["GMAC"],
            "bench_batch": s["bench_batch"],
            "bench_runs": s["bench_runs"],
            "bench_warmup": s["bench_warmup"],
            "lat_ms_p50": s["lat_ms_p50"],
            "lat_ms_p95": s["lat_ms_p95"],
            "lat_ms_mean": s["lat_ms_mean"],
            "fps_p50": s["fps_p50"],
            "fps_p95": s["fps_p95"],
            "fps_mean": s["fps_mean"],
        }
        pd.DataFrame([row]).to_csv(out_csv, index=False)
        return out_csv


# Lista modelos disponíveis -> Paths e nomes
model_names, model_files, model_paths = report.list_models(MODELS_PATH)
print(f"Available models: {model_names}")
print(f"Model files: {model_files}")
print(f"Model paths: {model_paths}")

# Processa TXT -> resultados por classe e métricas globais
result_file_names = report.list_results()
print(f"Result files: {result_file_names}")
results_byclass = process_results.process_txt(result_file_names)
results_macro = process_results.process_global_metrics(result_file_names)
print(results_byclass)
print(results_macro)
#

results = {}
for name, file, path in zip(model_names, model_files, model_paths):
    insp = KerasModelInspector(str(path)).load()
    s = (
        insp.summary_dict()
    )  # contém file, trainable, non_trainable, total, memory_MB, dtype_breakdown, inputs, outputs
    s["model_name"] = name
    s["path"] = str(path)
    results[file] = s
    insp.clear()

# 3) (Opcional) salvar JSON
with open("models_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
