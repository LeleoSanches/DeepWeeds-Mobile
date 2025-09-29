import time
import json
import numpy as np
import tensorflow as tf

class benchmark(object):
    @staticmethod
    def _percentiles(a, ps=(50,90,95,99)):
        return {f"p{p}": float(np.percentile(a, p)) for p in ps}

    @staticmethod
    def _one_sample_from_generator(gen):
        # usa um batch do val_generator (já preprocessado) e recorta para BS=1
        xb, _ = next(iter(gen))
        x1 = xb[0:1]  # shape (1,H,W,3)
        return x1
    

    @staticmethod
    def benchmark_keras_inference(model, sample, runs=100, warmup=20, use_predict=False, jit_compile=True):
        """
        Mede latência do Keras (GPU/CPU conforme dispositivo atual).
        - sample: tensor (1,H,W,3), já preprocessado como no treino
        - use_predict=True usa model.predict (mais overhead); False usa chamada direta.
        - jit_compile tenta XLA; se não rolar, cai para eager.
        Retorna dict com média, std, p50/p90/p95/p99 (ms).
        """
        fn = lambda x: model(x, training=False)
        if jit_compile:
            try:
                fn = tf.function(fn, jit_compile=True)
                _ = fn(sample)  # compila
            except Exception:
                pass

        # warmup
        for _ in range(warmup):
            y = fn(sample) if not use_predict else model.predict(sample, verbose=0)
            if hasattr(y, "numpy"): y.numpy()  # força sync

        # mede
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            y = fn(sample) if not use_predict else model.predict(sample, verbose=0)
            if hasattr(y, "numpy"): y.numpy()  # sincroniza GPU
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)

        times = np.asarray(times, dtype=np.float64)
        out = {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            **_percentiles(times),
            "runs": int(runs), "warmup": int(warmup),
        }
        return out

    
    @staticmethod
    def save_latency_row(csv_path, row_dict):
        import csv, os
        header = list(row_dict.keys())
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header: w.writeheader()
            w.writerow(row_dict)
        print(f"[OK] Linha salva em: {csv_path}")

    ### --- TFLite (float32 e INT8) ---
    @staticmethod
    def keras_to_tflite(model, tflite_path="model_fp32.tflite",
                        optim=None, representative_gen=None,
                        int8_io=False):
        """
        Converte Keras -> TFLite. 
        - optim="DEFAULT" ativa PTQ.
        - representative_gen: generator de dados representativos (necessário p/ INT8).
        - int8_io=True força entrada/saída INT8 (para devices microcontrolados).
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if optim == "DEFAULT":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_gen is not None:
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if int8_io:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[OK] TFLite salvo em: {tflite_path}")
        return tflite_path

    @staticmethod
    def _set_input(interpreter, arr):
        idx = interpreter.get_input_details()[0]["index"]
        # TFLite aceita float32 por padrão; se INT8, normalizamos para int8 simétrico.
        dtype = interpreter.get_input_details()[0]["dtype"]
        if dtype == np.float32:
            interpreter.set_tensor(idx, arr.astype(np.float32))
        elif dtype == np.int8:
            scale, zero = interpreter.get_input_details()[0]["quantization"]
            # arr aqui deve estar no mesmo espaço do treino (ex.: [-1,1] p/ MobileNetV2/V3).
            # Convertemos para int8 da forma padrão: int8 = arr/scale + zero_point
            # Se seu pipeline for diferente, ajuste esta linha.
            q = np.clip(np.round(arr/scale + zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(idx, q)
        else:
            raise ValueError(f"Entrada dtype não suportado: {dtype}")

    @staticmethod
    def benchmark_tflite(tflite_path, sample, runs=200, warmup=50):
        """
        Mede latência do TFLite no host atual (CPU).
        sample: numpy array (1,H,W,3) no espaço ORIGINAL do modelo Keras (ex.: [-1,1]).
        """
        interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=os.cpu_count())
        interpreter.allocate_tensors()

        # Warmup
        for _ in range(warmup):
            _set_input(interpreter, sample)
            interpreter.invoke()

        # Bench
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _set_input(interpreter, sample)
            interpreter.invoke()
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)

        times = np.asarray(times, dtype=np.float64)
        out = {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            **_percentiles(times),
            "runs": int(runs), "warmup": int(warmup),
        }
        return out
    

# 1) pegue 1 amostra pronta do val_generator (já preprocessada)
sample = _one_sample_from_generator(val_generator).numpy()

# 2) benchmark KERAS na GPU (ou CPU, conforme onde o modelo está)
lat_gpu = benchmark_keras_inference(model, sample, runs=200, warmup=30, jit_compile=True)
row_gpu = {"model":"MobileNetV3Large", "mode":"keras_gpu", "bs":1, "img":f"{IMG_SIZE[0]}x{IMG_SIZE[1]}", **lat_gpu}
save_latency_row("latency_results.csv", row_gpu)
print("KERAS/GPU:", row_gpu)

# (Opcional) forçar CPU (copia pesos, pode demorar no warmup; útil só para comparação)
with tf.device("/CPU:0"):
    lat_cpu = benchmark_keras_inference(model, sample, runs=200, warmup=50, jit_compile=False)
row_cpu = {"model":"MobileNetV3Large", "mode":"keras_cpu", "bs":1, "img":f"{IMG_SIZE[0]}x{IMG_SIZE[1]}", **lat_cpu}
save_latency_row("latency_results.csv", row_cpu)
print("KERAS/CPU:", row_cpu)

# 3) Converter para TFLite (FP32) e medir
tflite_fp32 = keras_to_tflite(model, "model_fp32.tflite")
lat_tflite_fp32 = benchmark_tflite(tflite_fp32, sample, runs=400, warmup=80)
row_tflite_fp32 = {"model":"MobileNetV3Large", "mode":"tflite_fp32", "bs":1, "img":f"{IMG_SIZE[0]}x{IMG_SIZE[1]}", **lat_tflite_fp32}
save_latency_row("latency_results.csv", row_tflite_fp32)
print("TFLite FP32:", row_tflite_fp32)

# 4) Converter para TFLite INT8 (PTQ) e medir
def representative_gen():
    # amostras representativas para calibrar INT8
    # use ~100-300 imagens; aqui só um exemplo curto
    for fp in df_train["Filename"].sample(200, random_state=42):
        img = tf.io.read_file(fp)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_fn(img)              # MESMO preprocess do treino
        yield [tf.expand_dims(img, 0)]

tflite_int8 = keras_to_tflite(model, "model_int8.tflite",
                              optim="DEFAULT",
                              representative_gen=representative_gen,
                              int8_io=False)  # mantenha FP32 I/O se preferir simplificar
lat_tflite_int8 = benchmark_tflite(tflite_int8, sample, runs=400, warmup=80)
row_tflite_int8 = {"model":"MobileNetV3Large", "mode":"tflite_int8", "bs":1, "img":f"{IMG_SIZE[0]}x{IMG_SIZE[1]}", **lat_tflite_int8}
save_latency_row("latency_results.csv", row_tflite_int8)
print("TFLite INT8:", row_tflite_int8)