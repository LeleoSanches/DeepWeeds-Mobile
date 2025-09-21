import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, out_png="curvas_treinamento.png", title="Treinamento"):
    """Plota e salva acurácia e loss (treino/val)."""
    hist = history.history
    epochs = range(1, len(hist["accuracy"]) + 1)

    plt.figure(figsize=(12,5))

    # Acurácia
    plt.subplot(1,2,1)
    plt.plot(epochs, hist["accuracy"], label="Treino")
    plt.plot(epochs, hist["val_accuracy"], label="Validação")
    plt.title("Acurácia")
    plt.xlabel("Época"); plt.ylabel("Acurácia")
    plt.legend(); plt.grid(True, alpha=0.3)

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, hist["loss"], label="Treino")
    plt.plot(epochs, hist["val_loss"], label="Validação")
    plt.title("Loss")
    plt.xlabel("Época"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Curvas salvas em: {out_png}")
    plt.close()

def save_history_csv(history, out_csv="historico_treinamento.csv"):
    """Salva o histórico completo em CSV."""
    df = pd.DataFrame(history.history)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Histórico salvo em: {out_csv}")

def plot_confusion_and_report(model, val_generator, class_indices,
                              cm_png="matriz_confusao.png", report_txt="relatorio_classificacao.txt",
                              normalize=True):
    """Gera preds no conjunto de validação, plota matriz de confusão e salva relatório."""
    # Verdadeiros e predições
    y_true = val_generator.classes
    probs = model.predict(val_generator, verbose=0)
    y_pred = probs.argmax(axis=1)

    # Nomes de classes na ordem correta
    inv_map = {v: k for k, v in class_indices.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # evita NaN se alguma classe não aparece

    # Plot
    plt.figure(figsize=(8,7))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusão" + (" (normalizada)" if normalize else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Anotações
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.savefig(cm_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Matriz de confusão salva em: {cm_png}")
    plt.close()

    # Relatório por classe
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Relatório salvo em: {report_txt}")
    # Também imprime no console
    print(report)


