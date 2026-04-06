from pathlib import Path
from PIL import Image
import csv

# ============================================================
# CONFIGURAÇÕES
# ============================================================
IMAGES_DIR = Path("WEED6C-Dataset")  # pasta com as imagens originais
ANNOTATIONS_FILE = Path(
    "WEED6C-Dataset/_annotations.txt"
)  # arquivo txt com as anotações
OUTPUT_DIR = Path("WEED6C_crops")  # pasta onde os crops serão salvos
CSV_OUTPUT = Path("weed6c_classification.csv")

# Se True, cria subpastas por classe: WEED6C_crops/0, WEED6C_crops/1, ...
SAVE_IN_CLASS_FOLDERS = False

# Se True, converte tudo para RGB antes de salvar
FORCE_RGB = True


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def parse_annotation_line(line: str):
    """
    Faz o parse de uma linha no formato:
    nome_imagem.jpg x1,y1,x2,y2,classe x1,y1,x2,y2,classe ...

    Retorna:
        image_name: str
        boxes: list[dict]
    """
    parts = line.strip().split()

    if not parts:
        return None, []

    image_name = parts[0]
    raw_boxes = parts[1:]

    boxes = []
    for item in raw_boxes:
        values = item.split(",")
        if len(values) != 5:
            # ignora entradas mal formatadas
            continue

        try:
            x1, y1, x2, y2, cls = map(int, values)
        except ValueError:
            continue

        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": cls})

    return image_name, boxes


def clamp_box(x1, y1, x2, y2, width, height):
    """
    Garante que a bounding box fique dentro da imagem.
    """
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


# ============================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    used_names = set()
    total_crops = 0
    total_images_with_annotations = 0
    skipped_images = 0
    skipped_boxes = 0

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de anotações não encontrado: {ANNOTATIONS_FILE}"
        )

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {IMAGES_DIR}")

    with ANNOTATIONS_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            image_name, boxes = parse_annotation_line(line)

            if not image_name or not boxes:
                continue

            image_path = IMAGES_DIR / image_name
            if not image_path.exists():
                print(
                    f"[AVISO] Imagem não encontrada (linha {line_number}): {image_path}"
                )
                skipped_images += 1
                continue

            try:
                with Image.open(image_path) as img:
                    if FORCE_RGB:
                        img = img.convert("RGB")

                    width, height = img.size
                    total_images_with_annotations += 1

                    for idx, box in enumerate(boxes, start=1):
                        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                        label = box["label"]

                        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)

                        # ignora bbox inválida
                        if x2 <= x1 or y2 <= y1:
                            print(
                                f"[AVISO] Bounding box inválida ignorada "
                                f"(linha {line_number}, imagem {image_name}, box {idx})"
                            )
                            skipped_boxes += 1
                            continue

                        crop = img.crop((x1, y1, x2, y2))

                        # nome base: {nome imagem original}_{ID}
                        stem = Path(image_name).stem
                        crop_name = f"{stem}_{idx:04d}.jpg"

                        # garantia extra de nome único
                        if crop_name in used_names:
                            suffix = 1
                            new_name = f"{stem}_{idx:04d}_{suffix}.jpg"
                            while new_name in used_names:
                                suffix += 1
                                new_name = f"{stem}_{idx:04d}_{suffix}.jpg"
                            crop_name = new_name

                        used_names.add(crop_name)

                        if SAVE_IN_CLASS_FOLDERS:
                            save_dir = OUTPUT_DIR / str(label)
                            save_dir.mkdir(parents=True, exist_ok=True)
                        else:
                            save_dir = OUTPUT_DIR

                        crop_path = save_dir / crop_name
                        crop.save(crop_path, quality=95)

                        records.append(
                            {
                                "Filename": crop_name,
                                "image_path": str(crop_path.as_posix()),
                                "Label": label,
                            }
                        )

                        total_crops += 1

            except Exception as e:
                print(f"[ERRO] Falha ao processar {image_name}: {e}")
                skipped_images += 1

    # salva CSV
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filename", "image_path", "Label"])
        writer.writeheader()
        writer.writerows(records)

    print("\n===== PROCESSAMENTO FINALIZADO =====")
    print(f"Imagens com anotações processadas: {total_images_with_annotations}")
    print(f"Crops gerados: {total_crops}")
    print(f"Imagens ignoradas: {skipped_images}")
    print(f"Bounding boxes ignoradas: {skipped_boxes}")
    print(f"Pasta de saída: {OUTPUT_DIR.resolve()}")
    print(f"CSV gerado: {CSV_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
