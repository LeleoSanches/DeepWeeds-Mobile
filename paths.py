from pathlib import Path

def get_project_dirs(must_exist=("images", "labels")):
    BASE_DIR = Path(__file__).resolve().parent
    dirs = {
        "images":  BASE_DIR / "images/",
        "labels":  BASE_DIR / "labels/",
        "results": BASE_DIR / "results/",
        "models":  BASE_DIR / "models/",
    }
    for key in must_exist:
        if not dirs[key].exists():
            raise FileNotFoundError(f"Pasta obrigatória não encontrada: {dirs[key]} \n Crie a pasta com as imagens do DeepWeeds antes de executar.")
    for key in ["results", "models"]:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs

