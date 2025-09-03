import os
import pandas as pd



def check_paths(df, directory, x_col='filename'):
    """Exibe diagnóstico dos caminhos usados pelo flow_from_dataframe."""
    missing = []
    valid = []

    for fname in df[x_col]:
        full_path = os.path.join(directory, fname) if directory else fname
        if os.path.isfile(full_path):
            valid.append(full_path)
        else:
            missing.append(full_path)

    print(f"✅ Imagens encontradas: {len(valid)}")
    print(f"❌ Imagens faltando: {len(missing)}")
    if missing:
        print("Exemplos faltando:")
        for m in missing[:5]:
            print(" -", m)
    return valid, missing


data = pd.read_csv("./labels/train_subset0.csv")
valid, missing = check_paths(data,
                             directory="/home/leo/Documentos/DeepWeeds-master/images2/images/",# ajuste aqui
                             x_col='Filename')







from PIL import Image
import pathlib
dire = "/home/leo/Documentos/DeepWeeds-master/images2/images/"
p = pathlib.Path(dire + data['Filename'].iloc[0])
try:
    img = Image.open(p)
    img.load()          # força leitura completa
    print("✅ PIL abriu:", p, "- modo:", img.mode, "- formato:", img.format)
except Exception as e:
    print("❌ PIL falhou:", e)