from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def print_distribution(name, data, label_col):
    dist = data[label_col].value_counts(normalize=True).sort_index() * 100
    print(f"\nDistribuição {name} (%):")
    print(dist.round(2))


def generate_3fold_datasets(
    csv_path: str,
    output_dir: str,
    label_col: str = "Label",
    random_state: int = 42,
    val_size: float = 0.2,
):
    """
    Gera 3 folds estratificados a partir de um CSV.

    Para cada fold:
      - 1 parte vira teste
      - as demais viram treino+validação
      - treino+validação é dividido estratificadamente em train e val

    Salva:
      output_dir/fold_1/train.csv
      output_dir/fold_1/val.csv
      output_dir/fold_1/test.csv
      ...
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Coluna de rótulo '{label_col}' não encontrada no CSV.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    X = df.index
    y = df[label_col]

    for fold_num, (train_val_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir = output_path / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_size,
            stratify=df_train_val[label_col],
            random_state=random_state,
        )

        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        df_train.to_csv(fold_dir / "train.csv", index=False)
        df_val.to_csv(fold_dir / "val.csv", index=False)
        df_test.to_csv(fold_dir / "test.csv", index=False)

        total_size = len(df)

        print(f"\nFold {fold_num}")
        print(f"Train: {len(df_train)} ({len(df_train)/total_size:.2%})")
        print(f"Val:   {len(df_val)} ({len(df_val)/total_size:.2%})")
        print(f"Test:  {len(df_test)} ({len(df_test)/total_size:.2%})")

        print_distribution("Train", df_train, label_col)
        print_distribution("Val", df_val, label_col)
        print_distribution("Test", df_test, label_col)


if __name__ == "__main__":
    generate_3fold_datasets(
        csv_path="labels/labels.csv",
        output_dir="labels/",
        label_col="Label",
        random_state=77,
        val_size=0.2,
    )
