import shutil
from pathlib import Path

import hydra
import pandas as pd
from sklearn.model_selection import KFold
from pytorch_lightning import seed_everything

def create_folds(df: pd.DataFrame, n_splits: int, seed: int = 42):
    df["fold"] = -1

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (_, val_idx) in enumerate(kf.split(X=df)):
        df.loc[val_idx, "fold"] = fold

    return df


def connect_text(df: pd.DataFrame, text_cols: list[str], sep: str = " "):
    text_df = df[text_cols].fillna("NAN").astype(str)
    connected_text = text_df[text_cols[0]].str.cat(text_df[text_cols[1:]], sep=sep)

    return connected_text


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg):
    seed_everything(cfg.seed)
    processed_dir = Path(cfg.dir.processed_dir)

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Delete {processed_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Create {processed_dir}")

    # load data
    data_dir = Path(cfg.dir.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    era_df = pd.read_csv(data_dir / "era.csv")

    # merge era
    train_df = pd.merge(train_df, era_df, on="時代", how="left")
    test_df = pd.merge(test_df, era_df, on="時代", how="left")

    # create folds
    train_df = create_folds(train_df, n_splits=cfg.n_splits, seed=cfg.seed)

    # create target
    train_df["labels"] = train_df[cfg.target_col]

    # connect text
    train_df["text"] = connect_text(train_df, text_cols=cfg.text_cols, sep=cfg.sep)
    test_df["text"] = connect_text(test_df, text_cols=cfg.text_cols, sep=cfg.sep)

    # save
    train_df[["text", "labels", "fold"]].to_csv(processed_dir / "train.csv", index=False)
    test_df[["text"]].to_csv(processed_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
