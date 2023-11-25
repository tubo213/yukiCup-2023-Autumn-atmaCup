from multiprocessing import cpu_count
from pathlib import Path

import hydra
import pandas as pd
from datasets import Dataset, DatasetDict
from pytorch_lightning import seed_everything
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from src.trainer import TreasureFGMTrainer
from src.utils import compute_metrics, sigmoid, threshold_search


def get_datasets(df: pd.DataFrame, tokenizer, max_len: int):
    def text_to_input_ids(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=max_len)

    ds = Dataset.from_pandas(df)
    return ds.map(text_to_input_ids, batched=True, num_proc=cpu_count())


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg):
    seed_everything(cfg.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1)

    # load data
    processed_dir = Path(cfg.dir.processed_dir)
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    train_ds = get_datasets(train_df, tokenizer, cfg.max_len)
    test_ds = get_datasets(test_df, tokenizer, cfg.max_len)

    # create dataset dict
    ds_dict = DatasetDict(
        {
            "train": train_ds.filter(lambda x: x["fold"] != cfg.fold),
            "eval": train_ds.filter(lambda x: x["fold"] == cfg.fold),
        }
    )

    # create trainer
    trainer = TreasureFGMTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=Path.cwd(),
            evaluation_strategy="epoch",
            learning_rate=cfg.lr,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            weight_decay=cfg.weight_decay,
            fp16=True,
            save_total_limit=1,
            save_strategy="epoch",
            metric_for_best_model="f1_score",
            load_best_model_at_end=True,
            greater_is_better=True,
            seed=cfg.seed,
        ),
        data_collator=data_collator,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["eval"],
        compute_metrics=compute_metrics,
        adv_start_epoch=cfg.adv_start_epoch,
        adv_epsilon=cfg.adv_epsilon,
    )

    # train
    trainer.train()

    # inference
    eval_pred_result = trainer.predict(ds_dict["eval"])
    eval_pred = sigmoid(eval_pred_result.predictions)
    eval_label = eval_pred_result.label_ids
    test_pred = sigmoid(trainer.predict(test_ds).predictions)

    # evaluate
    search_result = threshold_search(eval_label, eval_pred)
    print("OOF Score: ", search_result["f1"], "Threshold:", search_result["threshold"])

    # create submission
    test_df["is_kokuhou"] = (test_pred > search_result["threshold"]).astype(int)
    test_df[["is_kokuhou"]].to_csv(f"submission_{cfg.fold}.csv", index=False)

    # create submission
    test_df["is_kokuhou"] = (test_pred > search_result["threshold"]).astype(int)
    test_df[["is_kokuhou"]].to_csv(f"submission_{cfg.fold}.csv", index=False)


if __name__ == "__main__":
    main()
