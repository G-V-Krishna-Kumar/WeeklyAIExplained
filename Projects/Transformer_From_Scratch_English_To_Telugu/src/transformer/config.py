from typing import Any
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def get_config() -> dict[str, Any]:
    return {

        "batch_size": 4,
        "grad_accum_steps": 8,
        "num_epochs": 30,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "seq_len": 256,
        "mixed_precision": True,
        "max_grad_norm": 1.0,


        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,


        "vocab_size": 32000,
        "lang_src": "english",
        "lang_tgt": "telugu",
        "dataset_file": str(
            BASE_DIR / "data" / "english-telugu-translations-dataset.csv"
        ),


        "train_split": 0.90,
        "random_seed": 42,


        "model_folder": str(BASE_DIR / "weights"),
        "model_basename": "tmodel_",
        "checkpoint_every": 1000,


        "beam_size": 4,


        "tokenizer_file": str(
            BASE_DIR / "tokenizer" / "tokenizer_{0}.json"
        ),


        "preload": str(BASE_DIR / "weights" / "latest.pt"),


        "early_stopping_patience": 5,


        "scheduler_factor": 0.5,
        "scheduler_patience": 1,


        "eval_subset_size": 1000,
        "experiment_name": str(BASE_DIR / "runs" / "tmodel"),
    }



def get_weights_file_path(config: dict[str, Any], epoch: str) -> str:
    model_folder = Path(config["model_folder"])

    model_filename = f"{config['model_basename']}{epoch}.pt"

    return str(model_folder / model_filename)



def get_latest_checkpoint_path(config: dict[str, Any]) -> str:
    return str(Path(config["model_folder"]) / "latest.pt")



def get_best_model_path(config: dict[str, Any]) -> str:
    return str(Path(config["model_folder"]) / "best.pt")



def create_folders(config: dict[str, Any]) -> None:
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    Path(BASE_DIR / "tokenizer").mkdir(parents=True, exist_ok=True)
    Path(BASE_DIR / "runs" / "tmodel").mkdir(parents=True, exist_ok=True)