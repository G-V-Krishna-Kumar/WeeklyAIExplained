import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from src.transformer.dataset import BilingualDataset
from collections.abc import Iterator



def build_tokenizer(config: dict, ds: list, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    print(f"\nTraining tokenizer for {lang}...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[SOS]",
            "[EOS]"
        ]
    )

    def sentence_iterator() -> Iterator[str]:
        for item in ds:
            yield item["translation"][lang]

    tokenizer.train_from_iterator(
        sentence_iterator(),
        trainer=trainer
    )

    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer: {tokenizer_path}")

    return tokenizer



def load_data(config: dict) -> tuple[BilingualDataset,
                                    BilingualDataset,
                                    Tokenizer,
                                    Tokenizer]:

    print("\nLoading dataset...")

    df = pd.read_csv(config["dataset_file"])
    df = df.dropna()

    print(f"Loaded {len(df):,} sentence pairs")

    ds = []
    for _, row in df.iterrows():
        ds.append(
            {
                "translation":
                {
                    config["lang_src"] : str(row[config["lang_src"]]),
                    config["lang_tgt"] : str(row[config["lang_tgt"]])
                }
            }
        )


    train_raw, val_raw = train_test_split(
        ds,
        test_size=1.0 - config["train_split"],
        random_state=config["random_seed"],
        shuffle=True
    )

    print(f"Train samples: {len(train_raw):,}")

    print(f"Val samples: {len(val_raw):,}")


    tokenizer_src = build_tokenizer(
        config,
        ds,
        config["lang_src"]
    )

    tokenizer_tgt = build_tokenizer(
        config,
        ds,
        config["lang_tgt"]
    )

    print(f"\nSource vocab size: {tokenizer_src.get_vocab_size():,}")
    print(f"Target vocab size: {tokenizer_tgt.get_vocab_size():,}")


    train_ds = BilingualDataset(
        train_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    val_ds = BilingualDataset(
        val_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    return (
        train_ds,
        val_ds,
        tokenizer_src,
        tokenizer_tgt
    )   