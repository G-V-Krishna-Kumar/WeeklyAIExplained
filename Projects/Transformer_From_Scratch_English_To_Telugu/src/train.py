from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from pathlib import Path


import sacrebleu
import math

from src.transformer.data_loader import load_data
from src.transformer.config import get_config, get_weights_file_path
from src.transformer.dataset import BilingualDataset
from src.transformer.model import build_transformer
from src.transformer.translate import beam_search_decode


def compute_translation_metrics(
        model: nn.Module,
        val_loader: DataLoader,
        tokenizer_src: Any,
        tokenizer_tgt: Any,
        device: torch.device,
        beam_size: int) -> dict[str, float]:

    model.eval()

    preds = []
    refs = []

    exact_match = 0
    total = 0

    for batch in val_loader:

        src_text = batch["src_text"][0]
        tgt_text = batch["tgt_text"][0]

        pred_tokens = beam_search_decode(
            model,
            tokenizer_src,
            tokenizer_tgt,
            src_text,
            device,
            beam_size=beam_size
        )

        pred_text = tokenizer_tgt.decode(pred_tokens)

        preds.append(pred_text)
        refs.append([tgt_text])

        if pred_text.strip() == tgt_text.strip():
            exact_match += 1

        total += 1

    bleu = sacrebleu.corpus_bleu(
        preds,
        refs
    ).score

    chrf = sacrebleu.corpus_chrf(
        preds,
        refs
    ).score

    ter = sacrebleu.corpus_ter(
        preds,
        refs
    ).score

    exact_match_acc = (
        100.0 * exact_match / total
    )

    return {
        "bleu": bleu,
        "chrf": chrf,
        "ter": ter,
        "exact_match": exact_match_acc
    }


def compute_token_accuracy(
        model: nn.Module,
        val_loader: DataLoader,
        tokenizer_tgt: Any,
        device: torch.device
        ) -> float:
    
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in val_loader:

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            enc_out = model.encode(
                encoder_input,
                encoder_mask
            )

            dec_out = model.decode(
                decoder_input,
                enc_out,
                encoder_mask,
                decoder_mask
            )

            proj = model.project(dec_out)

            pred = proj.argmax(dim=-1)

            mask = (
                label !=
                tokenizer_tgt.token_to_id("[PAD]")
            )

            correct += (
                (pred == label) & mask
            ).sum().item()

            total += mask.sum().item()

    return 100.0 * correct / total


# ----------------------------
# VALIDATION LOSS
# ----------------------------
def validation_loss(
        model: nn.Module, 
        val_loader: DataLoader, 
        loss_fn: nn.Module, 
        tokenizer_tgt: Any, 
        device: torch.device) -> float:
    
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            enc_out = model.encode(encoder_input, encoder_mask)
            dec_out = model.decode(decoder_input, enc_out, encoder_mask, decoder_mask)
            proj = model.project(dec_out)

            loss = loss_fn(
                proj.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train() -> None:
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    Path(config["model_folder"]).mkdir(exist_ok=True)

    train_ds, val_ds, tokenizer_src, tokenizer_tgt = load_data(config)


    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    val_subset = Subset(val_ds, range(min(config["eval_subset_size"], len(val_ds))))
    val_subset_loader = DataLoader(val_subset, batch_size=1)

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    ).to(device)

    use_amp = (torch.cuda.is_available() and config["mixed_precision"])

    scaler = GradScaler("cuda", enabled=use_amp)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        eps=1e-9,
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"]
    )

    global_step = 0
    best_bleu = -1
    patience_counter = 0
    start_epoch = 0

    if config["preload"] is not None:

        checkpoint = torch.load(config["preload"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", 0)
        best_bleu = checkpoint.get("best_bleu", -1)

        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()

        for batch in batch_iter:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            with autocast(
                "cuda",
                enabled=use_amp
            ):

                enc_out = model.encode(
                    encoder_input,
                    encoder_mask
                )

                dec_out = model.decode(
                    decoder_input,
                    enc_out,
                    encoder_mask,
                    decoder_mask
                )

                proj = model.project(dec_out)

                loss = loss_fn(
                    proj.view(
                        -1,
                        tokenizer_tgt.get_vocab_size()
                    ),
                    label.view(-1)
                )

                loss = (
                    loss
                    / config["grad_accum_steps"]
                )


            scaler.scale(loss).backward()

            if (
                (global_step + 1)
                % config["grad_accum_steps"]
                == 0
            ):

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["max_grad_norm"]
                )

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad()

            batch_iter.set_postfix(
                loss=(
                    loss.item()
                    * config["grad_accum_steps"]
                )
            )

            global_step += 1

            # =====================================
            # Mid-epoch checkpoint
            # =====================================
            if (
                global_step > 0
                and
                global_step
                % config["checkpoint_every"]
                == 0
            ):

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "global_step":
                            global_step,
                        "best_bleu":
                            best_bleu
                    },
                    "weights/latest.pt"
                )

                print(
                    f"\nCheckpoint saved "
                    f"(step {global_step})"
                )

        if global_step % config["grad_accum_steps"] != 0:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["max_grad_norm"]
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # =====================================
        # Validation
        # =====================================
        val_loss = validation_loss(
            model,
            val_loader,
            loss_fn,
            tokenizer_tgt,
            device
        )

        scheduler.step(val_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        metrics = compute_translation_metrics(
            model,
            val_subset_loader,
            tokenizer_src,
            tokenizer_tgt,
            device,
            config["beam_size"]
        )

        token_acc = compute_token_accuracy(
            model,
            val_loader,
            tokenizer_tgt,
            device
        )

        perplexity = math.exp(val_loss)

        print(f"\nEpoch {epoch}")

        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"Perplexity     : "
            f"{perplexity:.2f}"
        )

        print(
            f"BLEU           : "
            f"{metrics['bleu']:.2f}"
        )

        print(
            f"chrF++         : "
            f"{metrics['chrf']:.2f}"
        )

        print(
            f"TER            : "
            f"{metrics['ter']:.2f}"
        )

        print(
            f"Token Accuracy : "
            f"{token_acc:.2f}%"
        )

        print(
            f"Exact Match    : "
            f"{metrics['exact_match']:.2f}%"
        )

        print(
            f"Learning Rate  : "
            f"{current_lr:.8f}"
        )

        # =====================================
        # Best model + early stopping
        # =====================================
        improved = (
            metrics["bleu"]
            > best_bleu
        )

        if improved:

            best_bleu = metrics["bleu"]

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "global_step":
                        global_step,
                    "best_bleu":
                        best_bleu
                },
                "weights/best.pt"
            )

            print(
                "New best model saved."
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= config[
                "early_stopping_patience"
            ]
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

        # =====================================
        # Epoch checkpoint
        # =====================================
        model_path = get_weights_file_path(
            config,
            f"{epoch:02d}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "global_step":
                    global_step,
                "best_bleu":
                    best_bleu
            },
            model_path
        )

        print(f"Saved: {model_path}")


if __name__ == "__main__":
    train()