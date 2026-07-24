import torch

from src.transformer.config import get_config
from src.transformer.data_loader import load_data
from src.transformer.model import build_transformer


config = get_config()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading model...")

# Load tokenizers
_, _, tokenizer_src, tokenizer_tgt = load_data(config)

# Build model
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

# Load checkpoint
checkpoint = torch.load(
    "weights/best.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# ---------------------------------------
# Save FP32 weights
# ---------------------------------------
torch.save(
    model.state_dict(),
    "weights/model_weights_fp32.pt"
)

print("Saved FP32 model.")

# ---------------------------------------
# Convert to FP16
# ---------------------------------------
model.half()

# ---------------------------------------
# Save FP16 weights
# ---------------------------------------
torch.save(
    model.state_dict(),
    "weights/model_weights_fp16.pt"
)

print("Saved FP16 model.")

print("\nDone!")