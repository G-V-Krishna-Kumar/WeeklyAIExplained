# Transformer from Scratch for English-to-Telugu Neural Machine Translation

An implementation of the **Transformer Encoder–Decoder architecture** built entirely from scratch in **PyTorch** for **English-to-Telugu Neural Machine Translation (NMT)**, based on the paper **"Attention Is All You Need"** by Vaswani et al.

The objective of this project is to understand the complete Transformer architecture, training pipeline, and inference process without relying on pre-trained translation models.

> **Project Status:** Educational implementation developed as part of the **WeeklyAIExplained** series. This project is intended for learning and experimentation rather than production deployment.

---

# Features

- Transformer Encoder–Decoder implemented completely from scratch
- Multi-Head Self-Attention
- Scaled Dot-Product Attention
- Positional Encoding
- Residual Connections and Layer Normalization
- Byte Pair Encoding (BPE) tokenizer
- English → Telugu Neural Machine Translation
- Greedy Search and Beam Search decoding
- Mixed Precision (FP16) training
- FP16 inference model for reduced storage
- Interactive command-line translation interface
- Evaluation using BLEU, chrF++, TER, Perplexity, Token Accuracy and Exact Match

---

# Dataset

This project uses the **HackHedron English-Telugu Parallel Corpus**.

Download:

https://huggingface.co/datasets/HackHedron/English_Telugu_Parallel_Corpus

Place the dataset as:

```text
data/
└── english-telugu-translations-dataset.csv
```

The local filename is renamed for project organization. The dataset contents remain unchanged from the original Hugging Face release.

---

# Tech Stack

- Python
- PyTorch
- Hugging Face Tokenizers
- Pandas
- SacreBLEU

---

# Project Structure

```text
Transformer_From_Scratch_English_To_Telugu/
│
├── src/
│   ├── train.py
│   └── transformer/
│       ├── __init__.py
│       ├── check_model.py
│       ├── config.py
│       ├── data_loader.py
│       ├── dataset.py
│       ├── model.py
│       └── translate.py
│
├── tokenizer/
│   ├── tokenizer_english.json
│   └── tokenizer_telugu.json
│
├── weights/
│   └── model_weights_fp16.pt
│
├── requirements.txt
├── README.md
└── LICENSE
```

# Installation

This project is part of the **WeeklyAIExplained** repository.

To download only this Transformer project instead of cloning the complete repository, use **Git sparse checkout**.

Clone the repository without checking out all files:

```bash
git clone --filter=blob:none --no-checkout https://github.com/G-V-Krishna-Kumar/WeeklyAIExplained.git
cd WeeklyAIExplained
git sparse-checkout init --cone
git sparse-checkout set "Projects/Transformer_From_Scratch_English_To_Telugu"
git checkout
cd "Projects/Transformer_From_Scratch_English_To_Telugu"
```

Create a virtual environment.

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Install PyTorch appropriate for your operating system and GPU.

https://pytorch.org/get-started/locally/

---

# Training

Train the Transformer.

```bash
python -m src.train
```

Training checkpoints will be saved inside the `weights/` directory.

---

# Running Translation

Launch the interactive translator.

```bash
python -m src.transformer.translate
```

Example

```text
English : How are you?
Telugu  : మీరు ఎలా ఉన్నారు?
```

Type `exit` to close the application.

---

# Model Configuration

| Component | Value |
|-----------|------:|
| Architecture | Transformer Encoder–Decoder |
| Vocabulary Size | 32,000 |
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Attention Heads | 8 |
| Model Dimension (d_model) | 256 |
| Feed Forward Dimension | 1024 |
| Maximum Sequence Length | 256 |
| Beam Size | 4 |

The released inference model uses **FP16 weights**, reducing storage requirements while preserving translation quality.

---

# Evaluation Metrics

The model was evaluated using multiple metrics to measure translation quality, confidence, and token-level prediction performance.

| Metric | Description |
|--------|-------------|
| Validation Loss | Measures the difference between predicted token probabilities and the actual target tokens during validation. Lower values indicate better model predictions. |
| Perplexity | Measures how uncertain the model is while predicting the next token. Lower perplexity indicates the model is more confident. |
| BLEU | Measures n-gram overlap between generated translations and reference translations. Higher BLEU scores indicate better translation similarity. |
| chrF++ | Character-level F-score metric that compares character n-grams between generated and reference translations. It is especially useful for morphologically rich languages such as Telugu. Higher values indicate better similarity. |
| TER (Translation Edit Rate) | Measures the number of edits required to transform the generated translation into the reference translation. Lower TER indicates fewer corrections are needed. |
| Token Accuracy | Measures the percentage of correctly predicted tokens compared to the reference tokens, excluding padding tokens. |
| Exact Match | Measures the percentage of translations where the generated output exactly matches the reference sentence. |

Final evaluation results:

| Metric | Score |
|--------|------:|
| Validation Loss | 2.0561 |
| Perplexity | 7.82 |
| BLEU | 33.44 |
| chrF++ | 82.55 |
| TER | 31.25 |
| Token Accuracy | 81.79% |
| Exact Match | 0.10% |
---

# References

**Attention Is All You Need**

https://arxiv.org/abs/1706.03762

**HackHedron English-Telugu Parallel Corpus**

https://huggingface.co/datasets/HackHedron/English_Telugu_Parallel_Corpus

**PyTorch Documentation**

https://pytorch.org/docs/

---

# License

This project is licensed under the **MIT License**.