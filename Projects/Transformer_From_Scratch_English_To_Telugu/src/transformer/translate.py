import torch


# ==========================================================
# Causal Mask
# ==========================================================
def causal_mask(size):

    return torch.tril(
        torch.ones(
            (size, size),
            dtype=torch.bool
        )
    )


# ==========================================================
# Greedy Decode
# ==========================================================
@torch.no_grad()
def greedy_decode(
    model,
    tokenizer_src,
    tokenizer_tgt,
    src_text,
    device,
    max_len=256
):

    model.eval()

    # ==========================================
    # Special Tokens
    # ==========================================
    src_sos = tokenizer_src.token_to_id("[SOS]")
    src_eos = tokenizer_src.token_to_id("[EOS]")
    src_pad = tokenizer_src.token_to_id("[PAD]")

    tgt_sos = tokenizer_tgt.token_to_id("[SOS]")
    tgt_eos = tokenizer_tgt.token_to_id("[EOS]")

    # ==========================================
    # Tokenize Source
    # ==========================================
    src_tokens = tokenizer_src.encode(
        src_text
    ).ids

    # ==========================================
    # IMPORTANT FIX
    #
    # Match training format
    #
    # [SOS] sentence [EOS]
    # ==========================================
    src_tokens = (
        [src_sos]
        + src_tokens
        + [src_eos]
    )

    src = torch.tensor(
        src_tokens,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    # ==========================================
    # Encoder Mask
    # ==========================================
    src_mask = (
        src != src_pad
    ).unsqueeze(1).unsqueeze(2)

    # ==========================================
    # Encoder Forward
    # ==========================================
    encoder_output = model.encode(
        src,
        src_mask
    )

    # ==========================================
    # Decoder Starts With SOS
    # ==========================================
    decoder_input = torch.tensor(
        [[tgt_sos]],
        dtype=torch.long,
        device=device
    )

    # ==========================================
    # Autoregressive Generation
    # ==========================================
    for _ in range(max_len):

        tgt_mask = causal_mask(
            decoder_input.size(1)
        )

        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(device)

        decoder_output = model.decode(
            decoder_input,
            encoder_output,
            src_mask,
            tgt_mask
        )

        logits = model.project(
            decoder_output[:, -1]
        )

        next_token = torch.argmax(
            logits,
            dim=-1
        )

        decoder_input = torch.cat(
            [
                decoder_input,
                next_token.unsqueeze(0)
            ],
            dim=1
        )

        if next_token.item() == tgt_eos:
            break

    return decoder_input.squeeze(0).tolist()


# ==========================================================
# Beam Search Decode
# ==========================================================
@torch.no_grad()
def beam_search_decode(
    model,
    tokenizer_src,
    tokenizer_tgt,
    src_text,
    device,
    beam_size=4,
    max_len=256
):

    model.eval()

    src_sos = tokenizer_src.token_to_id("[SOS]")
    src_eos = tokenizer_src.token_to_id("[EOS]")
    src_pad = tokenizer_src.token_to_id("[PAD]")

    tgt_sos = tokenizer_tgt.token_to_id("[SOS]")
    tgt_eos = tokenizer_tgt.token_to_id("[EOS]")

    # ==========================================
    # Source Encoding
    # ==========================================
    src_tokens = tokenizer_src.encode(
        src_text
    ).ids

    src_tokens = (
        [src_sos]
        + src_tokens
        + [src_eos]
    )

    src = torch.tensor(
        src_tokens,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    src_mask = (
        src != src_pad
    ).unsqueeze(1).unsqueeze(2)

    encoder_output = model.encode(
        src,
        src_mask
    )

    # ==========================================
    # Beam
    # ==========================================
    beams = [
        (
            torch.tensor(
                [[tgt_sos]],
                device=device
            ),
            0.0
        )
    ]

    for _ in range(max_len):

        candidates = []

        finished = True

        for seq, score in beams:

            if seq[0, -1].item() == tgt_eos:
                candidates.append(
                    (seq, score)
                )
                continue

            finished = False

            tgt_mask = causal_mask(
                seq.size(1)
            )

            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(device)

            decoder_output = model.decode(
                seq,
                encoder_output,
                src_mask,
                tgt_mask
            )

            logits = model.project(
                decoder_output[:, -1]
            )

            log_probs = torch.log_softmax(
                logits,
                dim=-1
            )

            topk_log_probs, topk_ids = torch.topk(
                log_probs,
                beam_size,
                dim=-1
            )

            for k in range(beam_size):

                token_id = topk_ids[0, k]

                token_score = topk_log_probs[0, k].item()

                new_seq = torch.cat(
                    [
                        seq,
                        token_id.view(1, 1)
                    ],
                    dim=1
                )

                candidates.append(
                    (
                        new_seq,
                        score + token_score
                    )
                )

        if finished:
            break

        beams = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )[:beam_size]

    best_sequence = beams[0][0]

    return best_sequence.squeeze(0).tolist()


# ==========================================================
# Translate Sentence
# ==========================================================
def translate_sentence(
    model,
    tokenizer_src,
    tokenizer_tgt,
    sentence,
    device,
    beam_size=4
):

    output_tokens = beam_search_decode(
        model,
        tokenizer_src,
        tokenizer_tgt,
        sentence,
        device,
        beam_size=beam_size
    )

    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")

    output_tokens = [
        token
        for token in output_tokens
        if token not in (sos_id, eos_id)
    ]

    return tokenizer_tgt.decode(
        output_tokens
    )

if __name__ == "__main__":

    from src.transformer.config import get_config
    from src.transformer.data_loader import load_data
    from src.transformer.model import build_transformer

    config = get_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, tokenizer_src, tokenizer_tgt = load_data(config)

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

    # ==========================================
    # Load FP16 Model
    # ==========================================

    model = model.half()

    state_dict = torch.load(
        "weights/model_weights_fp16.pt",
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print("Model loaded.")

    while True:

        sentence = input("\nEnglish: ")

        if sentence.lower() == "exit":
            break

        translation = translate_sentence(
            model,
            tokenizer_src,
            tokenizer_tgt,
            sentence,
            device,
            beam_size=4
        )

        print("Telugu:", translation)
        with open("out.txt", "a") as f:
            f.write(f"English: {sentence}\n")
            f.write(f"Telugu: {translation}\n\n")
        