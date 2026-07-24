from typing import Any

import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(
        self,
        ds: Any,
        tokenizer_src :Any,
        tokenizer_tgt: Any,
        src_lang: str,
        tgt_lang: str,
        seq_len:  int
    ):
        super().__init__()

        self.ds = ds

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.seq_len = seq_len

        self.src_sos = tokenizer_src.token_to_id("[SOS]")
        self.src_eos = tokenizer_src.token_to_id("[EOS]")
        self.src_pad = tokenizer_src.token_to_id("[PAD]")


        self.tgt_sos = tokenizer_tgt.token_to_id("[SOS]")
        self.tgt_eos = tokenizer_tgt.token_to_id("[EOS]")
        self.tgt_pad = tokenizer_tgt.token_to_id("[PAD]")


    def __len__(self) -> int:
        return len(self.ds)
    

    @staticmethod
    def causal_mask(size: int) -> torch.Tensor:
        mask = torch.tril(torch.ones((size, size),dtype=torch.bool))

        return mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.ds[idx]

        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids


        src_required_len = len(src_tokens) + 2
        tgt_required_len = len(tgt_tokens) + 1

        if src_required_len > self.seq_len:
            raise ValueError(f"Source sentence too long ({src_required_len} > {self.seq_len})")

        if tgt_required_len > self.seq_len:
            raise ValueError(f"Target sentence too long ({tgt_required_len} > {self.seq_len})")

 
        # Initialize encoder_input = tensor([PAD, PAD, ..., PAD]) with length = seq_len
        encoder_input = torch.full(
            (self.seq_len,),
            self.src_pad,
            dtype=torch.long
        ) 

        # encoder_input = tensor([SOS, TOK1, TOK2, EOS, PAD, ..., PAD]) if len(src_tokens) = 2
        encoder_input[0] = self.src_sos
        encoder_input[1:1 + len(src_tokens)] = torch.tensor(src_tokens, dtype=torch.long)
        encoder_input[1 + len(src_tokens)] = self.src_eos 

        

        # Initialize decoder_input = tensor([PAD, PAD, ..., PAD]) with length = seq_len
        decoder_input = torch.full(
            (self.seq_len,),
            self.tgt_pad,
            dtype=torch.long
        )

        # decoder_input = tensor([SOS, TOK1, TOK2, PAD, PAD, ..., PAD]) if len(tgt_tokens) = 2
        decoder_input[0] = self.tgt_sos
        decoder_input[1:1 + len(tgt_tokens)] = torch.tensor(tgt_tokens, dtype=torch.long)



        # Initialize label = tensor([PAD, PAD, ..., PAD]) with length = seq_len
        label = torch.full(
            (self.seq_len,),
            self.tgt_pad,
            dtype=torch.long
        )

        # label = tensor([TOK1, TOK2, EOS, PAD, PAD, ..., PAD]) if len(tgt_tokens) = 2)
        label[:len(tgt_tokens)] = torch.tensor(tgt_tokens, dtype=torch.long)
        label[len(tgt_tokens)] = self.tgt_eos


        encoder_mask = (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0) 
        # Now the encoder_mask = [True, True, True, False, False, ..., False]  if encoder_input = [SOS, TOK1, EOS, PAD, PAD, ... PAD]
        # encoder_mask.shape = (1, 1, seq_len)


        causal_mask = self.causal_mask(self.seq_len) # (seq_len, seq_len)

        decoder_padding_mask = (decoder_input != self.tgt_pad).unsqueeze(0)
        # Now the decoder_padding_mask = [True, True, True, False, False, ..., False]  if decoder_input = [SOS, TOK1, TOK2, PAD, PAD, ... PAD]
        # decoder_padding_mask.shape = (1, seq_len)

        decoder_mask = (decoder_padding_mask.unsqueeze(0) & causal_mask.unsqueeze(0)) 
        # (1, 1, seq_len) & (1, seq_len, seq_len) => PyTorch broadcasts decoder_padding_mask to (1, seq_len, seq_len) and the operation becomes ((1, seq_len, seq_len) & (1, seq_len, seq_len))
        # decoder_mask.shape will be (1, seq_len, seq_len)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }