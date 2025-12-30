from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64    # max # tokens looked at

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384       # size of embedding for each word

    dropout: float = 0.1
    bias: bool = True       # use bias terms in Linear/LayerNorm


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0                                 # embeddings divisible by heads
        self.cfg = cfg                                                      # the GPTConfig we set up earlier
        self.n_head = cfg.n_head                                            # 6 heads
        self.head_dim = cfg.n_embd // cfg.n_head                            # each head gets 384/6 = 64 numbers

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)  # for each token vector of length c, produce a qvk vector length 3c
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)      # combine info from all heads into a SINGLE vector of length c
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Casual mask using lower-triangular matrix
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()              # batch size, sequence length, embedding size
        qkv = self.c_attn(x)            # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)   # split into q, k, v vectors

        # Give each head its own Q, K, V
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Dot product for attention scores
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)
        att = att.masked_fill(                                    # blocks attention to future words
            self.bias_mask[:, :, :T, :T] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)                              # turn scores to probabilities
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)                             # each word gets a new representation based on earlier words
        y = y.transpose(1, 2).contiguous().view(B, T, C)          # recombine heads into full embedding size
        y = self.resid_dropout(self.c_proj(y))
        if return_attn:
            return y, att
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)      # expand the vector so it can think
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)    # compress back
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # given info gathered from attention, transform this word’s internal representation
        x = self.fc(x)          # expand so more combinations of features can interact
        x = F.gelu(x)           # thinking step: what features matter?
        x = self.proj(x)        # shrink back
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)    # normalizing only
        self.attn = CausalSelfAttention(cfg)                                # big step 1: attention
        self.ln2 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)    # normalizing again
        self.mlp = MLP(cfg)                                                 # big step 2: mlp

    # Two main steps: attention and mlp
    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if return_attn:
            a_out, att = self.attn(self.ln1(x), return_attn=True)
            x = x + a_out                   # take the old representation of each token & add what attention learned
            x = x + self.mlp(self.ln2(x))   # let each token privately think, and add that too
            return x, att
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x


class GPTSmall(nn.Module):
    """
    Minimal GPT decoder with:
    - forward(x) -> logits
    Optional:
    - forward(x, return_attn=True) -> (logits, attn_list)
    - forward(x, return_hidden=True) -> (logits, hidden_final)
    - forward_hidden(x) -> hidden_final
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)                     # word embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)                     # position embedding
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])   # creates n Blocks
        self.ln_f = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)       # normalization

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie weights

        self.apply(self._init_weights)

    # Initialize all weights randomly
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Returns internal token representations (hidden states) instead of predictions
    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  
        h = self.drop(self.wte(x) + self.wpe(pos))      # token vector + position vector
        for blk in self.blocks:                         # pass through blocks: each block does attention + residual + MLP + residual
            h = blk(h)                                  # type: ignore[assignment]
        h = self.ln_f(h)
        return h

    # Return predictions
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        return_hidden: bool = False,
    ):
        h = None
        attns: List[torch.Tensor] = []
        B, T = x.size()
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.drop(self.wte(x) + self.wpe(pos))

        if return_attn:
            for blk in self.blocks:                # pass through all blocks
                h, att = blk(h, return_attn=True)  # type: ignore[misc]
                attns.append(att)                  # keep track of attentions
        else:
            for blk in self.blocks:
                h = blk(h)                          # type: ignore[assignment]

        h = self.ln_f(h)
        logits = self.lm_head(h)

        if return_attn:
            return logits, attns
        if return_hidden:
            return logits, h
        return logits