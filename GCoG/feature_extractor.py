import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

def _infer_vocab_and_pad(tokenizer):
    # Be robust to attribute names
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        if hasattr(tokenizer, "token_to_id"):
            vocab_size = len(getattr(tokenizer, "token_to_id"))
        elif hasattr(tokenizer, "id_to_token"):
            vocab_size = len(getattr(tokenizer, "id_to_token"))
        else:
            vocab_size = 32768  # safe fallback
    pad_id = (
        getattr(tokenizer, "pad_token_id", None)
        if hasattr(tokenizer, "pad_token_id") else
        (tokenizer.token_to_id.get("<pad>", 0) if hasattr(tokenizer, "token_to_id") else 0)
    )
    return int(vocab_size), int(pad_id)

class TextNumericExtractor(BaseFeaturesExtractor):
    """
    Expects Dict obs with keys:
      - 'numbers': Box(..., shape=[D])
      - 'text_tokens': Box(..., shape=[L], dtype=int)
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        vocab_size: int,
        embed_dim: int = 128,
        num_hidden: int = 128,
        fusion_hidden: int = 256,
        features_dim: int = 256,
        padding_idx: int = 0,
    ):
        
        num_space = observation_space.spaces["numbers"]
        tok_space = observation_space.spaces["text_tokens"]
        assert isinstance(num_space, spaces.Box)
        assert isinstance(tok_space, spaces.Box)
        self.num_dim = int(num_space.shape[-1])
        out_dim = self.num_dim + num_hidden
        super().__init__(observation_space, features_dim)
        
        inferred_vocab = 0
        if hasattr(tok_space, "high"):
            high = tok_space.high
            inferred_vocab = int(np.max(high)) + 1  # ids are [0..max]
        num_embeddings = max(int(vocab_size), inferred_vocab, 1)
        self.pad_id = int(min(max(int(padding_idx), 0), num_embeddings - 1))



        self.num_norm = nn.LayerNorm(self.num_dim)

        self.embedding = nn.Embedding(num_embeddings, embed_dim, padding_idx=self.pad_id)
        self.txt_proj = nn.Linear(embed_dim, num_hidden)
        self.txt_norm = nn.LayerNorm(num_hidden)

        fused_in = self.num_dim + num_hidden
        self.fuse = nn.Sequential(                               # NEW
                    nn.Linear(fused_in, features_dim),
                    nn.SiLU(),
                    nn.Linear(features_dim, features_dim),
                )


        


    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # numbers
        raw_num = obs["numbers"].float()
        raw_num = self.num_norm(raw_num)        

        
        tokens = obs["text_tokens"].long()
        emb = self.embedding(tokens)                                        # [B, L, E]
        mask = (tokens != self.embedding.padding_idx).unsqueeze(-1)         # [B, L, 1]
        summed = (emb * mask).sum(dim=1)                                    # [B, E]
        lengths = mask.sum(dim=1).clamp(min=1)                              # [B, 1]
        x_txt = self.txt_norm(self.txt_proj(summed / lengths))

        return self.fuse(torch.cat([raw_num, x_txt], dim = -1))