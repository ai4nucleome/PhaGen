from transformers.configuration_utils import PretrainedConfig


class PhaGenConfig(PretrainedConfig):
    model_type = "phagen"
    architectures = ["PhaGenModel"]
    IGNORE_INDEX = -100
    
    def __init__(
        self,
        num_tokens: int = 9,
        pad_token_id: int = 7,
        dim = (512, 256, 196),
        depth = (8, 8, 8),
        max_seq_len = (128, 64, 16),
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        rel_pos: bool = True,
        pos_emb: bool = False,
        attn_implementation: str = "eager",
        **kwargs
    ):
        legacy_pad_id = kwargs.pop("pad_id", None)
        if legacy_pad_id is not None:
            if pad_token_id is not None and legacy_pad_id != pad_token_id:
                raise ValueError(f"Conflicting pad ids: pad_id={legacy_pad_id} vs pad_token_id={pad_token_id}")
            pad_token_id = legacy_pad_id

        self.num_tokens = num_tokens
        self.dim = tuple(dim) if isinstance(dim, list) else dim
        self.depth = tuple(depth) if isinstance(depth, list) else depth 
        self.max_seq_len = tuple(max_seq_len) if isinstance(max_seq_len, list) else max_seq_len
        self.dim_head = dim_head
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_mult = ff_mult
        self.ff_dropout = ff_dropout
        self.rel_pos = rel_pos
        self.pos_emb = pos_emb
        self.attn_implementation = attn_implementation
        self.tie_word_embeddings=kwargs.pop("tie_word_embeddings", False)  # PhaGenModel does not tie input and output embeddings
        kwargs.setdefault("architectures", ["PhaGenModel"])
        
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs
        )
