import torch
import torch.nn as nn
from .transformer import LayerNorm, TextTransformer
from .tokenizer import tokenize


class CLIP_text(nn.Module):
    # stolen text encoder from EVA02_CLIP_E_psz14_plus_s9B
    def __init__(self):
        super().__init__()
        text = TextTransformer(
            context_length=77,
            vocab_size=49408,
            width=1280,
            heads=20,
            layers=32,
            ls_init_value=None,
            output_dim=768, # use 768 for alignment
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
            xattn=False,
            attn_mask=True,
        )
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
    
    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = tokenize
    
    def forward(self, text):
        text = self.tokenizer(text)
        return text