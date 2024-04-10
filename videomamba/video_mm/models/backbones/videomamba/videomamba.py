# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
import numpy as np
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import logging
logger = logging.getLogger(__name__)


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, output_dim=768, embed_dim=768, norm_layer=nn.LayerNorm):
        super().__init__()

        self.head = nn.Linear(embed_dim, output_dim)
        self.norm = norm_layer(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))
        return x
    

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0) 


class PretrainVideoMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            drop_path_rate=0.,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            pool_type="cls+avg",
            # video
            kernel_size=1, 
            num_frames=8, 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            # clip,
            clip_decoder_embed_dim=768,
            clip_output_dim=512,
            clip_return_layer=1,
            clip_student_return_interval=1,
            add_pool_norm=True,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        logger.info(f'Use checkpoint: {use_checkpoint}')
        logger.info(f'Checkpoint number: {checkpoint_num}')
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        logger.info(f'Student return index: {self.return_index}')
        self.depth = depth
        self.pool_type = pool_type
        logger.info(f"Pool type: {pool_type}")

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # CLIP decoder
        self.clip_decoder = nn.ModuleList([
            Linear_Decoder(
                output_dim=clip_output_dim, 
                embed_dim=clip_decoder_embed_dim, 
                norm_layer=nn.LayerNorm, 
            ) for _ in range(clip_return_layer)
        ])

        self.clip_pos_embed = get_sinusoid_encoding_table(
            num_patches * num_frames // kernel_size + 1, 
            clip_decoder_embed_dim
        )
        self.clip_img_pos_embed = get_sinusoid_encoding_table(
            num_patches + 1, 
            clip_decoder_embed_dim
        )

        self.add_pool_norm = add_pool_norm
        if add_pool_norm:
            self.pool_norm = nn.LayerNorm(embed_dim)

        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, mask=None, use_image=False):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        if not use_image:
            # temporal pos
            cls_tokens = x[:B, :1, :]
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            x = x + self.temporal_pos_embedding
            x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        # mask
        if mask is not None:
            x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        else:
            x_vis = x
        x_clip_vis = []

        # mamba impl
        residual = None
        hidden_states = x_vis
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None
                )
            if (idx - 1) in self.return_index:
                x_clip_vis.append(self.norm(residual.to(dtype=self.norm.weight.dtype))) # share norm for mask

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if (self.depth - 1) in self.return_index:
            x_clip_vis.append(residual)
        
        x_vis = hidden_states
        x_clip_vis = torch.stack(x_clip_vis)

        return x_vis, x_clip_vis

    def forward(self, x, mask=None, use_image=False, keep_temporal=False):
        T = x.shape[2]
        x_vis, x_clip_vis = self.forward_features(x, mask, use_image)  # [B, N_vis, C_e]
        
        # align CLIP:
        if mask is not None and len(x_clip_vis) > 0:
            K, B, _, C_CLIP = x_clip_vis.shape
            if use_image:
                expand_clip_pos_embed = self.clip_img_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
            else:
                expand_clip_pos_embed = self.clip_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
            clip_pos_emd_vis = expand_clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
            x_clip_full = x_clip_vis + clip_pos_emd_vis # [K, B, N, C_d_clip]

            x_clip = []
            for idx, clip_decoder in enumerate(self.clip_decoder):
                x_clip.append(clip_decoder(x_clip_full[idx]))
            x_clip = torch.stack(x_clip) # align and normalize
        else:
            x_clip = None

        if self.add_pool_norm:
            x_vis_cls, x_vis = x_vis[:, :1], x_vis[:, 1:]
            if self.pool_type == "cls": # only return cls token
                x_pool_vis = self.pool_norm(x_vis_cls)
            else:
                if keep_temporal:
                    B, _, C_CLIP = x_vis.shape
                    if self.pool_type == "cls+avg":
                        x_pool_vis = self.pool_norm(x_vis_cls + x_vis.view(B, T, -1, C_CLIP).mean(2))  
                    elif self.pool_type == "cls_cat_avg":
                        x_pool_vis = self.pool_norm(torch.cat([x_vis_cls + x_vis.view(B, T, -1, C_CLIP).mean(2)], dim=1))
                    elif self.pool_type == "avg":
                        x_pool_vis = self.pool_norm(x_vis.view(B, T, -1, C_CLIP).mean(2))
                else:
                    if self.pool_type == "cls+avg":
                        x_pool_vis = self.pool_norm(x_vis_cls + x_vis.mean(1, keepdim=True))
                    elif self.pool_type == "cls_cat_avg":
                        x_pool_vis = self.pool_norm(torch.cat([x_vis_cls, x_vis.mean(1, keepdim=True)], dim=1))
                    elif self.pool_type == "avg":
                        x_pool_vis = self.pool_norm(x_vis.mean(1, keepdim=True))
            
            return x_vis, x_pool_vis, x_clip
        else:
            return x_vis, x_clip


def load_state_dict(pretrained_path, model, ckpt_num_frame, num_frames):
    logger.info(f"Loading pretrained weights from {pretrained_path}")
    checkpoint_model = torch.load(pretrained_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in checkpoint_model:
            checkpoint_model = checkpoint_model[model_key]
            logger.info(f"Load state_dict by model_key = {model_key}")
            break

    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
    num_patches = model.patch_embed.num_patches # 
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        logger.info("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # B, L, C -> B, H, W, C -> B, C, H, W
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # B, C, H, W -> B, H, W, C ->  B, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size, new_size, embedding_size) 
        pos_tokens = pos_tokens.flatten(1, 2) # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
    
    # we use 8 frames for pretraining
    temporal_pos_embed = checkpoint_model['temporal_pos_embedding']
    orig_t_size = ckpt_num_frame // model.patch_embed.tubelet_size
    new_t_size = num_frames // model.patch_embed.tubelet_size
    # height (== width) for the checkpoint position embedding
    if orig_t_size != new_t_size:
        logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        temporal_pos_embed = torch.nn.functional.interpolate(
            temporal_pos_embed, size=(new_t_size,), mode='linear', align_corners=False
        )
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        checkpoint_model['temporal_pos_embedding'] = temporal_pos_embed

    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)


def build_videomamba(config, add_pool_norm=True):
    model = PretrainVideoMamba(
        img_size=config.vision_encoder.img_size, 
        patch_size=config.vision_encoder.patch_size, 
        depth=config.vision_encoder.depth, 
        embed_dim=config.vision_encoder.embed_dim, 
        drop_path_rate=config.vision_encoder.drop_path_rate, 
        ssm_cfg=config.vision_encoder.ssm_cfg,
        norm_epsilon=config.vision_encoder.norm_epsilon,
        fused_add_norm=config.vision_encoder.fused_add_norm,
        rms_norm=config.vision_encoder.rms_norm,
        residual_in_fp32=config.vision_encoder.residual_in_fp32,
        bimamba=config.vision_encoder.bimamba,
        pool_type=config.vision_encoder.pool_type,
        kernel_size=config.vision_encoder.kernel_size,
        num_frames=config.vision_encoder.num_frames,
        use_checkpoint=config.vision_encoder.use_checkpoint,
        checkpoint_num=config.vision_encoder.checkpoint_num,
        clip_decoder_embed_dim=config.vision_encoder.clip_decoder_embed_dim,
        clip_output_dim=config.vision_encoder.clip_output_dim,
        clip_return_layer=config.vision_encoder.clip_return_layer,
        clip_student_return_interval=config.vision_encoder.clip_student_return_interval,
        add_pool_norm=add_pool_norm,
    )
    model.default_cfg = _cfg()
    if config.vision_encoder.pretrained is not None:
        load_state_dict(
            pretrained_path=config.vision_encoder.pretrained,
            model=model,
            ckpt_num_frame=config.vision_encoder.get('ckpt_num_frame', -1), 
            num_frames=config.vision_encoder.num_frames,
        )
    else:
        logger.info("No pretrained weights!!!")
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8

    config = {
        'vision_encoder':
            {
            "img_size": 224, 
            "patch_size": 16, 
            "depth": 32, 
            "embed_dim": 576, 
            "drop_path_rate": 0.,
            "ssm_cfg": None, 
            "norm_epsilon": 1e-5, 
            "fused_add_norm": True,
            "rms_norm": True, 
            "residual_in_fp32": True,
            "bimamba": True,
            "pool_type": "cls",
            "kernel_size": 1,
            "num_frames": 8, 
            "use_checkpoint": False,
            "checkpoint_num": 0,
            "clip_decoder_embed_dim": 576,
            "clip_output_dim": 512,
            "clip_return_layer": 1,
            "clip_student_return_interval": 1,
            "pretrained": "your_model_path/videomamba_m16_k400_mask_pt_f8_res224.pth",
        }
    }
    from easydict import EasyDict
    model = build_videomamba(EasyDict(config)).cuda()

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, 224, 224))
    # s = time.time()
    # logger.info(flop_count_table(flops, max_depth=1))
    # logger.info(time.time()-s)
    mask_token = num_frames * int(14 * 14 * 0.8)
    mask = torch.cat([
        torch.zeros(1, num_frames * 14 * 14 + 1 - mask_token),
        torch.ones(1, mask_token),
    ], dim=-1).to(torch.bool).cuda()
    logger.info(model(torch.rand(1, 3, num_frames, 224, 224).cuda(), mask)[1].shape)