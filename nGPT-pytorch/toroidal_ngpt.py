import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from functools import partial
from einops import rearrange, einsum
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

# Import the nGPT components you might want to build upon
try:
    from nGPT_pytorch import NormLinear, Scale, Attention, FeedForward
except ImportError:
    print("nGPT_pytorch not found in path. Using standalone implementation.")
    # You can then implement the necessary components here

# Helper functions
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# Toroidal normalization function
def toroidal_norm(
    t,
    dim = -1,
    R = 2.0,  # Major radius
    r = 1.0,  # Minor radius
    eps = 1e-10,
    groups = 1
):
    """
    Normalize vectors to lie on a torus instead of a sphere.
    This is a theoretical implementation and may need refinement.
    
    Args:
        t: Input tensor
        dim: Dimension to normalize along
        R: Major radius of the torus
        r: Minor radius of the torus
        eps: Small value for numerical stability
        groups: Number of groups to split the embeddings into
    
    Returns:
        Tensor normalized to lie (conceptually) on a torus
    """
    if groups > 1:
        t = t.chunk(groups, dim=dim)
        t = torch.stack(t)
    
    # First, get the magnitude of the vector
    norm = torch.norm(t, dim=dim, keepdim=True)
    
    # Normalize to unit sphere first
    sphere_points = t / norm.clamp(min=eps)
    
    # We need to project unit sphere to torus
    # In a real implementation, we'd map this properly
    # For now, we'll use a heuristic approach
    
    # Split the dimensions into two groups conceptually representing
    # the two circles of the torus
    half_dim = t.shape[dim] // 2
    
    if dim == -1 or dim == t.dim() - 1:
        first_half = sphere_points[..., :half_dim]
        second_half = sphere_points[..., half_dim:]
        
        # Scale the first half to represent the "major circle"
        first_half = first_half * R
        
        # Scale the second half to represent the "minor circle"
        second_half = second_half * r
        
        # Combine back
        toroidal_points = torch.cat([first_half, second_half], dim=dim)
    else:
        # For other dimensions, we'd need to implement slicing appropriately
        # This is a simplified version
        indices = torch.arange(t.shape[dim])
        first_half_indices = indices < half_dim
        second_half_indices = ~first_half_indices
        
        first_half = sphere_points.index_select(dim, torch.where(first_half_indices)[0]) * R
        second_half = sphere_points.index_select(dim, torch.where(second_half_indices)[0]) * r
        
        # We need to combine these back according to the original ordering
        toroidal_points = torch.zeros_like(sphere_points)
        toroidal_points.index_copy_(dim, torch.where(first_half_indices)[0], first_half)
        toroidal_points.index_copy_(dim, torch.where(second_half_indices)[0], second_half)
    
    # Renormalize to maintain constant magnitude
    # (optional, depending on what behavior you want)
    toroidal_points = toroidal_points / torch.norm(toroidal_points, dim=dim, keepdim=True).clamp(min=eps)
    
    if groups > 1:
        toroidal_points = torch.cat([*toroidal_points], dim=dim)
    
    return toroidal_points

# For use with parametrize
class ToroidalNorm(nn.Module):
    def __init__(self, dim=-1, R=2.0, r=1.0, groups=1):
        super().__init__()
        self.dim = dim
        self.R = R
        self.r = r
        self.groups = groups
    
    def forward(self, t):
        return toroidal_norm(t, dim=self.dim, R=self.R, r=self.r, groups=self.groups)

# Modified NormLinear for Toroidal space
class ToroidalLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in = True,
        parametrize = True,
        R = 2.0,
        r = 1.0,
        groups = 1
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)
        
        self.scale = groups ** -1
        self.parametrize = parametrize
        self.toroidal_norm = ToroidalNorm(dim=-1 if norm_dim_in else 0, R=R, r=r, groups=groups)
        
        if parametrize:
            register_parametrization(
                self.linear,
                'weight',
                self.toroidal_norm
            )
        
        self.norm_weights_()
    
    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original
            
            original.copy_(normed)
        else:
            self.weight.copy_(self.toroidal_norm(self.weight))
    
    @property
    def weight(self):
        return self.linear.weight
    
    def forward(self, x):
        return self.linear(x) * self.scale

# Toroidal Residual
class ToroidalResidual(nn.Module):
    def __init__(
        self,
        fn: nn.Module,
        dim: int,
        init: float,
        scale: float = None,
        R = 2.0,
        r = 1.0,
        groups = 1
    ):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim ** -0.5))
        self.toroidal_norm = partial(toroidal_norm, R=R, r=r, groups=groups)
    
    def forward(self, x, **kwargs):
        residual = x
        
        branch_out = self.fn(x, **kwargs)
        
        is_tuple_output = isinstance(branch_out, tuple)
        
        if is_tuple_output:
            branch_out, *rest = branch_out
        
        # Apply toroidal normalization
        branch_out = self.toroidal_norm(branch_out)
        
        # Use linear interpolation with learned scale
        # This could be replaced with a more toroidal-specific interpolation
        out = self.toroidal_norm(residual.lerp(branch_out, self.branch_scale()))
        
        if is_tuple_output:
            out = (out, *rest)
        
        return out

# Toroidal Attention - modified from original Attention
class ToroidalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        norm_qk = True,
        causal = True,
        manual_norm_weights = False,
        s_qk_init = 1.,
        s_qk_scale = None,
        R = 2.0,
        r = 1.0,
        groups = 1,
        flash_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True,
            enable_cudnn = True
        )
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        
        # Use toroidal versions of modules
        ToroidalLinear_ = partial(ToroidalLinear, parametrize=not manual_norm_weights, R=R, r=r, groups=groups)
        self.toroidal_norm = partial(toroidal_norm, R=R, r=r, groups=groups)
        
        dim_sqrt = dim ** 0.5
        self.dim_sqrt = dim_sqrt
        self.attn_scale = dim_head ** 0.5
        
        dim_inner = dim_head * heads
        self.to_q = ToroidalLinear_(dim, dim_inner)
        self.to_k = ToroidalLinear_(dim, dim_inner)
        self.to_v = ToroidalLinear_(dim, dim_inner)
        
        # Flash attention related context manager
        from torch.nn.attention import SDPBackend
        SDP_BACKEND_MAP = dict(
            enable_flash = SDPBackend.FLASH_ATTENTION,
            enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
            enable_math = SDPBackend.MATH,
            enable_cudnn = SDPBackend.CUDNN_ATTENTION
        )
        sdpa_backends = [SDP_BACKEND_MAP[enable_str] for enable_str, enable in flash_kwargs.items() if enable]
        self.sdpa_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
        
        # qk rmsnorm + scale
        self.norm_qk = norm_qk
        self.qk_scale = Scale(dim_inner, s_qk_init, default(s_qk_scale, dim ** -1))
        
        self.split_heads = nn.Sequential(*[
            nn.Unflatten(1, (1, -1)),
            nn.Unflatten(2, (heads, dim_head)),
            nn.Flatten(1, 2),
        ])
        
        self.merge_heads = nn.Sequential(*[
            nn.Unflatten(1, (-1, 1)),
            nn.Flatten(2, 3),
        ])
        
        self.to_out = ToroidalLinear_(dim_inner, dim, norm_dim_in=False)
    
    def forward(
        self,
        x,
        mask = None,
        rotary_embed = None,
        value_residual = None,
        return_values = False
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        
        # Split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        # Maybe value residual
        if exists(value_residual):
            v = 0.5 * (v + value_residual)
        
        # Rotary positions
        if exists(rotary_embed):
            q = rotary_embed.rotate_queries_or_keys(q)
            k = rotary_embed.rotate_queries_or_keys(k)
        
        # Maybe query key norm
        if self.norm_qk:
            q, k = map(self.toroidal_norm, (q, k))
        
        # Scaling queries and keys
        q = q * rearrange(self.qk_scale(), '(h d) -> h 1 d', h=self.heads)
        
        # For non-autoregressive masking
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
        
        # Scale is sqrt(dk)
        with self.sdpa_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                is_causal=self.causal,
                scale=self.attn_scale
            )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if not return_values:
            return out
        
        return out, v

# Toroidal nGPT
class ToroidalGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_norm_qk = True,
        ff_expand_factor = 4.,
        ce_ignore_index = -1,
        manual_norm_weights = False,
        tied_embedding = False,
        R = 2.0,  # Major radius
        r = 1.0,  # Minor radius
        groups = 1,
        causal = True,
        add_value_residual = True,
        # Scale hyperparameters
        alpha_init: float = None,
        s_logit_init: float = 1.,
        s_logit_scale = None,
        alpha_attn_init = None,
        alpha_attn_scale = None,
        alpha_ff_init = None,
        alpha_ff_scale = None,
        s_qk_init = 1.,
        s_qk_scale = None,
        s_ff_hidden_init = 1.,
        s_ff_hidden_scale = 1.,
        s_ff_gate_init = 1.,
        s_ff_gate_scale = 1.,
        attn_flash_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        ToroidalLinear_ = partial(ToroidalLinear, parametrize=not manual_norm_weights, R=R, r=r, groups=groups)
        self.toroidal_norm = partial(toroidal_norm, R=R, r=r, groups=groups)
        
        self.dim = dim
        self.causal = causal
        self.add_value_residual = add_value_residual
        alpha_init = default(alpha_init, 1. / depth)
        
        self.token_embed = ToroidalLinear_(dim, num_tokens)
        
        # For positional encoding
        from rotary_embedding_torch import RotaryEmbedding
        self.rotary_embed = RotaryEmbedding(dim_head)
        
        # Setup layers
        self.layers = nn.ModuleList([])
        
        # Handle scale hyperparameters
        def cast_tuple(t, length=1):
            out = t if isinstance(t, tuple) else ((t,) * length)
            assert len(out) == length
            return out
        
        scale_hparams = (
            alpha_attn_init,
            alpha_attn_scale,
            alpha_ff_init,
            alpha_ff_scale,
            s_qk_init,
            s_qk_scale,
            s_ff_hidden_init,
            s_ff_hidden_scale,
            s_ff_gate_init,
            s_ff_gate_scale
        )
        
        scale_hparams = tuple(cast_tuple(hparam, depth) for hparam in scale_hparams)
        
        for (
            alpha_attn_init_,
            alpha_attn_scale_,
            alpha_ff_init_,
            alpha_ff_scale_,
            s_qk_init_,
            s_qk_scale_,
            s_ff_hidden_init_,
            s_ff_hidden_scale_,
            s_ff_gate_init_,
            s_ff_gate_scale_
        ) in zip(*scale_hparams):
            
            attn = ToroidalAttention(
                dim,
                dim_head=dim_head,
                heads=heads,
                causal=causal,
                norm_qk=attn_norm_qk,
                manual_norm_weights=manual_norm_weights,
                s_qk_init=s_qk_init_,
                s_qk_scale=s_qk_scale_,
                R=R,
                r=r,
                groups=groups,
                flash_kwargs=attn_flash_kwargs
            )
            
            # Here we would adapt the FeedForward to be toroidal
            # For now, we'll assume FeedForward is still usable with our ToroidalLinear
            ff = FeedForward(
                dim,
                expand_factor=ff_expand_factor,
                manual_norm_weights=manual_norm_weights,
                s_hidden_init=s_ff_hidden_init_,
                s_hidden_scale=s_ff_hidden_scale_,
                s_gate_init=s_ff_gate_init_,
                s_gate_scale=s_ff_gate_scale_,
                norm_eps=0.,
                num_hyperspheres=groups
            )
            
            attn_with_residual = ToroidalResidual(
                attn,
                dim,
                default(alpha_attn_init_, alpha_init),
                default(alpha_attn_scale_, dim ** -0.5),
                R=R,
                r=r,
                groups=groups
            )
            
            ff_with_residual = ToroidalResidual(
                ff,
                dim,
                default(alpha_ff_init_, alpha_init),
                default(alpha_ff_scale_, dim ** -0.5),
                R=R,
                r=r,
                groups=groups
            )
            
            self.layers.append(nn.ModuleList([attn_with_residual, ff_with_residual]))
        
        self.to_logits = ToroidalLinear_(dim, num_tokens) if not tied_embedding else None
        
        self.logit_scale = Scale(num_tokens, s_logit_init, default(s_logit_scale, dim ** -0.5))
        
        self.ignore_index = ce_ignore_index
    
    @torch.no_grad()
    def norm_weights_(self):
        """Normalize all weights to lie on the torus"""
        for module in self.modules():
            if not isinstance(module, ToroidalLinear):
                continue
            
            module.norm_weights_()
    
    def register_step_post_hook(self, optimizer):
        """Register a hook to normalize weights after optimizer steps"""
        assert hasattr(optimizer, 'register_step_post_hook')
        
        def hook(*_):
            self.norm_weights_()
        
        return optimizer.register_step_post_hook(hook)
    
    def forward(
        self,
        ids,
        mask=None,
        return_loss=False
    ):
        token_embed, rotary_embed = self.token_embed.weight, self.rotary_embed
        
        if return_loss:
            assert self.causal
            ids, labels = ids[:, :-1], ids[:, 1:]
        
        tokens = token_embed[ids]
        
        first_values = None
        
        for attn, ff in self.layers:
            tokens, values = attn(tokens, mask=mask, rotary_embed=rotary_embed, return_values=True, 
                                value_residual=first_values if self.add_value_residual else None)
            
            first_values = default(first_values, values)
            
            tokens = ff(tokens)
        
        if exists(self.to_logits):
            logits = self.to_logits(tokens)
        else:
            # tied embeddings
            logits = einsum(tokens, token_embed, 'b n d, c d -> b n c')
        
        logits = logits * self.logit_scale()
        
        if not return_loss:
            return logits
        
        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index=self.ignore_index
        )
        
        return loss