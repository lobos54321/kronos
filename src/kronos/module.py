"""
Kronos Model — Module Components

Vendored from https://github.com/shiyu-coder/Kronos (MIT License)
Original authors: Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang, Jian Li
"""

import math
from einops import rearrange, reduce
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype), 0, zi.flatten(), torch.ones_like(zi.flatten()).to(zq.dtype), 'sum')
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta, input_format='bchw', soft_entropy=True, group_size=9, persample_entropy_compute='analytical', cb_entropy_compute='group', l2_norm=True, inv_temperature=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.input_format = input_format
        assert self.embed_dim % group_size == 0
        self.num_groups = self.embed_dim // group_size
        self.group_size = group_size
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature
        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))
        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim
        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)
        self.soft_entropy = soft_entropy

    def quantize(self, z):
        zhat = torch.where(z > 0, torch.tensor(1, dtype=z.dtype, device=z.device), torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def forward(self, z, collect_metrics=True):
        zq = self.quantize(z)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        zq = zq * q_scale
        if not collect_metrics:
            return zq, zq.new_zeros(()), {}
        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        used_codes = torch.unique(indices, return_counts=False) if not self.training else None
        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))
        return (zq, commit_loss + self.zeta * entropy_penalty / self.inv_temperature, {"H": cb_entropy, "used_codes": used_codes, "indices": indices, "group_indices": group_indices, "avg_prob": avg_prob})

    def soft_entropy_loss(self, z):
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)
        distance = -2 * torch.einsum('... g c, d c -> ... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob_stack = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob_stack, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        avg_prob = reduce(prob, '... g d -> g d', 'mean')
        codebook_entropy_val = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy_val.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        pe = -probs_per_dim * torch.log(probs_per_dim + 1e-8) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        return pe.sum(-1).mean()

    def codes_to_indexes(self, zhat):
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        return torch.remainder(torch.floor_divide(indices, self.basis), 2) * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        group_indices = group_indices.unsqueeze(-1)
        codes = torch.remainder(torch.floor_divide(group_indices, self.group_basis), 2)
        return rearrange(codes, 'b ... g c -> b ... (g c)') * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True) if normalize else count
        return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)


class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size)

    def bits_to_indices(self, bits):
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(0, bits.shape[-1], 1, dtype=torch.long, device=bits.device)
        return (bits * indices).sum(-1)

    def forward(self, z, half=False, collect_metrics=True):
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        if half:
            q_pre = quantized[:, :, :self.s1_bits]
            q_post = quantized[:, :, self.s1_bits:]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)
    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    def _update_cos_sin_cache(self, x, seq_len):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached
    def forward(self, q, k):
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)
    def forward(self, x, key_padding_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, S, -1) if key_padding_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=True)
        return self.resid_dropout(self.out_proj(out.transpose(1, 2).contiguous().view(B, S, self.d_model)))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)
    def forward(self, query, key, value, key_padding_mask=None):
        B, QL, _ = query.shape
        _, SL, _ = key.shape
        q = self.q_proj(query).view(B, QL, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, SL, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, SL, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, QL, -1) if key_padding_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=self.training)
        return self.resid_dropout(self.out_proj(out.transpose(1, 2).contiguous().view(B, QL, self.d_model)))


class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits, self.s2_bits, self.d_model = s1_bits, s2_bits, d_model
        self.emb_s1 = nn.Embedding(2 ** s1_bits, d_model)
        self.emb_s2 = nn.Embedding(2 ** s2_bits, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model ** -0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model ** -0.5)
    def forward(self, token_ids):
        if isinstance(token_ids, (tuple, list)):
            s1_ids, s2_ids = token_ids
        else:
            t = token_ids.long()
            s2_ids = t & ((1 << self.s2_bits) - 1)
            s1_ids = t >> self.s2_bits
        return self.fusion_proj(torch.cat([self.emb_s1(s1_ids) * math.sqrt(self.d_model), self.emb_s2(s2_ids) * math.sqrt(self.d_model)], dim=-1))


class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout)
        self.norm = RMSNorm(d_model)
    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        return self.norm(hidden_states + self.cross_attn(query=sibling_embed, key=hidden_states, value=hidden_states, key_padding_mask=key_padding_mask))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)
    def forward(self, x, key_padding_mask=None):
        x = x + self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask)
        return x + self.ffn(self.norm2(x))


class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        super().__init__()
        self.vocab_s1, self.vocab_s2 = 2 ** s1_bits, 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)
    def forward(self, x):
        return self.proj_s1(x)
    def cond_forward(self, x2):
        return self.proj_s2(x2)
    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        if padding_mask is not None:
            m = (padding_mask == 0)
            ce_s1 = F.cross_entropy(s1_logits[m], s1_targets[m])
            ce_s2 = F.cross_entropy(s2_logits[m], s2_targets[m])
        else:
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        return (ce_s1 + ce_s2) / 2, ce_s1, ce_s2


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        super().__init__()
        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(60, d_model)
        self.hour_embed = Embed(24, d_model)
        self.weekday_embed = Embed(7, d_model)
        self.day_embed = Embed(32, d_model)
        self.month_embed = Embed(13, d_model)
    def forward(self, x):
        x = x.long()
        return self.hour_embed(x[:,:,1]) + self.weekday_embed(x[:,:,2]) + self.day_embed(x[:,:,3]) + self.month_embed(x[:,:,4]) + self.minute_embed(x[:,:,0])
