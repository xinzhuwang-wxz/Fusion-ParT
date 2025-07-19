''' Fusion Particle Transformer
modify from
https://github.com/hqucms/weaver-core
https://www.youtube.com/watch?v=gY4-vVRTkpk
'''

import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from weaver.utils.logger import _logger


class CusMultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention module with optional bias and optional gating.
    """

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):
        """
        Initializes the module. MultiHeadAttention theoretically consists of
        N_head separate linear layers for the query, key and value embeddings.
        However, the embeddings can be computed jointly and split afterwards,
        so we only need one query, key and value layer with larger c_out.

        Args:
            c_in (int): Input dimension for the embeddings.
            c (int): Embedding dimension for each individual head.
            N_head (int): Number of heads.
            attn_dim (int): The dimension in the input tensor along which
                the attention mechanism is performed.
            gated (bool, optional): If True, an additional sigmoid-activated
                linear layer will be multiplicated against the weighted
                value vectors before feeding them through the output layer.
                Defaults to False.
            is_global (bool, optional): If True, global calculation will be performed.
                For global calculation, key and value embeddings will only use one head,
                and the q query vectors will be averaged to one query vector.
                Defaults to False.
            use_bias_for_embeddings (bool, optional): If True, query,
                key, and value embeddings will use bias, otherwise not.
                Defaults to False.
        """
        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global

        self.linear_q = nn.Linear(c_in, c * N_head, bias=use_bias_for_embeddings)

        c_kv = c if is_global else c * N_head
        self.linear_k = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)
        self.linear_v = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)

        self.linear_o = nn.Linear(c * N_head, c_in)

        if gated:
            self.linear_g = nn.Linear(c_in, c * N_head)

    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Splits the embeddings into individual heads and transforms the input
        shapes of form (*, q/k/v, *, N_head*c) into the shape
        (*, N_head, q/k/v, c). The position of the q/k/v dimension
        in the original tensors is given by attn_dim.

        Args:
            q (torch.Tensor): Query embedding of shape (*, q, *, N_head*c).
            k (torch.Tensor): Key embedding of shape (*, k, *, N_head*c).
            v (torch.Tensor): Value embedding of shape (*, v, *, N_head*c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, q/k/v, c) respectively.
        """

        # Transposing to [*, q/k/v, N_head*c]
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        # Unwrapping to [*, q/k/v, N_head, c]
        q_shape = q.shape[:-1] + (self.N_head, -1)
        k_shape = k.shape[:-1] + (self.N_head, -1)
        v_shape = v.shape[:-1] + (self.N_head, -1)

        q = q.view(q_shape)
        k = k.view(k_shape)
        v = v.view(v_shape)

        # Transposing to [*, N_head, q/k/v, c]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        return q, k, v

    def prepare_qkv_global(self, q, k, v):
        """
        Prepares the query, key and value embeddings with the following
        differences to the non-global version:
            - key and value embeddings use only one head.
            - the query vectors are contracted into one, average query vector.


        Args:
            q (torch.tensor): Query embeddings of shape (*, q, *, N_head*c).
            k (torch.tensor): Key embeddings of shape (*, k, *, c).
            v (torch.tensor): Value embeddings of shape (*, v, *, c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v.
        """
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        q_shape = q.shape[:-1] + (self.N_head, self.c)
        q = q.view(q_shape)

        q = q.transpose(-2, -3)
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        q = torch.mean(q, dim=-2, keepdim=True)

        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):
        """
        Forward pass through the MultiHeadAttention module.

        Args:
            x (torch.tensor): Input tensor of shape (*, q/k/v, *, c_in).
            bias (torch.tensor, optional): Optional bias tensor of shape
                (*, N_head, q, k) that will be added to the attention weights.
                Defaults to None.
            attention_mask (torch.tensor, optional): Optional attention mask
                of shape (*, k). If set, the keys with value 0 in the mask will
                not be attended to.

        Returns:
            torch.tensor: Output tensor of shape (*, q/k/v, *, c_in)
        """

        out = None

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.is_global:
            q, k, v = self.prepare_qkv_global(q, k, v)
        else:
            q, k, v = self.prepare_qkv(q, k, v)

        q = q / math.sqrt(self.c)

        a = torch.einsum('...qc,...kc->...qk', q, k)
        if bias is not None:
            bias_batch_shape = bias.shape[:-3]
            bias_bc_shape = bias_batch_shape + (1,) * (a.ndim - len(bias_batch_shape) - 3) + bias.shape[-3:]
            bias = bias.view(bias_bc_shape)

            a = a + bias

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            offset = (attention_mask == 0) * -1e8
            a = a + offset

        a = torch.softmax(a, dim=-1)
        # o has shape [*, N_head, q, c]
        o = torch.einsum('...qk,...kc->...qc', a, v)
        o = o.transpose(-3, -2)
        o = torch.flatten(o, start_dim=-2)
        o = o.moveaxis(-2, self.attn_dim)
        if self.gated:
            g = torch.sigmoid(self.linear_g(x))
            o = g * o

        out = self.linear_o(o)

        return out

class SharedDropout(nn.Module):
    """
    A module for dropout, that is shared along one dimension,
    i.e. for dropping out whole rows or columns.
    """
    def __init__(self, shared_dim: int, p: float):
        super().__init__()

        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x: torch.tensor):
        """
        Forward pass for shared dropout. The dropout mask is broadcasted along
        the shared dimension.

        Args:
            x (torch.tensor): Input tensor of arbitrary shape.

        Returns:
            torch.tensor: Output tensor of the same shape as x.
        """


        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)

        out = x * mask

        return out

class DropoutRowwise(SharedDropout):
    def __init__(self, p: float, kind: str):
        if kind == "atten":
            super().__init__(shared_dim=-2, p=p)
        elif kind == "mul":
            super().__init__(shared_dim=-2, p=p)
        else:
            raise ValueError(f"Unknown kind of dropout: {kind}. Use 'atten' or 'mul'.")


class DropoutColumnwise(SharedDropout):
    def __init__(self, p: float, kind: str):
        if kind == "atten":
            super().__init__(shared_dim=-1, p=p)
        elif kind == "mul":
            super().__init__(shared_dim=-3, p=p)
        else:
            raise ValueError(f"Unknown kind of dropout: {kind}. Use 'atten' or 'mul'.")



@torch.jit.script
def delta_phi(a, b):  #
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)  # x[:, :2] is (px, py), [N,2]; .sum(dim=1, keepdim=True) [N,2] -> [N,1]
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2  # [N,1]


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)  # x[:, 3:4] is E, x[:, :3] is (px, py, pz)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2  # [N,1]


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)  # shape of px is (N, 1, ...)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)  # [N, 4]


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    """
    xi shape: (N, 4, ...), dim1 : (px, py, pz, E)
    for num_outputs=4,
        outputs: (N, 4, ...), dim1 : (pt, rap, phi, m) for normal

    """
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:  #
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):  # used to trim the sequence length of input particles

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)  # C is the number of features, P is the number of particles, N is the batch size
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):  # 注意，x_input shape is (batch, embed_dim, seq_len)
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len) ，BatchNorm1d expects
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        """

        pairwise_lv_dim: number of pairwise features from lv (e.g. pt, rap, phi, m), lv means lorentz vector, i.e. px, py, pz, E
        pairwise_input_dim : extra number of pairwise features, if not, set to 0
        use_pre_activation_pair : whether to use pre-activation for pairwise features,
                                    if True, the last layer of pairwise features will be removed, prevent activation from interfering with message delivery



        """
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)  # partial function to avoid passing num_outputs,eps,for_onnx every time
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        """
        x: (batch, v_dim, seq_len)
        uu: (batch, v_dim, seq_len, seq_len)

        input shape [batch, v_dim, seq_len] or [batch, v_dim, seq_len, seq_len]
        output shape [batch, out_dim, seq_len, seq_len]

        """
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)  # repeat means from [batch, dim, seq_len] to [batch, dim, seq_len, seq_len]
                    xi = x[:, :, i, j]  # [batch, 4, i, j]
                    xj = x[:, :, j, i]  # [batch, 4, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y  # (batch, out_dim, seq_len, seq_len)

##### related to fusion
class RowAttentionWithInteractionBias(nn.Module):
    """
    Implementation of a row attention mechanism with interaction bias.
    """
    def __init__(self, c_p, c_i, c=32, N_head=8):
        """
        Initializes the RowAttentionWithPairBias module.
        Args:
            c_p (int): Particle embedding dimension.
            c_i (int): interaction embedding dimension.
            c (int, optional): embedding dimension. If not provided, defaults to 32
            N_head (int, optional): Number of attention heads. If not provided, defaults to 8
        """
        super().__init__()

        self.layer_norm_p = nn.LayerNorm(c_p)
        self.layer_norm_i = nn.LayerNorm(c_i)
        self.linear_i = nn.Linear(c_i, N_head, bias=False)
        self.mha = CusMultiHeadAttention(c_p, c, N_head, attn_dim=-2, gated=True)

        assert c_p % N_head == 0, f"c_p {c_p} must be divisible by N_head {N_head}"

    def forward(self, p, i):
            """
            Forward pass through the RowAttentionWithInteractionBias module.
            Args:
                p (torch.Tensor): Particle embeddings of shape (seq_len, batch, embed_dim).
                i (torch.Tensor): Interaction embeddings of shape (batch, embed_dim, seq_len, seq_len)

            Returns:
                torch.Tensor: Output tensor (seq_len, batch, embed_dim)
            """
            # switch to the correct shape
            p = p.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
            # print(f"i before permute: {i.shape}")
            i = i.permute(0, 2, 3, 1)  # (batch, seq_len, seq_len, embed_dim)
            # print(f"i after permute: {i.shape}")

            p = self.layer_norm_p(p)
            bias = self.linear_i(self.layer_norm_i(i))  # (batch, seq_len, seq_len, N_head)
            bias = bias.permute(0, 3, 1, 2)  # (batch, N_head, seq_len, seq_len)
            out = self.mha(p, bias=bias)  # (batch, seq_len, embed_dim)
            out = out.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
            

            return out

class ColAttention(nn.Module):
    """
    Implementation of a column attention mechanism.
    """
    def __init__(self, c_p, c=32, N_head=8):
        """
        Initializes the ColAttention module.
        Args:
            c_p (int): Particle embedding dimension.
            c (int, optional): embedding dimension. If not provided, defaults to 64
            N_head (int, optional): Number of attention heads. If not provided, defaults to 8
        """
        super().__init__()
        self.layer_norm_p = nn.LayerNorm(c_p)
        self.mha = CusMultiHeadAttention(c_p, c, N_head, attn_dim=-3, gated=True)

    def forward(self, p):
        """
        Forward pass through the ColAttention module.
        Args:
            p (torch.Tensor): after RawAttentionWithInteractionBias, shape (seq_len, batch, embed_dim)

        Returns:
            torch.Tensor: Output tensor (seq_len, batch, embed_dim)
        """
        p = p.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        p = self.layer_norm_p(p)  # (batch, seq_len, embed_dim)
        out = self.mha(p)  # (batch, seq_len, embed_dim)
        out = out.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        return out  # (seq_len, batch, embed_dim)

class ParticleTransition(nn.Module):
    """
    Implementation of a particle transition module.
    """
    def __init__(self, c_p, factor=4):
        """
        Initializes the ParticleTransition module.
        Args:
            c_p (int): Particle embedding dimension.
            factor (int, optional): Factor to multiply the input dimension. If not provided, defaults to 4
        """
        super().__init__()
        self.layer_norm_p = nn.LayerNorm(c_p)
        self.linear1 = nn.Linear(c_p, c_p * factor)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(c_p * factor, c_p)

    def forward(self, p):
        """
        Forward pass through the ParticleTransition module.
        Args:
            p (torch.Tensor): After ColAttention, shape (seq_len, batch, embed_dim)

        Returns:
            torch.Tensor: Output tensor (seq_len, batch, embed_dim)
        """

        out = self.linear2(self.relu(self.linear1(self.layer_norm_p(p))))
        return out

class OuterProductMean(nn.Module):
    """
    Implementation of an outer product mean module.
    """
    def __init__(self, c_p, c_i, c=32):
        """
        Initializes the OuterProductMean module.
        Args:
            c_p (int): Particle embedding dimension.
            c_i (int): Interaction embedding dimension.
            c (int, optional): embedding dimension. If not provided, defaults to 32
        """
        super().__init__()
        self.layer_norm_p = nn.LayerNorm(c_p)
        self.layer_1 = nn.Linear(c_p, c)
        self.layer_2 = nn.Linear(c_p, c)
        self.linear_out = nn.Linear(c*c, c_i)

    def forward(self, p):
        """
        Forward pass through the OuterProductMean module.
        Args:
            p (torch.Tensor): After ParticleTransition, shape (seq_len, batch, embed_dim).

        Returns:
            torch.Tensor: Output tensor (1, embed_dim, seq_len, seq_len)
        """
        Batch = p.size(-2)

        p = self.layer_norm_p(p)  # (seq_len, batch, embed_dim)
        p1 = self.layer_1(p).permute(-2 , -3, -1)  # (batch, seq_len, c)
        p2 = self.layer_2(p).permute(-2, -3, -1)  # (batch, seq_len, c)
        o = torch.einsum('...ic, ...jd -> ...ijcd', p1, p2)  # (batch, seq_len, seq_len, c, c)
        o = torch.flatten(o, start_dim=-2)  # (batch, seq_len, seq_len, c*c)
        o = self.linear_out(o)/Batch  # (batch, seq_len, seq_len, c_i)
        out = o.permute(0, 3, 1, 2)  # (batch, c_i, seq_len, seq_len)

        return out
# class OuterProductMean(nn.Module):
#     """
#     Implementation of an outer product mean module.
#     """
#     def __init__(self, c_p, c_i, c=32):
#         """
#         Initializes the OuterProductMean module.
#         Args:
#             c_p (int): Particle embedding dimension.
#             c_i (int): Interaction embedding dimension.
#             c (int, optional): embedding dimension. If not provided, defaults to 32
#         """
#         super().__init__()
#         self.layer_norm_p = nn.LayerNorm(c_p)
#         self.layer_1 = nn.Linear(c_p, c)
#         self.layer_2 = nn.Linear(c_p, c)
#         self.linear_out = nn.Linear(c*c, c_i)
#
#     def forward(self, p):
#         """
#         Forward pass through the OuterProductMean module.
#         Args:
#             p (torch.Tensor): After ParticleTransition, shape (seq_len, batch, embed_dim).
#
#         Returns:
#             torch.Tensor: Output tensor (1, embed_dim, seq_len, seq_len)
#         """
#         Batch = p.size(-2)
#
#         p = self.layer_norm_p(p)  # (seq_len, batch, embed_dim)
#         p1 = self.layer_1(p).permute(-2 , -3, -1)  # (batch, seq_len, c)
#         p2 = self.layer_2(p).permute(-2, -3, -1)  # (batch, seq_len, c)
#         o = torch.einsum('bic, bjd -> ijcd', p1, p2)  # (seq_len, seq_len, c, c)
#         o = torch.flatten(o, start_dim=-2)  # (seq_len, seq_len, c*c)
#         o = self.linear_out(o)/Batch  # (seq_len, seq_len, c_i)
#         out = o.permute(2, 0, 1)  # (c_i, seq_len, seq_len)
#
#         out = out.unsqueeze(0)  # (1, c_i, seq_len, seq_len)
#
#         return out

class TriangleMultiplication(nn.Module):
    """
    Implementation of triangle update using outgoing/incoming edges module.
    """
    def __init__(self, c_i, mult_type, c=32):
        """
        Initializes the TriangleMultiplication module.
        Arge:
            c_i (int): Interaction embedding dimension.
            mult_type (str): Type of multiplication to perform, either 'outgoing' or 'incoming'.
            c (int, optional): embedding dimension. If not provided, defaults to 32
        """

        super().__init__()
        if mult_type not in ['outgoing', 'incoming']:
            raise ValueError("mult_type must be either 'outgoing' or 'incoming'")
        self.mult_type = mult_type
        self.layer_norm_in = nn.LayerNorm(c_i)
        self.layer_norm_out = nn.LayerNorm(c)
        self.linear_a_p = nn.Linear(c_i, c)
        self.linear_a_g = nn.Linear(c_i, c)
        self.linear_b_p = nn.Linear(c_i, c)
        self.linear_b_g = nn.Linear(c_i, c)
        self.linear_g = nn.Linear(c_i, c_i)
        self.linear_i = nn.Linear(c, c_i)

    def forward(self, i):
        """
        Forward pass through the TriangleMultiplication module.
        Args:
            i (torch.Tensor): After OuterProductMean, shape (batch, c_i, seq_len, seq_len). same as Interaction embeddings.

        Returns:
            torch.Tensor: Output tensor (batch, c_i, seq_len, seq_len)
        """
        i = i.permute(0, 2, 3, 1)  # (batch, seq_len, seq_len, c_i)
        i = self.layer_norm_in(i)  # (batch, seq_len, seq_len, c_i)
        a = torch.sigmoid(self.linear_a_g(i)) * self.linear_a_p(i)  # (batch, seq_len, seq_len, c)
        b = torch.sigmoid(self.linear_b_g(i)) * self.linear_b_p(i)  # (batch, seq_len, seq_len, c)
        g = torch.sigmoid(self.linear_g(i))  # (batch, seq_len, seq_len, c_i)

        if self.mult_type == 'outgoing':
            i = torch.einsum('...ikc, ...jkc -> ...ijc', a, b)
        else:
            i = torch.einsum('...kic, ...kjc -> ...ijc', a, b)
        out = g * self.linear_i(self.layer_norm_out(i))  # (batch, seq_len, seq_len, c_i)
        out = out.permute(0, 3, 1, 2)  # (batch, c_i, seq_len, seq_len)

        return out

class TraingleAttention(nn.Module):
    """
    Implementation of triangular attention around starting/ending node module.
    """
    def __init__(self, c_i, node_type, c=32, N_head=4):
        """
        Initializes the TraingleAttention module.
        Args:
            c_i (int): Interaction embedding dimension.
            node_type (str): Type of node to perform attention on, either 'starting_node' or 'ending_node'.
            c (int, optional): embedding dimension. If not provided, defaults to 32
            N_head (int, optional): Number of attention heads. If not provided, defaults to 4
        """
        super().__init__()
        if node_type not in ['starting_node', 'ending_node']:
            raise ValueError("node_type must be either 'starting_node' or 'ending_node' but is {}".format(node_type))
        self.node_type = node_type
        self.layer_norm = nn.LayerNorm(c_i)
        if node_type == 'starting_node':
            attn_dim = -2
        else:
            attn_dim = -3

        self.mha = CusMultiHeadAttention(c_i, c, N_head, attn_dim=attn_dim, gated=True)
        self.linear = nn.Linear(c_i, N_head, bias=False)

    def forward(self, i):
            """
            Forward pass through the TraingleAttention module.
            Args:
                i (torch.Tensor): After TriangleMultiplication, shape # (batch, c_i, seq_len, seq_len)

            Returns:
                torch.Tensor: Output tensor (batch, c_i, seq_len, seq_len)
            """
            i = i.permute(0, 2, 3, 1)  # (batch, seq_len, seq_len, c_i)
            i = self.layer_norm(i)
            bias = self.linear(i)  # (batch, seq_len, seq_len, N_head)
            bias = bias.moveaxis(-1, -3)  # (batch, N_head， seq_len, seq_len)
            if self.node_type == "ending_node":
                bias = bias.transpose(-1, -2)

            out = self.mha(i, bias)  # (batch, seq_len, seq_len, c_i)
            out = out.permute(0, 3, 1, 2)

            return out  # (batch, c_i, seq_len, seq_len)

class InteractionTransition(nn.Module):
    """
    Implementation of interaction transition module.
    """
    def __init__(self, c_i, factor=4):
        """
        Initializes the InteractionTransition module.
        Args:
            c_i (int): Interaction embedding dimension.
            factor (int, optional): Factor to multiply the input dimension. If not provided, defaults to 4
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_i)
        self.linear1 = nn.Linear(c_i, c_i * factor)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(c_i * factor, c_i)

    def forward(self, i):
        """
        Forward pass through the InteractionTransition module.
        Args:
            i (torch.Tensor): After TraingleAttention, shape (batch, c_i, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor (batch, c_i seq_len, seq_len)
        """
        i = i.permute(0, 2, 3, 1)  # (batch, seq_len, seq_len, c_i)
        i = self.layer_norm(i)  # (batch, seq_len, seq_len, c_i)
        out = self.linear2(self.relu(self.linear1(i)))
        out = out.permute(0, 3, 1, 2)

        return out

class InteractionStack(nn.Module):
    """
    Implementation of interaction stack module.
    """
    def __init__(self, c_i):
        """
        Initializes the InteractionStack module.
        Args:
            c_i (int): Interaction embedding dimension.
        """
        super().__init__()
        self.dropout_rowwise = DropoutRowwise(p=0.25, kind="atten")
        self.dropout_colwise = DropoutColumnwise(p=0.25, kind="atten")
        self.tri_mul_out = TriangleMultiplication(c_i, 'outgoing')
        self.tri_mul_in = TriangleMultiplication(c_i, 'incoming')
        self.tri_att_start = TraingleAttention(c_i, 'starting_node')
        self.tri_att_end = TraingleAttention(c_i, 'ending_node')
        self.interaction_transition = InteractionTransition(c_i)

    def forward(self, i):
        """
        Forward pass through the InteractionStack module.
        Args:
            i (torch.Tensor): Interaction embeddings of shape (batch, c_i, seq_len, seq_len).
        Returns:
            torch.Tensor: Output tensor (batch, c_i, seq_len, seq_len)
        """
        # print("[Debug] In InteractionStack forward, i shape: ", i.shape)
        i = i + self.dropout_rowwise(self.tri_mul_out(i))  # (batch, c_i, seq_len, seq_len)
        i = i + self.dropout_rowwise(self.tri_mul_in(i))  # (batch, c_i, seq_len, seq_len)

        i = i + self.dropout_rowwise(self.tri_att_start(i))  # (batch, c_i, seq_len, seq_len)
        i = i + self.dropout_colwise(self.tri_att_end(i))
        i = i + self.interaction_transition(i)  # (batch, c_i, seq_len, seq_len)
        # print("[Debug] After InteractionStack forward, i shape: ", i.shape)

        return i  # (batch, c_i, seq_len, seq_len)

class FusionFormerBlock(nn.Module):
    """
    Implementation of a FusionFormer block.
    """
    def __init__(self, c_p, c_i):
        """
        Initializes the FusionFormerBlock module.
        Args:
            c_p (int): Particle embedding dimension.  shape is (seq_len, batch, c_p)
            c_i (int): Interaction embedding dimension.  shape is (batch, c_i, seq_len, seq_len)
        """
        super().__init__()
        self.dropout_rowwise_p = DropoutRowwise(p=0.15, kind="mul")
        self.row_att = RowAttentionWithInteractionBias(c_p, c_i)
        self.col_att = ColAttention(c_p)
        self.particle_transition = ParticleTransition(c_p)
        self.outer_product_mean = OuterProductMean(c_p, c_i)
        self.core = InteractionStack(c_i)

    def forward(self, p, i):
        """
        Forward pass through the FusionFormerBlock module.
        Args:
            p (torch.Tensor): Particle embeddings of shape (seq_len, batch, embed_dim).
            i (torch.Tensor): Interaction embeddings of shape (batch, embed_dim, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor p and i
        """
        # print("[Debug] In FusionFormerBlock forward, p shape: ", p.shape, " i shape: ", i.shape)

        p = p + self.dropout_rowwise_p(self.row_att(p, i))  # (seq_len, batch, embed_dim)
        p = p + self.col_att(p)   # (seq_len, batch, embed_dim)
        p = self.particle_transition(p)  # (seq_len, batch, embed_dim)

        i = i + self.outer_product_mean(p)  # (1, embed_dim, seq_len, seq_len)

        # print("[Debug] check input core i shape: ", i.shape)
        i = self.core(i)
        # print("[Debug] check output core i shape: ", i.shape)
        # print("[Debug] After FusionFormerBlock forward, p shape: ", p.shape, " i shape: ", i.shape)

        return p, i  # (seq_len, batch, embed_dim), (batch, embed_dim, seq_len, seq_len)

class FusionFormerStack(nn.Module):
    """
    Implementation of a FusionFormer stack.
    """
    def __init__(self, c_p, c_i, num_block):
        """
        Initializes the FusionFormerStack module.
        Args:
            c_p (int): Particle embedding dimension.
            c_i (int): Interaction embedding dimension.
            num_block (int): Number of FusionFormer blocks to stack.
        """
        super().__init__()
        self.fusionformer_blocks = nn.ModuleList([FusionFormerBlock(c_p, c_i) for _ in range(num_block)])

    def forward(self, p, i):
        """
        Forward pass through the FusionFormerStack module.
        Args:
            p (torch.Tensor): Particle embeddings of shape (seq_len, batch, embed_dim).
            i (torch.Tensor): Interaction embeddings of shape (batch, embed_dim, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor p and i
        """
        for block in self.fusionformer_blocks:
            block(p, i)

        return p, i




#######
class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None  # c_atten means class attention
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.
            attn_mask (ByteTensor, optional): binary

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class FusionParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_fusion_blocks=4,
                 num_heads=8,
                 num_layers=4,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)  #
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.fusion = FusionFormerStack(
            c_p=embed_dim,
            c_i=cfg_block['num_heads'],
            num_block=num_fusion_blocks
        )
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)  N means batch size, C means number of features, P means number of particles
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            # attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                interaction_embed = self.pair_embed(v, uu)  # (N, num_heads, P, P)
                # attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # fusion
            # print('~~~~~~~~~~~~~~before fusion x shape:', x.shape)
            # print('~~~~~~~~~~~~~~x shape:', x.shape)
            # print('~~~~~~~~~~~~~~~Interaction embedding shape:', interaction_embed.shape)

            x, attn_ = self.fusion(x, interaction_embed)
            attn_mask = attn_.view(-1, x.size(0), x.size(0)) if attn_ is not None else None  # (N*num_heads, P, P)
            # print('~~~~~~~~~~~~~~after fusion x shape:', x.shape)
            # print('~~~~~~~~~~~~~~attn_mask shape:', attn_mask.shape if attn_mask is not None else None)
            # print('~~~~~~~~~~~~~~padding_mask shape:', padding_mask.shape)
            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output

"""
main function 
"""
class FusionParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = FusionParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_fusion_blocks=4,
        num_heads=8,
        num_layers=4,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = FusionParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


# class ParticleTransformerTagger(nn.Module):
#
#     def __init__(self,
#                  pf_input_dim,
#                  sv_input_dim,
#                  num_classes=None,
#                  # network configurations
#                  pair_input_dim=4,
#                  pair_extra_dim=0,
#                  remove_self_pair=False,
#                  use_pre_activation_pair=True,
#                  embed_dims=[128, 512, 128],
#                  pair_embed_dims=[64, 64, 64],
#                  num_heads=8,
#                  num_layers=8,
#                  num_cls_layers=2,
#                  block_params=None,
#                  cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
#                  fc_params=[],
#                  activation='gelu',
#                  # misc
#                  trim=True,
#                  for_inference=False,
#                  use_amp=False,
#                  **kwargs) -> None:
#         super().__init__(**kwargs)
#
#         self.use_amp = use_amp
#
#         self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
#         self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
#
#         self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
#         self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)
#
#         self.part = ParticleTransformer(input_dim=embed_dims[-1],
#                                         num_classes=num_classes,
#                                         # network configurations
#                                         pair_input_dim=pair_input_dim,
#                                         pair_extra_dim=pair_extra_dim,
#                                         remove_self_pair=remove_self_pair,
#                                         use_pre_activation_pair=use_pre_activation_pair,
#                                         embed_dims=[],
#                                         pair_embed_dims=pair_embed_dims,
#                                         num_heads=num_heads,
#                                         num_layers=num_layers,
#                                         num_cls_layers=num_cls_layers,
#                                         block_params=block_params,
#                                         cls_block_params=cls_block_params,
#                                         fc_params=fc_params,
#                                         activation=activation,
#                                         # misc
#                                         trim=False,
#                                         for_inference=for_inference,
#                                         use_amp=use_amp)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'part.cls_token', }
#
#     def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
#         # x: (N, C, P)
#         # v: (N, 4, P) [px,py,pz,energy]
#         # mask: (N, 1, P) -- real particle = 1, padded = 0
#
#         with torch.no_grad():
#             pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
#             sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
#             v = torch.cat([pf_v, sv_v], dim=2)
#             mask = torch.cat([pf_mask, sv_mask], dim=2)
#
#         with torch.cuda.amp.autocast(enabled=self.use_amp):
#             pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
#             sv_x = self.sv_embed(sv_x)
#             x = torch.cat([pf_x, sv_x], dim=0)
#
#             return self.part(x, v, mask)
#
#
# class ParticleTransformerTaggerWithExtraPairFeatures(nn.Module):
#
#     def __init__(self,
#                  pf_input_dim,
#                  sv_input_dim,
#                  num_classes=None,
#                  # network configurations
#                  pair_input_dim=4,
#                  pair_extra_dim=0,
#                  remove_self_pair=False,
#                  use_pre_activation_pair=True,
#                  embed_dims=[128, 512, 128],
#                  pair_embed_dims=[64, 64, 64],
#                  num_heads=8,
#                  num_layers=8,
#                  num_cls_layers=2,
#                  block_params=None,
#                  cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
#                  fc_params=[],
#                  activation='gelu',
#                  # misc
#                  trim=True,
#                  for_inference=False,
#                  use_amp=False,
#                  **kwargs) -> None:
#         super().__init__(**kwargs)
#
#         self.use_amp = use_amp
#         self.for_inference = for_inference
#
#         self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
#         self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
#
#         self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
#         self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)
#
#         self.part = ParticleTransformer(input_dim=embed_dims[-1],
#                                         num_classes=num_classes,
#                                         # network configurations
#                                         pair_input_dim=pair_input_dim,
#                                         pair_extra_dim=pair_extra_dim,
#                                         remove_self_pair=remove_self_pair,
#                                         use_pre_activation_pair=use_pre_activation_pair,
#                                         embed_dims=[],
#                                         pair_embed_dims=pair_embed_dims,
#                                         num_heads=num_heads,
#                                         num_layers=num_layers,
#                                         num_cls_layers=num_cls_layers,
#                                         block_params=block_params,
#                                         cls_block_params=cls_block_params,
#                                         fc_params=fc_params,
#                                         activation=activation,
#                                         # misc
#                                         trim=False,
#                                         for_inference=for_inference,
#                                         use_amp=use_amp)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'part.cls_token', }
#
#     def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
#         # x: (N, C, P)
#         # v: (N, 4, P) [px,py,pz,energy]
#         # mask: (N, 1, P) -- real particle = 1, padded = 0
#
#         with torch.no_grad():
#             if not self.for_inference:
#                 if pf_uu_idx is not None:
#                     pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.size(-1))
#
#             pf_x, pf_v, pf_mask, pf_uu = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
#             sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
#             v = torch.cat([pf_v, sv_v], dim=2)
#             mask = torch.cat([pf_mask, sv_mask], dim=2)
#             uu = torch.zeros(v.size(0), pf_uu.size(1), v.size(2), v.size(2), dtype=v.dtype, device=v.device)
#             uu[:, :, :pf_x.size(2), :pf_x.size(2)] = pf_uu
#
#         with torch.cuda.amp.autocast(enabled=self.use_amp):
#             pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
#             sv_x = self.sv_embed(sv_x)
#             x = torch.cat([pf_x, sv_x], dim=0)
#
#             return self.part(x, v, mask, uu)
