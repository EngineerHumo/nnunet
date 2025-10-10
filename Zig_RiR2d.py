from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


@torch.jit.script
def _exp_clamped(tensor: torch.Tensor) -> torch.Tensor:
    clamp_min, clamp_max = -60.0, 60.0
    return torch.exp(torch.clamp(tensor, min=clamp_min, max=clamp_max))



up_kwargs = {'mode': 'bilinear', 'align_corners': True}
T_MAX = 512*64
from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["./cuda/wkv_op.cpp", "./cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def _wkv_fallback_scan_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    first: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    B = k.size(0)
    T = k.size(1)
    C = k.size(2)

    state_num = torch.zeros((B, C), dtype=k.dtype, device=k.device)
    state_den = torch.zeros((B, C), dtype=k.dtype, device=k.device)
    outputs = torch.empty((B, T, C), dtype=k.dtype, device=k.device)

    for t in range(T):
        k_t = k[:, t, :]
        v_t = v[:, t, :]
        exp_k = _exp_clamped(k_t)

        numerator = state_num + first * exp_k * v_t
        denominator = state_den + first * exp_k

        outputs[:, t, :] = numerator / (denominator + eps)

        state_num = state_num * decay + exp_k * v_t
        state_den = state_den * decay + exp_k

    return outputs


@torch.jit.script
def _wkv_fallback_scan(
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    first: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    B = k.size(0)
    T = k.size(1)
    C = k.size(2)

    exp_k = _exp_clamped(k)

    decay = decay.view(decay.size(0), 1, decay.size(1)).expand(B, T, C)
    first = first.view(first.size(0), 1, first.size(1)).expand(B, T, C)

    ones = torch.ones((B, 1, C), dtype=k.dtype, device=k.device)
    log_decay = torch.log(decay)
    decay_cumsum = torch.cumsum(log_decay, dim=1)
    decay_cumprod = torch.exp(decay_cumsum)
    decay_prefix = torch.cat((ones, decay_cumprod[:, :-1, :]), dim=1)
    decay_prefix_inv = torch.reciprocal(decay_prefix)

    weighted_v = exp_k * v

    state_num = torch.cumsum(weighted_v * decay_prefix_inv, dim=1) * decay_prefix
    state_den = torch.cumsum(exp_k * decay_prefix_inv, dim=1) * decay_prefix

    zero_state = torch.zeros((B, 1, C), dtype=k.dtype, device=k.device)
    state_num_prev = torch.cat((zero_state, state_num[:, :-1, :]), dim=1)
    state_den_prev = torch.cat((zero_state, state_den[:, :-1, :]), dim=1)

    numerator = state_num_prev + first * weighted_v
    denominator = state_den_prev + first * exp_k

    return numerator / (denominator + eps)


_wkv_fallback_scan_validated = False


def _validate_wkv_fallback_scan() -> None:
    global _wkv_fallback_scan_validated
    if _wkv_fallback_scan_validated:
        return
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return

    generator = torch.Generator()
    generator.manual_seed(0)
    B, T, C = 2, 4, 3
    k = torch.randn(B, T, C, generator=generator)
    v = torch.randn(B, T, C, generator=generator)
    decay = torch.rand(1, C, generator=generator)
    first = torch.rand(1, C, generator=generator)
    eps = float(torch.finfo(k.dtype).eps)

    ref = _wkv_fallback_scan_reference(k, v, decay, first, eps)
    vec = _wkv_fallback_scan(k, v, decay, first, eps)

    max_diff = torch.max(torch.abs(ref - vec))
    if bool(max_diff > 1e-5):
        raise RuntimeError("_wkv_fallback_scan vectorization mismatch: max diff {}".format(float(max_diff)))

    _wkv_fallback_scan_validated = True


_validate_wkv_fallback_scan()


def _run_wkv_fallback(B, T, C, w, u, k, v):
    device = k.device
    dtype = k.dtype

    w = w.to(device=device, dtype=dtype)
    u = u.to(device=device, dtype=dtype)
    k = k.to(dtype=dtype)
    v = v.to(dtype=dtype)

    decay = _exp_clamped(-w).unsqueeze(0)
    first = _exp_clamped(u).unsqueeze(0)

    eps = float(torch.finfo(dtype).eps)

    return _wkv_fallback_scan(k, v, decay, first, eps)


def RUN_CUDA(B, T, C, w, u, k, v):
    in_onnx_export = False
    if hasattr(torch.onnx, 'is_in_onnx_export'):
        in_onnx_export = torch.onnx.is_in_onnx_export()
    if (not k.is_cuda) or in_onnx_export or torch.jit.is_tracing():
        return _run_wkv_fallback(B, T, C, w, u, k, v)
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1 / 4):
    assert 0 <= gamma <= 1 / 4
    if shift_pixel <= 0:
        return input.clone()

    B, C, H, W = input.shape
    device = input.device

    ratios = input.new_tensor([gamma, gamma * 2, gamma * 3, gamma * 4], dtype=torch.float32)
    channel_count = ratios.new_full((), float(C))
    boundaries = torch.floor(ratios * channel_count).to(torch.long)
    boundaries = torch.clamp(boundaries, min=0, max=C)
    boundaries = torch.cummax(boundaries, dim=0)[0]
    boundaries = torch.cat([boundaries, boundaries.new_tensor([C])])

    idx = torch.arange(C, device=device, dtype=boundaries.dtype)
    masks = (
        idx < boundaries[0],
        (idx >= boundaries[0]) & (idx < boundaries[1]),
        (idx >= boundaries[1]) & (idx < boundaries[2]),
        (idx >= boundaries[2]) & (idx < boundaries[3]),
        idx >= boundaries[3],
    )

    output = torch.zeros_like(input)

    def shift_width(src, direction):
        if direction == "right":
            padded = F.pad(src, (shift_pixel, 0, 0, 0))
            return padded[..., :W]
        padded = F.pad(src, (0, shift_pixel, 0, 0))
        return padded[..., shift_pixel:]

    def shift_height(src, direction):
        if direction == "down":
            padded = F.pad(src, (0, 0, shift_pixel, 0))
            return padded[..., :H, :]
        padded = F.pad(src, (0, 0, 0, shift_pixel))
        return padded[..., shift_pixel:, :]

    output[:, masks[0], :, :] = shift_width(input[:, masks[0], :, :], "right")
    output[:, masks[1], :, :] = shift_width(input[:, masks[1], :, :], "left")
    output[:, masks[2], :, :] = shift_height(input[:, masks[2], :, :], "down")
    output[:, masks[3], :, :] = shift_height(input[:, masks[3], :, :], "up")
    output[:, masks[4], :, :] = input[:, masks[4], :, :]

    return output



class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False,
                 scan_schemes=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.device = None
        self.recurrence = 2
        self.scan_schemes = scan_schemes or [('top-left', 'horizontal'), ('bottom-right', 'vertical')]
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
        self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

    def get_zigzag_indices(self, h, w, start='top-left', direction='horizontal'):
        device = self.device if self.device is not None else torch.device('cpu')
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)

        if direction == 'horizontal':
            row_order = rows
            if start in ('bottom-left', 'bottom-right'):
                row_order = torch.flip(row_order, dims=(0,))

            base_cols = cols
            reverse_cols = torch.flip(base_cols, dims=(0,))
            if start in ('top-left', 'bottom-left'):
                even_cols = base_cols
                odd_cols = reverse_cols
            else:
                even_cols = reverse_cols
                odd_cols = base_cols

            even_cols_matrix = even_cols.unsqueeze(0).expand(h, -1)
            odd_cols_matrix = odd_cols.unsqueeze(0).expand(h, -1)
            is_even_row = (row_order % 2 == 0).unsqueeze(1)
            cols_matrix = torch.where(is_even_row, even_cols_matrix, odd_cols_matrix)
            rows_matrix = row_order.unsqueeze(1).expand(-1, w)
        else:
            col_order = cols
            if start in ('top-right', 'bottom-right'):
                col_order = torch.flip(col_order, dims=(0,))

            base_rows = rows
            reverse_rows = torch.flip(base_rows, dims=(0,))
            if start in ('top-left', 'top-right'):
                even_rows = base_rows
                odd_rows = reverse_rows
            else:
                even_rows = reverse_rows
                odd_rows = base_rows

            even_rows_matrix = even_rows.unsqueeze(0).expand(w, -1)
            odd_rows_matrix = odd_rows.unsqueeze(0).expand(w, -1)
            is_even_col = (col_order % 2 == 0).unsqueeze(1)
            rows_per_col = torch.where(is_even_col, even_rows_matrix, odd_rows_matrix)
            cols_per_col = col_order.unsqueeze(1).expand(-1, h)

            rows_matrix = rows_per_col.transpose(0, 1)
            cols_matrix = cols_per_col.transpose(0, 1)

        indices = rows_matrix * w + cols_matrix
        return indices.reshape(-1).to(dtype=torch.long)


    def jit_func(self, x, resolution, scan_scheme):
        h, w = resolution
        start, direction = scan_scheme
        zigzag_order = self.get_zigzag_indices(h, w, start=start, direction=direction)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)

        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x[..., zigzag_order]
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=h, w=w)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v


    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        selected_scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        sr, k, v = self.jit_func(x, resolution, selected_scheme)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:

                h, w = resolution
                new_h, new_w = (h, w) if selected_scheme[1] == 'horizontal' else (w, h)
                zigzag_order = self.get_zigzag_indices(new_h, new_w, start=selected_scheme[0],
                                                       direction=selected_scheme[1])
                k = rearrange(k, 'b (h w) c -> b c h w', h=h, w=w)
                k = rearrange(k, 'b c h w -> b c (h w)')[..., zigzag_order]
                k = rearrange(k, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
                v = rearrange(v, 'b c h w -> b c (h w)')[..., zigzag_order]
                v = rearrange(v, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (h w) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)


    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x



class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, layer_id, outer_head, inner_head, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = VRWKV_SpatialMix(n_embd=inner_dim, n_layer=None, layer_id=layer_id)
            self.inner_norm2 = norm_layer(num_words * inner_dim)
            self.inner_ffn = VRWKV_ChannelMix(n_embd=inner_dim, n_layer=None, layer_id=None)
            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = VRWKV_SpatialMix(n_embd=outer_dim, n_layer=None, layer_id=layer_id)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_ffn = VRWKV_ChannelMix(n_embd=outer_dim, n_layer=None, layer_id=1)



    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in):
        B, N, C = outer_tokens.size()
        if self.has_inner:
            inner_patch_resolution = [H_in, W_in]
            x = x + self.drop_path(self.inner_attn(self.inner_norm1(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1), inner_patch_resolution))
            x = x + self.drop_path(self.inner_ffn(self.inner_norm2(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1), inner_patch_resolution))
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B, N, -1))))
        outer_patch_resolution = [H_out, W_out]
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), outer_patch_resolution))
        outer_tokens = outer_tokens + self.drop_path(self.outer_ffn(self.outer_norm2(outer_tokens), outer_patch_resolution))
        return x, outer_tokens


class PatchMerging2D_sentence(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        _, _, new_h, new_w = x.shape
        x = x.reshape(B, -1, new_h * new_w).transpose(1, 2)
        return x, new_h, new_w


class PatchMerging2D_word(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)

        pad_h = H_out % 2
        pad_w = W_out % 2
        x = x.permute(0, 3, 4, 5, 1, 2)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 4, 5, 1, 2, 3)

        x1 = x[:, 0::2, 0::2, :, :, :]
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4)
        x = x.reshape(-1, 2*H_in, 2*W_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x


class Stem(nn.Module):
    def __init__(self, img_size=224, in_chans=1, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 16

        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 4, inner_dim * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)



    def forward(self, x):
        B, C, H, W = x.shape

        x = self.common_conv(x)

        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4

        inner_tokens = self.inner_convs(x)
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2)
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in * W_in).transpose(1, 2)


        outer_tokens = self.outer_convs(x)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


class Stage(nn.Module):
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, layer_id=j, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[j], act_layer=act_layer, norm_layer=norm_layer,
                se=se, sr_ratio=sr_ratio))

        self.blocks = nn.ModuleList(blocks)


    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        return inner_tokens, outer_tokens


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.gelu1 = nn.GELU()
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x


class PyramidRiR_enc(nn.Module):
    def __init__(self, img_size=512, outer_dims=None, in_chans=1, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        depths = [2, 4, 9, 2]
        inner_dims = [4, 4 * 2, 4 * 4, 4 * 8]
        outer_heads = [2, 2 * 2, 2 * 4, 2 * 8]
        inner_heads = [1, 1 * 2, 1 * 4, 1 * 8]
        sr_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.num_features = outer_dims[-1]


        self.patch_embed = Stem(img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        self.pos_embed_sentence = nn.Parameter(torch.zeros(1, num_patches, outer_dims[0]))
        self.pos_embed_word = nn.Parameter(torch.zeros(1, num_words, inner_dims[0]))
        self.interpolate_mode = 'bicubic'


        depth = 0
        self.word_merges = nn.ModuleList([])
        self.sentence_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                self.word_merges.append(PatchMerging2D_word(inner_dims[i - 1], inner_dims[i]))
                self.sentence_merges.append(PatchMerging2D_sentence(outer_dims[i-1], outer_dims[i]))
            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                                     outer_head=outer_heads[i], inner_head=inner_heads[i],
                                     num_patches=num_patches // (2 ** i) // (2 ** i), num_words=num_words,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[depth:depth + depths[i]], norm_layer=norm_layer, se=se,
                                     sr_ratio=sr_ratios[i])
                               )
            depth += depths[i]

        self.up_blocks = nn.ModuleList([])
        for i in range(4):
            self.up_blocks.append(UpsampleBlock(outer_dims[i], outer_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def forward_features(self, x):
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        outputs = []

        for i in range(4):
            if i > 0:
                inner_tokens = self.word_merges[i - 1](inner_tokens, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out = self.sentence_merges[i - 1](outer_tokens, H_out, W_out)
            inner_tokens, outer_tokens = self.stages[i](inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
            b, l, m = outer_tokens.shape
            mid_out = outer_tokens.reshape(b, H_out, W_out, m).permute(0, 3, 1, 2)
            mid_out = self.up_blocks[i](mid_out)
            outputs.append(mid_out)
        return outputs


    def forward(self, x):
        x = self.forward_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x


class ZRiR(nn.Module):
    def __init__(self, channels, num_classes=None, img_size=None, in_chans=3):
        super(ZRiR, self).__init__()

        self.RiR_backbone = PyramidRiR_enc(img_size=img_size, outer_dims=channels, in_chans=in_chans)
        self.decode4 = Decoder(channels[3], channels[2])
        self.decode3 = Decoder(channels[2], channels[1])
        self.decode2 = Decoder(channels[1], channels[0])
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Conv2d(channels[0], num_classes, kernel_size=1, bias=False))

    def forward(self, x):
        _, _, hei, wid = x.shape
        outputs = self.RiR_backbone(x)
        t1, t2, t3, t4 = outputs[0], outputs[1], outputs[2], outputs[3]
        d4 = self.decode4(t4, t3)
        d3 = self.decode3(d4, t2)
        d2 = self.decode2(d3, t1)
        out = self.decode0(d2)

        return out
