"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from mask2former.utils.image_utils import get_1d_coords_scale_from_h_w_ps, convert_1d_index_to_2d, convert_2d_index_to_1d, patches_to_images, convert_1d_patched_index_to_2d_org_index

from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class DownSampleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_dim, affine=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        #x = self.instance_norm(x)

        return x


class OverlapDownSample(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        n_layers = int(torch.log2(torch.tensor([patch_size])).item())
        conv_layers = []
        emb_dim_list = [channels] + [embed_dim]*(n_layers-1)
        for i in range(n_layers):
            conv = DownSampleConvBlock(emb_dim_list[i], embed_dim)
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, im):
        x = self.conv_layers(im)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)



class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        n_layers = int(torch.log2(torch.tensor([patch_size])).item())
        conv_layers = []
        emb_dim_list = [channels] + [embed_dim] * (n_layers - 1)
        for i in range(n_layers):
            conv = DownSampleConvBlock(emb_dim_list[i], embed_dim)
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.conv_layers(im).flatten(2).transpose(1, 2)
        return x


class TransformerLayer(nn.Module):
    def __init__(
            self,
            n_blocks,
            dim,
            n_heads,
            dim_ff,
            dropout=0.0,
            drop_path_rate=0.0,
    ):
        super().__init__()

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, dim_ff, dropout, dpr[i]) for i in range(n_blocks)]
        )

    def forward(self, x):
        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x)
        return x


class MixResDecoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_encoder,
        n_heads,
        n_query_tokens,
        dropout=0.0,
        drop_path_rate=0.0,
        split_ratio=4,
        n_scales=2
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_encoder = d_encoder
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_cls = n_cls
        self.scale = d_model ** -0.5
        self.n_query_tokens = n_query_tokens

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_model*4, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, self.n_query_tokens, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(self.n_query_tokens)


        minimum_resolution = image_size[0] // (patch_size // 2**(n_scales-1))

        self.n_patches = minimum_resolution ** 2
        self.pos_embed = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.apply(init_weights)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)


    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)


    def forward(self, x, patch_code):
        H, W = self.image_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        x = x + self.pos_embed(patch_code.float())
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, : -self.n_query_tokens], x[:, -self.n_query_tokens :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.unsqueeze(1)
        masks = rearrange(masks, "b h w c -> b c h w")

        max_scale = patch_code[:, 0].max()
        new_coord_PS = self.patch_size // 2**max_scale
        new_GS = H // new_coord_PS
        org_patch_codes_list = []
        for scale in range(0, max_scale + 1):
            indx_curr_scale = patch_code[:, 0] == scale
            coords_curr_scale = patch_code[indx_curr_scale, 1]
            scale_curr_scale = patch_code[indx_curr_scale, 0]
            org_patch_codes = convert_1d_patched_index_to_2d_org_index(coords_curr_scale, H, self.patch_size, scale, new_coord_PS).cuda()
            org_patch_codes = torch.cat([scale_curr_scale.unsqueeze(1), org_patch_codes], dim=1)
            org_patch_codes_list.append(org_patch_codes)
        patch_code_org = torch.cat(org_patch_codes_list, dim=0).unsqueeze(0)

        masks = patches_to_images(masks, patch_code_org, (new_GS, new_GS))

        return masks


class MixResEncoder(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            n_layers,
            d_model,
            n_heads,
            dropout=0.0,
            drop_path_rate=0.0,
            channels=1,
            split_ratio=4,
            n_scales=2
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbedding(
            image_size,
            patch_size,
            d_model[0],
            channels,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        # Pos Embs
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, d_model[0]))
        self.rel_pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.split_ratio, d_model[i])) for i in range(n_scales - 1)])
        self.scale_embs = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model[i])) for i in range(n_scales - 1)])

        # transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(n_layers[i], d_model[i], n_heads[i], d_model[i] * 4, dropout, drop_path_rate) for i in
             range(len(n_layers))]
        )

        # Downsamplers
        self.downsamplers = nn.ModuleList([nn.Linear(d_model[i], d_model[i + 1]) for i in range(len(n_layers) - 1)])

        # Split layers
        self.splits = nn.ModuleList(
            [nn.Linear(d_model[i], d_model[i] * self.split_ratio) for i in range(len(n_layers))]
        )

        # Metaloss predictions
        self.metalosses = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model[i], d_model[i]),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model[i]),
            nn.Linear(d_model[i], 1)) for i in range(len(n_layers))])

        self.high_res_patchers = nn.ModuleList(
            [nn.Conv2d(channels, d_model[i - 1], kernel_size=patch_size // (2 ** i), stride=patch_size // (2 ** i)) for
             i in
             range(1, len(n_layers))])

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def divide_tokens_to_split_and_keep(self, tokens_at_curr_scale, patches_scale_coords_curr_scale, curr_scale):
        k_split = tokens_at_curr_scale.shape[1] // self.split_ratio
        k_keep = tokens_at_curr_scale.shape[1] - k_split
        pred_meta_loss = self.metalosses[curr_scale](tokens_at_curr_scale).squeeze(2)
        tkv, tki = torch.topk(pred_meta_loss, k=k_split, dim=1, sorted=False)
        bkv, bki = torch.topk(pred_meta_loss, k=k_keep, dim=1, sorted=False, largest=False)

        batch_indices_k = torch.arange(tokens_at_curr_scale.shape[0]).unsqueeze(1).repeat(1, k_keep)
        batch_indices_s = torch.arange(tokens_at_curr_scale.shape[0]).unsqueeze(1).repeat(1, k_split)

        tokens_to_keep = tokens_at_curr_scale[batch_indices_k, bki]
        tokens_to_split = tokens_at_curr_scale[batch_indices_s, tki]
        coords_to_keep = patches_scale_coords_curr_scale[bki].squeeze(0)
        coords_to_split = patches_scale_coords_curr_scale[tki].squeeze(0)

        return tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss

    def divide_tokens_coords_on_scale(self, tokens, patches_scale_coords, curr_scale):
        indx_curr_scale = patches_scale_coords[:, 0] == curr_scale
        indx_old_scales = patches_scale_coords[:, 0] != curr_scale
        coords_at_curr_scale = patches_scale_coords[indx_curr_scale]
        coords_at_older_scales = patches_scale_coords[indx_old_scales]
        tokens_at_curr_scale = tokens[:, indx_curr_scale, :]
        tokens_at_older_scale = tokens[:, indx_old_scales, :]

        return tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales

    def split_tokens(self, tokens_to_split, curr_scale):
        x_splitted = self.splits[curr_scale](tokens_to_split)
        x_splitted = rearrange(x_splitted, 'b n (s d) -> b n s d', s=self.split_ratio)
        x_splitted = x_splitted + self.rel_pos_embs[curr_scale] + self.scale_embs[curr_scale]
        x_splitted = rearrange(x_splitted, 'b n s d -> b (n s) d', s=self.split_ratio)
        return x_splitted

    def split_coords(self, coords_to_split, patch_size, curr_scale):
        new_scale = curr_scale + 1
        new_coord_ratio = self.split_ratio // 2
        two_d_coords = convert_1d_index_to_2d(coords_to_split[:, 1], patch_size)
        a = torch.stack([two_d_coords[:, 0] * new_coord_ratio, two_d_coords[:, 1] * new_coord_ratio])
        b = torch.stack([two_d_coords[:, 0] * new_coord_ratio, two_d_coords[:, 1] * new_coord_ratio + 1])
        c = torch.stack([two_d_coords[:, 0] * new_coord_ratio + 1, two_d_coords[:, 1] * new_coord_ratio])
        d = torch.stack([two_d_coords[:, 0] * new_coord_ratio + 1, two_d_coords[:, 1] * new_coord_ratio + 1])

        new_coords_2dim = torch.stack([a, b, c, d]).permute(2, 0, 1)
        new_coords_2dim = rearrange(new_coords_2dim, 'n s c -> (n s) c', s=self.split_ratio, c=2)
        new_coords_1dim = convert_2d_index_to_1d(new_coords_2dim, patch_size * 2).unsqueeze(1).int()

        scale_lvl = torch.tensor([[new_scale]] * new_coords_1dim.shape[0]).to('cuda').int()
        patches_scale_coords = torch.cat([scale_lvl, new_coords_1dim], dim=1)

        return patches_scale_coords

    def add_high_res_features(self, tokens, coords, curr_scale, image):
        patched_im = self.high_res_patchers[curr_scale](image)
        patched_im = rearrange(patched_im, 'b c h w -> b (h w) c')
        patched_im = patched_im[:, coords]
        tokens = tokens + patched_im

        return tokens

    def split_input(self, tokens, patches_scale_coords, curr_scale, patch_size, im):
        tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales = self.divide_tokens_coords_on_scale(
            tokens, patches_scale_coords, curr_scale)
        meta_loss_coords = coords_at_curr_scale[:, 1]
        tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss = self.divide_tokens_to_split_and_keep(
            tokens_at_curr_scale, coords_at_curr_scale, curr_scale)
        tokens_after_split = self.split_tokens(tokens_to_split, curr_scale)
        coords_after_split = self.split_coords(coords_to_split, patch_size, curr_scale)

        tokens_after_split = self.add_high_res_features(tokens_after_split, coords_after_split[:, 1], curr_scale, im)

        all_tokens = torch.cat([tokens_at_older_scale, tokens_to_keep, tokens_after_split], dim=1)
        all_coords = torch.cat([coords_at_older_scales, coords_to_keep, coords_after_split], dim=0)

        return all_tokens, all_coords, pred_meta_loss, meta_loss_coords

    def forward(self, im):
        B, _, H, W = im.shape
        PS = self.patch_size
        patched_im_size = H // PS
        x = self.patch_embed(im)
        x = x + self.pos_embed

        patches_scale_coords = get_1d_coords_scale_from_h_w_ps(H, W, PS, 0).to('cuda')
        meta_losses = []
        meta_loss_coords = []
        for l_idx in range(len(self.layers)):
            x = self.layers[l_idx](x)
            # print("Current total number of tokens in layer {}: {}".format(blk_idx, x.shape[1]))
            '''
            for s in range(blk_idx + 1):
                indx_scale = patches_scale_coords[:, 0] == s
                coords_at_scale = patches_scale_coords[indx_scale]
                print("Current number of tokens at scale {} in layer {}: {}".format(s, blk_idx, len(coords_at_scale)))
            '''
            if l_idx < self.n_scales - 1:
                x, patches_scale_coords, meta_loss, meta_loss_coord = self.split_input(x, patches_scale_coords, l_idx,
                                                                                       patched_im_size, im)
                PS /= 2
                patched_im_size *= 2
                x = self.downsamplers[l_idx](x)
                meta_losses.append(meta_loss)
                meta_loss_coords.append(meta_loss_coord)

        return x, patches_scale_coords, meta_losses, meta_loss_coords

class MixResBackbone(nn.Module):
    def __init__(self,
        image_size=(512, 512),
        patch_size=128,
        channels=1,
        n_layers_encoder=[2, 2],
        d_encoder=[128, 64],
        n_heads_encoder=[16, 8],
        n_layers_decoder=4,
        d_decoder=64,
        n_heads_decoder=4,
        split_ratio=4,
        n_scales=2,
        n_query_tokens_decoder=32) -> None:
        super().__init__()

        self.encoder = MixResEncoder(image_size=image_size,
                    patch_size=patch_size,
                    channels=channels,
                    n_layers=n_layers_encoder,
                    d_model=d_encoder,
                    n_heads=n_heads_encoder,
                    split_ratio=split_ratio,
                    n_scales=n_scales)

        self.decoder = MixResDecoder(image_size=image_size,
                        patch_size=patch_size,
                        n_layers=n_layers_decoder,
                        d_model=d_decoder,
                        d_encoder=d_encoder[-1],
                        n_heads=n_heads_decoder,
                        n_query_tokens=n_query_tokens_decoder,
                        n_cls=n_cls,
                        split_ratio=split_ratio,
                        n_scales=n_scales)

        self.num_features = d_encoder + [n_query_tokens_decoder]

    def forward(self, im):
        enc_out, patches_scale_coords, meta_losses, meta_loss_coords = self.encoder(im)
        #print(enc_out.shape)
        dec_out = self.decoder(enc_out, patches_scale_coords)

        outs = {'res1': dec_out}

        return outs, meta_losses, meta_loss_coords


@BACKBONE_REGISTRY.register()
class D2MixResTransformer(MixResBackbone, Backbone):
    def __init__(self, cfg, input_shape):
        image_size=cfg.MODEL.MIXRES.IMG_SIZE
        patch_size=cfg.MODEL.MIXRES.PATCH_SIZE
        channels=3,
        n_layers_encoder=cfg.MODEL.MIXRES.ENC_DEPTHS
        d_encoder=cfg.MODEL.MIXRES.ENC_DIMS
        n_heads_encoder=cfg.MODEL.MIXRES.ENC_HEADS
        n_layers_decoder=cfg.MODEL.MIXRES.DEC_DEPTH
        d_decoder=cfg.MODEL.MIXRES.DEC_DIM
        n_heads_decoder=cfg.MODEL.MIXRES.DEC_HEADS
        split_ratio=cfg.MODEL.MIXRES.SPLIT_RATIO
        n_scales=cfg.MODEL.MIXRES.N_SCALES
        n_query_tokens_decoder=cfg.MODEL.MIXRES.DEC_QUERY_TOKENS


        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            n_layers_encoder=n_layers_encoder,
            d_encoder=d_encoder,
            n_heads_encoder=n_heads_encoder,
            n_layers_decoder=n_layers_decoder,
            d_decoder=d_decoder,
            n_heads_decoder=n_heads_decoder,
            split_ratio=split_ratio,
            n_scales=n_scales,
            n_query_tokens_decoder=n_query_tokens_decoder
        )

        self._out_features = ['res1']

        self._out_feature_strides = {
            "res1": 4
        }
        self._out_feature_channels = {
            "res1": self.num_features[-1]
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y, meta_losses, meta_loss_coords = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs, meta_losses, meta_loss_coords

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
