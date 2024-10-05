{
  "hidden_size": 128,
  "z_channels": 4,
  "hidden_size_mult": [1, 2, 4, 4],
  "attn_resolutions": [],
  "dropout": 0.0,
  "resolution": 256,
  "double_z": True,
  "embed_dim": 4,
  "num_res_blocks": 2,
  "q_conv": "CausalConv3d",
  "encoder_conv_in": "Conv2d",
  "encoder_conv_out": "CausalConv3d",
  "encoder_attention": "AttnBlock3DFix",
  # Encoder Downsample self.vae.down
  "in_ch_mult": (1, 1, 2, 4, 4),
  "hidden_size_mult": [1, 2, 4, 4],
  "encoder_resnet_blocks": [
    "ResnetBlock2D",
    "ResnetBlock2D",
    "ResnetBlock3D",
    "ResnetBlock3D",
  ],
  "encoder_spatial_downsample": [
    "Downsample",
    "Spatial2xTime2x3DDownsample",
    "Spatial2xTime2x3DDownsample",
    "",
  ],
  "encoder_temporal_downsample": ["", "", "", ""],
  "encoder_mid_resnet": "ResnetBlock3D",
  "decoder_conv_in": "CausalConv3d",
  "decoder_conv_out": "CausalConv3d",
  "decoder_attention": "AttnBlock3DFix",
  "decoder_resnet_blocks": [
    "ResnetBlock3D",
    "ResnetBlock3D",
    "ResnetBlock3D",
    "ResnetBlock3D",
  ],
  "decoder_spatial_upsample": [
    "",
    "SpatialUpsample2x",
    "Spatial2xTime2x3DUpsample",
    "Spatial2xTime2x3DUpsample",
  ],
  "decoder_temporal_upsample": ["", "", "", ""],
  "decoder_mid_resnet": "ResnetBlock3D",
  "tile_latent_min_size": 32,
  "tile_latent_min_size_t": 16,
  "tile_overlap_factor": 0.125,
  "tile_sample_min_size": 256,
  "tile_sample_min_size_t": 33,
  "training": False,
  "use_quant_layer": True,
  "use_tiling": True,
}

# %% tiled_encode2d(self, x, return_moments=False)
# (Pdb) self.tile_sample_min_size
# 256
# (Pdb) self.tile_latent_min_size
# 32
# (Pdb) overlap_size
# 224
# (Pdb) blend_extent
# 4
# (Pdb) row_limit
# 28


# %% Encoder
{
  "hidden_size_mult": [1, 2, 4, 4],
  "in_ch_mult": (1, 1, 2, 4, 4),
  "num_res_blocks": 2,
  "num_resolutions": 4,
  "resolution": 256,
  "training": False,
}

# Encoder(
#   (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (down): ModuleList(
#     (0): Module(
#       (block): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
#           (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#       (attn): ModuleList()
#       (downsample): Downsample(
#         (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
#       )
#     )
#     (1): Module(
#       (block): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
#           (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
#           (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#       (attn): ModuleList()
#       (downsample): Spatial2xTime2x3DDownsample(
#         (conv): CausalConv3d(
#           (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2))
#           (pad): ReplicationPad2d((0, 0, 2, 0))
#         )
#       )
#     )
#     (2): Module(
#       (block): ModuleList(
#         (0): ResnetBlock3D(
#           (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
#           (conv1): CausalConv3d(
#             (conv): Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 2, 0))
#           )
#           (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): CausalConv3d(
#             (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 2, 0))
#           )
#           (nin_shortcut): CausalConv3d(
#             (conv): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 0, 0))
#           )
#         )
#         (1): ResnetBlock3D(
#           (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
#           (conv1): CausalConv3d(
#             (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 2, 0))
#           )
#           (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): CausalConv3d(
#             (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 2, 0))
#           )
#         )
#       )
#       (attn): ModuleList()
#       (downsample): Spatial2xTime2x3DDownsample(
#         (conv): CausalConv3d(
#           (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2))
#           (pad): ReplicationPad2d((0, 0, 2, 0))
#         )
#       )
#     )
#     (3): Module(
#       (block): ModuleList(
#         (0-1): 2 x ResnetBlock3D(
#           (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
#           (conv1): CausalConv3d(
#             (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 2, 0))
#           )
#           (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): CausalConv3d(
#             (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#             (pad): ReplicationPad2d((0, 0, 2, 0))
#           )
#         )
#       )
#       (attn): ModuleList()
#     )
#   )
#   (mid): Module(
#     (block_1): ResnetBlock3D(
#       (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
#       (conv1): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 2, 0))
#       )
#       (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
#       (dropout): Dropout(p=0.0, inplace=False)
#       (conv2): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 2, 0))
#       )
#     )
#     (attn_1): AttnBlock3DFix(
#       (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
#       (q): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 0, 0))
#       )
#       (k): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 0, 0))
#       )
#       (v): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 0, 0))
#       )
#       (proj_out): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 0, 0))
#       )
#     )
#     (block_2): ResnetBlock3D(
#       (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
#       (conv1): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 2, 0))
#       )
#       (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
#       (dropout): Dropout(p=0.0, inplace=False)
#       (conv2): CausalConv3d(
#         (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#         (pad): ReplicationPad2d((0, 0, 2, 0))
#       )
#     )
#   )
#   (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
#   (conv_out): CausalConv3d(
#     (conv): Conv3d(512, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
#     (pad): ReplicationPad2d((0, 0, 2, 0))
#   )
# )

# ConvIn
# Shape bconf_in: torch.Size([1, 3, 33, 256, 256])
# Shape bconf_out: torch.Size([1, 128, 33, 256, 256])

# Downsample
# Level: 0
# Block: 0
# Shape Block0_out: torch.Size([1, 128, 33, 256, 256])
# Block: 1
# Shape Block1_out: torch.Size([1, 128, 33, 256, 256])
# Shape Downsample_out: torch.Size([1, 128, 33, 128, 128])

# Level: 1
# Block: 0
# Shape Block0_out: torch.Size([1, 256, 33, 128, 128])
# Block: 1
# Shape Block1_out: torch.Size([1, 256, 33, 128, 128])
# Shape Downsample_out: torch.Size([1, 256, 17, 64, 64])

# Level: 2
# Block: 0
# Shape Block0_out: torch.Size([1, 512, 17, 64, 64])
# Block: 1
# Shape Block1_out: torch.Size([1, 512, 17, 64, 64])
# Shape Downsample_out: torch.Size([1, 512, 9, 32, 32])

# Level: 3
# Block: 0
# Shape Block0_out: torch.Size([1, 512, 9, 32, 32])
# Block: 1
# Shape Block1_out: torch.Size([1, 512, 9, 32, 32])

# After Mid:
# torch.Size([1, 512, 9, 32, 32])

# Before Convout
# torch.Size([1, 512, 9, 32, 32])
# After Convout
# torch.Size([1, 8, 9, 32, 32])

# After quant layer:
# torch.Size([1, 8, 9, 32, 32])


# %% CausalConv3d
# fixed: K=3, S=2
# T' = floor[ (T + K-1 - D*(K-1) -1)/S + 1 ] # For T=33, T'=17
# K-1 because only pad front, not back along T dim.

# updownsample.py: CausalConv3d
# 113                 # 1 + 16   16 as video, 1 as image
# 114 ->              first_frame_pad = x[:, :, :1, :, :].repeat(
# 115                     (1, 1, self.time_kernel_size - 1, 1, 1)
# 116                 )  # b c t h w
# 117                 x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
# (Pdb) x.shape
# torch.Size([1, 256, 33, 129, 129])
# (Pdb) first_frame_pad.shape
# torch.Size([1, 256, 2, 129, 129])
# (pdb) x = torch.concatenate((first_frame_pad, x), dim=2)  # 2 + 33
# (Pdb) x.shape
# torch.Size([1, 256, 35, 129, 129])

# %% Tiled VAE Shape

# why [1:] ?, when is padded?
# def tiled_encode(self, x):
#             moment = self.tiled_encode2d(chunk_x, return_moments=True)[:, :, 1:]

# tiled_encode2d(self, x, return_moments=False)
# moments = torch.cat(result_rows, dim=3)
# (Pdb) pp moments.shape
# torch.Size([1, 8, 9, 60, 80])

# tiled_encode(self, x)
# moments = torch.cat(moments, dim=2)
# (Pdb) pp t_chunk_start_end
# [[0, 33], [32, 65], [64, 93]]
# (Pdb) moments[0].shape
# torch.Size([1, 8, 9, 60, 80])
# (Pdb)  moments[1].shape
# torch.Size([1, 8, 8, 60, 80])
# (Pdb)  moments[2].shape
# torch.Size([1, 8, 7, 60, 80])
