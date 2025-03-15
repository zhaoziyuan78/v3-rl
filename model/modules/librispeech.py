import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.resblocks import ResBlock, ResBlockTranspose


class Encoder(nn.Module):
    def __init__(self, n_channels, W, H, d_emb_c, d_emb_s):
        super().__init__()
        # define the CNN with ResBlocks
        self.d_emb_c = d_emb_c
        self.d_emb_s = d_emb_s

        n_channels_1 = n_channels // (2**4)

        self.cnn_1 = self._make_cnn(1, n_channels_1, 3, 4, (9, 3), (2, 2))
        W_1, H_1 = self._get_cnn_output_size(W, H, n_channels, 3, (2, 2))[1:3]
        self.cnn_2 = self._make_cnn(n_channels_1, n_channels, 4, 4, (5, 3), (2, 1))
        W_2, H_2, output_size = self._get_cnn_output_size(
            W_1, H_1, n_channels, 4, (2, 1)
        )[1:]  # output_size should be larger or equal to d_emb_c + d_emb_s

        self.linear_1 = nn.Linear(output_size, d_emb_c + d_emb_s)
        self.linear_c = nn.Linear(d_emb_c, d_emb_c)
        self.bn_c = nn.BatchNorm1d(d_emb_c)
        self.linear_s = nn.Linear(d_emb_s, d_emb_s)
        self.bn_s = nn.BatchNorm1d(d_emb_s)
        self.act = nn.GELU()

    def _make_cnn(
        self,
        n_input_channels,
        n_hidden_channels,
        n_layers,
        n_blocks_per_layer=3,
        kernel_size=(3, 3),
        stride_at_layer_start=(2, 1),
    ):
        """
        Build a resnet-like (but with downsampling or upsampling) cnn. Every "layer" here consists of four ResBlocks and one down/upsample.
        n_hidden_channels for each layer will start from n_hidden_channels / (2**n_layers) and increase by doubling.
        """
        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )

        n_hidden_channels_first = n_hidden_channels // (2**n_layers)

        # start layer
        cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                n_hidden_channels_first,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
            ),
            nn.BatchNorm2d(n_hidden_channels_first),
            nn.GELU(),
        )
        for j in range(n_blocks_per_layer):
            cnn.add_module(
                f"resblock_start_{j}",
                ResBlock(
                    n_hidden_channels_first,
                    n_hidden_channels_first,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                ),
            )
        # main layers
        for i in range(n_layers):
            cnn.add_module(
                f"resblock_{i}_0",
                ResBlock(
                    n_hidden_channels_first * (2**i),
                    n_hidden_channels_first * (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=stride_at_layer_start,
                    padding=padding,
                ),
            )
            for j in range(1, n_blocks_per_layer):
                cnn.add_module(
                    f"resblock_{i}_{j}",
                    ResBlock(
                        n_hidden_channels_first * (2 ** (i + 1)),
                        n_hidden_channels_first * (2 ** (i + 1)),
                        kernel_size=kernel_size,
                        stride=(1, 1),
                        padding=padding,
                    ),
                )

        return cnn

    @staticmethod
    def _get_cnn_output_size(
        w,
        h,
        n_channels,
        n_pooling_layers,
        pooling_kernel_size,
    ):
        """
        Only works for CNN models with padding.
        """
        if isinstance(pooling_kernel_size, int):
            for i in range(n_pooling_layers):
                w = int((w - pooling_kernel_size) // pooling_kernel_size + 1)
                h = int((h - pooling_kernel_size) // pooling_kernel_size + 1)
        elif isinstance(pooling_kernel_size, tuple):
            for i in range(n_pooling_layers):
                w = int((w - pooling_kernel_size[0]) // pooling_kernel_size[0] + 1)
                h = int((h - pooling_kernel_size[1]) // pooling_kernel_size[1] + 1)

        output_size = w * h * n_channels

        return [n_channels, w, h, output_size]

    def forward(self, x):
        """
        x: [batch_size, n_segments, n_feature, segment_len]
        """
        batch_size, n_segments = x.shape[0], x.shape[1]
        x = x.unsqueeze(2)  # [batch_size, n_segments, 1, n_feature, segment_len]
        x = x.reshape(
            -1, 1, x.shape[-2], x.shape[-1]
        )  # [batch_size * n_segments, 1, n_feature, segment_len]
        x = self.cnn_1(x)  # [batch_size * n_segments, ...]
        x = self.cnn_2(x)  # [batch_size * n_segments, ...]
        emb = x.reshape(x.shape[0], -1)  # [batch_size * n_segments, cnn_output_size]
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb_c = emb[:, : self.d_emb_c]
        emb_s = emb[:, self.d_emb_c :]
        emb_c = self.linear_c(emb_c)
        emb_c = emb_c.reshape(batch_size, n_segments, -1)
        emb_s = self.linear_s(emb_s)
        emb_s = emb_s.reshape(batch_size, n_segments, -1)

        return emb_c, emb_s


class Decoder(nn.Module):
    def __init__(self, n_channels, W, H, d_emb_c, d_emb_s):
        super().__init__()
        self.n_channels = n_channels
        self.W = W
        self.H = H
        self.linear_c = nn.Linear(d_emb_c, d_emb_c)
        self.linear_s = nn.Linear(d_emb_s, d_emb_s)
        self.act = nn.GELU()

        self.W_1, self.H_1 = Encoder._get_cnn_output_size(W, H, n_channels, 3, (2, 2))[
            1:3
        ]
        self.W_2, self.H_2, cnn_output_size = Encoder._get_cnn_output_size(
            self.W_1, self.H_1, n_channels, 4, (2, 1)
        )[1:]  # output_size should be larger or equal to d_emb_c + d_emb_s

        self.linear_0 = nn.Linear(d_emb_c + d_emb_s, self.W_2 * self.H_2 * n_channels)

        n_channels_1 = n_channels // (2**4)

        self.cnn_transpose_2 = self._make_cnn_transpose(
            n_channels, n_channels_1, 4, 4, (5, 3), (2, 1)
        )
        self.cnn_transpose_1 = self._make_cnn_transpose(
            n_channels_1, 1, 3, 4, (9, 3), (2, 2)
        )

    def _make_cnn_transpose(
        self,
        n_hidden_channels,
        n_output_channels,
        n_layers,
        n_blocks_per_layer=3,
        kernel_size=(3, 3),
        scale_factor=(2, 1),
    ):
        """
        Build a resnet-like (but with downsampling or upsampling) cnn. Every "layer" here consists of four ResBlocks and one down/upsample.
        n_hidden_channels for each layer will start from n_hidden_channels and decrease by halving.
        """
        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )

        # main layers
        cnn = nn.Sequential()
        for i in range(n_layers):
            for j in range(n_blocks_per_layer - 1):
                cnn.add_module(
                    f"resblock_{i}_{j}",
                    ResBlockTranspose(
                        n_hidden_channels // (2**i),
                        n_hidden_channels // (2**i),
                        kernel_size=kernel_size,
                        stride=(1, 1),
                        padding=padding,
                    ),
                )
            cnn.add_module(
                f"resblock_{i}_{n_blocks_per_layer - 1}",
                ResBlockTranspose(
                    n_hidden_channels // (2**i),
                    n_hidden_channels // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                ),
            )
            cnn.add_module(f"upsample_{i}", nn.Upsample(scale_factor=scale_factor))

        # last layer
        for j in range(n_blocks_per_layer):
            cnn.add_module(
                f"resblock_end_{j}",
                ResBlockTranspose(
                    n_hidden_channels // (2**n_layers),
                    n_hidden_channels // (2**n_layers),
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                ),
            )
        cnn.add_module(
            "output_conv",
            nn.ConvTranspose2d(
                n_hidden_channels // (2**n_layers),
                n_output_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
            ),
        )

        return cnn

    def forward(self, emb_c, emb_s):
        """
        emb_c: [batch_size, n_segments, d_emb_c]
        emb_s: [batch_size, n_segments, d_emb_s]
        """
        batch_size, n_segments = emb_c.shape[0], emb_c.shape[1]
        emb_c = self.linear_c(emb_c)
        emb_c = self.act(emb_c)
        emb_s = self.linear_s(emb_s)
        emb_s = self.act(emb_s)
        emb = torch.cat([emb_c, emb_s], dim=-1)
        emb = self.linear_0(emb)
        emb = self.act(emb)
        emb = emb.reshape(batch_size * n_segments, emb.shape[-1])
        emb = emb.reshape(batch_size * n_segments, self.n_channels, self.W_2, self.H_2)
        emb = self.cnn_transpose_2(
            emb
        )  # [batch_size * n_segments, n_channels, W_1, H_1]
        emb = self.cnn_transpose_1(emb)  # [batch_size * n_segments, 1, W, H]
        emb = emb.reshape(batch_size, n_segments, 1, self.W, self.H)
        output = emb.squeeze(2)

        return output
