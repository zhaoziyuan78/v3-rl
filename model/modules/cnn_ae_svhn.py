import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.resblocks import ResBlock, ResBlockTranspose


class CNNEncoder(nn.Module):
    def __init__(self, n_channels, W, H, d_emb_c, d_emb_s):
        super().__init__()
        # define the CNN with ResBlocks
        self.d_emb_c = d_emb_c
        self.d_emb_s = d_emb_s

        n_channels_1 = n_channels // (2**4)

        self.cnn_1 = self._make_cnn(3, n_channels_1, 2, 3, (5, 5), (2, 2))
        W_1, H_1 = self._get_cnn_output_size(W, H, n_channels, 2, (2, 2))[1:3]
        self.cnn_2 = self._make_cnn(n_channels_1, n_channels, 2, 3, (3, 3), (2, 2))
        W_2, H_2, output_size = self._get_cnn_output_size(
            W_1, H_1, n_channels, 2, (2, 2)
        )[1:]  # output_size should be larger or equal to d_emb_c + d_emb_s

        # self.linear_0 = nn.Linear(
        #     output_size,
        #     output_size,
        # )
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
        stride_at_layer_start=(2, 2),
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
        # x = x.unsqueeze(2)
        # torch cannot handle the case where the input is a 5D tensor, I hate this
        x = x.reshape(
            -1, x.shape[-3], x.shape[-2], x.shape[-1]
        )  # [batch_size * n_segments, 1, n_feature, segment_len]
        x = self.cnn_1(x)  # [batch_size * n_segments, ...]
        x = self.cnn_2(x)  # [batch_size * n_segments, ...]
        emb = x.reshape(x.shape[0], -1)  # [batch_size * n_segments, cnn_output_size]
        # x = x.reshape(
        #     batch_size, n_segments, -1
        # )  # [batch_size, n_segments, cnn_output_size]
        # emb = self.linear_0(x)  # [batch_size, n_segments, d_emb_c + d_emb_s]
        # emb = self.act(emb)
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb_c = emb[:, : self.d_emb_c]
        emb_s = emb[:, self.d_emb_c :]
        emb_c = self.linear_c(emb_c)
        emb_c = self.bn_c(emb_c)
        emb_c = emb_c.reshape(batch_size, n_segments, -1)
        emb_s = self.linear_s(emb_s)
        emb_s = self.bn_s(emb_s)
        emb_s = emb_s.reshape(batch_size, n_segments, -1)

        return emb_c, emb_s


class ResBlockTranspose(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size, stride, padding):
        super().__init__()

        n_channels_in = int(n_channels_in)
        n_channels_out = int(n_channels_out)

        self.conv_0 = nn.ConvTranspose2d(
            n_channels_in, n_channels_in, kernel_size, stride, padding
        )
        self.bn_0 = nn.BatchNorm2d(n_channels_in)
        self.conv_1 = nn.ConvTranspose2d(
            n_channels_in, n_channels_out, kernel_size, stride, padding
        )
        self.bn_1 = nn.BatchNorm2d(n_channels_out)
        self.act = nn.GELU()

        self.is_bottleneck = False
        if n_channels_in != n_channels_out:
            # shortcut
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    n_channels_in, n_channels_out, (1, 1), 1, (0, 0)
                ),  # no activation
                nn.BatchNorm2d(n_channels_out),
            )
            self.is_bottleneck = True

    def forward(self, x):
        if self.is_bottleneck:
            residual = self.shortcut(x)
        else:
            residual = x
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = x + residual
        x = self.act(x)

        return x


class CNNDecoder(nn.Module):
    def __init__(self, n_channels, W, H, d_emb_c, d_emb_s):
        super().__init__()
        self.n_channels = n_channels
        self.W = W
        self.H = H
        # self.linear_c = nn.Linear(d_emb_c, d_emb_c)
        # self.linear_s = nn.Linear(d_emb_s, d_emb_s)
        self.act = nn.GELU()

        self.W_1, self.H_1 = CNNEncoder._get_cnn_output_size(
            W, H, n_channels, 2, (2, 2)
        )[1:3]
        self.W_2, self.H_2, cnn_output_size = CNNEncoder._get_cnn_output_size(
            self.W_1, self.H_1, n_channels, 2, (2, 2)
        )[1:]  # output_size should be larger or equal to d_emb_c + d_emb_s

        self.linear_0 = nn.Linear(
            d_emb_c, self.W_2 * self.H_2 * n_channels
        )  # assume d_emb_c == d_emb_s
        # self.linear_1 = nn.Linear(
        #     self.W_2 * self.H_2 * n_channels, self.W_2 * self.H_2 * n_channels
        # )

        n_channels_1 = n_channels // (2**4)

        self.cnn_transpose_2 = self._make_cnn_transpose(
            n_channels, n_channels_1, 2, 2, (3, 3), (2, 2)
        )
        self.cnn_transpose_1 = self._make_cnn_transpose(
            n_channels_1,
            3,
            2,
            2,
            (5, 5),
            (2, 2),  # the first number 1 or 3 is the number of channels
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
        # emb_c = self.linear_c(emb_c)
        # emb_c = self.act(emb_c)
        # emb_s = self.linear_s(emb_s)
        # emb_s = self.act(emb_s)
        # emb = torch.cat([emb_c, emb_s], dim=-1)
        emb = emb_c + emb_s

        emb = self.linear_0(emb)
        emb = self.act(emb)  # [batch_size, n_segments, w_0 * h_0 * n_channels]

        emb = emb.reshape(batch_size * n_segments, emb.shape[-1])
        emb = emb.reshape(batch_size * n_segments, self.n_channels, self.W_2, self.H_2)
        emb = self.cnn_transpose_2(
            emb
        )  # [batch_size * n_segments, n_channels, W_1, H_1]
        emb = self.cnn_transpose_1(emb)  # [batch_size * n_segments, 1, W, H]
        # emb = emb.reshape(
        #     batch_size * n_segments, 1, self.W * self.H
        # )  # [batch_size * n_segments, 1, W * H]
        # emb = self.output_fc(emb)  # [batch_size * n_segments, 1, W * H]
        # output = emb.reshape(batch_size, n_segments, self.W, self.H)
        output = emb.reshape(batch_size, n_segments, 3, self.W, self.H)

        return output


class CSC_CNN_VQ(nn.Module):
    def __init__(self, config):
        """
        CSC means Content-Style-Changes
        """
        super().__init__()
        n_channels = config["n_channels"]
        W = config["n_feature"]
        H = config["segment_len"]
        d_emb_c = config["d_emb_c"]
        d_emb_s = config["d_emb_s"]
        self.encoder = CNNEncoder(n_channels, W, H, d_emb_c, d_emb_s)
        self.vq = VectorQuantize(
            dim=d_emb_c,
            codebook_size=config["n_atoms"],
            commitment_weight=1,
            decay=config["vq_ema_decay"] if "vq_ema_decay" in config else 0.98,
            kmeans_init=True,
            ema_update=True,
            threshold_ema_dead_code=config["threshold_ema_dead_code"]
            if "threshold_ema_dead_code" in config
            else 0,
            # learnable_codebook=True,
            # affine_param=True,
            # orthogonal_reg_weight=1,
        )
        # x = torch.randn(config["n_atoms"], config["d_emb_c"])
        # q, _ = torch.linalg.qr(x.T)
        # ovq = q.T.clone()
        # self.vq.codebook = ovq
        # self.sparsemax = Sparsemax(dim=-1)
        self.decoder = CNNDecoder(n_channels, W, H, d_emb_c, d_emb_s)

    def encode(self, x):
        emb_c, emb_s = self.encoder(x)

        return emb_c, emb_s

    def quantize(self, x, freeze_codebook=False):
        quantized, indices, commit_loss = self.vq(x, freeze_codebook=freeze_codebook)

        return quantized, indices, commit_loss

    def decode(self, emb_c, emb_s):
        output = self.decoder(emb_c, emb_s)

        return output

    def forward(self, x, freeze_codebook=False):
        emb_c, emb_s = self.encoder(x)
        emb_c_vq, vq_indices, commit_loss = self.quantize(
            emb_c, freeze_codebook=freeze_codebook
        )
        output = self.decoder(emb_c_vq, emb_s)

        return output, emb_c, emb_c_vq, vq_indices, commit_loss, emb_s

    def get_model_size(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params

        message = (
            f"Encoder params: {encoder_params}\n"
            + f"Decoder params: {decoder_params}\n"
            + f"Total params: {total_params}"
        )

        return message


def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # from dataloader.svhnplural_dataloader import get_dataloader
    from dataloader.digitstring_dataloader import get_dataloader

    dataloader = get_dataloader(
        "../data/DigitString3Krist/train",
        batch_size=32,
        n_segments=config["n_segments"],
        portion=1,
        shuffle=True,
    )

    model = CSC_CNN_VQ(config).to(device)
    print(model.get_model_size())

    for batch, c_labels, s_labels in dataloader:
        batch = batch.to(device)
        output, emb_c, emb_c_vq, emb_s, vq_indices, commit_loss = model(batch)
        print(batch.shape)
        print(output.shape)
        print(emb_c.shape)
        print(emb_s.shape)
        print(vq_indices.shape)
        print(commit_loss.shape)
        break


if __name__ == "__main__":
    config = {
        "n_channels": 256,
        "n_segments": 10,
        "segment_len": 32,
        "n_feature": 48,
        "d_emb_c": 384,
        "d_emb_s": 384,
        "n_atoms": 10,
        "threshold_ema_dead_code": 0,
    }
    test_model(config)
