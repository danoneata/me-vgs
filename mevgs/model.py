import math

from functools import partial
from toolz import dissoc

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    unpack_sequence,
    pad_packed_sequence,
)
from torchvision.models import alexnet


def get_average_pooling_layer(**_):
    def func(x, lens):
        return x.sum(dim=-1, keepdims=True) / lens

    return func


def get_two_layer_mlp_pooling_layer(*, frame_dim):
    layer = nn.Sequential(
        nn.Linear(frame_dim // 2, frame_dim // 4),
        nn.LeakyReLU(),
        nn.Linear(frame_dim // 4, 1),
        nn.LeakyReLU(),
    )

    def func(x, lens):
        return layer(x)

    return func


AUDIO_POOLING_LAYERS = {
    "average": get_average_pooling_layer,
    "two-layer-mlp": get_two_layer_mlp_pooling_layer,
}


class AudioEncoderLeanne(nn.Module):
    def __init__(
        self,
        *,
        in_channels=40,
        num_channels=64,
        kernel_size=4,
        stride=2,
        padding=1,
        z_dim=64,
        c_dim=512,
        frame_dim=256,
        use_pretrained_cpc=False,
        output_dim=2048,
        pooling_layer="two-layer-mlp",
    ):
        super(AudioEncoderLeanne, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            num_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )

        self.encoder = nn.Sequential(
            # 1
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 2
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 3
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 4
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # 5
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, z_dim),
        )
        self.rnn1 = nn.LSTM(z_dim, c_dim, batch_first=True)
        self.rnn2 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn3 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn4 = nn.LSTM(c_dim, c_dim, batch_first=True)

        self.english_rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.english_rnn2 = nn.LSTM(
            1024, output_dim // 2, batch_first=True, bidirectional=True
        )
        self.relu = nn.ReLU()

        self.pooling_layer = AUDIO_POOLING_LAYERS[pooling_layer](frame_dim=frame_dim)

        if use_pretrained_cpc:

            def drop_prefix(s, prefix, sep="."):
                fst, *rest = s.split(sep)
                assert fst == prefix
                return ".".join(rest)

            print("Using pretrained CPC")
            path = "checkpoints/audio-model-cpc-epoch-1500.pt"
            model_weights = torch.load(path)
            model_weights = model_weights["acoustic_model"]
            model_weights = {
                drop_prefix(key, "module"): value
                for key, value in model_weights.items()
            }
            model_dict = self.state_dict()

            for key in model_weights:
                if key in model_dict:
                    model_dict[key] = model_weights[key]
                else:
                    print(f"WARN · Missing key: {key}")
            self.load_state_dict(model_dict)

    def forward(self, mels, lengths):
        # mels: B × D × T

        # mels = mels.transpose(1, 2)  # B × D × T

        z = self.conv(mels)
        z = self.relu(z)

        z = z.transpose(1, 2)  # B × T × D
        z = self.encoder(z)

        lengths = lengths // self.conv.stride[0]
        z = pack_padded_sequence(
            z,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        c, _ = self.rnn1(z)
        c, _ = self.rnn2(c)
        c, _ = self.rnn3(c)
        c, _ = self.rnn4(c)

        s, _ = self.english_rnn1(c)
        s, _ = self.english_rnn2(s)

        s, _ = pad_packed_sequence(s, batch_first=True)
        s = s.transpose(1, 2)  # B × D × T
        s = self.pooling_layer(s, lengths)

        # return z, z, s
        return s


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class TransformerPooling(nn.Module):
    def __init__(self, input_dim, width, output_dim):
        super(TransformerPooling, self).__init__()
        # transformer hyperparameters
        assert width % 64 == 0
        nhead = width // 64
        dim_feedforward = 4 * width

        self.proj_in = nn.Linear(input_dim, width)
        self.class_embedding = nn.Parameter(input_dim**-0.5 * torch.randn(width))
        self.layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.ln_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, output_dim, bias=False)

    def forward(self, x, lengths=None):
        # x: B × T × D
        B, T, _ = x.shape
        x = self.proj_in(x)
        x = torch.cat([_expand_token(self.class_embedding, B), x], dim=1)
        if lengths is not None:
            mask = torch.arange(1 + T).expand(B, -1).to(x.device)
            mask = mask >= (lengths + 1).unsqueeze(1)
        else:
            mask = None
        out = self.layer(x, src_key_padding_mask=mask)
        out = out[:, 0]
        out = self.ln_out(out)
        out = self.proj_out(out)
        return out.unsqueeze(-1)


class AudioEncoderTransformer(nn.Module):
    def __init__(self, input_dim, width, output_dim):
        super(AudioEncoderTransformer, self).__init__()
        self.pool = TransformerPooling(input_dim, width, output_dim)

    def forward(self, x, lengths):
        # x: B × D × T
        x = x.permute(0, 2, 1)
        return self.pool(x, lengths)


AUDIO_ENCODERS = {
    "leanne": AudioEncoderLeanne,
    "transformer": AudioEncoderTransformer,
}


class ImageBackboneAlexNet(nn.Module):
    def __init__(self, to_freeze=False, use_pretrained=True):
        super(ImageBackboneAlexNet, self).__init__()
        self.model = alexnet(pretrained=False).features

        if use_pretrained:
            print("Using pretrained AlexNet")
            path = "checkpoints/alexnet-self-supervised.pth"
            model_weights = torch.load(path, map_location="cpu")

            fmt_key = lambda key: ".".join(key.split(".")[2:])
            model_dict = {
                fmt_key(k): v
                for k, v in model_weights["model"].items()
                if k.split(".")[1] == "features"
            }
            self.model.load_state_dict(model_dict)

        if to_freeze:
            assert use_pretrained
            for _, param in self.model.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class ImageBackboneDINO(nn.Module):
    def __init__(self, type_, to_freeze=False, use_pretrained=True):
        assert type_ == "resnet50"
        super(ImageBackboneDINO, self).__init__()

        self.model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_" + type_,
            pretrained=use_pretrained,
        )
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        if to_freeze:
            assert use_pretrained
            for _, param in self.model.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


IMAGE_BACKBONES = {
    "alexnet": ImageBackboneAlexNet,
    "dino-resnet50": partial(ImageBackboneDINO, type_="resnet50"),
    "identity": lambda **kwargs: nn.Identity(),
}


IMAGE_BACKBONE_FEATURE_DIM = {
    "alexnet": 256,
    "dino-resnet50": 2048,
}


class ImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        output_dim,
        width,
        backbone_type,
        to_freeze_backbone,
        use_pretrained_backbone,
        input_dim=None,
    ):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = IMAGE_BACKBONES[backbone_type](
            to_freeze=to_freeze_backbone,
            use_pretrained=use_pretrained_backbone,
        )

        input_dim = input_dim or IMAGE_BACKBONE_FEATURE_DIM[backbone_type]
        self.pool = TransformerPooling(input_dim, width, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)  # B × D × W x H
        B, D, _, _ = x.shape
        x = x.view(B, D, -1)  # B × D × (WH)
        x = x.permute(0, 2, 1)  # B × (WH) × D
        return self.pool(x)


class MattNet(nn.Module):
    def __init__(self, embed_dim, pooling, audio_encoder_kwargs, image_encoder_kwargs):
        super(MattNet, self).__init__()
        SCORE_EMB_FUNCS = {
            "features-avg": self.score_emb_pool_features_avg,
            "scores-max": self.score_emb_pool_scores_max,
        }
        audio_encoder_type = audio_encoder_kwargs["type"]
        audio_encoder_kwargs_1 = dissoc(audio_encoder_kwargs, "type")
        self.audio_enc = AUDIO_ENCODERS[audio_encoder_type](
            output_dim=embed_dim, **audio_encoder_kwargs_1
        )
        self.image_enc = ImageEncoder(output_dim=embed_dim, **image_encoder_kwargs)
        self.score_emb = SCORE_EMB_FUNCS[pooling]
        # self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    # def forward(self, audio, image):
    #     return self.score(audio, image, type="cross")

    def l2_normalize(self, x, dim):
        return x / x.norm(dim=dim, keepdim=True)

    def score(self, audio, audio_length, image, type):
        audio_emb = self.audio_enc(audio, audio_length)
        image_emb = self.image_enc(image)
        return self.score_emb(audio_emb, image_emb, type)

    def score_emb_pool_features_avg(self, audio_emb, image_emb, type):
        EINSUM_OP = {
            "pair": "bd,bd->b",
            "cross": "xd,yd->xy",
            "pair-and-cross": "bnd,bmd->bnm",
        }
        audio_emb = audio_emb.mean(dim=-1)
        image_emb = image_emb.mean(dim=-1)
        return torch.einsum(EINSUM_OP[type], audio_emb, image_emb)

    def score_emb_pool_scores_max(self, audio_emb, image_emb, type):
        # def compute1(a, i):
        #     att = torch.bmm(a.unsqueeze(0).transpose(1, 2), i.unsqueeze(0))
        #     s = att.max()
        #     return s

        EINSUM_OP = {
            "pair": "bda,bdi->bai",
            "cross": "xda,ydi->xyai",
            "pair-and-cross": "bnda,bmdi->bnmai",
        }
        op = EINSUM_OP[type]

        # audio_embedding = self.audio_enc(audio)
        # image_embedding = self.image_enc(image)

        # sim = torch.stack([torch.stack([compute1(a, i) for i in image_embedding]) for a in audio_embedding])
        # sim = torch.cat(att).to(audio.device)

        # audio_embedding = self.l2_normalize(audio_embedding, dim=1)
        # image_embedding = self.l2_normalize(image_embedding, dim=1)

        sim = torch.einsum(op, audio_emb, image_emb)
        sim, _ = sim.max(dim=-1)
        sim, _ = sim.max(dim=-1)

        # τ = torch.maximum(self.logit_scale.exp(), torch.tensor(100.0))
        τ = 1.0
        return τ * sim

    def forward(self, audio, audio_length, image, labels):
        """Input shapes:

        - audio:        B × (pos + neg) × D × T
        - audio_length: B × (pos + neg)
        - image:        B × (pos + neg) × 3 × H × W
        - labels:       B × (pos + neg)

        Currently, we assume that there is a single positive: pos = 1.

        """

        B1, N1, *_ = audio.shape
        B2, N2, *_ = image.shape

        assert B1 == B2 and N1 == N2
        B = B1
        N = N1

        audio = audio.view(B * N, *audio.shape[2:])
        audio_length = audio_length.view(B * N)
        audio_emb = self.audio_enc(audio, audio_length)
        audio_emb = audio_emb.view(B, N, *audio_emb.shape[1:])

        image = image.view(B * N, *image.shape[2:])
        image_emb = self.image_enc(image)
        image_emb = image_emb.view(B, N, *image_emb.shape[1:])

        # assume one positive per batch and all positives are in first position
        assert labels.sum() == B
        assert labels[:, 0].sum() == B
        assert labels[:, 1:].sum() == 0
        # assert labels == torch.cat([torch.ones(B, 1), torch.zeros(B, N - 1)], dim=1)

        sim1 = self.score_emb(audio_emb[:, :1], image_emb, "pair-and-cross")
        sim1 = sim1.squeeze(1)
        sim2 = self.score_emb(audio_emb, image_emb[:, :1], "pair-and-cross")
        sim2 = sim2.squeeze(2)

        pred = torch.cat([sim1, sim2], dim=0)

        true = torch.zeros(2 * B).to(labels.device).long()
        return F.cross_entropy(pred, true)

    def predict_paired_test(self, audio, audio_length, image_pos, image_neg):
        """Input shapes:

        - audio:        B × D × T
        - audio_length: B
        - image-pos:    B × 3 × H × W
        - image-neg:    B × 3 × H × W

        """
        audio_emb = self.audio_enc(audio, audio_length)

        image_pos_emb = self.image_enc(image_pos)
        image_neg_emb = self.image_enc(image_neg)

        scores_pos = self.score_emb(audio_emb, image_pos_emb, type="pair")
        scores_neg = self.score_emb(audio_emb, image_neg_emb, type="pair")

        scores = torch.stack([scores_pos, scores_neg], dim=1)
        return F.softmax(scores, dim=1)


class CLIP(nn.Module):
    def __init__(self, embed_dim, audio_encoder_kwargs, image_encoder_kwargs):
        super(CLIP, self).__init__()
        audio_encoder_type = audio_encoder_kwargs["type"]
        audio_encoder_kwargs_1 = dissoc(audio_encoder_kwargs, "type")
        self.audio_enc = AUDIO_ENCODERS[audio_encoder_type](
            output_dim=embed_dim,
            **audio_encoder_kwargs_1,
        )
        self.image_enc = ImageEncoder(
            output_dim=embed_dim,
            **image_encoder_kwargs,
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def l2_normalize(self, x, dim):
        return x / x.norm(dim=dim, keepdim=True)

    def score_emb(self, audio_emb, image_emb, type):
        EINSUM_OP = {
            "pair": "bd,bd->b",
            "cross": "xd,yd->xy",
            "pair-and-cross": "bnd,bmd->bnm",
        }
        audio_emb = audio_emb.mean(dim=-1)
        audio_emb = self.l2_normalize(audio_emb, dim=-1)

        image_emb = image_emb.mean(dim=-1)
        image_emb = self.l2_normalize(image_emb, dim=-1)

        sim = torch.einsum(EINSUM_OP[type], audio_emb, image_emb)
        τ = torch.maximum(self.logit_scale.exp(), torch.tensor(100.0))
        # τ = 1

        return τ * sim

    def score(self, audio, audio_length, image, type):
        audio_emb = self.audio_enc(audio, audio_length)
        image_emb = self.image_enc(image)
        return self.score_emb(audio_emb, image_emb, type)

    def forward(self, audio, audio_length, image, labels):
        """Input shapes:

        - audio:        B × (pos + neg) × D × T
        - audio_length: B × (pos + neg)
        - image:        B × (pos + neg) × 3 × H × W
        - labels:       B × (pos + neg)

        Currently, we assume that there is a single positive: pos = 1.

        """

        B1, N1, *_ = audio.shape
        B2, N2, *_ = image.shape

        assert B1 == B2 and N1 == N2
        B = B1
        N = N1

        audio = audio.view(B * N, *audio.shape[2:])
        audio_length = audio_length.view(B * N)
        audio_emb = self.audio_enc(audio, audio_length)
        audio_emb = audio_emb.view(B, N, *audio_emb.shape[1:])

        image = image.view(B * N, *image.shape[2:])
        image_emb = self.image_enc(image)
        image_emb = image_emb.view(B, N, *image_emb.shape[1:])

        # assume one positive per batch and all positives are in first position
        assert labels.sum() == B
        assert labels[:, 0].sum() == B
        assert labels[:, 1:].sum() == 0
        # assert labels == torch.cat([torch.ones(B, 1), torch.zeros(B, N - 1)], dim=1)

        sim1 = self.score_emb(audio_emb[:, :1], image_emb, "pair-and-cross")
        sim1 = sim1.squeeze(1)
        sim2 = self.score_emb(audio_emb, image_emb[:, :1], "pair-and-cross")
        sim2 = sim2.squeeze(2)

        pred = torch.cat([sim1, sim2], dim=0)

        true = torch.zeros(2 * B).to(labels.device).long()
        return F.cross_entropy(pred, true)

    def predict_paired_test(self, audio, audio_length, image_pos, image_neg):
        """Input shapes:

        - audio:        B × D × T
        - audio_length: B
        - image-pos:    B × 3 × H × W
        - image-neg:    B × 3 × H × W

        """
        audio_emb = self.audio_enc(audio, audio_length)

        image_pos_emb = self.image_enc(image_pos)
        image_neg_emb = self.image_enc(image_neg)

        scores_pos = self.score_emb(audio_emb, image_pos_emb, type="pair")
        scores_neg = self.score_emb(audio_emb, image_neg_emb, type="pair")

        scores = torch.stack([scores_pos, scores_neg], dim=1)
        return F.softmax(scores, dim=1)


class BarLIP(nn.Module):
    def __init__(self, embed_dim, audio_encoder_kwargs, image_encoder_kwargs, λ):
        super(BarLIP, self).__init__()
        audio_encoder_type = audio_encoder_kwargs["type"]
        audio_encoder_kwargs_1 = dissoc(audio_encoder_kwargs, "type")
        self.audio_enc = AUDIO_ENCODERS[audio_encoder_type](
            output_dim=embed_dim,
            **audio_encoder_kwargs_1,
        )
        self.image_enc = ImageEncoder(
            output_dim=embed_dim,
            **image_encoder_kwargs,
        )
        self.bn_audio = nn.BatchNorm1d(embed_dim, affine=False)
        self.bn_image = nn.BatchNorm1d(embed_dim, affine=False)
        self.λ = λ

    def score_emb(self, audio_emb, image_emb):
        audio_emb = audio_emb.mean(dim=-1)
        audio_emb = self.bn_audio(audio_emb)
        image_emb = image_emb.mean(dim=-1)
        image_emb = self.bn_image(image_emb)
        return - (audio_emb - image_emb).pow(2).sum(dim=-1)

    # def std_normalize(self, x, dim):
    #     μ = x.mean(dim=dim, keepdim=True)
    #     σ = x.std(dim=dim, keepdim=True)
    #     return (x - μ) / σ

    def forward(self, audio, audio_length, image, labels):
        """Input shapes:

        - audio:        B × D × T
        - audio_length: B
        - image:        B × 3 × H × W
        - labels:       B

        Currently, we assume that there is a single positive: pos = 1.

        """
        audio_emb = self.audio_enc(audio, audio_length)
        audio_emb = audio_emb.mean(dim=-1)
        # audio_emb = self.std_normalize(audio_emb, dim=0)
        audio_emb = self.bn_audio(audio_emb)

        image_emb = self.image_enc(image)
        image_emb = image_emb.mean(dim=-1)
        # image_emb = self.std_normalize(image_emb, dim=0)
        image_emb = self.bn_image(image_emb)

        B, D = audio_emb.shape

        corr = torch.einsum("bi, bj -> ij", audio_emb, image_emb) / B
        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        cdif[~diag.bool()] *= self.λ
        return cdif.sum()

    def predict_paired_test(self, audio, audio_length, image_pos, image_neg):
        """Input shapes:

        - audio:        B × D × T
        - audio_length: B
        - image-pos:    B × 3 × H × W
        - image-neg:    B × 3 × H × W

        """
        audio_emb = self.audio_enc(audio, audio_length)

        image_pos_emb = self.image_enc(image_pos)
        image_neg_emb = self.image_enc(image_neg)

        scores_pos = self.score_emb(audio_emb, image_pos_emb)
        scores_neg = self.score_emb(audio_emb, image_neg_emb)

        scores = torch.stack([scores_pos, scores_neg], dim=1)
        return scores


class CLIPTwoAudios(CLIP):
    def compute_audio_emb(self, audio, audio_length, B, N):
        audio = audio.view(B * N, *audio.shape[2:])
        audio_length = audio_length.view(B * N)
        audio_emb = self.audio_enc(audio, audio_length)
        audio_emb = audio_emb.view(B, N, *audio_emb.shape[1:])
        return audio_emb

    def compute_image_emb(self, image, B, N):
        image = image.view(B * N, *image.shape[2:])
        image_emb = self.image_enc(image)
        image_emb = image_emb.view(B, N, *image_emb.shape[1:])
        return image_emb

    def compute_loss(self, emb1, emb2, labels):
        B = emb1.shape[0]

        # assume one positive per batch and all positives are in first position
        assert labels.sum() == B
        assert labels[:, 0].sum() == B
        assert labels[:, 1:].sum() == 0

        sim1 = self.score_emb(emb1[:, :1], emb2, "pair-and-cross")
        sim1 = sim1.squeeze(1)
        sim2 = self.score_emb(emb1, emb2[:, :1], "pair-and-cross")
        sim2 = sim2.squeeze(2)

        pred = torch.cat([sim1, sim2], dim=0)
        true = torch.zeros(2 * B).to(labels.device).long()
        return F.cross_entropy(pred, true)

    def forward(self, audio1, audio1_length, audio2, audio2_length, image, labels):
        B0, N0, *_ = audio1.shape
        B1, N1, *_ = audio2.shape
        B2, N2, *_ = image.shape

        assert B0 == B1 == B2 and N0 == N1 == N2
        B = B0
        N = N0

        audio1_emb = self.compute_audio_emb(audio1, audio1_length, B, N)
        audio2_emb = self.compute_audio_emb(audio2, audio2_length, B, N)
        image_emb = self.compute_image_emb(image, B, N)

        loss1 = self.compute_loss(audio1_emb, image_emb, labels)
        loss2 = self.compute_loss(audio2_emb, image_emb, labels)
        loss3 = self.compute_loss(audio1_emb, audio2_emb, labels)
        return (loss1 + loss2 + loss3) / 3


MODELS = {
    "mattnet": MattNet,
    "clip": CLIP,
    "clip-two-audios": CLIPTwoAudios,
    "barlip": BarLIP,
}


def setup_model(*, model_name, **kwargs):
    return MODELS[model_name](**kwargs)
