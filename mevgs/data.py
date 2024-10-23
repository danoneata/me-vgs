from functools import partial
from itertools import groupby
from typing import Literal, Tuple

import json
import random
import pdb

import h5py
import librosa
import numpy as np
import scipy

import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from PIL import Image
from torchvision import transforms as T

from toolz import concat, dissoc
from mevgs.utils import read_file, read_json


Split = Literal["train", "valid", "test"]
Language = Literal["english", "dutch", "french"]


def load_dictionary():
    def parse_line(line):
        en, nl, fr = line.strip().split(",")
        return {
            "english": en,
            "dutch": nl,
            "french": fr,
        }

    return read_file("data/concepts.csv", parse_line)


IMAGE_SIZE = 256

transform_image_norm = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

transform_image_train_1 = T.Compose(
    [
        T.RandomAffine(scale=(0.75, 1.00), degrees=5, fill=255),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.00)),
        T.RandomHorizontalFlip(),
    ]
)

transform_image_train = T.Compose(
    [
        transform_image_train_1,
        T.ToTensor(),
        transform_image_norm,
    ]
)

transform_image_test = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        transform_image_norm,
    ]
)

TRANSFORM_IMAGE = {
    "train": transform_image_train,
    "valid": transform_image_test,
    "test": transform_image_test,
}


def get_image_path(datum: dict):
    name = datum["name"]
    return f"data/images/{name}.jpg"


def get_audio_path(datum: dict):
    name = datum["name"]
    lang = datum["lang"]
    return f"data/{lang}_words/{name}.wav"


def load_image(datum: dict, transform_image=transform_image_test) -> torch.Tensor:
    path = get_image_path(datum)
    img = Image.open(path).convert("RGB")
    return transform_image(img)


def load_audio(datum: dict):
    def preemphasis(signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def pad_to_length(data, len_target, pad_value=0):
        # data: T × D
        len_pad = len_target - len(data)
        if len_pad > 0:
            return np.pad(
                data,
                ((0, len_pad), (0, 0)),
                "constant",
                constant_values=pad_value,
            )
        else:
            return data[:len_target]

    CONFIG = {
        "preemph-coef": 0.97,
        "sample-rate": 16_000,
        "window-size": 0.025,
        "window-stride": 0.01,
        "window": scipy.signal.hamming,
        "num-mel-bins": 40,
        "target-length": 256,
        "use-raw_length": False,
        "pad-value": 0,
        "fmin": 20,
    }

    path = get_audio_path(datum)
    y, sr = librosa.load(path, sr=CONFIG["sample-rate"])
    y = y - y.mean()
    y = preemphasis(y, CONFIG["preemph-coef"])

    n_fft = int(CONFIG["sample-rate"] * CONFIG["window-size"])
    win_length = n_fft
    hop_length = int(CONFIG["sample-rate"] * CONFIG["window-stride"])

    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=CONFIG["num-mel-bins"],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hamming",
        fmin=CONFIG["fmin"],
    )

    # logspec = melspec
    logspec = librosa.power_to_db(melspec, ref=np.max)  # D × T
    # audio_length = min(logspec.shape[1], CONFIG["target-length"])

    # logspec = logspec.T  # T × D
    # logspec = pad_to_length(logspec, CONFIG["target-length"], CONFIG["pad-value"])
    # logspec = logspec.T

    # D × T
    return torch.tensor(logspec)


class AudioFeaturesLoader:
    def __init__(self, feature_type, split, langs):
        path = "output/features-audio/me-dataset-{split}-{lang}-{feature_type}.h5"
        self.files = {
            lang: h5py.File(
                path.format(split=split, lang=lang, feature_type=feature_type), "r"
            )
            for lang in langs
        }

    def __call__(self, datum):
        name = datum["name"]
        lang = datum["lang"]
        data = np.array(self.files[lang][name]["feature"])
        data = data.T
        # D × T
        return torch.tensor(data)


class ImageRawLoader:
    def __init__(self, split):
        self.transform = TRANSFORM_IMAGE[split]
        # self.transform = transform_image_test

    def __call__(self, datum):
        return load_image(datum, self.transform)


class ImageFeaturesLoader:
    def __init__(self, feature_type, split):
        path = f"output/features-image/me-dataset-{split}-{feature_type}.h5"
        self.file = h5py.File(path, "r")

    def __call__(self, datum):
        name = datum["name"]
        data = np.array(self.file[name]["feature"])
        return torch.tensor(data)


def get_image_loader(feature_type, split):
    if feature_type == "raw":
        return ImageRawLoader(split)
    else:
        return ImageFeaturesLoader(feature_type, split)


def group_by_word(data):
    get_word = lambda datum: datum["word-en"]
    data1 = sorted(data, key=get_word)
    return {key: list(group) for key, group in groupby(data1, get_word)}


class MEDataset:
    def __init__(self, split: Split, langs: Tuple[Language]):
        self.split = split
        self.langs = langs

        self.words_seen = read_file("data/words-seen.txt")
        self.words_unseen = read_file("data/words-unseen.txt")

        image_files = read_json(f"data/filelists/image-{split}.json")
        audio_files = read_json(f"data/filelists/audio-{split}-3.json")
        audio_files = [datum for datum in audio_files if datum["lang"] in langs]

        self.image_files = image_files
        self.audio_files = audio_files

        self.word_to_images = group_by_word(image_files)
        self.word_to_audios = group_by_word(audio_files)


class SimplePairedMEDataset(Dataset):
    def __init__(
        self,
        split,
        langs,
        feature_type_audio: str,
        feature_type_image: str,
        to_fix_validation_samples: bool = True,
    ):
        super(SimplePairedMEDataset).__init__()

        assert split in ("train", "valid")
        self.split = split
        self.dataset = MEDataset(split, langs)

        # Use Leanne's order
        self.word_audio = [
            (word, audio)
            for word, audios in self.dataset.word_to_audios.items()
            for audio in audios
        ]
        self.word_audio = sorted(self.word_audio, key=lambda x: x[0])
        self.load_audio = AudioFeaturesLoader(feature_type_audio, split, langs)
        self.load_image = get_image_loader(feature_type_image, split)

        if split == "valid" and to_fix_validation_samples:
            suffix = "_".join(langs)
            path = f"data/filelists/validation-samples-simple-{suffix}.json"
            with open(path, "r") as f:
                validation_samples = json.load(f)
            self.get_positive_pair = lambda i: validation_samples[i]

    def get_positive_pair(self, i):
        word, audio = self.word_audio[i]
        image = random.choice(self.dataset.word_to_images[word])
        return audio, image

    def __getitem__(self, i):
        audio_name, image_name = self.get_positive_pair(i)
        return {
            "index": i,
            "audio": self.load_audio(audio_name),
            "image": self.load_image(image_name),
            "label": 1,
        }

    def __len__(self):
        return len(self.word_audio)


class PairedMEDataset(Dataset):
    def __init__(
        self,
        split,
        langs,
        num_pos: int,
        num_neg: int,
        feature_type_audio: str,
        feature_type_image: str,
        # num_word_repeats: int,
        to_fix_validation_samples: bool = True,
    ):
        super(PairedMEDataset).__init__()

        assert split in ("train", "valid")
        self.split = split
        self.dataset = MEDataset(split, langs)

        self.n_pos = num_pos
        self.n_neg = num_neg

        # num_word_repeats = num_word_repeats if split == "train" else 1
        # words_seen = self.dataset.words_seen
        # self.words = [word for word in words_seen for _ in range(num_word_repeats)]

        # Use Leanne's order
        self.word_audio = [
            (word, audio)
            for word, audios in self.dataset.word_to_audios.items()
            for audio in audios
        ]
        self.word_audio = sorted(self.word_audio, key=lambda x: x[0])
        self.load_audio = AudioFeaturesLoader(feature_type_audio, split, langs)
        self.load_image = get_image_loader(feature_type_image, split)

        if split == "valid" and to_fix_validation_samples:
            suffix = "{}-P{}-N{}".format("_".join(langs), num_pos, num_neg)
            path = f"data/filelists/validation-samples-{suffix}.json"
            with open(path, "r") as f:
                validation_samples = json.load(f)
            self.get_positives_and_negatives = lambda i: validation_samples[i]

    def get_positives_and_negatives(self, i):
        def sample_neg(data, word):
            words = set(self.dataset.words_seen) - set([word])
            words = random.choices(list(words), k=self.n_neg)
            return [random.choice(data[word]) for word in words]

        word, audio_pos = self.word_audio[i]
        images_pos = random.choices(self.dataset.word_to_images[word], k=self.n_pos)
        audios_pos = random.choices(self.dataset.word_to_audios[word], k=self.n_pos - 1)
        audios_pos = [audio_pos] + audios_pos

        images_neg = sample_neg(self.dataset.word_to_images, word)
        audios_neg = sample_neg(self.dataset.word_to_audios, word)

        return {
            "audios-pos": audios_pos,
            "audios-neg": audios_neg,
            "images-pos": images_pos,
            "images-neg": images_neg,
        }

    def __getitem__(self, i):
        # worker_info = torch.utils.data.get_worker_info()
        # print("worker:", worker_info.id)
        # print("index: ", i)
        # print()

        transform_image = TRANSFORM_IMAGE[self.split]
        samples = self.get_positives_and_negatives(i)

        data_pos = [
            {
                "index": i,
                "audio": self.load_audio(audio_name),
                "image": self.load_image(image_name),
                "label": 1,
            }
            for image_name, audio_name in zip(
                samples["images-pos"], samples["audios-pos"]
            )
        ]

        data_neg = [
            {
                "index": i,
                "audio": self.load_audio(audio_name),
                "image": self.load_image(image_name),
                "label": 0,
            }
            for image_name, audio_name in zip(
                samples["images-neg"], samples["audios-neg"]
            )
        ]

        return data_pos + data_neg

    def __len__(self):
        return len(self.word_audio)


class TripletMEDataset(Dataset):
    def __init__(
        self,
        split,
        langs,
        num_pos: int,
        num_neg: int,
        feature_type_audio: str,
        feature_type_image: str,
        # num_word_repeats: int,
        to_fix_validation_samples: bool = True,
    ):
        super(TripletMEDataset).__init__()

        assert split in ("train", "valid")
        assert len(langs) in {1, 2}

        lang1, *lang2 = langs

        if len(lang2) == 0:
            lang2 = lang1
        elif len(lang2) == 1:
            lang2 = lang2[0]
        else:
            assert False

        self.lang1 = lang1
        self.lang2 = lang2

        self.split = split
        self.dataset = MEDataset(split, langs)

        self.n_pos = num_pos
        self.n_neg = num_neg

        # num_word_repeats = num_word_repeats if split == "train" else 1
        # words_seen = self.dataset.words_seen
        # self.words = [word for word in words_seen for _ in range(num_word_repeats)]

        # Use Leanne's order
        self.word_audio = [
            (word, audio)
            for word, audios in self.dataset.word_to_audios.items()
            for audio in audios
            # if audio["lang"] == lang1
        ]
        self.word_audio = sorted(self.word_audio, key=lambda x: x[0])
        self.load_audio = AudioFeaturesLoader(feature_type_audio, split, langs)
        self.load_image = get_image_loader(feature_type_image, split)

        self.word_to_audios_per_lang = {
            lang: group_by_word(
                [audio for audio in self.dataset.audio_files if audio["lang"] == lang]
            )
            for lang in langs
        }

        # if split == "valid" and to_fix_validation_samples:
        #     suffix = "{}-P{}-N{}".format("_".join(langs), num_pos, num_neg)
        #     path = f"data/filelists/validation-samples-{suffix}.json"
        #     with open(path, "r") as f:
        #         validation_samples = json.load(f)
        #     self.get_positives_and_negatives = lambda i: validation_samples[i]

    def get_other_lang(self, lang):
        return self.lang2 if lang == self.lang1 else self.lang1

    def get_positives_and_negatives(self, i):
        def sample_neg(data, word):
            words = set(self.dataset.words_seen) - set([word])
            words = random.choices(list(words), k=self.n_neg)
            return [random.choice(data[word]) for word in words]

        word, audio1_pos = self.word_audio[i]
        lang1 = audio1_pos["lang"]
        lang2 = self.get_other_lang(lang1)

        images_pos = random.choices(self.dataset.word_to_images[word], k=self.n_pos)
        images_neg = sample_neg(self.dataset.word_to_images, word)

        audios1_pos = random.choices(
            self.word_to_audios_per_lang[lang1][word],
            k=self.n_pos - 1,
        )
        audios1_pos = [audio1_pos] + audios1_pos
        audios1_neg = sample_neg(self.word_to_audios_per_lang[lang1], word)

        audios2_pos = random.choices(
            self.word_to_audios_per_lang[lang2][word],
            k=self.n_pos,
        )
        audios2_neg = sample_neg(self.word_to_audios_per_lang[lang2], word)

        return {
            "audios1-pos": audios1_pos,
            "audios1-neg": audios1_neg,
            "audios2-pos": audios2_pos,
            "audios2-neg": audios2_neg,
            "images-pos": images_pos,
            "images-neg": images_neg,
        }

    def __getitem__(self, i):
        samples = self.get_positives_and_negatives(i)

        data_pos = [
            {
                "index": i,
                "image": self.load_image(image_name),
                "audio1": self.load_audio(audio1_name),
                "audio2": self.load_audio(audio2_name),
                "label": 1,
            }
            for image_name, audio1_name, audio2_name in zip(
                samples["images-pos"],
                samples["audios1-pos"],
                samples["audios2-pos"],
            )
        ]

        data_neg = [
            {
                "index": i,
                "image": self.load_image(image_name),
                "audio1": self.load_audio(audio1_name),
                "audio2": self.load_audio(audio2_name),
                "label": 0,
            }
            for image_name, audio1_name, audio2_name in zip(
                samples["images-neg"],
                samples["audios1-neg"],
                samples["audios2-neg"],
            )
        ]

        return data_pos + data_neg

    def __len__(self):
        return len(self.word_audio)


class PairedTestDataset(Dataset):
    def __init__(self, langs, feature_type_audio, feature_type_image, test_name):
        # assert test_name in {"familiar-familiar", "novel-familiar"}
        super(PairedTestDataset).__init__()

        split = "test"
        self.load_audio = AudioFeaturesLoader(feature_type_audio, split, langs)
        self.load_image = get_image_loader(feature_type_image, split)

        langs_str = "_".join(langs)
        with open(f"data/filelists/{test_name}-{langs_str}-test.json", "r") as f:
            self.data_pairs = json.load(f)

    def __getitem__(self, index: int):
        datum = self.data_pairs[index]
        assert datum["audio"]["word-en"] == datum["image-pos"]["word-en"]
        assert datum["audio"]["word-en"] != datum["image-neg"]["word-en"]
        return {
            "audio": self.load_audio(datum["audio"]),
            "image-pos": self.load_image(datum["image-pos"]),
            "image-neg": self.load_image(datum["image-neg"]),
        }

    def __len__(self):
        return len(self.data_pairs)



def collate_audio(batch, key="audio"):
    audio = pad_sequence([datum[key].T for datum in batch], batch_first=True)
    audio = audio.permute(0, 2, 1)
    audio_length = torch.tensor([datum[key].shape[1] for datum in batch])
    return {key: audio, f"{key}-length": audio_length}


def collate_with_audio(batch):
    audio = collate_audio(batch, key="audio")
    rest = [dissoc(datum, "audio") for datum in batch]
    rest = default_collate(rest)
    return {**audio, **rest}


def collate_nested(batch):
    B = len(batch)
    N = len(batch[0])
    batch = [datum for data in batch for datum in data]
    batch = collate_with_audio(batch)
    return {key: data.view(B, N, *data.shape[1:]) for key, data in batch.items()}


def collate_nested_two_audios(batch):
    B = len(batch)
    N = len(batch[0])
    batch = [datum for data in batch for datum in data]
    audio1 = collate_audio(batch, key="audio1")
    audio2 = collate_audio(batch, key="audio2")
    rest = [dissoc(datum, "audio1", "audio2") for datum in batch]
    rest = default_collate(rest)
    batch = {**audio1, **audio2, **rest}
    return {key: data.view(B, N, *data.shape[1:]) for key, data in batch.items()}


if __name__ == "__main__":
    num_pos = 1
    num_neg = 9
    dataset = TripletMEDataset(
        split="train",
        langs=("english", "french"),
        num_pos=num_pos,
        num_neg=num_neg,
        feature_type_audio="wavlm-base-plus",
        feature_type_image="dino-resnet50",
    )
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=4,
        collate_fn=collate_nested_two_audios,
    )

    for batch in dataloader:
        import pdb

        pdb.set_trace()
