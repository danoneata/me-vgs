import click
import pdb
import random

import h5py
import librosa
import torch
import numpy as np

from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
from tqdm import tqdm

from mevgs.data import MEDataset, get_audio_path, Split


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            # padding=True,
            # max_length=16_000,
            # truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_attentions=True,
                # output_hidden_states=False,
            )
        return outputs.last_hidden_state


FEATURE_EXTRACTORS = {
    "wav2vec2-base": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-base"
    ),
    "wav2vec2-large": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large"
    ),
    "wav2vec2-large-lv60": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-lv60"
    ),
    "wav2vec2-large-robust": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-robust"
    ),
    "wav2vec2-large-xlsr-53": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-xlsr-53"
    ),
    "wav2vec2-xls-r-300m": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-300m"
    ),
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
    "wavlm-base": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base"
    ),
    "wavlm-base-sv": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-sv"
    ),
    "wavlm-base-plus": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-plus"
    ),
    "wavlm-large": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-large"
    ),
}


SPLITS = "train valid test".split()
@click.command()
@click.option("-s", "--split", type=click.Choice(SPLITS), required=True)
@click.option("-f", "--feature-type", type=str, required=True)
def main(split: Split, feature_type: str):
    SAMPLING_RATE = 16_000
    DATASET_NAME = "me-dataset"
    LANGS = ("english",)
    dataset = MEDataset(split, LANGS)
    dataset = dataset.audio_files

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()

    def load_audio(datum):
        path = get_audio_path(datum)
        y, _ = librosa.load(path, sr=SAMPLING_RATE)
        return y

    def extract1(audio):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.squeeze(dim=0)
        feature = feature.cpu().numpy()
        return feature

    langs = "_".join(LANGS)
    path_hdf5 = f"output/features/{DATASET_NAME}-{split}-{langs}-{feature_type}.h5"

    with h5py.File(path_hdf5, "a") as f:
        for i in tqdm(range(len(dataset))):
            datum = dataset[i]
            try:
                group = f.create_group(datum["name"])
            except ValueError:
                group = f[datum["name"]]
                if "feature" in group:
                    continue
                else:
                    pass

            audio = load_audio(datum)
            feature = extract1(audio)
            group.create_dataset("feature", data=feature)

    # def get_feature_dim(f):
    #     for key in f.keys():
    #         return f[key]["feature"].shape[0]

    # with h5py.File(path_hdf5, "r") as f:
    #     num_samples = len(dataset)
    #     X = np.zeros((num_samples, get_feature_dim(f)))

    #     for i in tqdm(range(len(dataset))):
    #         filename = datum["name"]
    #         group = f[filename]
    #         X[i] = np.array(group["feature"])

    # path = f"output/features/{DATASET_NAME}-{split}-{feature_type}.h5"
    # np.savez(path, X=X)


if __name__ == "__main__":
    main()
