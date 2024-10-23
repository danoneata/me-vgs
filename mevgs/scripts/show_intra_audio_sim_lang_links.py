import random
import pdb

from copy import deepcopy

import click
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from toolz import concat
from matplotlib import pyplot as plt

import torch
from torchdata.datapipes.map import SequenceWrapper

from mevgs.config import CONFIGS
from mevgs.data import MEDataset, AudioFeaturesLoader, collate_audio
from mevgs.predict import load_model
from mevgs.utils import cache_json, cache_np


SPLIT = "test"
RUN = "a"


MODEL_SPECS = {
    "en": {
        "num-langs": 1,
        "lang-links": False,
        "config-name": f"26{RUN}",
    },
    "fr": {
        "num-langs": 1,
        "lang-links": False,
        "config-name": f"26{RUN}-fr",
    },
    "en-fr": {
        "num-langs": 2,
        "lang-links": False,
        "config-name": f"26{RUN}-en-fr",
    },
    "en-links": {
        "num-langs": 1,
        "lang-links": True,
        "config-name": f"clip-lang-links-{RUN}-md-en",
    },
    "fr-links": {
        "num-langs": 1,
        "lang-links": True,
        "config-name": f"clip-lang-links-{RUN}-md-fr",
    },
    "en-fr-links": {
        "num-langs": 2,
        "lang-links": True,
        "config-name": f"clip-lang-links-{RUN}-md-en-fr",
    },
}


LINKS_SUFFIX = {
    False: "",
    True: "-links",
}


LINKS_TO_BILINGUAL_CONFIG = {
    False: "26{}-en-fr",
    True: "clip-lang-links-{}-md-en-fr",
}


for v in "abcde":
    for links in [False, True]:
        suffix = LINKS_SUFFIX[links]
        MODEL_SPECS[f"en-fr{suffix}-{v}"] = {
            "num-langs": 2,
            "lang-links": links,
            "config-name": LINKS_TO_BILINGUAL_CONFIG[links].format(v),
        }


def select_audios(dataset, num_audios_per_word):
    words = dataset.words_seen + dataset.words_unseen
    audios = (
        random.choices(dataset.word_to_audios[word], k=num_audios_per_word)
        for word in words
    )
    return list(concat(audios))


def extract_embeddings(dataset, audios, model_spec):
    config_name = MODEL_SPECS[model_spec]["config-name"]
    config = CONFIGS[config_name]
    config = deepcopy(config)

    feature_type_audio = config["data"]["feature_type_audio"]
    load_audio = AudioFeaturesLoader(feature_type_audio, SPLIT, dataset.langs)
    batch_size = 64

    dp = SequenceWrapper(audios)
    dp = dp.map(lambda audio: {"audio": load_audio(audio)})
    dp = dp.batch(batch_size)
    dp = dp.map(collate_audio)

    model = load_model(config_name, config)

    with torch.no_grad():
        embs = [
            model.audio_enc(batch["audio"], batch["audio-length"])
            for batch in dp
        ]
        embs = torch.cat(embs, dim=0)
        embs = torch.squeeze(embs, -1)
        embs = model.l2_normalize(embs, dim=1)
        embs = embs.numpy()

    return embs


def compute_intra_similiarities(audios, embs):
    def sim1(embs, word):
        idxs = [i for i, audio in enumerate(audios) if audio["word-en"] == word]
        embs_word = embs[idxs]
        sims = embs_word @ embs_word.T
        sims = np.triu(sims, k=1)
        N = len(idxs)
        M = N * (N - 1) // 2
        sim = sims.sum() / M
        return {
            "word": word,
            "sim": sim,
        }

    words = set(audio["word-en"] for audio in audios)
    return [sim1(embs, word) for word in words]


def compute_intra_similiarities_model_spec(dataset, audios, model_spec):
    assert len(dataset.langs) == 1
    lang = dataset.langs[0]
    path = f"output/scripts-cache/features-{model_spec}-{lang}.npy"
    embs = cache_np(path, extract_embeddings, dataset, audios, model_spec=model_spec)
    sims = compute_intra_similiarities(audios, embs)
    return [{**sim, **MODEL_SPECS[model_spec], "model": model_spec} for sim in sims]


def plot(sims, order):
    df = pd.DataFrame(sims)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="word", y="sim", hue="model", order=order, data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    st.set_page_config(layout="wide")
    st.write(df)
    st.pyplot(fig)


@click.command()
@click.option("--lang")
def main(lang):
    assert lang in ("english", "french")
    dataset = MEDataset("test", (lang,))

    path = f"output/scripts-cache/selected-audios-{lang}.json"
    audios = cache_json(path, select_audios, dataset, num_audios_per_word=30)

    lang_short = lang[:2]
    sims = [
        sim
        for model_spec in [lang_short, lang_short + "-links", "en-fr", "en-fr-links"]
        for sim in compute_intra_similiarities_model_spec(dataset, audios, model_spec)
    ]
    order = sorted(dataset.words_seen) + sorted(dataset.words_unseen)
    plot(sims, order)


if __name__ == "__main__":
    main()
