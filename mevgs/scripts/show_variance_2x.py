"""
Quantify the variance for the experimens with double data for the monolingual models.
"""

import pdb
import os

from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt

from mevgs.constants import LANGS_SHORT, LANG_LONG_TO_SHORT
from mevgs.predict import CONFIGS, MODELS
from mevgs.utils import read_json, cache_np
from mevgs.scripts.show_model_comparison import (
    compute_audio_embeddings_given_model,
    compute_image_embeddings_given_model,
)
from mevgs.scripts.show_interspeech25_sec42 import (
    compute_var,
    filter,
    select_by_word,
    WORDS,
)


def compute_variance_full(data, embs, *, words_type):
    to_keep = lambda datum: datum["word-en"] in WORDS[words_type]
    data, embs = filter(data, embs, to_keep)
    return compute_var(embs)


def compute_variance_concept(data, embs, *, words_type):
    vars = [compute_var(select_by_word(data, embs, word)) for word in WORDS[words_type]]
    return np.mean(vars)


COMPUTE_VARIANCE_FUNCS = {
    "full": compute_variance_full,
    "concept": compute_variance_concept,
}


CACHE_NAMES = {
    "-".join(langs): "{}_links-no_size-md".format("-".join(sorted(langs)))
    for langs in list(combinations(LANGS_SHORT, 1)) + list(combinations(LANGS_SHORT, 2))
}


MODEL_TO_CONFIG = {
    t + lang: t + lang + "_size-md_seed-{}"
    for lang in LANGS_SHORT
    for t in "2x 1x".split()
}


def load_model_and_config(model_name, seed):
    config_name = MODEL_TO_CONFIG[model_name].format(seed)
    config = CONFIGS[config_name] 
    model = MODELS[config_name]()
    model.to("cuda")
    return model, config


def load_data(model_name, modality, seed):
    FUNCS = {
        "audio": compute_audio_embeddings_given_model,
        "image": compute_image_embeddings_given_model,
    }

    def compute_embeddings(model_name, data, seed):
        model, config = load_model_and_config(model_name, seed)
        return FUNCS[modality](model, config, data)

    path = f"output/show-model-comparison/{modality}-data-ss-30.json"
    data = read_json(path)

    name = CACHE_NAMES.get(model_name, model_name)
    path = f"output/show-model-comparison/embeddings-{modality}/{name}_seed-{seed}.npy"
    embs = cache_np(
        path,
        compute_embeddings,
        model_name=model_name,
        data=data,
        seed=seed,
    )

    return data, embs


def do(lang):
    models = [
        lang,
        "2x" + lang,
        "1x" + lang,
    ] + ["{}-{}".format(*sorted([lang, lang2])) for lang2 in LANGS_SHORT if lang2 != lang]

    def to_keep_lang(modality, datum):
        return modality == "image" or LANG_LONG_TO_SHORT[datum["lang"]] in lang

    results = [
        {
            "model": model,
            "modality": modality,
            "words_type": words_type,
            "var_type": var_type,
            "value": COMPUTE_VARIANCE_FUNCS[var_type](
                *filter(
                    *load_data(model, modality, seed),
                    partial(to_keep_lang, modality),
                ),
                words_type=words_type,
            ),
        }
        for model in models
        for modality in ["audio", "image"]
        for seed in "abcde"
        for words_type in ["seen", "unseen"]
        for var_type in ["full"]
    ]
    df = pd.DataFrame(results)
    sns.set(style="whitegrid", font="Arial", context="poster")
    fig = sns.catplot(
        df,
        x="modality",
        y="value",
        hue="words_type",
        col="model",
        kind="bar",
        errorbar="sd",
        aspect=0.7,
    )
    st.pyplot(fig)


def main():
    do("en")
    do("fr")
    do("nl")


if __name__ == "__main__":
    main()
