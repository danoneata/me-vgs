import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

from functools import partial
from matplotlib import pyplot as plt
from procrustes import orthogonal
from sklearn import manifold

from adjustText import adjust_text
from toolz import first

from mevgs.data import MEDataset, load_dictionary
from mevgs.utils import cache_np, read_json, mapt, implies
from mevgs.scripts.show_model_comparison import (
    compute_audio_embeddings,
    LANG_SHORT_TO_LONG,
)


def aggregate_by_word(emb, meta, lang=None):
    def aggregate1(emb, meta, word):
        idxs = [
            i
            for i, m in enumerate(meta)
            if m["word-en"] == word
            and implies(lang is not None, m["lang"] == LANG_SHORT_TO_LONG[lang])
        ]
        return emb[idxs].mean(0)

    return np.vstack([aggregate1(emb, meta, word) for word in WORDS_SEEN])


def align(emb1, emb2, meta1, meta2, lang):
    def error(emb1, emb2):
        return np.linalg.norm(emb1 - emb2)

    emb_agg_1 = aggregate_by_word(emb1, meta1, lang)
    emb_agg_2 = aggregate_by_word(emb2, meta2, lang)

    result = orthogonal(emb_agg_1, emb_agg_2)
    emb_agg_1_out = emb_agg_1 @ result.t
    emb_1_out = emb1 @ result.t

    error_in = error(emb_agg_1, emb_agg_2)
    error_out = error(emb_agg_1_out, emb_agg_2)

    return emb_1_out, error_out


def add_texts(ax, df):
    def find_closest_point(df, row):
        # df_ = df[df.word == row.word]
        df_ = df.loc[df.word == row.word]
        df_["dist"] = (df_.x - row.x).abs() + (df_.y - row.y).abs()
        return df_.sort_values("dist").iloc[0][["x", "y"]].values

    def find_random_point(df, row):
        df_ = df[df.word == row.word]
        return df_.sample(1).iloc[0][["x", "y"]].values

    locs = df.groupby(["word", "lang"])["x", "y"].mean()
    locs = locs.reset_index()
    texts = [
        ax.text(
            # *find_random_point(df, row),
            *find_closest_point(df, row),
            row.word,
            ha="center",
            va="center",
            size=10,
        )
        for _, row in locs.iterrows()
    ]
    adjust_text(
        texts,
        x=df.x.values,
        y=df.y.values,
        ax=ax,
        force_points=0.5,
        # force_text=0.5,
        # lim=15,
        # ha="center",
        arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
    )


VOCAB = load_dictionary()


def translate(word_en, lang):
    entry = first(entry for entry in VOCAB if entry["english"] == word_en)
    return entry[lang]


def show_tsne(embs, data, langs, ax):
    tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
    embs_tsne = tsne.fit_transform(embs)

    df = pd.DataFrame(embs_tsne, columns=["x", "y"])
    df["word-en"] = [d["word-en"] for d in data]
    df["lang"] = [d["lang"] for d in data]
    df["word"] = [translate(d["word-en"], d["lang"]) for d in data]

    markers = {
        "english": "o",
        "french": "X",
        "dutch": "s",
    }

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="word-en",
        style="lang",
        legend=False,
        markers=markers,
        ax=ax,
    )
    ax.set_title("-".join(langs))
    add_texts(ax, df)


LANGS_LONG = ["english", "french", "dutch"]
DATASET = MEDataset("test", LANGS_LONG)  # type: ignore
WORDS_SEEN = set(DATASET.words_seen)


def load_data_and_embs(langs, size, variant):
    path = f"output/show-model-comparison/audio-data-ss-30.json"
    audio_data = read_json(path)

    langs_str = "-".join(langs)
    model_name = f"{langs_str}_links-no_size-{size}"
    path = f"output/show-model-comparison/embeddings/{model_name}_seed-{variant}.npy"
    embs = cache_np(
        path,
        compute_audio_embeddings,
        model_name=model_name,
        test_lang=langs[0],
        audio_data=audio_data,
        seed=variant,
    )

    # Filter by language and seen words
    langs_long = [LANG_SHORT_TO_LONG[lang] for lang in langs]
    idxs_audio_data = [
        (idx, datum)
        for idx, datum in enumerate(audio_data)
        if datum["lang"] in langs_long and datum["word-en"] in WORDS_SEEN
    ]
    idxs, audio_data = mapt(list, zip(*idxs_audio_data))
    embs = embs[idxs]

    return audio_data, embs


def do(langs, size, seed):
    VARIANTS = "abcde"
    variant = VARIANTS[seed]

    lang1, lang2 = langs
    load_data_and_embs_1 = partial(load_data_and_embs, size=size, variant=variant)

    data1, embs1 = load_data_and_embs_1([lang1])
    data2, embs2 = load_data_and_embs_1([lang2])
    data3, embs3 = load_data_and_embs_1(langs)

    # embs1, error1 = align(embs1, embs3, data1, data3, lang1)
    # embs2, error2 = align(embs2, embs3, data2, data3, lang2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    show_tsne(embs1, data1, [lang1], axs[0])
    show_tsne(embs3, data3, langs, axs[1])
    show_tsne(embs2, data2, [lang2], axs[2])
    st.pyplot(fig)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    with st.sidebar:
        langs = st.multiselect(
            "Languages:",
            ["en", "fr", "nl"],
            help="Select two languages to compare",
        )
        size = st.selectbox("Size:", ["sm", "md", "lg"])
        seed = st.selectbox("Seed:", list(range(5)))

    if len(langs) != 2:
        st.error("Please select exactly two languages.")
    else:
        do(langs, size, seed)
