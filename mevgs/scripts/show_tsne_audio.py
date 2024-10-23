import random
import pdb

import click
import colorcet as cc
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from toolz import concat
from matplotlib import pyplot as plt

from sklearn import manifold

from mevgs.data import Language, load_dictionary
from mevgs.utils import cache_json, cache_np
from mevgs.scripts.show_intra_audio_sim_lang_links import (
    MEDataset,
    MODEL_SPECS,
    extract_embeddings,
    select_audios,
)


# st.set_page_config(layout="wide")
sns.set_theme(context="talk", style="whitegrid", font="Arial")

LANGS = ["english", "french"]  # type: list[Language]
datasets = {lang: MEDataset("test", (lang,)) for lang in LANGS}

paths = {lang: f"output/scripts-cache/selected-audios-{lang}.json" for lang in LANGS}
audios = {
    lang: cache_json(paths[lang], select_audios, datasets[lang], num_audios_per_word=30)
    for lang in LANGS
}

dictionary = load_dictionary()
en_to_fr = {row["english"]: row["french"] for row in dictionary}


def get_type(dataset, word_en):
    if word_en in dataset.words_seen:
        return "familiar"
    elif word_en in dataset.words_unseen:
        return "novel"
    else:
        assert False


def translate(word_en, lang):
    if lang == "english":
        return word_en
    elif lang == "french":
        return en_to_fr[word_en]
    else:
        assert False


def make_plot(model_spec, word_type, ax):

    def find_closest_point(df, row):
        df_ = df[df.word == row.word]
        df_["dist"] = (df_.x - row.x).abs() + (df_.y - row.y).abs()
        return df_.sort_values("dist").iloc[0][["x", "y"]].values

    def find_random_point(df, row):
        df_ = df[df.word == row.word]
        return df_.sample(1).iloc[0][["x", "y"]].values

    def add_texts(ax, df):
        from adjustText import adjust_text

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

    paths = {
        lang: f"output/scripts-cache/features-{model_spec}-{lang}.npy" for lang in LANGS
    }
    embs = {
        lang: cache_np(
            paths[lang], extract_embeddings, datasets[lang], audios[lang], model_spec
        )
        for lang in LANGS
    }
    embs = np.concatenate([embs[lang] for lang in LANGS])

    t_sne = manifold.TSNE()
    embs_2d = t_sne.fit_transform(embs)
    x, y = embs_2d.T

    metadata = [
        {
            "word-en": audio["word-en"],
            "type": get_type(datasets[lang], audio["word-en"]),
            "lang": audio["lang"],
            "word": translate(audio["word-en"], lang)
        }
        for lang in LANGS
        for audio in audios[lang]
    ]

    df1 = pd.DataFrame({"x": x, "y": y})
    df2 = pd.DataFrame(metadata)
    df = pd.concat([df1, df2], axis=1)

    df1 = df[df["type"] == word_type]
    df2 = df[df["type"] != word_type]

    n_words = len(set(df1["word-en"]))
    palette = sns.color_palette(cc.glasbey, n_colors=n_words)

    sns.scatterplot(
        df2,
        x="x",
        y="y",
        color="gray",
        style="lang",
        # alpha=0.5,
        legend=False,
        ax=ax,
    )
    sns.scatterplot(
        data=df1,
        x="x",
        y="y",
        hue="word-en",
        style="lang",
        palette=palette,
        legend=False,
        ax=ax,
    )
    add_texts(ax, df1)


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


# fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
# for r, links in enumerate([False, True]):
#     for c, word_type in enumerate(["familiar", "novel"]):
#         suffix_links = "-links" if links else ""
#         model_spec = f"en-fr{suffix_links}"
#         ax = axs[r, c]
#         make_plot(model_spec, word_type, ax)
#         clean_axes(ax)
#         ax.set_title(
#             "links: {} · words: {}".format(
#                 "✓" if links else "✗",
#                 word_type,
#             )
#         )
fig, ax = plt.subplots(figsize=(6, 6))
make_plot("en-fr", "familiar", ax)
clean_axes(ax)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 6))
make_plot("en-fr-links", "familiar", ax)
clean_axes(ax)
st.pyplot(fig)

# fig.savefig("output/plots/tsne-audio-familiar.pdf")
# fig.savefig("output/plots/tsne-audio-familiar.png")