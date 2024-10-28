import pdb
import random

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import networkx as nx

from itertools import combinations, product
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr
from tqdm import tqdm

from sklearn.decomposition import PCA
from adjustText import adjust_text
from toolz import first, concat

# from mevgs.data import MEDataset, load_dictionary
from mevgs.utils import cache_np, read_json, mapt
from scipy.stats import kendalltau
from mevgs.scripts.show_crosslingual_audio_alignment import (
    align,
    load_data_and_embs,
    aggregate_by_word,
    WORDS_SEEN,
)

# from mevgs.scripts.show_model_comparison import (
#     compute_audio_embeddings,
#     LANG_SHORT_TO_LONG,
# )


sns.set(style="whitegrid")

SEEDS = "abcde"
SIZES = "sm md lg".split()
LANGS = "en fr nl".split()
LANGS_COMBS = list(concat(combinations(LANGS, n) for n in [1, 2]))


def compute_sims(modality, train_langs, test_lang, size, seed):
    data, embs = load_data_and_embs(modality, train_langs, size, seed)
    embs = aggregate_by_word(embs, data, test_lang)
    # sims = np.corrcoef(embs)
    return embs @ embs.T


def compute_rsm(modality, train_langs, test_lang, size, seed):
    sims = compute_sims(modality, train_langs, test_lang, size, seed)
    idxs = np.triu_indices_from(sims, k=1)
    return sims[idxs]


def compute_rsa(model_params_1, model_params_2, use_same_lang=True):
    if use_same_lang:
        langs1 = model_params_1["train_langs"]
        langs2 = model_params_2["train_langs"]
        lang_common = set(langs1) & set(langs2)
        lang_common = first(lang_common)
    else:
        lang_common = None
    rsm1 = compute_rsm(**model_params_1, test_lang=lang_common)
    rsm2 = compute_rsm(**model_params_2, test_lang=lang_common)
    return spearmanr(rsm1, rsm2).correlation
    # return np.corrcoef(rsm1, rsm2)[0, 1]
    # return kendalltau(rsm1, rsm2).correlation


def show_across_langs():
    with st.sidebar:
        size = st.selectbox("Size", SIZES)
        seed = st.selectbox("Seed", SEEDS)

    common_params = {
        "modality": "audio",
        "size": size,
        "seed": seed,
    }

    data = [
        {
            "langs1": langs1,
            "langs2": langs2,
            "rsa": compute_rsa(
                {**common_params, "train_langs": langs1},
                {**common_params, "train_langs": langs2},
            ),
        }
        for langs1, langs2 in tqdm(product(LANGS_COMBS, LANGS_COMBS))
        if langs1 != langs2 and set(langs1) & set(langs2)
    ]

    df = pd.DataFrame(data)
    st.write(df)

    graph = nx.Graph()
    for entry in data:
        langs1_str = "-".join(entry["langs1"])
        langs2_str = "-".join(entry["langs2"])
        rsa_value = entry["rsa"]
        graph.add_edge(langs1_str, langs2_str, weight=rsa_value)

    pos = {
        "en": (-1, -1),
        "fr": (+1, -1),
        "nl": (0, +1),
        "en-fr": (0, -2),
        "en-nl": (-1, 0),
        "fr-nl": (+1, 0),
    }
    edges, weights = zip(*nx.get_edge_attributes(graph, "weight").items())

    fig, ax = plt.subplots()

    nx.draw_networkx(
        graph,
        pos,
        node_color="lightblue",
        with_labels=True,
        node_size=3000,
        font_size=10,
        edgelist=edges,
        edge_color=weights,
        edge_cmap=cm.Blues,
        width=2,
        ax=ax,
    )

    edge_labels = {
        (langs1, langs2): f"{weight:.2f}"
        for (langs1, langs2), weight in nx.get_edge_attributes(graph, "weight").items()
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=8, ax=ax
    )

    fig.tight_layout()
    st.pyplot(fig)


def show_across_seeds():
    def do1(langs, size):
        common_params = {
            "modality": "audio",
            "size": size,
            "train_langs": langs,
        }
        results = [
            {
                "seed1": seed1,
                "seed2": seed2,
                "rsa": compute_rsa(
                    {**common_params, "seed": seed1},
                    {**common_params, "seed": seed2},
                ),
            }
            # for seed1, seed2 in list(combinations(SEEDS, 2))
            for seed1, seed2 in list(product(SEEDS, SEEDS))
        ]
        df = pd.DataFrame(results)
        df = df.pivot(index="seed1", columns="seed2", values="rsa")
        return df

    def get_rsa_mean_std(df):
        idxs = np.triu_indices_from(df, k=1)
        rsa_values = df.values[idxs]
        return rsa_values.mean(), rsa_values.std()

    st.markdown("## RSA across seeds")
    st.markdown(
        "Measure how similar are the audio embeddings spaces for the familiar words for two different seeds, given a training language and a model size."
    )

    rsa_agg = []

    for langs in LANGS_COMBS:
        cols = st.columns(len(SIZES))
        for col, size in zip(cols, SIZES):
            df = do1(langs, size)
            fig, ax = plt.subplots()
            rsa_mean, rsa_std = get_rsa_mean_std(df)
            sns.heatmap(
                df,
                annot=True,
                square=True,
                vmin=0,
                vmax=1,
                cbar=False,
                fmt=".1f",
                ax=ax,
            )
            ax.set_title(
                "{} · {} · {:.2f}±{:.1f}".format(
                    "-".join(langs),
                    size,
                    rsa_mean,
                    2 * rsa_std,
                )
            )
            col.pyplot(fig)
            rsa_agg.append(
                {
                    "langs": "-".join(langs),
                    "size": size,
                    "mean": rsa_mean,
                }
            )

    df = pd.DataFrame(rsa_agg)
    df = df.pivot(index="size", columns="langs", values="mean")
    fig, ax = plt.subplots()
    sns.heatmap(
        df, annot=True, square=True, vmin=0, vmax=1, cbar=False, fmt=".2f", ax=ax
    )
    ax.set_title("RSA mean")

    st.markdown("Aggregated results across the ten seed combinations (5 choose 2).")
    st.pyplot(fig)


CONFIGS1 = [
    {"modality": "audio", "train_langs": None},
    {"modality": "audio", "train_langs": ("en",)},
    {"modality": "audio", "train_langs": ("fr",)},
    {"modality": "audio", "train_langs": ("nl",)},
    {"modality": "image", "train_langs": None},
    {"modality": "image", "train_langs": ("en",)},
    {"modality": "image", "train_langs": ("fr",)},
    {"modality": "image", "train_langs": ("nl",)},
]
CONFIGS2 = [
    {"modality": "audio", "train_langs": None},
    {"modality": "audio", "train_langs": ("en",)},
    {"modality": "audio", "train_langs": ("fr",)},
    {"modality": "audio", "train_langs": ("nl",)},
    {"modality": "audio", "train_langs": ("en", "fr")},
    {"modality": "audio", "train_langs": ("en", "nl")},
    {"modality": "audio", "train_langs": ("fr", "nl")},
    {"modality": "image", "train_langs": None},
    {"modality": "image", "train_langs": ("en",)},
    {"modality": "image", "train_langs": ("fr",)},
    {"modality": "image", "train_langs": ("nl",)},
]


def cfg_to_str(cfg):
    modality = cfg["modality"]
    train_langs = cfg["train_langs"]
    if not train_langs:
        langs = "random"
    else:
        langs = "-".join(cfg["train_langs"])
    return "{}/{}".format(modality, langs)


def get_seed_comparison(config1, config2):
    modalities = config1["modality"], config2["modality"]
    have_different_modalities = set(modalities) == {"audio", "image"}
    have_same_lang = config1["train_langs"] == config2["train_langs"]

    if config1 == config2:
        return "combinations"
    elif have_different_modalities and have_same_lang:
        return "zip"
    else:
        return "product"


def get_use_same_language(config1, config2):
    langs1 = config1["train_langs"]
    langs2 = config2["train_langs"]

    are_both_audio = config1["modality"] == config2["modality"] == "audio"
    have_both_langs = langs1 is not None and langs2 is not None
    have_common_lang = have_both_langs and set(langs1) & set(langs2)
    are_mono_and_bi = have_both_langs and (
        len(langs1) == 1 and len(langs2) == 2 or len(langs1) == 2 and len(langs2) == 1
    )

    if are_both_audio and have_common_lang:
        return True
    elif are_both_audio and not have_common_lang and are_mono_and_bi:
        return None
    else:
        return False


def show_aggregated(size):

    def compute_rsa_agg(config1, config2):
        # return use_same_lang
        SEED_PAIRS = {
            "product": list(product(SEEDS, SEEDS)),
            "combinations": list(combinations(SEEDS, 2)),
            "zip": list(zip(SEEDS, SEEDS)),
        }
        seed_pairs_type = get_seed_comparison(config1, config2)
        seed_pairs = SEED_PAIRS[seed_pairs_type]

        use_same_lang = get_use_same_language(config1, config2)
        if use_same_lang is None:
            return None

        results = [
            compute_rsa(
                {**config1, "size": size, "seed": seed1},
                {**config2, "size": size, "seed": seed2},
                use_same_lang=use_same_lang,
            )
            for seed1, seed2 in seed_pairs
        ]
        # return "{:.2f}±{:.1f}".format(np.mean(results), 2 * np.std(results))
        # return "{:.2f}".format(np.mean(results))
        return np.mean(results)

    results = [
        {
            "config1": cfg_to_str(config1),
            "config2": cfg_to_str(config2),
            "rsa": compute_rsa_agg(config1, config2),
        }
        for config1, config2 in tqdm(product(CONFIGS1, CONFIGS2))
    ]
    idxs = [cfg_to_str(cfg) for cfg in CONFIGS1]
    cols = [cfg_to_str(cfg) for cfg in CONFIGS2]
    df = pd.DataFrame(results)
    df = df.pivot(index="config1", columns="config2", values="rsa")
    df = df[cols]
    df = df.loc[idxs]
    st.write(df)
    print(df.to_csv(index=False))


def show_rsa_detailed():
    with st.sidebar:
        size = st.selectbox("Size", SIZES)
        config1 = st.selectbox("Config 1", CONFIGS1, format_func=cfg_to_str)
        config2 = st.selectbox("Config 2", CONFIGS2, format_func=cfg_to_str)

    # return use_same_lang
    SEED_PAIRS = {
        "product": list(product(SEEDS, SEEDS)),
        "combinations": list(combinations(SEEDS, 2)),
        "zip": list(zip(SEEDS, SEEDS)),
    }
    seed_pairs_type = get_seed_comparison(config1, config2)
    seed_pairs = SEED_PAIRS[seed_pairs_type]

    use_same_lang = get_use_same_language(config1, config2)
    if use_same_lang is None:
        return None

    if use_same_lang:
        langs1 = config1["train_langs"]
        langs2 = config1["train_langs"]
        test_lang = set(langs1) & set(langs2)
        test_lang = first(test_lang)
    else:
        test_lang = None

    rsms = {
        (seed1, seed2): (
            compute_rsm(**config1, size=size, test_lang=test_lang, seed=seed1),
            compute_rsm(**config2, size=size, test_lang=test_lang, seed=seed2),
        )
        for seed1, seed2 in seed_pairs
    }

    n_seeds = len(SEEDS)
    S = 1.5
    fig, axs = plt.subplots(
        nrows=n_seeds,
        ncols=n_seeds,
        sharex=True,
        sharey=True,
        figsize=(S * n_seeds, S * n_seeds),
    )
    corrs = []
    for r in range(n_seeds):
        for c in range(n_seeds):
            seed1 = SEEDS[r]
            seed2 = SEEDS[c]
            if (seed1, seed2) not in rsms:
                # axs[r, c].axis("off")
                pass
            else:
                rsm1, rsm2 = rsms[(seed1, seed2)]
                ρ = spearmanr(rsm1, rsm2).correlation
                corrs.append(ρ)
                axs[r, c].scatter(rsm1, rsm2)
                axs[r, c].set_title("ρ = {:.2f}".format(ρ))
            if r == n_seeds - 1:
                axs[r, c].set_xlabel(f"Seed: {seed2}")
            if c == 0:
                axs[r, c].set_ylabel(f"Seed: {seed1}")
    fig.tight_layout()

    st.markdown("## Scatter plots of the embeddings similarities")
    st.markdown(
        """
        - Test lang: `{}`.
        - Correlation (mean and 2 × std): {:.2f}±{:.1}.
        - Model `{}` is across rows.
        - Model `{}` is across columns.
        """.format(
            test_lang,
            np.mean(corrs),
            2 * np.std(corrs),
            cfg_to_str(config1),
            cfg_to_str(config2),
        )
    )
    st.pyplot(fig)
    st.markdown("---")

    cols = st.columns(2)
    seed1 = cols[0].selectbox("Seed 1", SEEDS)
    seed2 = cols[1].selectbox("Seed 2", [s2 for s1, s2 in seed_pairs if s1 == seed1])

    def show_rsm(config, ax):
        sims = compute_sims(**config)
        df = pd.DataFrame(data=sims, index=WORDS_SEEN, columns=WORDS_SEEN)
        mask = np.triu(sims)
        sns.heatmap(
            df,
            square=True,
            annot=True,
            annot_kws={"size": 7},
            fmt=".1f",
            cbar=False,
            vmin=0.5,
            vmax=1.0,
            mask=mask,
            ax=ax,
        )
        ax.set_title("{} · seed: {}".format(cfg_to_str(config), config["seed"]))

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 10),
        sharex=True,
        sharey=True,
    )
    show_rsm({**config1, "test_lang": test_lang, "size": size, "seed": seed1}, axs[0])
    show_rsm({**config2, "test_lang": test_lang, "size": size, "seed": seed2}, axs[1])
    st.markdown("## Similarity matrices")
    st.markdown(
        "Similarity matrices for familiar words for the two selected configurations."
    )
    st.pyplot(fig)
    st.markdown("---")

    data1, embs1 = load_data_and_embs(**config1, size=size, seed=seed1)
    data2, embs2 = load_data_and_embs(**config2, size=size, seed=seed2)

    embs2, _ = align(embs2, embs1, data2, data1, test_lang)

    proj = PCA(n_components=2)
    proj.fit(embs1)

    embs1_2d = proj.transform(embs1)
    embs2_2d = proj.transform(embs2)

    def add_texts(ax, df):
        df1 = df.groupby("word-en").mean().reset_index()
        for _, row in df1.iterrows():
            xy = df.loc[df["word-en"] == row["word-en"], ["x", "y"]].values
            xy = random.choice(xy.tolist())
            ax.annotate(
                row["word-en"],
                xy=xy,
                xytext=(row["x"], row["y"]),
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

    def show_2d(embs_2d, data, ax):
        # tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
        # embs_tsne = tsne.fit_transform(embs)
        df = pd.DataFrame(embs_2d, columns=["x", "y"])
        df["word-en"] = [d["word-en"] for d in data]
        # df["lang"] = [d["lang"] for d in data]
        # df["word"] = [translate(d["word-en"], d["lang"]) for d in data]
        # df["word"] = df["word-en"]

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
            # style="lang",
            legend=False,
            markers=markers,
            ax=ax,
        )
        # ax.set_title("-".join(langs))
        add_texts(ax, df)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    show_2d(embs1_2d, data1, axs[0])
    show_2d(embs2_2d, data2, axs[1])
    axs[0].set_title("{} · seed: {}".format(cfg_to_str(config1), seed1))
    axs[1].set_title("{} · seed: {}".format(cfg_to_str(config2), seed2))

    st.markdown("## Low-dimensional projections")
    st.markdown(
        """
    - Align the embedding from model 2 to those of model 1.
    - Learn PCA on embeddings from model 2 and project 2D embeddings from both models.
    """
    )
    st.pyplot(fig)


if __name__ == "__main__":
    # st.set_page_config(layout="wide")
    # show_across_langs()
    # st.markdown("---")
    # show_across_seeds()
    # for s in SIZES:
    #     st.markdown(f"## {s}")
    #     show_aggregated(s)

    show_rsa_detailed()
