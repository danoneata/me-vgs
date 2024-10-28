import pdb

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

from adjustText import adjust_text
from toolz import first, concat

# from mevgs.data import MEDataset, load_dictionary
from mevgs.utils import cache_np, read_json, mapt
from scipy.stats import kendalltau
from mevgs.scripts.show_crosslingual_audio_alignment import (
    load_data_and_embs,
    aggregate_by_word,
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


def compute_rsm(modality, train_langs, test_lang, size, seed):
    data, embs = load_data_and_embs(modality, train_langs, size, seed)
    embs = aggregate_by_word(embs, data, test_lang)
    # sims = np.corrcoef(embs)
    sims = embs @ embs.T
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


def show_aggregated():
    size = "sm"
    CONFIGS1 = [
        {"modality": "audio", "train_langs": ("en",)},
        {"modality": "audio", "train_langs": ("fr",)},
        {"modality": "audio", "train_langs": ("nl",)},
    ]
    CONFIGS2 = [
        {"modality": "audio", "train_langs": ("en",)},
        {"modality": "audio", "train_langs": ("fr",)},
        {"modality": "audio", "train_langs": ("nl",)},
        {"modality": "audio", "train_langs": ("en", "fr")},
        {"modality": "audio", "train_langs": ("en", "nl")},
        {"modality": "audio", "train_langs": ("fr", "nl")},
    ]

    def cfg_to_str(cfg):
        modality = cfg["modality"]
        langs = "-".join(cfg["train_langs"])
        return "{}/{}".format(modality, langs)

    def compute_rsa_agg(config1, config2):
        langs1 = config1["train_langs"]
        langs2 = config2["train_langs"]

        if config1 == config2:
            seed_pairs = list(combinations(SEEDS, 2))
        else:
            seed_pairs = list(product(SEEDS, SEEDS))

        are_both_audio = config1["modality"] == config2["modality"] == "audio"
        are_both_mono = len(langs1) == len(langs2) == 1
        have_common_lang = set(langs1) & set(langs2)

        if are_both_audio and are_both_mono:
            use_same_lang = False
        elif are_both_audio and not have_common_lang:
            return None
        else:
            use_same_lang = True

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
    cols = [cfg_to_str(cfg) for cfg in CONFIGS2]
    df = pd.DataFrame(results)
    df = df.pivot(index="config1", columns="config2", values="rsa")
    df = df[cols]
    st.write(df)


if __name__ == "__main__":
    # st.set_page_config(layout="wide")
    # show_across_langs()
    # st.markdown("---")
    # show_across_seeds()
    show_aggregated()
